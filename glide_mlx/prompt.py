# Copied unchanged from fastino/GLiDE @ d199b6d7 (glide/prompt.py), Fastino Community License (see LICENSE).
"""Prompt construction and request parsing.

Every question becomes one prompt:

    <chat template user turn>
    Context:
    <state>

    Question: <instructions>

    Options:
    (A) key: description
    (B) ...

    Think it through briefly, then answer with the option letter only.
    <assistant turn with an empty thought> Answer: (

The answer is read from the decision readout at the final "(" token; nothing is generated.
"""
from __future__ import annotations

import json

MAX_OPTIONS = 255
QUESTION_TYPES = ("choice", "noul", "score")
REASONING_EFFORT = "medium"
ANSWER_CUE = "Answer: ("
BODY_TAIL = "\n\nThink it through briefly, then answer with the option letter only."

IMAGE_TOKEN, VISION_START, VISION_END = "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"
IMAGE_PART_TYPES = ("image_url", "image", "input_image")
TEXT_PART_TYPES = ("text", "input_text")


class RequestError(ValueError):
    """A request the model cannot serve. The HTTP server answers these with status 422."""


def as_text(x):
    """Instructions that are not strings are rendered as readable text."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return "\n".join(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}" for k, v in x.items())
    return json.dumps(x, ensure_ascii=False)


def options_for(q):
    """Question -> [(option key, description)] in prompt order.

    choice: the criteria object's keys, in order; non-string descriptions are rendered as JSON.
    noul:   ("true", criteria.true or "Yes"), ("false", criteria.false or "No").
    score:  levels 0..n-1 from a list of descriptions, or from an object keyed by numeric level (ascending).
    """
    if not isinstance(q, dict):
        raise RequestError("each question must be an object")
    t = q.get("type")
    if t not in QUESTION_TYPES:
        raise RequestError(f"unsupported question type {t!r} (expected choice, noul or score)")
    crit = q.get("criteria") or {}
    if t == "noul":
        c = crit if isinstance(crit, dict) else {}
        return [("true", c.get("true") or "Yes"), ("false", c.get("false") or "No")]
    if t == "score":
        if isinstance(crit, list):
            levels = crit
        elif isinstance(crit, dict):
            try:
                levels = [crit[k] for k in sorted(crit, key=lambda x: float(x))]
            except (TypeError, ValueError):
                raise RequestError("score criteria keys must be numeric levels") from None
        else:
            raise RequestError("score criteria must be a list or an object keyed by numeric level")
        if not levels:
            raise RequestError("a score question needs at least one level")
        return [(str(i), str(d)) for i, d in enumerate(levels)]
    if not isinstance(crit, dict) or not crit:
        raise RequestError("a choice question needs a non-empty criteria object")
    return [(k, v if isinstance(v, str) and v else (json.dumps(v, ensure_ascii=False, separators=(", ", ": ")) if v not in (None, "") else k))
            for k, v in crit.items()]


def split_state(state):
    """System One multimodal state -> (text state, [image references]).

    A multimodal state is a list of parts: strings, {"type": "text", "text": ...} and
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} (also "image" / "input_image" with url / data).
    Any other state (string, object, list without image parts) is returned unchanged with no images.
    """
    if not isinstance(state, list) or not any(isinstance(p, dict) and p.get("type") in IMAGE_PART_TYPES for p in state):
        return state, []
    texts, images = [], []
    for p in state:
        if isinstance(p, dict) and p.get("type") in IMAGE_PART_TYPES:
            u = p.get("image_url", p.get("url", p.get("data", p.get("image"))))
            images.append(u.get("url") if isinstance(u, dict) else u)
        elif isinstance(p, dict) and p.get("type") in TEXT_PART_TYPES:
            texts.append(str(p.get("text", "")))
        elif isinstance(p, str):
            texts.append(p)
        else:
            texts.append(json.dumps(p, ensure_ascii=False))
    return (texts[0] if len(texts) == 1 else "\n".join(texts)), images


class Prompter:
    """Builds question prompts. The chat-template wrapper is rendered once and verified against a full template render on the
    first prompts, so the per-question cost is string concatenation."""

    def __init__(self, tokenizer, codes):
        self.tokenizer, self.codes = tokenizer, list(codes)
        mark = "\x00BODY\x00"
        chat = self._chat(mark)
        self.pre, post = chat.split(mark)
        self.post = post + self._opener(chat) + "\n</think>\n\n" + ANSWER_CUE
        self._checked = 0

    def _chat(self, body):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=True, reasoning_effort=REASONING_EFFORT)

    def _opener(self, chat):
        if chat.rstrip().endswith("<think>"):
            return ""
        return "<think>\n" if "<think>" in self.tokenizer.get_vocab() else ""

    def body(self, state_text, instructions, options):
        lines = [f"({c}) {k}" if d == k else f"({c}) {k}: {d}" for c, (k, d) in zip(self.codes, options)]
        text = f"Context:\n{state_text}\n\nQuestion: {instructions}\n\nOptions:\n" + "\n".join(lines)
        return text.rstrip() + BODY_TAIL

    def text(self, state, question, options, image_tokens=None):
        """Full prompt for one question. image_tokens: per-image visual token counts; each image becomes
        <|vision_start|><|image_pad|> x n<|vision_end|>, all placed before the state text."""
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        if image_tokens:
            st = "".join(VISION_START + IMAGE_TOKEN * int(n) + VISION_END for n in image_tokens) + "\n" + st
        body = self.body(st, as_text(question.get("instructions", "")), options)
        out = self.pre + body + self.post
        if self._checked < 3:
            chat = self._chat(body)
            if out != chat + self._opener(chat) + "\n</think>\n\n" + ANSWER_CUE:
                raise RuntimeError("cached chat-template wrapper differs from a full template render")
            self._checked += 1
        return out
