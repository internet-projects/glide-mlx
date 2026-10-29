"""GLiDE on MLX: loading, the two-way forward pass and the decision readout.

Mirrors fastino/GLiDE glide/model.py (PyTorch) on Apple Silicon:
  * one forward pass per question prompt, nothing generated, no KV cache;
  * full-attention layers attend over the whole prompt (no causal mask); linear-attention (Gated DeltaNet) layers keep their
    left-to-right recurrence, exactly as upstream ("noncausal_full_attention");
  * the final-normed hidden state of the last token (the "(" of "Answer: (") goes through the 255-way readout
    (readout.safetensors, kept in its original precision and applied in it, as upstream); a softmax over the question's options,
    divided by the temperature in decision_config.json, gives the probabilities;
  * images go through the vision encoder and are placed into the same forward pass with multimodal rotary positions.

The network is mlx-vlm's qwen3_5 model. Prompts, option handling and image preprocessing are GLiDE's own code (prompt.py and
images.py, copied unchanged), so the token ids fed to MLX are the ones the PyTorch reference sees.

Loads either the original Hugging Face folder (bf16, sanitized on the fly) or a folder written by convert.py (8- or 4-bit).
"""
from __future__ import annotations

import collections
import contextlib
import json
import math
import os
import threading
import time

import numpy as np

from .images import image_inputs, load_image
from .prompt import MAX_OPTIONS, Prompter, RequestError, options_for, split_state

DECISION_CONFIG = "decision_config.json"
READOUT_FILE = "readout.safetensors"
DEFAULT_NAME = "glide-v1.2"
DEFAULT_MAX_LEN = 40000

_lock = threading.Lock()          # mlx-vlm's mask helper is patched process-wide while a GLiDE forward runs
_two_way = threading.local()


def _install_two_way_patch():
    """While a GLiDE forward runs, the full-attention layers get no mask (they see every token); otherwise mlx-vlm's own
    mask is used, so other qwen3_5 models in the same process are unaffected."""
    from mlx_vlm.models.qwen3_5 import language as L
    if getattr(L, "_glide_patched", False):
        return
    orig = L._create_qwen3_5_attention_mask

    def mask(h, cache):
        if getattr(_two_way, "on", False):
            if cache is not None:
                raise RuntimeError("GLiDE runs without a KV cache")
            return None
        return orig(h, cache)

    L._create_qwen3_5_attention_mask = mask
    L._glide_patched = True


@contextlib.contextmanager
def two_way():
    _two_way.on = True
    try:
        yield
    finally:
        _two_way.on = False


def set_memory_limits():
    """Keep MLX from holding freed buffers: each prompt length allocates differently sized activations, and with MLX's
    default cache limit (as large as the memory limit) those pile up into tens of GB over a benchmark run.
    GLIDE_CACHE_LIMIT_GB [2] caps the buffer cache; GLIDE_MEMORY_LIMIT_GB (unset = MLX default) caps total MLX memory
    (allocations above it wait for running work and then fail instead of pushing macOS into swap)."""
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("GLIDE_CACHE_LIMIT_GB", "2")) * 1e9))
    if os.environ.get("GLIDE_MEMORY_LIMIT_GB"):
        mx.set_memory_limit(int(float(os.environ["GLIDE_MEMORY_LIMIT_GB"]) * 1e9))


def mrope_positions(ids, grids, image_token_id, merge):
    """[3, 1, T] rotary positions (t, h, w) for one unpadded sequence; same rule as upstream mrope_positions: text tokens
    advance all three axes together; an image's merged tokens get (start, start + row, start + col) and the following text
    starts at start + max(rows, cols)."""
    n = len(ids)
    pos = np.zeros((3, n), dtype=np.int32)
    gi, cur, i = 0, 0, 0
    while i < n:
        if ids[i] == image_token_id:
            t, gh, gw = grids[gi]
            gi += 1
            lh, lw = gh // merge, gw // merge
            cnt = t * lh * lw
            tt = np.repeat(np.arange(t), lh * lw)
            hh = np.tile(np.repeat(np.arange(lh), lw), t)
            ww = np.tile(np.arange(lw), t * lh)
            pos[:, i:i + cnt] = np.stack([tt, hh, ww]) + cur
            cur += max(lh, lw)
            i += cnt
        else:
            j = i
            while j < n and ids[j] != image_token_id:
                j += 1
            pos[:, i:j] = np.arange(cur, cur + j - i)
            cur += j - i
            i = j
    if gi != len(grids):
        raise ValueError(f"{len(grids)} images but {gi} image-token runs")
    return pos[:, None, :]


def _softmax(logits, n, temperature):
    v = [x / temperature for x in logits[:n]]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    z = sum(e)
    return [x / z for x in e]


def _answer(qtype, keys, probs):
    """Same fields and formulas as upstream _answer."""
    pmap = dict(zip(keys, probs))
    top = max(pmap, key=pmap.get)
    conf = pmap[top] - sorted(pmap.values())[-2] if len(pmap) > 1 else pmap[top]
    if qtype == "noul":
        return {"type": "noul", "noul": pmap["true"], "confidence": abs(pmap["true"] - 0.5) * 2}
    if qtype == "score":
        return {"type": "score", "score": int(top), "expected_level": sum(float(k) * p for k, p in pmap.items()), "confidence": conf,
                "probabilities": pmap}
    return {"type": "choice", "choice": top, "confidence": conf, "probabilities": pmap}


def _read_local_image(x):
    if isinstance(x, os.PathLike) or (isinstance(x, str) and not x.startswith("data:") and len(x) < 4096 and os.path.isfile(x)):
        with open(x, "rb") as f:
            return f.read()
    return x


def _check_codes(tokenizer, config):
    codes, ids = config.get("codes") or [], config.get("token_ids") or []
    if len(codes) != MAX_OPTIONS or len(ids) != MAX_OPTIONS:
        raise ValueError(f"{DECISION_CONFIG} must list {MAX_OPTIONS} codes and token ids")
    for c, i in zip(codes, ids):
        if tokenizer.encode(c, add_special_tokens=False) != [i]:
            raise ValueError(f"label code {c!r} does not match the tokenizer")


def load_network(path, vision=True, lazy=False):
    """-> (mlx-vlm qwen3_5 Model without lm_head, raw config dict, readout weight [255, hidden] as mx.array)."""
    import glob
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.utils import get_model_and_args, update_module_configs

    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("text_config", {})
    cfg.setdefault("vision_config", {})
    cfg.setdefault("audio_config", {})
    cfg["text_config"]["tie_word_embeddings"] = True      # GLiDE has no lm_head: build none (logits are never computed)
    arch, _ = get_model_and_args(config=cfg)
    mc = update_module_configs(arch.ModelConfig.from_dict(cfg), arch, cfg, ["text", "vision", "perceiver", "projector", "audio"])
    model = arch.Model(mc)

    files = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {path}")
    weights = {}
    for f in files:
        weights.update(mx.load(f))
    if any(k.startswith("model.") for k in weights):          # original Hugging Face checkpoint
        weights = model.sanitize(weights)
    if not vision or not any(k.startswith("vision_tower.") for k in weights):
        weights = {k: v for k, v in weights.items() if not k.startswith("vision_tower.")}
        model.vision_tower = None
    if (q := config.get("quantization")) is not None:
        def pred(p, m):
            if p in q:
                return q[p]
            return f"{p}.scales" in weights
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"), class_predicate=pred)
    model.load_weights(list(weights.items()), strict=True)
    if not lazy:
        mx.eval(model.parameters())
    readout = mx.load(os.path.join(path, READOUT_FILE))["weight"]
    mx.eval(readout)
    return model, config, readout


class GlideMLX:
    """Decision model on MLX. GlideMLX.from_pretrained(path); decide() / decide_batch() as upstream Glide."""

    def __init__(self, model, config, readout, tokenizer, decision_config, max_len=DEFAULT_MAX_LEN, name=DEFAULT_NAME):
        import mlx.core as mx
        self.mx = mx
        self.model, self.config, self.tokenizer, self.dconf = model, config, tokenizer, decision_config
        # upstream applies the readout as a Linear in the model dtype (bf16): same here, so near-ties round the same way;
        # GLIDE_READOUT_FP32=1 applies it in float32 instead
        self.readout_fp32 = os.environ.get("GLIDE_READOUT_FP32", "0") == "1"
        self.readout_t = (readout.astype(mx.float32) if self.readout_fp32 else readout).T   # [hidden, 255]
        mx.eval(self.readout_t)
        self.max_len, self.name = int(max_len), name
        self.codes = list(decision_config["codes"])
        self.temperature = float(decision_config.get("temperature", 1.0))
        if decision_config.get("attention_mode", "noncausal_full_attention") != "noncausal_full_attention":
            raise ValueError("unsupported attention_mode in decision_config.json")
        if decision_config.get("pooling", "last") != "last" or decision_config.get("prompt_format", "glide-v1") != "glide-v1":
            raise ValueError("unsupported pooling / prompt_format in decision_config.json")
        self.prompter = Prompter(tokenizer, self.codes)
        self.vision = model.vision_tower is not None
        vc = config.get("vision_config") or {}
        self.image_token_id = int(config.get("image_token_id", 248056))
        self._patch = (int(vc.get("patch_size", 16)), int(vc.get("temporal_patch_size", 2)), int(vc.get("spatial_merge_size", 2)))
        _install_two_way_patch()
        set_memory_limits()

    @classmethod
    def from_pretrained(cls, path, vision=True, max_len=DEFAULT_MAX_LEN, name=DEFAULT_NAME, lazy=False):
        from transformers import AutoTokenizer
        t0 = time.time()
        with open(os.path.join(path, DECISION_CONFIG)) as f:
            dconf = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(path)
        _check_codes(tokenizer, dconf)
        model, config, readout = load_network(path, vision=vision, lazy=lazy)
        g = cls(model, config, readout, tokenizer, dconf, max_len=max_len, name=name)
        g.load_s = time.time() - t0
        return g

    # ------------------------------------------------------------------------------------------------- public API (upstream)
    def decide(self, state, questions, images=None):
        req = {"state": state, "questions": questions}
        if images:
            req["images"] = [_read_local_image(x) for x in images]
        r = self.decide_batch([req])[0]
        if isinstance(r, RequestError):
            raise r
        return r

    def decide_batch(self, requests):
        """One item per request: a response dict or a RequestError instance (not raised). Questions run one at a time."""
        results = []
        for req in requests:
            try:
                entries, vis = self._prepare(req)
            except RequestError as e:
                results.append(e)
                continue
            ids = self.tokenizer([e[3] for e in entries], add_special_tokens=False)["input_ids"] if entries else []
            if any(len(x) + 8 > self.max_len for x in ids):
                results.append(RequestError("prompt exceeds the maximum context length"))
                continue
            answers = {}
            feats = self._image_features(vis) if vis else None
            for (qk, qtype, keys, _), x in zip(entries, ids):
                row = self.logits(x, vis, feats)
                answers[qk] = _answer(qtype, keys, _softmax(row, len(keys), self.temperature))
            results.append({"model": self.name, "answers": answers, "usage": {"input_tokens": sum(len(x) for x in ids), "output_tokens": 0}})
            self.mx.clear_cache()                # every prompt length allocates new buffer sizes: do not keep them
        return results

    # ------------------------------------------------------------------------------------------------------------- internals
    def _prepare(self, req):
        """Same checks and prompt text as upstream Glide._prepare."""
        if not isinstance(req, dict):
            raise RequestError("request must be an object")
        questions = req.get("questions")
        if not isinstance(questions, dict):
            raise RequestError("questions must be an object mapping keys to questions")
        state, refs = split_state(req.get("state"))
        refs = list(refs) + list(req.get("images") or [])
        vis = None
        if refs:
            if not self.vision:
                raise RequestError("images are not supported by this deployment")
            patch, temporal, merge = self._patch
            try:
                vis = [image_inputs(load_image(u), patch=patch, temporal=temporal, merge=merge) for u in refs]
            except Exception as e:  # noqa: BLE001
                raise RequestError(f"invalid image: {type(e).__name__}: {str(e)[:120]}") from None
        ntok = [v[2] for v in vis] if vis else None
        entries = []
        for qk, q in questions.items():
            opts = options_for(q)
            if len(opts) > MAX_OPTIONS:
                raise RequestError(f"options per choice exceeds the label set ({len(opts)} > {MAX_OPTIONS})")
            entries.append((qk, q["type"], [key for key, _ in opts], self.prompter.text(state, q, opts, ntok)))
        return entries, vis

    def _image_features(self, vis):
        """Vision encoder output for a request's images (computed once, shared by its questions)."""
        mx = self.mx
        pv = mx.array(np.concatenate([p for p, _, _ in vis]))
        grid = mx.array(np.array([g for _, g, _ in vis], dtype=np.int32))
        dtype = self.model.vision_tower.patch_embed.proj.weight.dtype
        feats, _ = self.model.vision_tower(pv.astype(dtype), grid)
        mx.eval(feats)
        return feats

    def hidden_last(self, ids, vis=None, feats=None):
        """Final-normed hidden state of the last token, [hidden] in the model dtype (one two-way forward pass)."""
        mx = self.mx
        lm = self.model.language_model
        x = mx.array(np.asarray(ids, dtype=np.int32))[None]
        merge = self._patch[2]
        grids = [g for _, g, _ in vis] if vis else []
        if vis:
            pos = mx.array(mrope_positions(list(ids), grids, self.image_token_id, merge))
        else:                                    # text only: plain positions on all three axes (as upstream's text path)
            pos = mx.array(np.broadcast_to(np.arange(len(ids), dtype=np.int32), (3, 1, len(ids))).copy())
        emb = lm.model.embed_tokens(x)
        if vis:
            m = np.asarray(ids) == self.image_token_id
            if int(m.sum()) != feats.shape[0]:
                raise ValueError(f"image tokens {int(m.sum())} != image features {feats.shape[0]}")
            idx = mx.array(np.cumsum(m) - 1)
            emb = mx.where(mx.array(m)[None, :, None], feats.astype(emb.dtype)[idx][None], emb)
        with _lock, two_way():
            out = lm(x, inputs_embeds=emb, cache=None, position_ids=pos, return_hidden=True, skip_logits=True)
            h = out.hidden_states[-1][0, -1]
            mx.eval(h)
        return h

    def logits(self, ids, vis=None, feats=None):
        """[255] readout logits for one prompt (python floats)."""
        mx = self.mx
        if vis and feats is None:
            feats = self._image_features(vis)
        h = self.hidden_last(ids, vis, feats)
        if self.readout_fp32:
            z = h.astype(mx.float32) @ self.readout_t
        else:
            z = (h.astype(self.readout_t.dtype) @ self.readout_t).astype(mx.float32)
        mx.eval(z)
        return z.tolist()
