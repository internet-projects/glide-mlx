# GLiDE on Apple Silicon (MLX)

An MLX port of [fastino/GLiDE](https://huggingface.co/fastino/GLiDE), Fastino's typed decision model (Qwen3.8-27B based):
give it a state (text, JSON or images) and typed questions (`choice`, `noul`, `score`) and it returns a probability for every
option, in one forward pass per question. Not affiliated with Fastino.

**Built with Fastino GLiDE.** GLiDE and this port are licensed under the Fastino Community License (see `LICENSE`, `NOTICE`,
`licenses/`); commercial use requires the attribution in section 4, and entities above the revenue threshold in section 2 need
a commercial agreement with Fastino.

## What the port does

- Same reading as upstream: the 16 full-attention layers see the whole prompt (no causal mask), the 48 Gated DeltaNet layers
  keep their left-to-right recurrence; the last token's hidden state goes through GLiDE's 255-way readout (kept at its original
  precision, applied in float32); softmax over the question's options at the temperature in `decision_config.json`.
- Same prompts and image preprocessing: `glide_mlx/prompt.py` and `glide_mlx/images.py` are GLiDE's own files, unchanged.
- The network is mlx-vlm's `qwen3_5` model; the vision encoder runs on MLX in its original precision.
- Same HTTP API as upstream: `POST /v1/systemone`, `GET /health`.

## Use

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
hf download fastino/GLiDE --local-dir glide-hf                     # 52 GB, bf16
.venv/bin/python convert.py --hf glide-hf --out glide-mlx-q8 --bits 8  # or --bits 4
.venv/bin/python -m glide_mlx.server --model glide-mlx-q8 --port 8790
curl -s localhost:8790/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Order 123 arrived broken and the customer wants their money back.",
  "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
                           "criteria": {"refund": "A refund", "exchange": "An exchange", "info": "Information"}},
                "angry": {"type": "noul", "instructions": "Is the customer angry?"}}}'
```

Python: `from glide_mlx import GlideMLX; g = GlideMLX.from_pretrained("glide-mlx-q8"); g.decide(state, questions)`.
The original bf16 folder loads directly too (`--model glide-hf`).

## Checks

- `parity.py`: runs the same requests through Fastino's PyTorch code and through this port and compares top answers,
  probabilities and KL.
- `speed.py`: one question at 1K / 8K / 32K prompt tokens.
- `scripts/make_tiny.py` + `tests/`: a tiny random model with GLiDE's exact layout, for checking the port against the upstream
  code without the 52 GB download.
