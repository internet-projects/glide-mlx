# GLiDE on Apple Silicon (MLX)

**License:** [Fastino Community License](LICENSE) (GLiDE and this port; see [`NOTICE`](NOTICE)) · the Qwen3.8-27B base keeps its [Apache-2.0 license](licenses/Qwen3.8-27B-LICENSE). Commercial use requires "Built with Fastino GLiDE" (section 4); see [License](#license).

An MLX port of [fastino/GLiDE](https://huggingface.co/fastino/GLiDE), Fastino's decision model built on Qwen3.8-27B. Give it a
state (text, JSON or images) and typed questions (`choice`, `noul` for yes / no, `score`) and it returns a probability for every
option, in one forward pass per question.

**Built with Fastino GLiDE.** Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with Fastino.

| Build | Size | Same top answer as Fastino's code | JevBench public (231) |
|---|---|---|---|
| [netprojects/GLiDE-mlx-8bit](https://huggingface.co/netprojects/GLiDE-mlx-8bit) (recommended) | 28.2 GB | 202 / 204 | **215 / 231** |
| [netprojects/GLiDE-mlx-4bit](https://huggingface.co/netprojects/GLiDE-mlx-4bit) | 15.3 GB | 197 / 204 | 213 / 231 |
| original weights ([fastino/GLiDE](https://huggingface.co/fastino/GLiDE), bf16) on MLX | 52 GB | 202 / 204 | — |

Measured on a MacBook Pro M5 Max (128 GB), 29 October 2026, GLiDE v1.2 (revision `d199b6d`). "Same top answer" compares each
build with Fastino's own PyTorch code (bf16) on 72 requests / 204 questions (text, JSON and 12 image requests; all three question
types); the 8-bit build also gives the same top answer as the original weights on MLX on 204 / 204. JevBench v1.2.1, public set,
`--adapter typesafe`: 8-bit 48/48 · 72/72 · 95/111, 4-bit 48/48 · 72/72 · 93/111 (easy · standard · hard); median 0.74 s (8-bit)
and 0.45 s (4-bit) per item. Details in [RESULTS.md](RESULTS.md).

## How it works

- **Same reading as GLiDE:** the 16 full-attention layers see the whole prompt (no causal mask); the 48 Gated DeltaNet layers keep
  their left-to-right recurrence (`noncausal_full_attention`). The port runs mlx-vlm's `qwen3_5` network and switches off the
  causal mask of the full-attention layers only while a GLiDE forward pass runs.
- **Same answer head:** the last token's hidden state goes through GLiDE's 255-way readout (`readout.safetensors`, kept at its
  original precision and applied in it, as the original does); softmax over the question's options at the temperature in
  `decision_config.json`; `choice`, `noul` and `score` answers with the same fields.
- **Same prompts and images:** `glide_mlx/prompt.py` and `glide_mlx/images.py` are GLiDE's own files, unchanged; images go
  through the vision encoder (kept at its original precision) with GLiDE's multimodal positions.
- **Same API:** `POST /v1/systemone` and `GET /health`, as GLiDE's server.
- **Memory:** the MLX buffer cache is capped (`GLIDE_CACHE_LIMIT_GB`, default 2) and cleared after each request;
  `GLIDE_MEMORY_LIMIT_GB` caps MLX memory if set.

## Use

```bash
git clone https://github.com/internet-projects/glide-mlx && cd glide-mlx
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
hf download netprojects/GLiDE-mlx-8bit --local-dir glide-mlx-8bit
.venv/bin/python -m glide_mlx.server --model glide-mlx-8bit --port 8790     # http://127.0.0.1:8790/v1/systemone
curl -s localhost:8790/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Order 123 arrived broken and the customer wants their money back.",
  "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
                           "criteria": {"refund": "A refund", "exchange": "An exchange", "info": "Information"}},
                "angry": {"type": "noul", "instructions": "Is the customer angry?"}}}'
```

Python: `from glide_mlx import GlideMLX; g = GlideMLX.from_pretrained("glide-mlx-8bit"); g.decide(state, questions)`.
Server settings: `GLIDE_API_KEY` (bearer key), `GLIDE_MAX_LEN` (40,000), `GLIDE_VISION` (1), `GLIDE_REQUEST_TIMEOUT_S` (600),
`GLIDE_MIN_PIXELS` / `GLIDE_MAX_PIXELS` (as in GLiDE).

## Build and check it yourself

```bash
hf download fastino/GLiDE --local-dir glide-hf                                  # original weights, bf16, 52 GB
.venv/bin/python convert.py --hf glide-hf --out glide-mlx-8bit --bits 8          # or --bits 4
.venv/bin/python parity.py cases --out cases.jsonl
.venv/bin/python parity.py ref --model glide-hf --cases cases.jsonl --out ref.jsonl --device mps   # Fastino's PyTorch code
.venv/bin/python parity.py mlx --model glide-mlx-8bit --cases cases.jsonl --out q8.jsonl
.venv/bin/python parity.py compare ref.jsonl q8.jsonl
tests/test_tiny.sh glide-hf      # tiny random model with GLiDE's layout vs Fastino's code, no 52 GB load
```

Run the reference and the MLX side one after the other: the bf16 reference alone needs about 55 GB.

## License

GLiDE and its Derivative Works, including these conversions and this repository (which contains GLiDE's `prompt.py` and
`images.py`), are under the **Fastino Community License** (`LICENSE`, `NOTICE`, `licenses/`). Commercial use requires displaying
"Built with Fastino GLiDE" (section 4); production commercial use by an entity with annual revenue above USD 10 million requires a
separate agreement with Fastino (sections 2–3). Read the license for the full terms.

Credits: Fastino (GLiDE), Qwen team (Qwen3.8-27B, Apache-2.0), Apple MLX and mlx-vlm. Port by Paris Michalas, Partner at
[NetProjects](https://netprojects.com). Not affiliated with Fastino.
