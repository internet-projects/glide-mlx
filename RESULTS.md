# Results

MacBook Pro M5 Max, 128 GB, macOS; mlx 0.32.4, mlx-vlm 0.7.11, transformers 5.17.0, torch 2.14.0 (reference on MPS).
GLiDE v1.2, `fastino/GLiDE` revision `d199b6d7e04523d87986b8b68cd62272ed1cefc0`. 29 October 2026.

## Same answers as Fastino's code

`parity.py`: 72 requests, 204 questions (72 `choice` with 2–40 options, 72 `noul`, 60 `score`); states of 30–1,500 words of text
or JSON, and 12 image requests (shapes and receipts). Reference: Fastino's own `Glide` class, bf16 on MPS, one request at a time.

| vs Fastino's PyTorch code (bf16) | same top answer | probability difference mean / max | KL mean |
|---|---|---|---|
| original weights on MLX (bf16) | 202 / 204 | 0.0016 / 0.025 | 0.00016 |
| 8-bit | 202 / 204 | 0.0020 / 0.034 | 0.00023 |
| 4-bit | 197 / 204 | 0.0115 / 0.124 | 0.0064 |

8-bit vs the original weights on MLX: 204 / 204 same top answer. All differing answers are `choice` questions; `noul` and
`score` answers are identical in every build. The two answers where MLX bf16 and 8-bit differ from the reference are near-ties in
the reference (top-two margin 0.018 and 0.000).

## JevBench public

JevBench v1.2.1, public set (231 items), `--adapter typesafe`, through `glide_mlx.server`.

| build | total | easy | standard | hard | median time per item |
|---|---|---|---|---|---|
| 8-bit | **215 / 231** | 48 / 48 | 72 / 72 | 95 / 111 | 0.74 s |
| 4-bit | 213 / 231 | 48 / 48 | 72 / 72 | 93 / 111 | 0.45 s |

## Build sizes

| build | weights |
|---|---|
| 8-bit (affine, group size 64) | 28.2 GB |
| 4-bit (affine, group size 64) | 15.3 GB |
| original (bf16) | 52.2 GB |

Kept at original precision in both builds: the vision encoder, norms, DeltaNet `conv1d` / `A_log` / `dt_bias`, and the readout.
