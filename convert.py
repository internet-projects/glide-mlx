#!/usr/bin/env python3
"""Convert the original fastino/GLiDE folder (bf16) to an MLX 8- or 4-bit build.

    python convert.py --hf ~/local-llms/glide/hf --out ~/local-llms/glide/mlx-q8 --bits 8
    python convert.py --hf ~/local-llms/glide/hf --out ~/local-llms/glide/mlx-q4 --bits 4

Quantization: MLX affine, group size 64, on every Linear and the token embedding of the language model (the same set
mlx-vlm's default convert quantizes). Kept at their original precision: the vision encoder, all norms, the Gated DeltaNet
conv1d / A_log / dt_bias, and the 255-way readout (a separate 2.6 MB file).
Optional per-part overrides (16 = not quantized): --attn-bits for the 16 full-attention layers' projections,
--gate-bits for the DeltaNet in_proj_a / in_proj_b gate projections, --embed-bits for the token embedding.

The output folder is self-contained: weights, config.json (+ "quantization"), tokenizer, chat template, decision_config.json,
readout.safetensors, and Fastino's LICENSE / NOTICE / licenses/ (required by the Fastino Community License, section 1).
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COPY = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "decision_config.json", "readout.safetensors",
        "preprocessor_config.json", "generation_config.json", "LICENSE", "NOTICE", ".glide_revision"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="original fastino/GLiDE folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, required=True, choices=[4, 5, 6, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--attn-bits", type=int, default=None)
    ap.add_argument("--gate-bits", type=int, default=None)
    ap.add_argument("--embed-bits", type=int, default=None)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "config.json")):
        sys.exit(f"{a.out} already has a build; remove it first")

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.utils import save_weights
    from glide_mlx.model import load_network

    t0 = time.time()
    model, config, _ = load_network(a.hf, vision=True, lazy=True)

    def override(path):
        if a.embed_bits is not None and path.endswith("embed_tokens"):
            return a.embed_bits
        if a.attn_bits is not None and ".self_attn." in path:
            return a.attn_bits
        if a.gate_bits is not None and (path.endswith("in_proj_a") or path.endswith("in_proj_b")):
            return a.gate_bits
        return None

    per_module = {}

    def pred(path, m):
        if not path.startswith("language_model.") or not hasattr(m, "to_quantized"):
            return False
        if hasattr(m, "weight") and m.weight.shape[-1] % a.group_size != 0:
            return False
        b = override(path)
        if b is None:
            return True
        if b >= 16:
            per_module[path] = False
            return False
        per_module[path] = {"group_size": a.group_size, "bits": b}
        return per_module[path]

    nn.quantize(model, group_size=a.group_size, bits=a.bits, mode="affine", class_predicate=pred)
    os.makedirs(a.out, exist_ok=True)
    save_weights(a.out, model, donate_weights=True)
    cfg = dict(config)
    q = {"group_size": a.group_size, "bits": a.bits, "mode": "affine"}
    q.update({k: v for k, v in per_module.items() if v})
    cfg["quantization"] = q
    cfg["glide_mlx"] = {"source": "fastino/GLiDE", "revision": open(os.path.join(a.hf, ".glide_revision")).read().strip()
                        if os.path.exists(os.path.join(a.hf, ".glide_revision")) else None,
                        "bits": a.bits, "attn_bits": a.attn_bits, "gate_bits": a.gate_bits, "embed_bits": a.embed_bits,
                        "kept_original_precision": ["vision encoder", "norms", "DeltaNet conv1d/A_log/dt_bias", "readout"]
                        + [k for k, v in per_module.items() if v is False]}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(dict(sorted(cfg.items())), f, indent=2)
    for name in COPY:
        src = os.path.join(a.hf, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out, name))
    if os.path.isdir(os.path.join(a.hf, "licenses")):
        shutil.copytree(os.path.join(a.hf, "licenses"), os.path.join(a.out, "licenses"), dirs_exist_ok=True)
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.endswith(".safetensors")) / 1e9
    print(f"wrote {a.out}: {size:.2f} GB of weights in {time.time() - t0:.0f} s; quantization {json.dumps(q)}")


if __name__ == "__main__":
    main()
