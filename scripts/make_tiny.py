#!/usr/bin/env python3
"""Test fixture: a tiny random model with GLiDE's exact layout (qwen3_5, DeltaNet + full attention every 4th layer, vision
encoder, 255-way readout) written in GLiDE's checkpoint format, using GLiDE's real tokenizer, chat template and
decision_config. Lets the MLX port be checked against the upstream PyTorch code without the 52 GB download.

    python scripts/make_tiny.py --src <folder with GLiDE's config/tokenizer/decision_config/glide/*.py> --out /tmp/tiny-glide
"""
import argparse
import json
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from safetensors.torch import save_file
    sys.path.insert(0, a.src)
    from glide.model import _Network  # upstream code
    from transformers import AutoConfig

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    t = cfg["text_config"]
    t.update(hidden_size=128, intermediate_size=256, num_hidden_layers=8, layer_types=["linear_attention"] * 3 + ["full_attention"]
             + ["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4, num_key_value_heads=2, head_dim=32,
             linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
             mtp_num_hidden_layers=0)
    t["rope_parameters"]["mrope_section"] = [2, 1, 1]          # head_dim 32 * 0.25 = 8 rotary dims -> 4 frequency pairs
    v = cfg["vision_config"]
    v.update(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=128)
    os.makedirs(a.out, exist_ok=True)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "decision_config.json", "preprocessor_config.json"]:
        shutil.copy2(os.path.join(a.src, f), os.path.join(a.out, f))
    full = AutoConfig.from_pretrained(a.out)
    vc = full.vision_config
    vc.image_token_id = int(full.image_token_id)
    torch.manual_seed(a.seed)
    net = _Network(full.text_config, vc).float()
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n.endswith("A_log") or n.endswith("dt_bias"):
                continue
            if p.ndim == 1:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if "visual" in n and "norm" in n and n.endswith("weight") else 0.0))
            else:
                p.normal_(0, 0.08)
        net.readout.weight.normal_(0, 1.0)
    dt = getattr(torch, a.dtype)
    sd = {}
    for n, p in net.backbone.state_dict().items():
        if "rotary" in n or "inv_freq" in n:
            continue
        sd["model.language_model." + n] = p.to(dt).contiguous()
    for n, p in net.visual.state_dict().items():
        if "rotary" in n or "inv_freq" in n:
            continue
        sd["model.visual." + n] = p.to(dt).contiguous()
    save_file(sd, os.path.join(a.out, "model-00001-of-00001.safetensors"))
    json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"))
    save_file({"weight": net.readout.weight.to(dt).contiguous()}, os.path.join(a.out, "readout.safetensors"))
    print(f"tiny GLiDE-shaped model in {a.out}: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M params ({a.dtype})")


if __name__ == "__main__":
    main()
