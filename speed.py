#!/usr/bin/env python3
"""Read-out speed of one question at fixed prompt lengths (one two-way forward pass each).

    python speed.py --model ~/local-llms/glide/mlx-q8 --lengths 1024,8192,32768,40000 --repeats 3 --json speed-q8.json

The state is filler text cut to the token count that makes the whole prompt (chat template, question, options) the
target length. One warm-up run per length is not counted. Reports p50 / min seconds, prompt tokens per second and MLX peak
memory for that length.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FILLER = ("The quarterly report lists shipments, refunds, delays and customer messages from several regions. "
          "Some orders arrived late because of weather; others were refunded after quality complaints. ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="1024,8192,32768")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json")
    a = ap.parse_args()
    import mlx.core as mx
    from glide_mlx.model import GlideMLX
    from glide_mlx.prompt import options_for
    g = GlideMLX.from_pretrained(a.model, vision=False)
    q = {"type": "noul", "instructions": "Does the report mention late deliveries?"}
    opts = options_for(q)
    base = len(g.tokenizer(g.prompter.text("", q, opts), add_special_tokens=False)["input_ids"])
    filler_ids = g.tokenizer(FILLER * 4000, add_special_tokens=False)["input_ids"]
    out = {"model": a.model, "bits": (g.config.get("quantization") or {}).get("bits", 16), "load_s": round(g.load_s, 1), "runs": []}
    for L in [int(x) for x in a.lengths.split(",")]:
        n = max(1, L - base)
        state = g.tokenizer.decode(filler_ids[:n])
        ids = g.tokenizer(g.prompter.text(state, q, opts), add_special_tokens=False)["input_ids"]
        g.logits(ids)                                   # warm-up
        mx.reset_peak_memory()
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            g.logits(ids)
            ts.append(time.perf_counter() - t0)
        r = {"target": L, "tokens": len(ids), "p50_s": round(statistics.median(ts), 3), "min_s": round(min(ts), 3),
             "tok_per_s": round(len(ids) / statistics.median(ts)), "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
        out["runs"].append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
