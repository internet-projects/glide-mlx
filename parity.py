#!/usr/bin/env python3
"""Parity between the upstream PyTorch GLiDE and this MLX port, request by request.

    python parity.py cases --out cases.jsonl [--di-rows sample.jsonl.gz --di-n 300]   # build the test requests
    python parity.py ref   --model ~/local-llms/glide/hf --cases cases.jsonl --out ref.jsonl --device mps   # upstream code
    python parity.py mlx   --model ~/local-llms/glide/mlx-q8 --cases cases.jsonl --out q8.jsonl                # this port
    python parity.py compare ref.jsonl q8.jsonl [--json out.json]

Each side runs in its own process (the bf16 reference alone needs about 55 GB). The reference is fastino's own Glide class,
imported from the model folder's glide/ package and run one request at a time (no micro-batch padding). Compared per
question: the top answer, the probability of every option (noul: P(true) and P(false)) and KL(ref || mlx).
"""
import argparse
import base64
import gzip
import io
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------------------------------ cases
WORDS = ("market price order shipment customer refund invoice delay quality support account login error payment travel "
         "flight hotel match team score league goal player injury weather rain sunny report risk claim evidence policy").split()


def _img(kind, seed):
    from PIL import Image, ImageDraw
    rnd = random.Random(seed)
    w, h = rnd.choice([(320, 240), (512, 512), (640, 360), (300, 900)])
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    color = rnd.choice(["red", "green", "blue", "orange"])
    if kind == "shape":
        x0, y0 = rnd.randint(0, w // 3), rnd.randint(0, h // 3)
        (d.ellipse if rnd.random() < 0.5 else d.rectangle)([x0, y0, x0 + w // 2, y0 + h // 2], fill=color)
    else:
        for i in range(6):
            d.text((10, 10 + i * 18), f"ITEM {i + 1} .... {rnd.randint(1, 99)}.{rnd.randint(10, 99)}", fill="black")
        d.text((10, 10 + 7 * 18), f"TOTAL {rnd.randint(100, 999)}.00", fill=color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return color, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def builtin_cases(n_text=60, n_img=12, seed=7):
    rnd = random.Random(seed)
    out = []
    for i in range(n_text):
        nwords = rnd.choice([30, 120, 400, 1500])
        state = " ".join(rnd.choice(WORDS) for _ in range(nwords))
        if i % 5 == 0:
            state = {"ticket": state[:300], "priority_hint": rnd.choice(["low", "high"]), "amount": rnd.randint(1, 900)}
        k = rnd.choice([2, 3, 5, 12, 40])
        crit = {f"opt_{j}_{rnd.choice(WORDS)}": f"The text is mostly about {rnd.choice(WORDS)} and {rnd.choice(WORDS)}" for j in range(k)}
        out.append({"id": f"text-{i}", "state": state, "questions": {
            "topic": {"type": "choice", "instructions": "Which option fits the context best?", "criteria": crit},
            "urgent": {"type": "noul", "instructions": "Is this urgent?", "criteria": {"true": "Needs action today", "false": "Can wait"}},
            "quality": {"type": "score", "instructions": "How well written is it?", "criteria": ["poor", "fair", "good", "excellent"]}}})
    for i in range(n_img):
        kind = "shape" if i % 2 == 0 else "receipt"
        color, url = _img(kind, seed * 100 + i)
        q = {"color": {"type": "choice", "instructions": "What color is the " + ("large shape" if kind == "shape" else "TOTAL line") + "?",
                       "criteria": {"red": "red", "green": "green", "blue": "blue", "orange": "orange"}},
             "receipt": {"type": "noul", "instructions": "Is this a receipt?"}}
        out.append({"id": f"image-{i}", "expect": {"color": color},
                    "state": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": f"Picture {i}."}],
                    "questions": q})
    return out


def di_cases(path, n, seed=11):
    """Requests from a Decision Index sample file (rows with state + questions), stratified by benchmark."""
    rows = [json.loads(l) for l in gzip.open(path, "rt")] if path.endswith(".gz") else [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        req = r.get("request") or r.get("payload") or r
        if not isinstance(req, dict) or "state" not in req or "questions" not in req:
            continue
        by.setdefault(r.get("benchmark") or r.get("dataset") or "?", []).append((r, req))
    rnd = random.Random(seed)
    per = max(1, n // max(1, len(by)))
    out = []
    for b, items in sorted(by.items()):
        for r, req in rnd.sample(items, min(per, len(items))):
            out.append({"id": f"di-{b}-{r.get('run_id') or r.get('id') or len(out)}", "state": req["state"], "questions": req["questions"]})
    return out[:n] if n else out


# ------------------------------------------------------------------------------------------------------------------- runs
def _probs(ans):
    if ans.get("type") == "noul":
        return {"true": ans["noul"], "false": 1 - ans["noul"]}
    return ans.get("probabilities") or {}


def run(engine, cases, out_path, limit=None):
    done = set()
    if os.path.exists(out_path):                 # resume: keep answered requests; errors and a torn last line are redone
        good = []
        for l in open(out_path):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if "answers" in r:
                good.append(l if l.endswith("\n") else l + "\n")
                done.add(r["id"])
        open(out_path, "w").writelines(good)
    with open(out_path, "a") as f:
        for c in cases[:limit] if limit else cases:
            if c["id"] in done:
                continue
            t0 = time.perf_counter()
            r = engine([{"state": c["state"], "questions": c["questions"]}])[0]
            dt = time.perf_counter() - t0
            if isinstance(r, Exception):
                rec = {"id": c["id"], "error": f"{type(r).__name__}: {r}", "s": dt}
            else:
                rec = {"id": c["id"], "s": dt, "tokens": r["usage"]["input_tokens"],
                       "answers": {k: {"type": a["type"], "top": a.get("choice", a.get("score", a.get("noul") >= 0.5 if a["type"] == "noul" else None)),
                                       "p": _probs(a)} for k, a in r["answers"].items()}}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{c['id']}: {dt:.2f}s {rec.get('tokens', '')} {rec.get('error', '')}", flush=True)


def load_cases(path):
    return [json.loads(l) for l in open(path)]


def cmd_ref(a):
    import torch
    sys.path.insert(0, a.model)                     # the model folder ships fastino's glide/ package
    from glide.model import Glide
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[a.dtype]
    g = Glide.from_pretrained(a.model, device=a.device, dtype=dtype, token_budget=10 ** 9)
    run(g.decide_batch, load_cases(a.cases), a.out, a.limit)


def cmd_mlx(a):
    sys.path.insert(0, HERE)
    from glide_mlx.model import GlideMLX
    g = GlideMLX.from_pretrained(a.model)
    print(f"loaded in {g.load_s:.0f} s", flush=True)
    run(g.decide_batch, load_cases(a.cases), a.out, a.limit)


def compare(ref_path, test_path):
    ref = {json.loads(l)["id"]: json.loads(l) for l in open(ref_path)}
    tst = {json.loads(l)["id"]: json.loads(l) for l in open(test_path)}
    n_ref = sum(len(r.get("answers") or {}) for r in ref.values())
    n = agree = 0
    dmax, dsum, dcount, kls, flips = 0.0, 0.0, 0, [], []
    errors = [i for i in tst if "error" in tst[i]] + [i for i in ref if "error" in ref[i]]
    by_type = {}
    for i, r in ref.items():
        t = tst.get(i)
        if not t or "answers" not in t or "answers" not in r:
            continue
        for k, ra in r["answers"].items():
            ta = t["answers"].get(k)
            if ta is None:
                continue
            n += 1
            same = ra["top"] == ta["top"]
            agree += same
            bt = by_type.setdefault(ra["type"], [0, 0])
            bt[0] += 1
            bt[1] += same
            kl = 0.0
            for o, p in ra["p"].items():
                q = ta["p"].get(o, 0.0)
                d = abs(p - q)
                dmax, dsum, dcount = max(dmax, d), dsum + d, dcount + 1
                if p > 0:
                    kl += p * (math.log(p) - math.log(max(q, 1e-12)))
            kls.append(kl)
            if not same:
                top2 = sorted(ra["p"].values())[-2:] if len(ra["p"]) > 1 else [0, 1]
                flips.append({"id": i, "q": k, "ref": ra["top"], "test": ta["top"], "ref_margin": round(top2[-1] - top2[0], 4)})
    kls.sort()
    res = {"questions_ref": n_ref, "questions": n, "missing_in_test": n_ref - n, "same_top": agree, "same_top_pct": round(100 * agree / max(n, 1), 2),
           "by_type": {k: f"{v[1]}/{v[0]}" for k, v in by_type.items()},
           "prob_abs_diff_mean": round(dsum / max(dcount, 1), 5), "prob_abs_diff_max": round(dmax, 5),
           "kl_mean": round(sum(kls) / max(len(kls), 1), 6), "kl_p95": round(kls[int(0.95 * (len(kls) - 1))], 6) if kls else None,
           "flips": flips, "errors": errors}
    ref_s = [r["s"] for r in ref.values() if "s" in r]
    tst_s = [r["s"] for r in tst.values() if "s" in r]
    res["seconds_ref_mlx"] = [round(sum(ref_s), 1), round(sum(tst_s), 1)]
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--out", required=True)
    c.add_argument("--di-rows")
    c.add_argument("--di-n", type=int, default=300)
    c.add_argument("--n-text", type=int, default=60)
    c.add_argument("--n-img", type=int, default=12)
    for name in ("ref", "mlx"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--cases", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int)
        if name == "ref":
            p.add_argument("--device", default="mps")
            p.add_argument("--dtype", default="bfloat16")
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("test")
    p.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "cases":
        cases = builtin_cases(a.n_text, a.n_img)
        if a.di_rows:
            cases += di_cases(a.di_rows, a.di_n)
        with open(a.out, "w") as f:
            for x in cases:
                f.write(json.dumps(x) + "\n")
        print(f"{len(cases)} requests -> {a.out}")
    elif a.cmd == "ref":
        cmd_ref(a)
    elif a.cmd == "mlx":
        cmd_mlx(a)
    else:
        res = compare(a.ref, a.test)
        print(json.dumps({k: v for k, v in res.items() if k != "flips"}, indent=1))
        for f in res["flips"][:20]:
            print("flip:", f)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
