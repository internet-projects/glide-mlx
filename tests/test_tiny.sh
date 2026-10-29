#!/bin/bash
# Port check without the 52 GB download: a tiny random model with GLiDE's layout (float32), run through Fastino's PyTorch
# code and through this port on the same requests (text, JSON, images; choice / noul / score). On CPU the difference is
# about 1e-5; on the Mac GPU float32 rounding differs a little and the random model is very sensitive, so the check allows
# small drift: >= 95% same top answer, max probability difference < 0.05, and flips only on near-ties (margin < 0.02).   tests/test_tiny.sh <folder with fastino's glide/ package, config,
# tokenizer and decision_config (e.g. the downloaded model folder)>
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="$1"; PY="${PY:-.venv/bin/python}"; T="$(mktemp -d)"
"$PY" scripts/make_tiny.py --src "$SRC" --out "$T/tiny"
cp -R "$SRC/glide" "$T/tiny/glide"
"$PY" parity.py cases --out "$T/cases.jsonl" --n-text 16 --n-img 6
"$PY" parity.py ref --model "$T/tiny" --cases "$T/cases.jsonl" --out "$T/ref.jsonl" --device cpu --dtype float32 > /dev/null
GLIDE_READOUT_FP32=1 "$PY" parity.py mlx --model "$T/tiny" --cases "$T/cases.jsonl" --out "$T/mlx.jsonl" > /dev/null
"$PY" parity.py compare "$T/ref.jsonl" "$T/mlx.jsonl" --json "$T/compare.json"
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); ok = r['same_top_pct'] >= 95 and r['prob_abs_diff_max'] < 0.05 and all(f['ref_margin'] < 0.02 for f in r['flips']) and not r['errors'] and r['missing_in_test'] == 0; print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)" "$T/compare.json"
