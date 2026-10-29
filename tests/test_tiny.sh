#!/bin/bash
# Port check without the 52 GB download: a tiny random model with GLiDE's layout (float32), run through Fastino's PyTorch
# code and through this port on the same requests (text, JSON, images; choice / noul / score). Expect 100% same top answers
# and a max probability difference around 1e-5.   tests/test_tiny.sh <folder with fastino's glide/ package, config,
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
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); ok = r['same_top_pct'] == 100 and r['prob_abs_diff_max'] < 1e-3 and not r['errors'] and r['missing_in_test'] == 0; print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)" "$T/compare.json"
