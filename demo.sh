#!/usr/bin/env bash
# Runnable, offline, ML-free demo of the Kubeflow LLM Trainer.
#
# Exercises the full surface with ZERO ML libraries installed:
#   1. list the pluggable backends
#   2. validate the unified config
#   3. dry-run plan the SAME config against TWO different backends
#   4. scaffold a fresh starter config and validate it
#   5. generate a Kubeflow Trainer v2 TrainJob manifest
#   6. validate that the emitted YAML actually parses
#
# Usage:  ./demo.sh        (uses ./.venv if present, else `python3`)
set -euo pipefail

cd "$(dirname "$0")"

# Pick an interpreter: prefer the project venv, fall back to python3.
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi
KLT="$PY -m kubeflow_llm_trainer.cli"
CFG="examples/lora_small.yaml"

hr() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }

hr "1. list-backends"
$KLT list-backends

hr "2. validate the unified config"
$KLT validate "$CFG"

hr "3. dry-run plan for backend: trl"
$KLT plan --backend trl "$CFG"

hr "3b. dry-run plan for backend: torchtune (SAME config)"
$KLT plan --backend torchtune "$CFG"

hr "4. scaffold a fresh starter config (klt init) and validate it"
TMP_CFG="$(mktemp --suffix=.yaml)"
$KLT init --backend trl --force "$TMP_CFG"
$KLT validate "$TMP_CFG"
rm -f "$TMP_CFG"

hr "5. generate a Kubeflow Trainer v2 TrainJob manifest"
MANIFEST="$(mktemp --suffix=.yaml)"
$KLT manifest --backend trl --kind trainjob --gpus 1 "$CFG" -o "$MANIFEST"
cat "$MANIFEST"

hr "6. validate the generated YAML actually parses (multi-doc)"
$PY - "$MANIFEST" <<'PYEOF'
import sys, yaml
docs = list(yaml.safe_load_all(open(sys.argv[1])))
kinds = [d["kind"] for d in docs]
assert kinds == ["ConfigMap", "TrainingRuntime", "TrainJob"], kinds
print("OK: valid multi-document YAML ->", " + ".join(kinds))
PYEOF
rm -f "$MANIFEST"

hr "demo complete"
echo "All steps ran offline with no ML libraries installed."
