#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
STRUCTEVAL_ROOT="${STRUCTEVAL_ROOT:-$PROJECT_DIR/third_party/StructEval-litellm}"
ENV_NAME="${ENV_NAME:-structeval-eval}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/toy_kv_experiments/results/structeval_evaluator_smoke/$RUN_LABEL}"

mkdir -p "$RUN_DIR/images" "$RUN_DIR/non_renderable_format_files"

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONPATH="$PROJECT_DIR:$STRUCTEVAL_ROOT:${PYTHONPATH:-}"

python -m toy_kv_experiments.make_structeval_evaluator_smoke \
  --dataset "$STRUCTEVAL_ROOT/dataset/StructEval_dataset.json" \
  --out "$RUN_DIR/inference.json"

INPUT_PATH="$RUN_DIR/inference.json" \
IMG_OUTPUT_PATH="$RUN_DIR/images" \
NON_RENDERABLE_OUTPUT_DIR="$RUN_DIR/non_renderable_format_files" \
RENDER_CONCURRENCY=4 \
  bash "$PROJECT_DIR/toy_kv_experiments/server/render_structeval_isolated.sh"

python -m structeval.cli evaluate \
  --input-path "$RUN_DIR/inference.json" \
  --output-path "$RUN_DIR/evaluation.json" \
  --img-path "$RUN_DIR/images" \
  --non-renderable-output-dir "$RUN_DIR/non_renderable_format_files"

python - "$RUN_DIR/inference.json" "$RUN_DIR/evaluation.json" <<'PY'
import json
import sys

inference = json.load(open(sys.argv[1], encoding="utf-8"))
evaluation = json.load(open(sys.argv[2], encoding="utf-8"))
failed = [
    {
        "task_id": row.get("task_id"),
        "output_type": row.get("output_type"),
        "render_score": row.get("render_score"),
        "extract_error": row.get("extract_error"),
        "render_error": row.get("render_error"),
    }
    for row in inference
    if float(row.get("render_score") or 0.0) != 1.0
]
print("rendered", len(inference) - len(failed), "of", len(inference))
print("evaluated", len(evaluation), "of", len(inference))
if failed:
    print(json.dumps(failed, indent=2))
    raise SystemExit(1)
if len(evaluation) != len(inference):
    raise SystemExit("official evaluator returned the wrong row count")
PY

echo "[structeval-evaluator-smoke] all 18 output formats passed"
