#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
STRUCTEVAL_ROOT="${STRUCTEVAL_ROOT:-$PROJECT_DIR/third_party/StructEval-litellm}"
ENV_NAME="${ENV_NAME:-structeval-eval}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
INFERENCE_PATH="${INFERENCE_PATH:?Set INFERENCE_PATH to a completed inference.json file.}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a unique evaluation directory.}"
EXPECTED_ROWS="${EXPECTED_ROWS:-950}"
RENDER_CONCURRENCY="${RENDER_CONCURRENCY:-4}"

if [[ ! -f "$INFERENCE_PATH" ]]; then
  echo "Inference artifact not found: $INFERENCE_PATH" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/rendered_images" "$RUN_DIR/non_renderable_format_files"
cp "$INFERENCE_PATH" "$RUN_DIR/inference.json"

# shellcheck disable=SC1091
set +u
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
export PYTHONPATH="$PROJECT_DIR:$STRUCTEVAL_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_DIR"

python - "$RUN_DIR/inference.json" "$EXPECTED_ROWS" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
if not isinstance(rows, list) or len(rows) != expected:
    raise SystemExit(f"expected {expected} inference rows, found {len(rows) if isinstance(rows, list) else 'non-list'}")
task_ids = [str(row.get("task_id") or "") for row in rows]
if len(set(task_ids)) != expected or not all(task_ids):
    raise SystemExit("inference artifact has missing or duplicate task IDs")
rendered = [task_id for task_id, row in zip(task_ids, rows) if bool(row.get("rendering", False))]
if rendered:
    raise SystemExit(f"StructEval-T artifact contains rendered tasks: {rendered[:5]}")
for required in ("generation", "run_fingerprint", "key_bits", "value_bits"):
    missing = [task_id for task_id, row in zip(task_ids, rows) if required not in row]
    if missing:
        raise SystemExit(f"missing {required} in tasks: {missing[:5]}")
print(f"StructEval-T inference gate: {expected} unique non-rendered rows")
PY

INPUT_PATH="$RUN_DIR/inference.json" \
IMG_OUTPUT_PATH="$RUN_DIR/rendered_images" \
NON_RENDERABLE_OUTPUT_DIR="$RUN_DIR/non_renderable_format_files" \
RENDER_CONCURRENCY="$RENDER_CONCURRENCY" \
  bash "$PROJECT_DIR/toy_kv_experiments/server/render_structeval_isolated.sh"

python -m structeval.cli evaluate \
  --input-path "$RUN_DIR/inference.json" \
  --output-path "$RUN_DIR/evaluation.json" \
  --img-path "$RUN_DIR/rendered_images" \
  --non-renderable-output-dir "$RUN_DIR/non_renderable_format_files"

python -m toy_kv_experiments.summarize_structeval_t \
  "$RUN_DIR/evaluation.json" \
  --expected-rows "$EXPECTED_ROWS" \
  --out "$RUN_DIR/summary.json"

date -Iseconds > "$RUN_DIR/COMPLETE"
echo "[StructEval-T evaluator] complete: $RUN_DIR/summary.json"
