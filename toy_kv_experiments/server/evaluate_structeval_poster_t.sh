#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
STRUCTEVAL_ROOT="${STRUCTEVAL_ROOT:-$PROJECT_DIR/third_party/StructEval-litellm}"
ENV_NAME="${ENV_NAME:-structeval-eval}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
INFERENCE_PATH="${INFERENCE_PATH:-}"
RUN_DIR="${RUN_DIR:-}"
RENDER_CONCURRENCY="${RENDER_CONCURRENCY:-4}"
EXPECTED_TEXT_ROWS="${EXPECTED_TEXT_ROWS:-50}"
EXPECTED_CATEGORY_COUNT="${EXPECTED_CATEGORY_COUNT:-25}"

if [ -z "$INFERENCE_PATH" ] || [ ! -f "$INFERENCE_PATH" ]; then
  echo "INFERENCE_PATH must point to a completed poster inference JSON" >&2
  exit 2
fi
if [ -z "$RUN_DIR" ]; then
  base_name="$(basename "$INFERENCE_PATH" .json)"
  RUN_DIR="$(dirname "$INFERENCE_PATH")/${base_name}_poster_t_eval"
fi

mkdir -p "$RUN_DIR/rendered_images" "$RUN_DIR/non_renderable_format_files"

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONPATH="$PROJECT_DIR:$STRUCTEVAL_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_DIR"

python - "$INFERENCE_PATH" "$RUN_DIR/inference_t.json" "$EXPECTED_TEXT_ROWS" <<'PY'
import json
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = [row for row in source if not bool(row.get("rendering", False))]
expected_rows = int(sys.argv[3])
if len(rows) != expected_rows:
    raise SystemExit(f"expected {expected_rows} non-renderable rows, found {len(rows)}")
Path(sys.argv[2]).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PY

INPUT_PATH="$RUN_DIR/inference_t.json" \
IMG_OUTPUT_PATH="$RUN_DIR/rendered_images" \
NON_RENDERABLE_OUTPUT_DIR="$RUN_DIR/non_renderable_format_files" \
RENDER_CONCURRENCY="$RENDER_CONCURRENCY" \
  bash "$PROJECT_DIR/toy_kv_experiments/server/render_structeval_isolated.sh"

python -m structeval.cli evaluate \
  --input-path "$RUN_DIR/inference_t.json" \
  --output-path "$RUN_DIR/evaluation_t.json" \
  --img-path "$RUN_DIR/rendered_images" \
  --non-renderable-output-dir "$RUN_DIR/non_renderable_format_files"

python -m toy_kv_experiments.summarize_structeval_poster \
  "$RUN_DIR/evaluation_t.json" \
  --expected-category-count "$EXPECTED_CATEGORY_COUNT" \
  --out "$RUN_DIR/poster_t_summary.json"

echo "[poster-t-eval] complete: $RUN_DIR/poster_t_summary.json"
