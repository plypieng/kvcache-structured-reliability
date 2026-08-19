#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
STRUCTEVAL_ROOT="${STRUCTEVAL_ROOT:-$PROJECT_DIR/third_party/StructEval-litellm}"
ENV_NAME="${ENV_NAME:-structeval-eval}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
INFERENCE_PATH="${INFERENCE_PATH:-}"
RUN_DIR="${RUN_DIR:-}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4.1-mini}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
RENDER_CONCURRENCY="${RENDER_CONCURRENCY:-4}"
ALLOW_UNTRUSTED_RENDERING="${ALLOW_UNTRUSTED_RENDERING:-0}"

if [ -z "$INFERENCE_PATH" ]; then
  echo "INFERENCE_PATH must point to a completed 2,035-row inference JSON file" >&2
  exit 2
fi
if [ ! -f "$INFERENCE_PATH" ]; then
  echo "inference file not found: $INFERENCE_PATH" >&2
  exit 2
fi
if [ "$ALLOW_UNTRUSTED_RENDERING" != "1" ]; then
  cat >&2 <<'EOF'
Refusing to execute model-generated code without ALLOW_UNTRUSTED_RENDERING=1.
Run this evaluator only inside an isolated environment: StructEval renders
HTML/framework code and executes generated Matplotlib code.
EOF
  exit 2
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is required for the official GPT-4.1-mini VQA stage" >&2
  exit 2
fi

if [ -z "$RUN_DIR" ]; then
  base_name="$(basename "$INFERENCE_PATH" .json)"
  RUN_DIR="$(dirname "$INFERENCE_PATH")/${base_name}_official_eval"
fi

mkdir -p "$RUN_DIR/rendered_images" "$RUN_DIR/non_renderable_format_files"
cp "$INFERENCE_PATH" "$RUN_DIR/inference.json"

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONPATH="$PROJECT_DIR:$STRUCTEVAL_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_DIR"

INPUT_PATH="$RUN_DIR/inference.json" \
IMG_OUTPUT_PATH="$RUN_DIR/rendered_images" \
NON_RENDERABLE_OUTPUT_DIR="$RUN_DIR/non_renderable_format_files" \
RENDER_CONCURRENCY="$RENDER_CONCURRENCY" \
  bash "$PROJECT_DIR/toy_kv_experiments/server/render_structeval_isolated.sh"

python -m structeval.cli evaluate \
  --input-path "$RUN_DIR/inference.json" \
  --output-path "$RUN_DIR/evaluation.json" \
  --img-path "$RUN_DIR/rendered_images" \
  --non-renderable-output-dir "$RUN_DIR/non_renderable_format_files" \
  --judge-model "$JUDGE_MODEL" \
  --api-key-env OPENAI_API_KEY \
  --concurrency "$JUDGE_CONCURRENCY" \
  --judge-temperature 0.0

python -m toy_kv_experiments.summarize_structeval_official \
  "$RUN_DIR/evaluation.json" \
  --out "$RUN_DIR/leaderboard_summary.json"

echo "[structeval-official-eval] official summary: $RUN_DIR/summary.json"
echo "[structeval-official-eval] leaderboard categories: $RUN_DIR/leaderboard_summary.json"
