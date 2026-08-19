#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/toy_kv_experiments/results/structeval_full_official}"
RUN_LABEL="${RUN_LABEL:-qwen2_5_7b_official_v1}"
EVALUATOR="$PROJECT_DIR/toy_kv_experiments/server/evaluate_structeval_official.sh"
FORCE_EVALUATION="${FORCE_EVALUATION:-0}"
STRUCTEVAL_JSONL="${STRUCTEVAL_JSONL:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/structeval_test.jsonl}"

if [ "${ALLOW_UNTRUSTED_RENDERING:-0}" != "1" ]; then
  echo "Set ALLOW_UNTRUSTED_RENDERING=1 after reviewing the isolated renderer." >&2
  exit 2
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is required for official StructEval-V scoring." >&2
  exit 2
fi

cd "$PROJECT_DIR"

conditions=(
  "FP16 16 16"
  "K8_V8 8 8"
  "K8_V4 8 4"
  "K4_V8 4 8"
  "K4_V4 4 4"
  "K2_V2 2 2"
)
matrix_arguments=()

for condition in "${conditions[@]}"; do
  read -r label key_bits value_bits <<<"$condition"
  inference_path="$RESULT_DIR/${RUN_LABEL}_${label}_inference.json"
  run_dir="$RESULT_DIR/${RUN_LABEL}_${label}_official_eval"
  summary_path="$run_dir/leaderboard_summary.json"

  python -m toy_kv_experiments.check_structeval_official_artifact inference \
    "$inference_path" \
    --dataset "$STRUCTEVAL_JSONL" \
    --expected-key-bits "$key_bits" \
    --expected-value-bits "$value_bits"

  if [ "$FORCE_EVALUATION" != "1" ] && \
    [ -f "$summary_path" ] && \
    python -m toy_kv_experiments.check_structeval_official_artifact summary "$summary_path"; then
    echo "[structeval-eval-matrix] reuse completed score: $label"
  else
    echo "[structeval-eval-matrix] evaluate: $label"
    INFERENCE_PATH="$inference_path" \
    RUN_DIR="$run_dir" \
    ALLOW_UNTRUSTED_RENDERING=1 \
      bash "$EVALUATOR"
    python -m toy_kv_experiments.check_structeval_official_artifact summary "$summary_path"
  fi

  matrix_arguments+=(--condition "$label=$summary_path")
done

python -m toy_kv_experiments.summarize_structeval_matrix \
  "${matrix_arguments[@]}" \
  --out "$RESULT_DIR/${RUN_LABEL}_official_matrix_summary.json" \
  --markdown-out "$RESULT_DIR/${RUN_LABEL}_official_matrix_summary.md"

echo "[structeval-eval-matrix] complete official matrix: $RESULT_DIR"
