#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
ENV_NAME="${ENV_NAME:-kvcache-py311}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
SOURCE="${SOURCE:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/structeval_test.jsonl}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/manifests/allformats_100_category_balanced_seed42.json}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/toy_kv_experiments/results/poster_pilot_allformats100}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"

cd "$PROJECT_DIR"
mkdir -p "$RESULT_DIR"

# The validator imports the same tokenizer/cache modules as the runner.
# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m toy_kv_experiments.check_structeval_poster_manifest \
  --source "$SOURCE" \
  --manifest "$MANIFEST"

run_case() {
  local label="$1"
  local bits="$2"
  local key_bits="$3"
  local value_bits="$4"
  local output_path="$RESULT_DIR/${RUN_LABEL}_${label}_inference.json"

  echo "[poster-pilot] start $label -> $output_path"
  LIMIT=100 \
  OUTPUT_TYPE=ALL \
  STRUCTEVAL_MANIFEST="$MANIFEST" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  DISABLE_END_CODE_STOP=0 \
  LOOP_NGRAM_SIZE=0 \
  LOOP_REPEAT_THRESHOLD=0 \
  OFFICIAL_STRUCTEVAL=1 \
  BITS="$bits" \
  KEY_BITS="$key_bits" \
  VALUE_BITS="$value_bits" \
  RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
  KV_GROUP_SIZE="$KV_GROUP_SIZE" \
  CACHE_QUANTIZATION_MODE=real-blockwise \
  STRUCTURE_TOKEN_PROTECTION=none \
  PROTECTION_SIGNAL_SOURCE=prompt-visible \
  RESUME=1 \
  OUT_PATH="$output_path" \
  CHECKPOINT_DIR="$output_path.checkpoints" \
    bash "$RUNNER"

  python -m toy_kv_experiments.check_structeval_poster_result \
    "$output_path" \
    --source "$SOURCE" \
    --manifest "$MANIFEST" \
    --key-bits "$key_bits" \
    --value-bits "$value_bits"
  echo "[poster-pilot] complete $label"
}

echo "[poster-pilot] run label: $RUN_LABEL"
echo "[poster-pilot] frozen manifest: $MANIFEST"
echo "[poster-pilot] per-task output cap: $MAX_NEW_TOKENS"
echo "$RUN_LABEL" > "$RESULT_DIR/latest_run_label.txt"

run_case FP16 16 16 16
run_case K8_V8 8 8 8
run_case K8_V4 8 8 4
run_case K4_V8 8 4 8
run_case K4_V4 4 4 4

echo "[poster-pilot] uniform characterization complete"
