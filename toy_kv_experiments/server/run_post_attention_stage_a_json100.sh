#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/manifests/json_100_stratified_seed42.json}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/toy_kv_experiments/results/post_attention_stage_a}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"

cd "$PROJECT_DIR"
mkdir -p "$RESULT_DIR"

run_case() {
  local label="$1"
  local bits="$2"
  local key_bits="$3"
  local value_bits="$4"
  local output_path="$RESULT_DIR/post_attention_v3_json100_${label}_${RUN_LABEL}.json"

  echo "[stage-a-v3] start $label -> $output_path"
  LIMIT=100 \
  STRUCTEVAL_MANIFEST="$MANIFEST" \
  OUT_PATH="$output_path" \
  CHECKPOINT_DIR="$output_path.checkpoints" \
  RESUME=1 \
  BITS="$bits" \
  KEY_BITS="$key_bits" \
  VALUE_BITS="$value_bits" \
  RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
  KV_GROUP_SIZE="$KV_GROUP_SIZE" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  CACHE_QUANTIZATION_MODE=real-blockwise \
  STRUCTURE_TOKEN_PROTECTION=none \
  PROTECTION_SIGNAL_SOURCE=prompt-visible \
  TARGET_AVERAGE_BITS='' \
  PROTECTED_BITS='' \
    bash "$RUNNER"
  echo "[stage-a-v3] complete $label"
}

echo "[stage-a-v3] run label: $RUN_LABEL"
echo "[stage-a-v3] frozen manifest: $MANIFEST"
echo "$RUN_LABEL" > "$RESULT_DIR/latest_run_label.txt"
run_case FP16 16 16 16
run_case K8_V8 8 8 8
run_case K8_V4 8 8 4
run_case K4_V8 8 4 8
run_case K4_V4 4 4 4
run_case K2_V2 8 2 2

echo "[stage-a-v3] queue complete"
