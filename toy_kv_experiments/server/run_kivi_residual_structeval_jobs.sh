#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
OUT_DIR="$PROJECT_DIR/toy_kv_experiments/results"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"

LIMIT="${LIMIT:-20}"
STRUCTEVAL_SAMPLING="${STRUCTEVAL_SAMPLING:-stratified}"
STRUCTEVAL_SEED="${STRUCTEVAL_SEED:-42}"
BITS="${BITS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTHS="${RESIDUAL_LENGTHS:-0 32 64 128}"
QUANTIZE_BLOCK_SIZE="${QUANTIZE_BLOCK_SIZE:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"
CACHE_QUANTIZATION_MODE="${CACHE_QUANTIZATION_MODE:-real-blockwise}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
echo "[residual] StructEval JSON residual-cache sweep"
echo "[residual] limit: $LIMIT"
echo "[residual] StructEval sampling: $STRUCTEVAL_SAMPLING"
echo "[residual] StructEval seed: $STRUCTEVAL_SEED"
echo "[residual] bits: $BITS"
echo "[residual] max_new_tokens: $MAX_NEW_TOKENS"
echo "[residual] residual lengths: $RESIDUAL_LENGTHS"
echo "[residual] quantization mode: $CACHE_QUANTIZATION_MODE"
echo "[residual] block size: $QUANTIZE_BLOCK_SIZE"
echo "[residual] KIVI group size: $KV_GROUP_SIZE"

for residual in $RESIDUAL_LENGTHS; do
  out_path="$OUT_DIR/qwen2_5_7b_structeval_official_json_real_blockwise_residual_limit${LIMIT}_sample_${STRUCTEVAL_SAMPLING}_seed_${STRUCTEVAL_SEED}_bits_${BITS}_residual_${residual}_block_${QUANTIZE_BLOCK_SIZE}_${timestamp}.json"
  log_path="$LOG_DIR/real_blockwise_residual_limit${LIMIT}_sample_${STRUCTEVAL_SAMPLING}_seed_${STRUCTEVAL_SEED}_bits_${BITS}_residual_${residual}_block_${QUANTIZE_BLOCK_SIZE}_${timestamp}.log"

  echo "[residual] running residual_length=$residual"
  echo "[residual] output: $out_path"
  echo "[residual] log:    $log_path"

  LIMIT="$LIMIT" \
  STRUCTEVAL_SAMPLING="$STRUCTEVAL_SAMPLING" \
  STRUCTEVAL_SEED="$STRUCTEVAL_SEED" \
  BITS="$BITS" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  RESIDUAL_LENGTH="$residual" \
  QUANTIZE_BLOCK_SIZE="$QUANTIZE_BLOCK_SIZE" \
  KV_GROUP_SIZE="$KV_GROUP_SIZE" \
  CACHE_QUANTIZATION_MODE="$CACHE_QUANTIZATION_MODE" \
  OFFICIAL_STRUCTEVAL=1 \
  OUT_PATH="$out_path" \
    bash "$RUNNER" 2>&1 | tee "$log_path"
done

echo "[residual] complete"
