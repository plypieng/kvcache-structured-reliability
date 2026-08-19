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
KEY_BITS="${KEY_BITS:-}"
VALUE_BITS="${VALUE_BITS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
QUANTIZE_BLOCK_SIZE="${QUANTIZE_BLOCK_SIZE:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"
STRUCTURE_TOKEN_PROTECTION="${STRUCTURE_TOKEN_PROTECTION:-none}"
PROTECTED_BITS="${PROTECTED_BITS:-}"
TARGET_AVERAGE_BITS="${TARGET_AVERAGE_BITS:-}"
PROTECTION_BUDGET_ORDER="${PROTECTION_BUDGET_ORDER:-prefix}"
PROTECTION_TARGET="${PROTECTION_TARGET:-both}"
CACHE_QUANTIZATION_MODE="${CACHE_QUANTIZATION_MODE:-real-blockwise}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
safe_key_bits="${KEY_BITS:-same}"
safe_value_bits="${VALUE_BITS:-same}"
safe_protected_bits="${PROTECTED_BITS:-fp16}"
safe_target_average_bits="${TARGET_AVERAGE_BITS:-none}"
safe_protection_target="$(printf "%s" "$PROTECTION_TARGET" | tr -c 'A-Za-z0-9_' '_')"
safe_mode="$(printf "%s" "$CACHE_QUANTIZATION_MODE" | tr -c 'A-Za-z0-9_' '_')"
out_path="$OUT_DIR/qwen2_5_7b_structeval_official_json_cache_reference_${safe_mode}_limit${LIMIT}_sample_${STRUCTEVAL_SAMPLING}_seed_${STRUCTEVAL_SEED}_bits_${BITS}_K_${safe_key_bits}_V_${safe_value_bits}_residual_${RESIDUAL_LENGTH}_group_${KV_GROUP_SIZE}_protection_${STRUCTURE_TOKEN_PROTECTION}_protecttarget_${safe_protection_target}_protected_${safe_protected_bits}_target_${safe_target_average_bits}_${timestamp}.json"
log_path="$LOG_DIR/cache_reference_${safe_mode}_limit${LIMIT}_sample_${STRUCTEVAL_SAMPLING}_seed_${STRUCTEVAL_SEED}_bits_${BITS}_K_${safe_key_bits}_V_${safe_value_bits}_residual_${RESIDUAL_LENGTH}_group_${KV_GROUP_SIZE}_protection_${STRUCTURE_TOKEN_PROTECTION}_protecttarget_${safe_protection_target}_protected_${safe_protected_bits}_target_${safe_target_average_bits}_${timestamp}.log"

echo "[kivi-block] StructEval JSON block-cache reference test (not exact KIVI)"
echo "[kivi-block] limit: $LIMIT"
echo "[kivi-block] StructEval sampling: $STRUCTEVAL_SAMPLING"
echo "[kivi-block] StructEval seed: $STRUCTEVAL_SEED"
echo "[kivi-block] bits: $BITS"
echo "[kivi-block] key bits override: ${KEY_BITS:-same as bits}"
echo "[kivi-block] value bits override: ${VALUE_BITS:-same as bits}"
echo "[kivi-block] max_new_tokens: $MAX_NEW_TOKENS"
echo "[kivi-block] residual length: $RESIDUAL_LENGTH"
echo "[kivi-block] quantize block size: $QUANTIZE_BLOCK_SIZE"
echo "[kivi-block] KIVI group size: $KV_GROUP_SIZE"
echo "[kivi-block] cache quantization mode: $CACHE_QUANTIZATION_MODE"
echo "[kivi-block] protection: $STRUCTURE_TOKEN_PROTECTION"
echo "[kivi-block] protected bits: ${PROTECTED_BITS:-fp16 copy-back}"
echo "[kivi-block] target average bits: ${TARGET_AVERAGE_BITS:-none}"
echo "[kivi-block] protection budget order: $PROTECTION_BUDGET_ORDER"
echo "[kivi-block] protection target: $PROTECTION_TARGET"
echo "[kivi-block] output: $out_path"
echo "[kivi-block] log:    $log_path"

LIMIT="$LIMIT" \
STRUCTEVAL_SAMPLING="$STRUCTEVAL_SAMPLING" \
STRUCTEVAL_SEED="$STRUCTEVAL_SEED" \
BITS="$BITS" \
KEY_BITS="$KEY_BITS" \
VALUE_BITS="$VALUE_BITS" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
STRUCTURE_TOKEN_PROTECTION="$STRUCTURE_TOKEN_PROTECTION" \
PROTECTED_BITS="$PROTECTED_BITS" \
TARGET_AVERAGE_BITS="$TARGET_AVERAGE_BITS" \
PROTECTION_BUDGET_ORDER="$PROTECTION_BUDGET_ORDER" \
PROTECTION_TARGET="$PROTECTION_TARGET" \
CACHE_QUANTIZATION_MODE="$CACHE_QUANTIZATION_MODE" \
QUANTIZE_BLOCK_SIZE="$QUANTIZE_BLOCK_SIZE" \
KV_GROUP_SIZE="$KV_GROUP_SIZE" \
OFFICIAL_STRUCTEVAL=1 \
OUT_PATH="$out_path" \
  bash "$RUNNER" 2>&1 | tee "$log_path"

echo "[kivi-block] complete"
