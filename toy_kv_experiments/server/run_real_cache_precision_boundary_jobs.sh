#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_kivi_block_structeval_jobs.sh"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"

LIMIT="${LIMIT:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
QUANTIZE_BLOCK_SIZE="${QUANTIZE_BLOCK_SIZE:-128}"
STRUCTURE_TOKEN_PROTECTION="${STRUCTURE_TOKEN_PROTECTION:-none}"
PROTECTED_BITS="${PROTECTED_BITS:-}"
TARGET_AVERAGE_BITS="${TARGET_AVERAGE_BITS:-}"
PROTECTION_BUDGET_ORDER="${PROTECTION_BUDGET_ORDER:-score}"

mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_DIR/real_cache_precision_boundary_${timestamp}.log"

echo "[real-cache] StructEval JSON real-cache precision-boundary sweep" | tee "$summary_log"
echo "[real-cache] limit: $LIMIT" | tee -a "$summary_log"
echo "[real-cache] max_new_tokens: $MAX_NEW_TOKENS" | tee -a "$summary_log"
echo "[real-cache] residual length: $RESIDUAL_LENGTH" | tee -a "$summary_log"
echo "[real-cache] block size: $QUANTIZE_BLOCK_SIZE" | tee -a "$summary_log"
echo "[real-cache] protection: $STRUCTURE_TOKEN_PROTECTION" | tee -a "$summary_log"
echo "[real-cache] target average bits: ${TARGET_AVERAGE_BITS:-none}" | tee -a "$summary_log"

run_case() {
  local label="$1"
  local bits="$2"
  local key_bits="$3"
  local value_bits="$4"

  echo | tee -a "$summary_log"
  echo "[real-cache] running $label" | tee -a "$summary_log"

  LIMIT="$LIMIT" \
  BITS="$bits" \
  KEY_BITS="$key_bits" \
  VALUE_BITS="$value_bits" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
  QUANTIZE_BLOCK_SIZE="$QUANTIZE_BLOCK_SIZE" \
  STRUCTURE_TOKEN_PROTECTION="$STRUCTURE_TOKEN_PROTECTION" \
  PROTECTED_BITS="$PROTECTED_BITS" \
  TARGET_AVERAGE_BITS="$TARGET_AVERAGE_BITS" \
  PROTECTION_BUDGET_ORDER="$PROTECTION_BUDGET_ORDER" \
  CACHE_QUANTIZATION_MODE=real-blockwise \
    bash "$RUNNER" 2>&1 | tee -a "$summary_log"
}

run_case "K8/V8 real packed cache" 8 "" ""
run_case "K6/V6 real int8-backed numeric 6-bit cache" 6 "" ""
run_case "K4/V8 real mixed precision cache" 8 4 8
run_case "K8/V4 real mixed precision cache" 8 8 4
run_case "K4/V4 real packed cache" 4 "" ""

echo | tee -a "$summary_log"
echo "[real-cache] complete" | tee -a "$summary_log"
echo "[real-cache] summary log: $summary_log"
