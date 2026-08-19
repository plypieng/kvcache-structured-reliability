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

mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_DIR/kivi_precision_boundary_${timestamp}.log"

echo "[precision-boundary] StructEval JSON precision-boundary sweep" | tee "$summary_log"
echo "[precision-boundary] limit: $LIMIT" | tee -a "$summary_log"
echo "[precision-boundary] max_new_tokens: $MAX_NEW_TOKENS" | tee -a "$summary_log"
echo "[precision-boundary] residual length: $RESIDUAL_LENGTH" | tee -a "$summary_log"
echo "[precision-boundary] block size: $QUANTIZE_BLOCK_SIZE" | tee -a "$summary_log"
echo "[precision-boundary] protection: $STRUCTURE_TOKEN_PROTECTION" | tee -a "$summary_log"
echo "[precision-boundary] protected bits: ${PROTECTED_BITS:-fp16 copy-back}" | tee -a "$summary_log"

run_case() {
  local label="$1"
  local bits="$2"
  local key_bits="$3"
  local value_bits="$4"

  echo | tee -a "$summary_log"
  echo "[precision-boundary] running $label" | tee -a "$summary_log"

  LIMIT="$LIMIT" \
  BITS="$bits" \
  KEY_BITS="$key_bits" \
  VALUE_BITS="$value_bits" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
  QUANTIZE_BLOCK_SIZE="$QUANTIZE_BLOCK_SIZE" \
  STRUCTURE_TOKEN_PROTECTION="$STRUCTURE_TOKEN_PROTECTION" \
  PROTECTED_BITS="$PROTECTED_BITS" \
    bash "$RUNNER" 2>&1 | tee -a "$summary_log"
}

run_case "K6/V6 blockwise" 6 "" ""
run_case "K4/V8 blockwise mixed precision" 8 4 8
run_case "K8/V4 blockwise mixed precision" 8 8 4

echo | tee -a "$summary_log"
echo "[precision-boundary] complete" | tee -a "$summary_log"
echo "[precision-boundary] summary log: $summary_log"
