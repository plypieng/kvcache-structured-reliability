#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_kivi_block_structeval_jobs.sh"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"

LIMIT="${LIMIT:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
QUANTIZE_BLOCK_SIZE="${QUANTIZE_BLOCK_SIZE:-128}"
TARGET_AVERAGE_BITS="${TARGET_AVERAGE_BITS:-}"
PROTECTION_BUDGET_ORDER="${PROTECTION_BUDGET_ORDER:-score}"
PROTECTION_SIGNAL_SOURCE="${PROTECTION_SIGNAL_SOURCE:-prompt-visible}"

mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_DIR/constraint_uep_${timestamp}.log"

echo "[constraint-uep] StructEval JSON schema/path-token UEP sweep" | tee "$summary_log"
echo "[constraint-uep] limit: $LIMIT" | tee -a "$summary_log"
echo "[constraint-uep] max_new_tokens: $MAX_NEW_TOKENS" | tee -a "$summary_log"
echo "[constraint-uep] residual length: $RESIDUAL_LENGTH" | tee -a "$summary_log"
echo "[constraint-uep] block size: $QUANTIZE_BLOCK_SIZE" | tee -a "$summary_log"
echo "[constraint-uep] target average bits: ${TARGET_AVERAGE_BITS:-none}" | tee -a "$summary_log"
echo "[constraint-uep] protection budget order: $PROTECTION_BUDGET_ORDER" | tee -a "$summary_log"
echo "[constraint-uep] protection signal source: $PROTECTION_SIGNAL_SOURCE" | tee -a "$summary_log"

run_case() {
  local label="$1"
  local bits="$2"
  local protection="$3"
  local protected_bits="$4"

  echo | tee -a "$summary_log"
  echo "[constraint-uep] running $label" | tee -a "$summary_log"

  LIMIT="$LIMIT" \
  BITS="$bits" \
  KEY_BITS="" \
  VALUE_BITS="" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
  QUANTIZE_BLOCK_SIZE="$QUANTIZE_BLOCK_SIZE" \
  STRUCTURE_TOKEN_PROTECTION="$protection" \
  PROTECTED_BITS="$protected_bits" \
  TARGET_AVERAGE_BITS="$TARGET_AVERAGE_BITS" \
  PROTECTION_BUDGET_ORDER="$PROTECTION_BUDGET_ORDER" \
  PROTECTION_SIGNAL_SOURCE="$PROTECTION_SIGNAL_SOURCE" \
    bash "$RUNNER" 2>&1 | tee -a "$summary_log"
}

run_case "K4/V4 no protection" 4 "none" ""
run_case "K4/V4, schema/path tokens at 8-bit" 4 "constraint-paths" 8
run_case "K4/V4, JSON syntax + schema/path tokens at 8-bit" 4 "json-syntax+constraint-paths" 8
run_case "K8/V8 no protection" 8 "none" ""
run_case "K8/V8, schema/path tokens at 8-bit" 8 "constraint-paths" 8

echo | tee -a "$summary_log"
echo "[constraint-uep] complete" | tee -a "$summary_log"
echo "[constraint-uep] summary log: $summary_log"
