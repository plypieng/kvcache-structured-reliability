#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
KIVI_RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_kivi_block_structeval_jobs.sh"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"

RECOVERY_LIMIT="${RECOVERY_LIMIT:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
QUANTIZE_BLOCK_SIZE="${QUANTIZE_BLOCK_SIZE:-128}"

mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_DIR/remaining_reliability_allocation_${timestamp}.log"

run_logged() {
  local label="$1"
  shift
  echo | tee -a "$summary_log"
  echo "[remaining] $(date '+%Y-%m-%d %H:%M:%S') running: $label" | tee -a "$summary_log"
  "$@" 2>&1 | tee -a "$summary_log"
}

run_real_case() {
  local label="$1"
  local bits="$2"
  local key_bits="$3"
  local value_bits="$4"
  local protection="$5"
  local protected_bits="$6"
  local protection_target="$7"

  run_logged "$label limit=${RECOVERY_LIMIT}" env \
    LIMIT="$RECOVERY_LIMIT" \
    BITS="$bits" \
    KEY_BITS="$key_bits" \
    VALUE_BITS="$value_bits" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
    QUANTIZE_BLOCK_SIZE="$QUANTIZE_BLOCK_SIZE" \
    STRUCTURE_TOKEN_PROTECTION="$protection" \
    PROTECTED_BITS="$protected_bits" \
    TARGET_AVERAGE_BITS="" \
    PROTECTION_BUDGET_ORDER=score \
    PROTECTION_SIGNAL_SOURCE=prompt-visible \
    PROTECTION_TARGET="$protection_target" \
    CACHE_QUANTIZATION_MODE=real-blockwise \
    bash "$KIVI_RUNNER"
}

echo "[remaining] StructEval reliability-allocation recovery queue" | tee "$summary_log"
echo "[remaining] recovery limit: $RECOVERY_LIMIT" | tee -a "$summary_log"
echo "[remaining] max_new_tokens: $MAX_NEW_TOKENS" | tee -a "$summary_log"
echo "[remaining] residual length: $RESIDUAL_LENGTH" | tee -a "$summary_log"
echo "[remaining] block size: $QUANTIZE_BLOCK_SIZE" | tee -a "$summary_log"
echo "[remaining] prior K6/K7 results were numerical-only and are not valid packed-storage treatments" | tee -a "$summary_log"

run_real_case "K4/V8 key-only JSON syntax + prompt paths at 8-bit" 8 4 8 json-syntax+constraint-paths 8 keys
run_real_case "K4/V8 key-only JSON syntax + prompt paths at FP16" 8 4 8 json-syntax+constraint-paths "" keys
run_real_case "K4/V4 key-only JSON syntax + prompt paths at 8-bit" 4 4 4 json-syntax+constraint-paths 8 keys
run_real_case "K4/V4 both-side JSON syntax + prompt paths at 8-bit" 4 4 4 json-syntax+constraint-paths 8 both
run_real_case "recover K8/V2 value-side JSON syntax + paths at 8-bit" 8 8 2 json-syntax+constraint-paths 8 values

echo | tee -a "$summary_log"
echo "[remaining] complete: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$summary_log"
echo "[remaining] summary log: $summary_log" | tee -a "$summary_log"
