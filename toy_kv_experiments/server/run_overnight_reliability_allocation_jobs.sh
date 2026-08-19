#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
BASE_RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
KIVI_RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_kivi_block_structeval_jobs.sh"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"

CONFIRM_LIMIT="${CONFIRM_LIMIT:-100}"
RECOVERY_LIMIT="${RECOVERY_LIMIT:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
QUANTIZE_BLOCK_SIZE="${QUANTIZE_BLOCK_SIZE:-128}"

mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_DIR/overnight_reliability_allocation_${timestamp}.log"

run_logged() {
  local label="$1"
  shift
  echo | tee -a "$summary_log"
  echo "[overnight] $(date '+%Y-%m-%d %H:%M:%S') running: $label" | tee -a "$summary_log"
  "$@" 2>&1 | tee -a "$summary_log"
}

run_fp16() {
  run_logged "confirm FP16 baseline limit=${CONFIRM_LIMIT}" env \
    LIMIT="$CONFIRM_LIMIT" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    BITS=16 \
    KEY_BITS="" \
    VALUE_BITS="" \
    RESIDUAL_LENGTH=0 \
    STRUCTURE_TOKEN_PROTECTION=none \
    PROTECTED_BITS="" \
    TARGET_AVERAGE_BITS="" \
    PROTECTION_BUDGET_ORDER=prefix \
    PROTECTION_TARGET=both \
    CACHE_QUANTIZATION_MODE=repeated \
    QUANTIZE_BLOCK_SIZE=1 \
    OFFICIAL_STRUCTEVAL=1 \
    bash "$BASE_RUNNER"
}

run_real_case() {
  local label="$1"
  local limit="$2"
  local bits="$3"
  local key_bits="$4"
  local value_bits="$5"
  local protection="$6"
  local protected_bits="$7"
  local target_average_bits="$8"
  local protection_order="$9"
  local protection_target="${10}"

  run_logged "$label limit=${limit}" env \
    LIMIT="$limit" \
    BITS="$bits" \
    KEY_BITS="$key_bits" \
    VALUE_BITS="$value_bits" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
    QUANTIZE_BLOCK_SIZE="$QUANTIZE_BLOCK_SIZE" \
    STRUCTURE_TOKEN_PROTECTION="$protection" \
    PROTECTED_BITS="$protected_bits" \
    TARGET_AVERAGE_BITS="$target_average_bits" \
    PROTECTION_BUDGET_ORDER="$protection_order" \
    PROTECTION_TARGET="$protection_target" \
    CACHE_QUANTIZATION_MODE=real-blockwise \
    bash "$KIVI_RUNNER"
}

echo "[overnight] StructEval reliability allocation queue" | tee "$summary_log"
echo "[overnight] confirm limit: $CONFIRM_LIMIT" | tee -a "$summary_log"
echo "[overnight] recovery limit: $RECOVERY_LIMIT" | tee -a "$summary_log"
echo "[overnight] max_new_tokens: $MAX_NEW_TOKENS" | tee -a "$summary_log"
echo "[overnight] residual length: $RESIDUAL_LENGTH" | tee -a "$summary_log"
echo "[overnight] block size: $QUANTIZE_BLOCK_SIZE" | tee -a "$summary_log"

echo | tee -a "$summary_log"
echo "[overnight] phase A: 100-task boundary confirmation" | tee -a "$summary_log"
run_fp16
run_real_case "confirm K8/V8" "$CONFIRM_LIMIT" 8 "" "" none "" "" prefix both
run_real_case "confirm K8/V4" "$CONFIRM_LIMIT" 8 8 4 none "" "" prefix both
run_real_case "confirm K8/V3" "$CONFIRM_LIMIT" 8 8 3 none "" "" prefix both
run_real_case "confirm K7/V4" "$CONFIRM_LIMIT" 8 7 4 none "" "" prefix both
run_real_case "confirm K6/V8 failing boundary" "$CONFIRM_LIMIT" 8 6 8 none "" "" prefix both
run_real_case "confirm K6/V4 failing boundary" "$CONFIRM_LIMIT" 6 6 4 none "" "" prefix both

echo | tee -a "$summary_log"
echo "[overnight] phase B: first fidelity-allocation recovery probes" | tee -a "$summary_log"
run_real_case "recover K6/V8 key-only constraint paths at 8-bit" "$RECOVERY_LIMIT" 8 6 8 constraint-paths 8 "" score keys
run_real_case "recover K6/V8 key-only JSON syntax + paths at 8-bit" "$RECOVERY_LIMIT" 8 6 8 json-syntax+constraint-paths 8 "" score keys
run_real_case "recover K6/V8 key-only JSON syntax + paths at FP16" "$RECOVERY_LIMIT" 8 6 8 json-syntax+constraint-paths "" "" score keys
run_real_case "recover K6/V4 key-only JSON syntax + paths at 8-bit" "$RECOVERY_LIMIT" 6 6 4 json-syntax+constraint-paths 8 "" score keys
run_real_case "recover K6/V4 key-only JSON syntax + paths at FP16" "$RECOVERY_LIMIT" 6 6 4 json-syntax+constraint-paths "" "" score keys
run_real_case "recover K6/V4 both-side JSON syntax + paths at 8-bit" "$RECOVERY_LIMIT" 6 6 4 json-syntax+constraint-paths 8 "" score both
run_real_case "recover K8/V2 value-side JSON syntax + paths at 8-bit" "$RECOVERY_LIMIT" 8 8 2 json-syntax+constraint-paths 8 "" score values

echo | tee -a "$summary_log"
echo "[overnight] complete: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$summary_log"
echo "[overnight] summary log: $summary_log" | tee -a "$summary_log"
