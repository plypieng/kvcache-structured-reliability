#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/manifests/json_50_stratified_seed42.json}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_kivi_reference_json100.sh"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"

RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_DIR/validity_gate_json50_${timestamp}.log"

run_case() {
  local label="$1"
  local key_bits="$2"
  local value_bits="$3"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] start $label" | tee -a "$summary_log"
  MANIFEST="$MANIFEST" \
  KEY_BITS="$key_bits" \
  VALUE_BITS="$value_bits" \
  BITS=8 \
  RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
  KV_GROUP_SIZE="$KV_GROUP_SIZE" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    bash "$RUNNER" 2>&1 | tee -a "$summary_log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] complete $label" | tee -a "$summary_log"
}

echo "Validity-gate JSON-50 queue" | tee "$summary_log"
echo "manifest: $MANIFEST" | tee -a "$summary_log"
echo "residual/group: $RESIDUAL_LENGTH/$KV_GROUP_SIZE" | tee -a "$summary_log"

# Run the key-sensitive case first so the most informative comparison finishes first.
run_case "K4/V8" 4 8
run_case "K8/V4" 8 4
run_case "K4/V4" 4 4

echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue complete" | tee -a "$summary_log"
