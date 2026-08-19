#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
OUT_DIR="$PROJECT_DIR/toy_kv_experiments/results"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"

LIMIT="${LIMIT:-20}"
BITS="${BITS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-0}"
STRUCTURE_TOKEN_PROTECTION="${STRUCTURE_TOKEN_PROTECTION:-json-syntax}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
out_path="$OUT_DIR/qwen2_5_7b_structeval_official_json_structure_fidelity_limit${LIMIT}_bits_${BITS}_residual_${RESIDUAL_LENGTH}_protection_${STRUCTURE_TOKEN_PROTECTION}_${timestamp}.json"
log_path="$LOG_DIR/structure_fidelity_limit${LIMIT}_bits_${BITS}_residual_${RESIDUAL_LENGTH}_protection_${STRUCTURE_TOKEN_PROTECTION}_${timestamp}.log"

echo "[structure-fidelity] StructEval JSON syntax-token fidelity test"
echo "[structure-fidelity] limit: $LIMIT"
echo "[structure-fidelity] bits: $BITS"
echo "[structure-fidelity] max_new_tokens: $MAX_NEW_TOKENS"
echo "[structure-fidelity] residual length: $RESIDUAL_LENGTH"
echo "[structure-fidelity] protection: $STRUCTURE_TOKEN_PROTECTION"
echo "[structure-fidelity] output: $out_path"
echo "[structure-fidelity] log:    $log_path"

LIMIT="$LIMIT" \
BITS="$BITS" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
STRUCTURE_TOKEN_PROTECTION="$STRUCTURE_TOKEN_PROTECTION" \
OFFICIAL_STRUCTEVAL=1 \
OUT_PATH="$out_path" \
  bash "$RUNNER" 2>&1 | tee "$log_path"

echo "[structure-fidelity] complete"
