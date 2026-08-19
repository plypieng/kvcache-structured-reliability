#!/usr/bin/env bash
set -euo pipefail

# Controlled KIVI-style storage/fidelity reference on the frozen balanced JSON set.
PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/manifests/json_100_stratified_seed42.json}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"

BITS="${BITS:-8}"
KEY_BITS="${KEY_BITS:-}"
VALUE_BITS="${VALUE_BITS:-}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

echo "[kivi-reference] frozen manifest: $MANIFEST"
echo "[kivi-reference] K/V bits: ${KEY_BITS:-$BITS}/${VALUE_BITS:-$BITS}"
echo "[kivi-reference] residual/group: $RESIDUAL_LENGTH/$KV_GROUP_SIZE"

LIMIT=100 \
STRUCTEVAL_MANIFEST="$MANIFEST" \
STRUCTURE_TOKEN_PROTECTION=none \
TARGET_AVERAGE_BITS='' \
PROTECTED_BITS='' \
CACHE_QUANTIZATION_MODE=real-blockwise \
QUANTIZE_BLOCK_SIZE=1 \
KV_GROUP_SIZE="$KV_GROUP_SIZE" \
BITS="$BITS" \
KEY_BITS="$KEY_BITS" \
VALUE_BITS="$VALUE_BITS" \
RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  bash "$RUNNER"
