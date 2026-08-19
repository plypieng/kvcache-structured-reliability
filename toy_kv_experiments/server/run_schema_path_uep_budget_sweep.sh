#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/toy_kv_experiments/models/qwen2_5_7b_instruct}"
STRUCTEVAL_JSONL="${STRUCTEVAL_JSONL:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/structeval_test.jsonl}"
LIMIT="${LIMIT:-20}"
BASE_BITS="${BASE_BITS:-4}"
PROTECTED_BITS="${PROTECTED_BITS:-8}"
PROTECTION_MODE="${PROTECTION_MODE:-constraint-paths}"
PROTECTION_BUDGET_ORDER="${PROTECTION_BUDGET_ORDER:-score}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/toy_kv_experiments/results}"

mkdir -p "$OUT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"

run_case() {
  local residual_length="$1"
  local target_average_bits="$2"
  local safe_target="${target_average_bits:-none}"
  local out_path="$OUT_DIR/schema_path_uep_budget_limit${LIMIT}_base${BASE_BITS}_protected${PROTECTED_BITS}_residual${residual_length}_target_${safe_target}_${timestamp}.json"

  echo
  echo "[budget-sweep] residual length: $residual_length"
  cmd=(
    python -m toy_kv_experiments.analyze_schema_path_uep_budget
    --model-dir "$MODEL_DIR" \
    --structeval-jsonl "$STRUCTEVAL_JSONL" \
    --output-type JSON \
    --limit "$LIMIT" \
    --base-bits "$BASE_BITS" \
    --protected-bits "$PROTECTED_BITS" \
    --residual-length "$residual_length" \
    --protection-mode "$PROTECTION_MODE" \
    --protection-budget-order "$PROTECTION_BUDGET_ORDER" \
    --official-structeval-prompt \
    --out "$out_path"
  )
  if [ -n "$target_average_bits" ]; then
    cmd+=(--target-average-bits "$target_average_bits")
  fi
  "${cmd[@]}"
}

echo "[budget-sweep] Schema-path UEP tokenizer-only budget sweep"
echo "[budget-sweep] limit: $LIMIT"
echo "[budget-sweep] base bits: $BASE_BITS"
echo "[budget-sweep] protected bits: $PROTECTED_BITS"
echo "[budget-sweep] protection mode: $PROTECTION_MODE"
echo "[budget-sweep] protection budget order: $PROTECTION_BUDGET_ORDER"

run_case 128 ""
run_case 64 ""
run_case 64 "6.5"
run_case 32 ""
run_case 32 "6.0"

echo
echo "[budget-sweep] complete"
