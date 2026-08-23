#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
SOURCE_JSONL="${SOURCE_JSONL:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/structeval_test.jsonl}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/manifests/protection_pilot_failure_control_8.json}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/toy_kv_experiments/results/protection_pilot}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/toy_kv_experiments/logs/protection_pilot}"
MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.2}"
MODEL_DIR="${MODEL_DIR:-$HOME/official_baselines/model_cache/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873}"

LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
QUANTIZE_BLOCK_SIZE="${QUANTIZE_BLOCK_SIZE:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"
TARGET_AVERAGE_BITS="${TARGET_AVERAGE_BITS:-10.5}"
PROTECTED_BITS="${PROTECTED_BITS:-8}"
PROTECTION_RANDOM_SEED="${PROTECTION_RANDOM_SEED:-20260823}"
GENERATION_SEED="${GENERATION_SEED:-123}"
TASK_COUNT="${TASK_COUNT:-$(grep -c '"task_id"' "$MANIFEST")}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
summary_log="$LOG_DIR/matched_budget_protection_pilot_${timestamp}.log"

echo "[protection-pilot] matched-budget StructEval failure/control pilot" | tee "$summary_log"
echo "[protection-pilot] source: $SOURCE_JSONL" | tee -a "$summary_log"
echo "[protection-pilot] manifest: $MANIFEST" | tee -a "$summary_log"
echo "[protection-pilot] model: $MODEL_ID" | tee -a "$summary_log"
echo "[protection-pilot] model directory: $MODEL_DIR" | tee -a "$summary_log"
echo "[protection-pilot] tasks: $TASK_COUNT (failure/control manifest)" | tee -a "$summary_log"
echo "[protection-pilot] cache: real-blockwise K4/V4, residual=$RESIDUAL_LENGTH, group=$KV_GROUP_SIZE, block=$QUANTIZE_BLOCK_SIZE" | tee -a "$summary_log"
echo "[protection-pilot] protected bits: $PROTECTED_BITS; storage ceiling: $TARGET_AVERAGE_BITS average bits" | tee -a "$summary_log"
echo "[protection-pilot] generation seed: $GENERATION_SEED; random protection seed: $PROTECTION_RANDOM_SEED" | tee -a "$summary_log"
echo "[protection-pilot] note: the ceiling is a fixed upper bound; the uniform KIVI-4 condition remains the lower-storage baseline" | tee -a "$summary_log"

run_case() {
  local label="$1"
  local condition="$2"
  local protection="$3"
  local order="$4"
  local protected_bits="$5"
  local target_bits="$6"
  local random_seed="$7"
  local safe_condition
  safe_condition="$(printf '%s' "$condition" | tr -c 'A-Za-z0-9_' '_')"
  local out_path="$OUT_DIR/${condition}_${timestamp}.json"

  echo | tee -a "$summary_log"
  echo "[protection-pilot] $(date '+%Y-%m-%d %H:%M:%S') start: $label" | tee -a "$summary_log"
  LIMIT="$LIMIT" \
  MODEL_ID="$MODEL_ID" \
  MODEL_DIR="$MODEL_DIR" \
  STRUCTEVAL_JSONL="$SOURCE_JSONL" \
  STRUCTEVAL_MANIFEST="$MANIFEST" \
  OUTPUT_TYPE=ALL \
  STRUCTEVAL_SAMPLING=stratified \
  STRUCTEVAL_SEED=42 \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  BITS=4 \
  KEY_BITS=4 \
  VALUE_BITS=4 \
  RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
  QUANTIZE_BLOCK_SIZE="$QUANTIZE_BLOCK_SIZE" \
  KV_GROUP_SIZE="$KV_GROUP_SIZE" \
  STRUCTURE_TOKEN_PROTECTION="$protection" \
  PROTECTED_BITS="$protected_bits" \
  TARGET_AVERAGE_BITS="$target_bits" \
  PROTECTION_BUDGET_ORDER="$order" \
  PROTECTION_RANDOM_SEED="$random_seed" \
  PROTECTION_TARGET=both \
  PROTECTION_SIGNAL_SOURCE=prompt-visible \
  CACHE_QUANTIZATION_MODE=real-blockwise \
  OFFICIAL_STRUCTEVAL=1 \
  OFFICIAL_MODEL_NAME="$MODEL_ID" \
  GENERATION_SEED="$GENERATION_SEED" \
  RESUME=1 \
  OUT_PATH="$out_path" \
    bash "$RUNNER" 2>&1 | tee -a "$summary_log"
  echo "[protection-pilot] $(date '+%Y-%m-%d %H:%M:%S') complete: $condition" | tee -a "$summary_log"
}

# Every condition uses the same 8-task manifest, model, prompt wrapper, greedy
# decoding, residual window, and packed-cache storage ceiling.
run_case "uniform KIVI-4 baseline" "uniform_kivi4" "none" "prefix" "" "$TARGET_AVERAGE_BITS" "$PROTECTION_RANDOM_SEED"
run_case "random protection" "random_protection" "all" "random" "$PROTECTED_BITS" "$TARGET_AVERAGE_BITS" "$PROTECTION_RANDOM_SEED"
run_case "recency protection" "recency_protection" "all" "recent" "$PROTECTED_BITS" "$TARGET_AVERAGE_BITS" "$PROTECTION_RANDOM_SEED"
run_case "structure-aware syntax plus prompt paths" "structure_aware_protection" "json-syntax+constraint-paths" "score" "$PROTECTED_BITS" "$TARGET_AVERAGE_BITS" "$PROTECTION_RANDOM_SEED"

echo | tee -a "$summary_log"
echo "[protection-pilot] complete: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$summary_log"
echo "[protection-pilot] summary log: $summary_log" | tee -a "$summary_log"
