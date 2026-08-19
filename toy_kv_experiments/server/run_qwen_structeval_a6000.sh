#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
ENV_NAME="${ENV_NAME:-kvcache-py311}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/toy_kv_experiments/models/qwen2_5_7b_instruct}"
STRUCTEVAL_JSONL="${STRUCTEVAL_JSONL:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/structeval_test.jsonl}"
STRUCTEVAL_MANIFEST="${STRUCTEVAL_MANIFEST:-}"
LIMIT="${LIMIT:-10}"
STRUCTEVAL_SAMPLING="${STRUCTEVAL_SAMPLING:-stratified}"
STRUCTEVAL_SEED="${STRUCTEVAL_SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
OUTPUT_TYPE="${OUTPUT_TYPE:-JSON}"
BITS="${BITS:-16}"
KEY_BITS="${KEY_BITS:-}"
VALUE_BITS="${VALUE_BITS:-}"
OFFICIAL_STRUCTEVAL="${OFFICIAL_STRUCTEVAL:-1}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-0}"
STRUCTURE_TOKEN_PROTECTION="${STRUCTURE_TOKEN_PROTECTION:-none}"
PROTECTION_SIGNAL_SOURCE="${PROTECTION_SIGNAL_SOURCE:-prompt-visible}"
ALLOW_ORACLE_PROTECTION="${ALLOW_ORACLE_PROTECTION:-0}"
PROTECTED_BITS="${PROTECTED_BITS:-}"
TARGET_AVERAGE_BITS="${TARGET_AVERAGE_BITS:-}"
PROTECTION_BUDGET_ORDER="${PROTECTION_BUDGET_ORDER:-prefix}"
PROTECTION_TARGET="${PROTECTION_TARGET:-both}"
CACHE_QUANTIZATION_MODE="${CACHE_QUANTIZATION_MODE:-repeated}"
QUANTIZE_BLOCK_SIZE="${QUANTIZE_BLOCK_SIZE:-1}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"
DISABLE_END_CODE_STOP="${DISABLE_END_CODE_STOP:-0}"
LOOP_NGRAM_SIZE="${LOOP_NGRAM_SIZE:-16}"
LOOP_REPEAT_THRESHOLD="${LOOP_REPEAT_THRESHOLD:-6}"
GENERATION_SEED="${GENERATION_SEED:-123}"
RESUME="${RESUME:-1}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/toy_kv_experiments/results}"

timestamp="$(date +%Y%m%d_%H%M%S)"
safe_bits="$(echo "$BITS" | tr ' ' '_')"
safe_key_bits="${KEY_BITS:-same}"
safe_value_bits="${VALUE_BITS:-same}"
safe_protection="$(echo "$STRUCTURE_TOKEN_PROTECTION" | tr -c 'A-Za-z0-9_' '_')"
safe_mode="$(echo "$CACHE_QUANTIZATION_MODE" | tr -c 'A-Za-z0-9_' '_')"
OUT_PATH="${OUT_PATH:-$OUT_DIR/qwen2_5_7b_structeval_json_bits_${safe_bits}_K_${safe_key_bits}_V_${safe_value_bits}_mode_${safe_mode}_residual_${RESIDUAL_LENGTH}_group_${KV_GROUP_SIZE}_block_${QUANTIZE_BLOCK_SIZE}_protection_${safe_protection}_${timestamp}.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUT_PATH.checkpoints}"

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
mkdir -p "$MODEL_DIR" "$OUT_DIR" "$PROJECT_DIR/toy_kv_experiments/logs"

echo "[run] host: $(hostname)"
echo "[run] model id: $MODEL_ID"
echo "[run] model dir: $MODEL_DIR"
echo "[run] structeval jsonl: $STRUCTEVAL_JSONL"
if [ -n "$STRUCTEVAL_MANIFEST" ]; then
  echo "[run] StructEval manifest: $STRUCTEVAL_MANIFEST"
fi
echo "[run] limit: $LIMIT"
echo "[run] StructEval sampling: $STRUCTEVAL_SAMPLING"
echo "[run] StructEval seed: $STRUCTEVAL_SEED"
if [ "$MAX_NEW_TOKENS" -le 0 ]; then
  echo "[run] max new tokens: remaining model context (official protocol has no explicit output cap)"
else
  echo "[run] max new tokens: $MAX_NEW_TOKENS (fixed experimental cap)"
fi
echo "[run] output type filter: $OUTPUT_TYPE"
echo "[run] bits: $BITS"
echo "[run] key bits override: ${KEY_BITS:-same as bits}"
echo "[run] value bits override: ${VALUE_BITS:-same as bits}"
echo "[run] residual length: $RESIDUAL_LENGTH"
echo "[run] structure token protection: $STRUCTURE_TOKEN_PROTECTION"
echo "[run] protection signal source: $PROTECTION_SIGNAL_SOURCE"
echo "[run] oracle evaluator-path override: $ALLOW_ORACLE_PROTECTION"
echo "[run] protected bits: ${PROTECTED_BITS:-fp16 copy-back}"
echo "[run] target average bits: ${TARGET_AVERAGE_BITS:-none}"
echo "[run] protection budget order: $PROTECTION_BUDGET_ORDER"
echo "[run] protection target: $PROTECTION_TARGET"
echo "[run] cache quantization mode: $CACHE_QUANTIZATION_MODE"
echo "[run] quantize block size: $QUANTIZE_BLOCK_SIZE"
echo "[run] KIVI group size: $KV_GROUP_SIZE"
echo "[run] stop on END_CODE: $([ "$DISABLE_END_CODE_STOP" = "1" ] && echo disabled || echo enabled)"
echo "[run] loop detector: ngram=$LOOP_NGRAM_SIZE repeat=$LOOP_REPEAT_THRESHOLD"
echo "[run] generation seed: $GENERATION_SEED"
echo "[run] official StructEval prompt/extraction: $OFFICIAL_STRUCTEVAL"
echo "[run] output: $OUT_PATH"
echo "[run] checkpoints: $CHECKPOINT_DIR (resume=$RESUME)"
nvidia-smi

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[run] downloading model"
  hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
else
  echo "[run] local model already exists"
fi

official_args=()
if [ "$OFFICIAL_STRUCTEVAL" = "1" ]; then
  official_args+=(--official-structeval-prompt)
  official_args+=(--official-structeval-extraction)
  official_args+=(--official-model-name "$MODEL_ID")
fi

manifest_args=()
if [ -n "$STRUCTEVAL_MANIFEST" ]; then
  manifest_args+=(--structeval-manifest "$STRUCTEVAL_MANIFEST")
fi

mixed_bits_args=()
if [ -n "$KEY_BITS" ]; then
  mixed_bits_args+=(--key-bits "$KEY_BITS")
fi
if [ -n "$VALUE_BITS" ]; then
  mixed_bits_args+=(--value-bits "$VALUE_BITS")
fi
if [ -n "$PROTECTED_BITS" ]; then
  mixed_bits_args+=(--protected-bits "$PROTECTED_BITS")
fi
if [ -n "$TARGET_AVERAGE_BITS" ]; then
  mixed_bits_args+=(--target-average-bits "$TARGET_AVERAGE_BITS")
fi
mixed_bits_args+=(--protection-budget-order "$PROTECTION_BUDGET_ORDER")
mixed_bits_args+=(--protection-target "$PROTECTION_TARGET")
mixed_bits_args+=(--loop-ngram-size "$LOOP_NGRAM_SIZE")
mixed_bits_args+=(--loop-repeat-threshold "$LOOP_REPEAT_THRESHOLD")
mixed_bits_args+=(--generation-seed "$GENERATION_SEED")
mixed_bits_args+=(--checkpoint-dir "$CHECKPOINT_DIR")
if [ "$RESUME" = "1" ]; then
  mixed_bits_args+=(--resume)
fi
if [ "$DISABLE_END_CODE_STOP" = "1" ]; then
  mixed_bits_args+=(--disable-end-code-stop)
fi
if [ "$ALLOW_ORACLE_PROTECTION" = "1" ]; then
  mixed_bits_args+=(--allow-oracle-protection)
fi

python -m toy_kv_experiments.pretrained_kv_quantization \
  --model-dir "$MODEL_DIR" \
  --structeval-jsonl "$STRUCTEVAL_JSONL" \
  --output-type "$OUTPUT_TYPE" \
  --structeval-limit "$LIMIT" \
  --structeval-sampling "$STRUCTEVAL_SAMPLING" \
  --structeval-seed "$STRUCTEVAL_SEED" \
  "${manifest_args[@]}" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --bits $BITS \
  "${mixed_bits_args[@]}" \
  --residual-length "$RESIDUAL_LENGTH" \
  --structure-token-protection "$STRUCTURE_TOKEN_PROTECTION" \
  --protection-signal-source "$PROTECTION_SIGNAL_SOURCE" \
  --cache-quantization-mode "$CACHE_QUANTIZATION_MODE" \
  --quantize-block-size "$QUANTIZE_BLOCK_SIZE" \
  --kv-group-size "$KV_GROUP_SIZE" \
  "${official_args[@]}" \
  --out "$OUT_PATH"

echo "[run] saved result: $OUT_PATH"
