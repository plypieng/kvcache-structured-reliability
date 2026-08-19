#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/toy_kv_experiments/results/structeval_full_official}"
RUN_LABEL="${RUN_LABEL:-qwen2_5_7b_official_v1}"
OUT_PATH="${OUT_PATH:-$RESULT_DIR/${RUN_LABEL}_FP16_inference.json}"

cd "$PROJECT_DIR"
mkdir -p "$RESULT_DIR"

echo "[structeval-full] FP16 official-protocol reproduction"
echo "[structeval-full] output: $OUT_PATH"

LIMIT=2035 \
OUTPUT_TYPE=ALL \
STRUCTEVAL_SAMPLING=head \
MAX_NEW_TOKENS=0 \
DISABLE_END_CODE_STOP=1 \
LOOP_NGRAM_SIZE=0 \
LOOP_REPEAT_THRESHOLD=0 \
OFFICIAL_STRUCTEVAL=1 \
BITS=16 \
KEY_BITS=16 \
VALUE_BITS=16 \
RESIDUAL_LENGTH=128 \
KV_GROUP_SIZE=32 \
CACHE_QUANTIZATION_MODE=real-blockwise \
STRUCTURE_TOKEN_PROTECTION=none \
PROTECTION_SIGNAL_SOURCE=prompt-visible \
RESUME=1 \
OUT_PATH="$OUT_PATH" \
CHECKPOINT_DIR="$OUT_PATH.checkpoints" \
  bash "$RUNNER"

echo "[structeval-full] inference complete: $OUT_PATH"
echo "[structeval-full] next step: render and score this file with the official evaluator"
