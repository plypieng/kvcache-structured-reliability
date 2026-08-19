#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
ENV_NAME="${ENV_NAME:-kvcache-py311}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
SOURCE="${SOURCE:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/structeval_test.jsonl}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/manifests/text20_from_allformats100_seed42.json}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/toy_kv_experiments/results/fidelity_dev_text20}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="$RESULT_DIR/${RUN_LABEL}_K4_V4_inference.json"

cd "$PROJECT_DIR"
mkdir -p "$RESULT_DIR"

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m toy_kv_experiments.check_structeval_text_dev_manifest \
  --source "$SOURCE" \
  --manifest "$MANIFEST" \
  --expected-rows 20

echo "[text20-k4v4] run label: $RUN_LABEL"
echo "[text20-k4v4] frozen manifest: $MANIFEST"
echo "[text20-k4v4] output: $OUTPUT_PATH"

LIMIT=20 \
OUTPUT_TYPE=ALL \
STRUCTEVAL_MANIFEST="$MANIFEST" \
MAX_NEW_TOKENS=2048 \
DISABLE_END_CODE_STOP=0 \
LOOP_NGRAM_SIZE=0 \
LOOP_REPEAT_THRESHOLD=0 \
OFFICIAL_STRUCTEVAL=1 \
BITS=4 \
KEY_BITS=4 \
VALUE_BITS=4 \
RESIDUAL_LENGTH=128 \
KV_GROUP_SIZE=32 \
CACHE_QUANTIZATION_MODE=real-blockwise \
STRUCTURE_TOKEN_PROTECTION=none \
PROTECTION_SIGNAL_SOURCE=prompt-visible \
RESUME=1 \
OUT_PATH="$OUTPUT_PATH" \
CHECKPOINT_DIR="$OUTPUT_PATH.checkpoints" \
  bash "$RUNNER"

conda activate "$ENV_NAME"
python -m toy_kv_experiments.check_structeval_text_dev_result \
  "$OUTPUT_PATH" \
  --source "$SOURCE" \
  --manifest "$MANIFEST" \
  --expected-rows 20 \
  --key-bits 4 \
  --value-bits 4

INFERENCE_PATH="$OUTPUT_PATH" \
RUN_DIR="$RESULT_DIR/${RUN_LABEL}_K4_V4_text_eval" \
EXPECTED_TEXT_ROWS=20 \
EXPECTED_CATEGORY_COUNT=10 \
  bash "$PROJECT_DIR/toy_kv_experiments/server/evaluate_structeval_poster_t.sh"

echo "[text20-k4v4] complete: $OUTPUT_PATH"
