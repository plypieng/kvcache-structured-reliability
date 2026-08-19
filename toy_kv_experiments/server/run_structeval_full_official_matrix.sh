#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/toy_kv_experiments/results/structeval_full_official}"
RUN_LABEL="${RUN_LABEL:-qwen2_5_7b_official_v1}"
STRUCTEVAL_JSONL="${STRUCTEVAL_JSONL:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/structeval_test.jsonl}"
ALLOW_LONG_MATRIX="${ALLOW_LONG_MATRIX:-0}"

if [ "$ALLOW_LONG_MATRIX" != "1" ]; then
  cat >&2 <<'EOF'
Refusing to start the full compressed matrix without ALLOW_LONG_MATRIX=1.
At the measured unfused reference-cache throughput, the compressed conditions
can occupy one RTX A6000 for multiple weeks. Run and score FP16 first.
EOF
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p "$RESULT_DIR"

run_case() {
  local label="$1"
  local bits="$2"
  local key_bits="$3"
  local value_bits="$4"
  local output_path="$RESULT_DIR/${RUN_LABEL}_${label}_inference.json"

  echo "[structeval-full-matrix] start $label -> $output_path"
  LIMIT=2035 \
  OUTPUT_TYPE=ALL \
  STRUCTEVAL_SAMPLING=head \
  MAX_NEW_TOKENS=0 \
  DISABLE_END_CODE_STOP=1 \
  LOOP_NGRAM_SIZE=0 \
  LOOP_REPEAT_THRESHOLD=0 \
  OFFICIAL_STRUCTEVAL=1 \
  BITS="$bits" \
  KEY_BITS="$key_bits" \
  VALUE_BITS="$value_bits" \
  RESIDUAL_LENGTH=128 \
  KV_GROUP_SIZE=32 \
  CACHE_QUANTIZATION_MODE=real-blockwise \
  STRUCTURE_TOKEN_PROTECTION=none \
  PROTECTION_SIGNAL_SOURCE=prompt-visible \
  RESUME=1 \
  OUT_PATH="$output_path" \
  CHECKPOINT_DIR="$output_path.checkpoints" \
    bash "$RUNNER"
  echo "[structeval-full-matrix] complete $label"

  python -m toy_kv_experiments.check_structeval_official_artifact inference \
    "$output_path" \
    --dataset "$STRUCTEVAL_JSONL" \
    --expected-key-bits "$key_bits" \
    --expected-value-bits "$value_bits"
}

echo "[structeval-full-matrix] run label: $RUN_LABEL"
fp16_output="$RESULT_DIR/${RUN_LABEL}_FP16_inference.json"
if [ -f "$fp16_output" ] && \
  python -m toy_kv_experiments.check_structeval_official_artifact inference \
    "$fp16_output" \
    --dataset "$STRUCTEVAL_JSONL" \
    --expected-key-bits 16 \
    --expected-value-bits 16; then
  echo "[structeval-full-matrix] reuse completed FP16 inference: $fp16_output"
else
  if [ -f "$fp16_output" ]; then
    echo "[structeval-full-matrix] FP16 artifact is incomplete; resume it before compression"
  fi
  run_case FP16 16 16 16
fi
run_case K8_V8 8 8 8
run_case K8_V4 8 8 4
run_case K4_V8 8 4 8
run_case K4_V4 4 4 4
run_case K2_V2 2 2 2

echo "[structeval-full-matrix] all inference conditions complete"
