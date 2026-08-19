#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
OUT_DIR="$PROJECT_DIR/toy_kv_experiments/results"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
baseline_out="$OUT_DIR/qwen2_5_7b_structeval_official_json_limit50_bits_16_8_${timestamp}.json"
baseline_log="$LOG_DIR/next_seminar_limit50_bits_16_8_${timestamp}.log"
collapse_out="$OUT_DIR/qwen2_5_7b_structeval_official_json_limit20_bits_4_2_${timestamp}.json"
collapse_log="$LOG_DIR/next_seminar_limit20_bits_4_2_${timestamp}.log"

echo "[seminar] job 1/2: baseline stability, LIMIT=50, BITS=16 8"
echo "[seminar] output: $baseline_out"
echo "[seminar] log:    $baseline_log"
LIMIT=50 \
BITS="16 8" \
MAX_NEW_TOKENS=1024 \
OFFICIAL_STRUCTEVAL=1 \
OUT_PATH="$baseline_out" \
  bash "$RUNNER" 2>&1 | tee "$baseline_log"

echo "[seminar] job 2/2: low-bit collapse check, LIMIT=20, BITS=4 2"
echo "[seminar] output: $collapse_out"
echo "[seminar] log:    $collapse_log"
LIMIT=20 \
BITS="4 2" \
MAX_NEW_TOKENS=1024 \
OFFICIAL_STRUCTEVAL=1 \
OUT_PATH="$collapse_out" \
  bash "$RUNNER" 2>&1 | tee "$collapse_log"

echo "[seminar] complete"
echo "[seminar] baseline result: $baseline_out"
echo "[seminar] collapse result: $collapse_out"
