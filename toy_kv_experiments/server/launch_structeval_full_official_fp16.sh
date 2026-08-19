#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUN_LABEL="${RUN_LABEL:-qwen2_5_7b_official_v1}"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"
STATE_DIR="$PROJECT_DIR/toy_kv_experiments/results/structeval_full_official"
PID_FILE="$STATE_DIR/${RUN_LABEL}_FP16.pid"
LATEST_LOG_FILE="$STATE_DIR/${RUN_LABEL}_FP16.latest_log.txt"

mkdir -p "$LOG_DIR" "$STATE_DIR"

if [ -f "$PID_FILE" ]; then
  existing_pid="$(cat "$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "FP16 full StructEval run is already active with PID $existing_pid"
    echo "log: $(cat "$LATEST_LOG_FILE")"
    exit 0
  fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/structeval_full_official_FP16_${timestamp}.nohup.log"

cd "$PROJECT_DIR"
nohup env PROJECT_DIR="$PROJECT_DIR" RUN_LABEL="$RUN_LABEL" \
  bash toy_kv_experiments/server/run_structeval_full_official_fp16.sh \
  >"$log_file" 2>&1 &
pid=$!

printf '%s\n' "$pid" >"$PID_FILE"
printf '%s\n' "$log_file" >"$LATEST_LOG_FILE"

echo "started PID $pid"
echo "run label: $RUN_LABEL"
echo "log: $log_file"
echo "status: $STATE_DIR/${RUN_LABEL}_FP16_inference.json.checkpoints/status.json"
