#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
LOG_DIR="$PROJECT_DIR/toy_kv_experiments/logs"
RESULT_DIR="$PROJECT_DIR/toy_kv_experiments/results/poster_pilot_allformats100"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="$LOG_DIR/poster_pilot_allformats100_${RUN_LABEL}.nohup.log"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_DIR"

nohup setsid env PYTHONUNBUFFERED=1 RUN_LABEL="$RUN_LABEL" \
  bash "$PROJECT_DIR/toy_kv_experiments/server/run_poster_pilot_allformats100.sh" \
  </dev/null >"$LOG_FILE" 2>&1 &
pid=$!

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "[poster-pilot-launch] launch failed; log follows" >&2
  tail -50 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "$pid" > "$RESULT_DIR/latest_queue.pid"
echo "$LOG_FILE" > "$RESULT_DIR/latest_log.txt"
echo "[poster-pilot-launch] pid: $pid"
echo "[poster-pilot-launch] log: $LOG_FILE"
