#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-$HOME/official_baselines/results}"
CURRENT_FILE="$RESULT_ROOT/CURRENT_KIVI_RUN"
WATCH="${WATCH:-0}"
INTERVAL="${INTERVAL:-30}"

show_status() {
  if [[ ! -f "$CURRENT_FILE" ]]; then
    echo "No official KIVI run has been registered."
    return
  fi

  local run_root
  run_root="$(cat "$CURRENT_FILE")"
  echo "run_root: $run_root"

  if [[ -f "$run_root/COMPLETE" ]]; then
    echo "queue_status: complete ($(cat "$run_root/COMPLETE"))"
  elif [[ -f "$run_root/queue.pid" ]] && kill -0 "$(cat "$run_root/queue.pid")" 2>/dev/null; then
    echo "queue_status: running (pid $(cat "$run_root/queue.pid"))"
  else
    echo "queue_status: not running or interrupted"
  fi

  local metadata
  while IFS= read -r metadata; do
    python3 - "$metadata" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

mean = data.get("mean_score", "pending")
elapsed = int(data.get("elapsed_seconds", 0))
print(
    f"condition={data.get('condition', 'unknown')} "
    f"status={data.get('status', 'unknown')} "
    f"mean={mean} elapsed_s={elapsed}"
)

active = [
    (task, status)
    for task, status in data.get("task_status", {}).items()
    if status.get("status") != "complete"
]
if active:
    task, status = active[-1]
    print(
        f"  active_task={task} "
        f"progress={status.get('completed', 0)}/{status.get('total', 0)} "
        f"last_s={int(status.get('last_sample_seconds', 0))}"
    )
PY
  done < <(find "$run_root" -name run_metadata.json -type f | sort)

  if [[ -f "$run_root/comparison.md" ]]; then
    echo
    sed -n '1,12p' "$run_root/comparison.md"
  fi
}

if [[ "$WATCH" == "1" ]]; then
  while true; do
    clear
    date -Iseconds
    show_status
    sleep "$INTERVAL"
  done
else
  show_status
fi
