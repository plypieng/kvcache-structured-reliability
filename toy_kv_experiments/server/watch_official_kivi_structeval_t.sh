#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-$HOME/official_baselines/results}"
CURRENT_FILE="$RESULT_ROOT/CURRENT_KIVI_STRUCTEVAL_T_RUN"

if [[ ! -f "$CURRENT_FILE" ]]; then
  echo "No official KIVI StructEval-T run is registered."
  exit 0
fi

run_root="$(cat "$CURRENT_FILE")"
echo "run_root: $run_root"
if [[ -f "$run_root/COMPLETE" ]]; then
  echo "queue_status: complete ($(cat "$run_root/COMPLETE"))"
elif [[ -f "$run_root/FAILED" ]]; then
  echo "queue_status: failed ($(cat "$run_root/FAILED"))"
elif [[ -f "$run_root/queue.pid" ]] && kill -0 "$(cat "$run_root/queue.pid")" 2>/dev/null; then
  echo "queue_status: running (pid $(cat "$run_root/queue.pid"))"
else
  echo "queue_status: interrupted or not running"
fi

while IFS= read -r metadata; do
  python3 - "$metadata" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
print(
    f"{sys.argv[1]}: status={data.get('status')} "
    f"progress={data.get('completed', 0)}/{data.get('total', 0)} "
    f"last_task={data.get('last_task_id', '-')} "
    f"last_s={float(data.get('last_sample_seconds', 0)):.1f} "
    f"elapsed_h={float(data.get('elapsed_seconds', 0))/3600:.2f}"
)
PY
done < <(find "$run_root" -name run_metadata.json -type f | sort)

if [[ -f "$run_root/comparison.md" ]]; then
  echo
  cat "$run_root/comparison.md"
fi
