#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$HOME/official_baselines/results/kivi2_longbench_mistral_v02_$RUN_ID}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_official_kivi_longbench.sh"
PREFLIGHT_TASK="${PREFLIGHT_TASK:-qasper}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
FULL_TASKS="${FULL_TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,musique,2wikimqa,gov_report,qmsum,multi_news,lcc,repobench-p,triviaqa,samsum,trec,passage_retrieval_en}"

mkdir -p "$RUN_ROOT"
exec 9>"$HOME/official_baselines/results/.official_kivi_reproduction.lock"
if ! flock -n 9; then
  echo "Another official KIVI reproduction holds the lock." >&2
  exit 3
fi

gpu_processes="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
)"
if [[ -n "$gpu_processes" && "$ALLOW_BUSY_GPU" != "1" ]]; then
  echo "The GPU already has compute processes; refusing to start." >&2
  echo "$gpu_processes" >&2
  exit 4
fi

echo "$$" > "$RUN_ROOT/queue.pid"
printf '%s\n' "$RUN_ROOT" > "$HOME/official_baselines/results/CURRENT_KIVI2_RUN"

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  echo "[official-kivi2] One-example preflight"
  OUTPUT_DIR="$RUN_ROOT/preflight_kivi2" \
  CONDITION=kivi \
  K_BITS=2 \
  V_BITS=2 \
  TASKS="$PREFLIGHT_TASK" \
  LIMIT_PER_TASK=1 \
    bash "$RUNNER"
fi

echo "[official-kivi2] Full KIVI-2 LongBench-15"
OUTPUT_DIR="$RUN_ROOT/full_kivi2" \
CONDITION=kivi \
K_BITS=2 \
V_BITS=2 \
TASKS="$FULL_TASKS" \
LIMIT_PER_TASK=0 \
  bash "$RUNNER"

date -Iseconds > "$RUN_ROOT/COMPLETE"
echo "[official-kivi2] Complete: $RUN_ROOT"
