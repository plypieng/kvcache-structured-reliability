#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$HOME/official_baselines/results/kivi_longbench_mistral_v02_$RUN_ID}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_official_kivi_longbench.sh"
COMPARATOR="$PROJECT_DIR/toy_kv_experiments/server/compare_official_kivi_longbench.py"
PREFLIGHT_TASK="${PREFLIGHT_TASK:-qasper}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
FULL_TASKS="${FULL_TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,musique,2wikimqa,gov_report,qmsum,multi_news,lcc,repobench-p,triviaqa,samsum,trec,passage_retrieval_en}"

mkdir -p "$RUN_ROOT"
exec 9>"$HOME/official_baselines/results/.official_kivi_reproduction.lock"
if ! flock -n 9; then
  echo "Another official KIVI reproduction queue holds the lock." >&2
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
printf '%s\n' "$RUN_ROOT" > "$HOME/official_baselines/results/CURRENT_KIVI_RUN"

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  echo "[official-kivi] FP16 one-example preflight"
  OUTPUT_DIR="$RUN_ROOT/preflight_fp16" \
  CONDITION=fp16 \
  K_BITS=16 \
  V_BITS=16 \
  TASKS="$PREFLIGHT_TASK" \
  LIMIT_PER_TASK=1 \
    bash "$RUNNER"

  echo "[official-kivi] KIVI-4 one-example preflight"
  OUTPUT_DIR="$RUN_ROOT/preflight_kivi4" \
  CONDITION=kivi \
  K_BITS=4 \
  V_BITS=4 \
  TASKS="$PREFLIGHT_TASK" \
  LIMIT_PER_TASK=1 \
    bash "$RUNNER"

  python "$COMPARATOR" \
    --fp16-dir "$RUN_ROOT/preflight_fp16" \
    --kivi4-dir "$RUN_ROOT/preflight_kivi4" \
    --output-json "$RUN_ROOT/preflight_comparison.json" \
    --output-md "$RUN_ROOT/preflight_comparison.md"
fi

echo "[official-kivi] Full FP16 LongBench-15"
OUTPUT_DIR="$RUN_ROOT/full_fp16" \
CONDITION=fp16 \
K_BITS=16 \
V_BITS=16 \
TASKS="$FULL_TASKS" \
LIMIT_PER_TASK=0 \
  bash "$RUNNER"

echo "[official-kivi] Full KIVI-4 LongBench-15"
OUTPUT_DIR="$RUN_ROOT/full_kivi4" \
CONDITION=kivi \
K_BITS=4 \
V_BITS=4 \
TASKS="$FULL_TASKS" \
LIMIT_PER_TASK=0 \
  bash "$RUNNER"

python "$COMPARATOR" \
  --fp16-dir "$RUN_ROOT/full_fp16" \
  --kivi4-dir "$RUN_ROOT/full_kivi4" \
  --output-json "$RUN_ROOT/comparison.json" \
  --output-md "$RUN_ROOT/comparison.md"

date -Iseconds > "$RUN_ROOT/COMPLETE"
echo "[official-kivi] Complete: $RUN_ROOT"
