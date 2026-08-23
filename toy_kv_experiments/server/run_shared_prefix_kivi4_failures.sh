#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
KIVI_REPO="${KIVI_REPO:-$HOME/official_baselines/KIVI-paper}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/official_baselines/model_cache}"
BASELINE_ROOT="${BASELINE_ROOT:-$HOME/official_baselines/results/kivi_structeval_t_20260729_170807}"
SELECTION_CSV="${SELECTION_CSV:-$PROJECT_DIR/artifacts/structeval_t_20260731/analysis/parse_failure_annotations.csv}"
SELECTION_LABEL="${SELECTION_LABEL:-KIVI-4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$HOME/official_baselines/results/shared_prefix_kivi4_failures_$RUN_ID}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-kivi-paper}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/shared_prefix_sensitivity.py"

mkdir -p "$RUN_ROOT"
exec 9>"$HOME/official_baselines/results/.shared_prefix_sensitivity.lock"
if ! flock -n 9; then
  echo "Another shared-prefix experiment holds the lock." >&2
  exit 3
fi

gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
if [[ -n "$gpu_processes" ]]; then
  echo "The GPU already has compute processes; refusing to start." >&2
  echo "$gpu_processes" >&2
  exit 4
fi

test -f "$SELECTION_CSV"
test -f "$BASELINE_ROOT/full_fp16/eval/evaluation.json"
test -f "$RUNNER"

printf '%s\n' "$RUN_ROOT" > "$HOME/official_baselines/results/CURRENT_SHARED_PREFIX_RUN"
date -Iseconds > "$RUN_ROOT/STARTED"
echo "$$" > "$RUN_ROOT/queue.pid"

mark_failed() {
  local exit_code=$?
  printf '%s exit=%s\n' "$(date -Iseconds)" "$exit_code" > "$RUN_ROOT/FAILED"
  exit "$exit_code"
}
trap mark_failed ERR

# shellcheck disable=SC1091
set +u
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

common_args=(
  --kivi-repo "$KIVI_REPO"
  --model-cache-dir "$MODEL_CACHE_DIR"
  --fp16-evaluation "$BASELINE_ROOT/full_fp16/eval/evaluation.json"
  --selection-csv "$SELECTION_CSV"
  --selection-condition "$SELECTION_LABEL"
  --seed 42
  --top-k 5
)

python "$RUNNER" \
  "${common_args[@]}" \
  --condition fp16 \
  --output "$RUN_ROOT/fp16_trace.json"

python "$RUNNER" \
  "${common_args[@]}" \
  --condition kivi \
  --k-bits 4 \
  --v-bits 4 \
  --fp16-trace "$RUN_ROOT/fp16_trace.json" \
  --output "$RUN_ROOT/kivi4_trace.json"

date -Iseconds > "$RUN_ROOT/COMPLETE"
rm -f "$RUN_ROOT/FAILED"
trap - ERR
echo "complete: $RUN_ROOT"
