#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$HOME/official_baselines/results/kivi_structeval_t_$RUN_ID}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_official_kivi_structeval_t.sh"
EVALUATOR="$PROJECT_DIR/toy_kv_experiments/server/evaluate_structeval_t.sh"
COMPARATOR="$PROJECT_DIR/toy_kv_experiments/server/compare_official_kivi_structeval_t.py"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

mkdir -p "$RUN_ROOT"
exec 9>"$HOME/official_baselines/results/.official_kivi_reproduction.lock"
if ! flock -n 9; then
  echo "Another official KIVI experiment holds the shared lock." >&2
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
printf '%s\n' "$RUN_ROOT" > "$HOME/official_baselines/results/CURRENT_KIVI_STRUCTEVAL_T_RUN"
date -Iseconds > "$RUN_ROOT/STARTED"

mark_failed() {
  local exit_code=$?
  printf '%s exit=%s\n' "$(date -Iseconds)" "$exit_code" > "$RUN_ROOT/FAILED"
  exit "$exit_code"
}
trap mark_failed ERR

run_condition() {
  local label="$1"
  local condition="$2"
  local bits="$3"
  local limit="$4"
  local expected_rows="$5"
  local output_dir="$RUN_ROOT/$label"

  echo "[StructEval-T] inference $label"
  OUTPUT_DIR="$output_dir" \
  CONDITION="$condition" \
  K_BITS="$bits" \
  V_BITS="$bits" \
  LIMIT="$limit" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    bash "$RUNNER"

  echo "[StructEval-T] evaluation $label"
  INFERENCE_PATH="$output_dir/inference.json" \
  RUN_DIR="$output_dir/eval" \
  EXPECTED_ROWS="$expected_rows" \
    bash "$EVALUATOR"
}

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  run_condition preflight_fp16 fp16 16 2 2
  run_condition preflight_kivi4 kivi 4 2 2
  run_condition preflight_kivi2 kivi 2 2 2
fi

run_condition full_fp16 fp16 16 0 950
run_condition full_kivi4 kivi 4 0 950
run_condition full_kivi2 kivi 2 0 950

PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 "$COMPARATOR" \
  --fp16 "$RUN_ROOT/full_fp16/eval/evaluation.json" \
  --kivi4 "$RUN_ROOT/full_kivi4/eval/evaluation.json" \
  --kivi2 "$RUN_ROOT/full_kivi2/eval/evaluation.json" \
  --output-json "$RUN_ROOT/comparison.json" \
  --output-md "$RUN_ROOT/comparison.md"

date -Iseconds > "$RUN_ROOT/COMPLETE"
rm -f "$RUN_ROOT/FAILED"
trap - ERR
echo "[StructEval-T] complete: $RUN_ROOT"
