#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
ENV_NAME="${ENV_NAME:-kvcache-py311}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/manifests/json_10_stratified_seed42.json}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/run_qwen_structeval_a6000.sh"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/toy_kv_experiments/results/post_attention_gate}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-32}"

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_DIR"
mkdir -p "$RESULT_DIR" "$PROJECT_DIR/toy_kv_experiments/logs"

fp16_path="$RESULT_DIR/post_attention_v3_json10_fp16_${RUN_LABEL}.json"
k2_path="$RESULT_DIR/post_attention_v3_json10_K2_V2_${RUN_LABEL}.json"
paired_path="$RESULT_DIR/post_attention_v3_json10_K2_V2_paired_${RUN_LABEL}.json"

run_case() {
  local output_path="$1"
  local bits="$2"
  local key_bits="$3"
  local value_bits="$4"

  LIMIT=10 \
  STRUCTEVAL_MANIFEST="$MANIFEST" \
  OUT_PATH="$output_path" \
  CHECKPOINT_DIR="$output_path.checkpoints" \
  RESUME=1 \
  BITS="$bits" \
  KEY_BITS="$key_bits" \
  VALUE_BITS="$value_bits" \
  RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
  KV_GROUP_SIZE="$KV_GROUP_SIZE" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  CACHE_QUANTIZATION_MODE=real-blockwise \
  STRUCTURE_TOKEN_PROTECTION=none \
  PROTECTION_SIGNAL_SOURCE=prompt-visible \
  TARGET_AVERAGE_BITS='' \
  PROTECTED_BITS='' \
    bash "$RUNNER"
}

echo "[validity-gate-v3] run label: $RUN_LABEL"
echo "[validity-gate-v3] manifest: $MANIFEST"
echo "[validity-gate-v3] FP16: $fp16_path"
run_case "$fp16_path" 16 16 16

echo "[validity-gate-v3] K2/V2: $k2_path"
run_case "$k2_path" 8 2 2

python -m toy_kv_experiments.analyze_paired_structeval \
  --fp16 "$fp16_path" \
  --candidate "$k2_path" \
  --out "$paired_path"

python - "$fp16_path" "$k2_path" <<'PY'
import json
import sys
from pathlib import Path

fp16 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
k2 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert len(fp16) == len(k2) == 10, "validity gate must contain exactly 10 paired tasks"
assert {row["task_id"] for row in fp16} == {row["task_id"] for row in k2}, "task IDs are not paired"
assert {row["cache_algorithm_revision"] for row in k2} == {"kivi-post-attention-finalize-v3"}
assert not any(row.get("uses_evaluator_path_leakage") for row in fp16 + k2)
assert all(row.get("run_fingerprint") for row in fp16 + k2)
print("[validity-gate-v3] provenance and pairing checks passed")
PY

echo "$RUN_LABEL" > "$RESULT_DIR/latest_run_label.txt"
echo "[validity-gate-v3] complete"
echo "[validity-gate-v3] paired report: $paired_path"
