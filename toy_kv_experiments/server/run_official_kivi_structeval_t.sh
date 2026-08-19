#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
KIVI_REPO="${KIVI_REPO:-$HOME/official_baselines/KIVI-paper}"
ENV_NAME="${ENV_NAME:-kivi-paper}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
MODEL_REVISION="${MODEL_REVISION:-41b61a33a2483885c981aa79e0df6b32407ed873}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/official_baselines/model_cache}"
STRUCTEVAL_SOURCE="${STRUCTEVAL_SOURCE:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/structeval_test.jsonl}"
STRUCTEVAL_MANIFEST="${STRUCTEVAL_MANIFEST:-$PROJECT_DIR/toy_kv_experiments/data/structeval_full/manifests/structeval_t_complete_950.json}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a unique result directory.}"
CONDITION="${CONDITION:?Set CONDITION to fp16 or kivi.}"
K_BITS="${K_BITS:-4}"
V_BITS="${V_BITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-32}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-42}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-67aba607a1deaeb18b70ae796ab25d05a08b3345}"
RUNNER="$PROJECT_DIR/toy_kv_experiments/server/official_kivi_structeval_t.py"

# shellcheck disable=SC1091
set +u
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

actual_commit="$(git -C "$KIVI_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "KIVI commit mismatch: expected $EXPECTED_COMMIT, found $actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$KIVI_REPO" status --short --untracked-files=no)" ]]; then
  echo "Frozen KIVI checkout has tracked changes." >&2
  git -C "$KIVI_REPO" status --short --untracked-files=no >&2
  exit 2
fi

python - <<'PY'
import datasets
import flash_attn
import kivi_gemv
import torch
import transformers

assert torch.cuda.is_available()
assert torch.__version__ == "2.1.2+cu121"
assert transformers.__version__ == "4.36.2"
assert flash_attn.__version__ == "2.5.6"
assert datasets.__version__ == "2.16.1"
print("official KIVI environment: OK")
print("gpu", torch.cuda.get_device_name(0))
print("kivi_gemv", kivi_gemv.__file__)
PY

mkdir -p "$OUTPUT_DIR"
python "$RUNNER" \
  --kivi-repo "$KIVI_REPO" \
  --expected-kivi-commit "$EXPECTED_COMMIT" \
  --model-name-or-path "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --model-cache-dir "$MODEL_CACHE_DIR" \
  --structeval-source "$STRUCTEVAL_SOURCE" \
  --structeval-manifest "$STRUCTEVAL_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --condition "$CONDITION" \
  --k-bits "$K_BITS" \
  --v-bits "$V_BITS" \
  --group-size "$GROUP_SIZE" \
  --residual-length "$RESIDUAL_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --seed "$SEED"
