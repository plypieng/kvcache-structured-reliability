#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
KIVI_REPO="${KIVI_REPO:-$HOME/official_baselines/KIVI-paper}"
KIVI_REMOTE="${KIVI_REMOTE:-https://github.com/jy-yuan/KIVI.git}"
KIVI_COMMIT="${KIVI_COMMIT:-67aba607a1deaeb18b70ae796ab25d05a08b3345}"
ENV_NAME="${ENV_NAME:-kivi-paper}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
REQUIREMENTS="$PROJECT_DIR/toy_kv_experiments/server/requirements_official_kivi_repro.txt"

# shellcheck disable=SC1091
set +u
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
set -u

if [[ ! -d "$KIVI_REPO/.git" ]]; then
  git clone "$KIVI_REMOTE" "$KIVI_REPO"
fi
git -C "$KIVI_REPO" fetch origin "$KIVI_COMMIT"
git -C "$KIVI_REPO" checkout --detach "$KIVI_COMMIT"
if [[ -n "$(git -C "$KIVI_REPO" status --short --untracked-files=no)" ]]; then
  echo "Paper-era KIVI checkout has tracked changes." >&2
  exit 2
fi

python -m pip install \
  torch==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$REQUIREMENTS"
python -m pip install -e "$KIVI_REPO" --no-deps

set +u
conda install -y -c nvidia -c conda-forge \
  cuda-version=12.1 \
  cuda-nvcc=12.1 \
  cuda-cccl=12.1.109 \
  cuda-cudart=12.1.105 \
  cuda-cudart-dev=12.1.105 \
  gcc_linux-64=12 \
  gxx_linux-64=12 \
  ninja
set -u

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export TORCH_CUDA_ARCH_LIST=8.6

# CUDA 12.1 cannot parse one explicit dependent conversion in the pybind11
# header bundled with PyTorch 2.1. The implicit conversion is equivalent and
# affects extension compilation only.
python - <<'PY'
from pathlib import Path
import torch

header = (
    Path(torch.__file__).parent
    / "include"
    / "pybind11"
    / "cast.h"
)
old = (
    "    return caster.operator typename "
    "make_caster<T>::template cast_op_type<T>();"
)
new = "    return caster;  // CUDA 12.1 / PyTorch 2.1 build compatibility"
text = header.read_text(encoding="utf-8")
if old in text:
    header.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise RuntimeError(f"Unexpected pybind11 cast helper in {header}")
print("pybind11 CUDA 12.1 build compatibility: OK")
PY

(
  cd "$KIVI_REPO/quant"
  python setup.py clean --all >/dev/null 2>&1 || true
)
python -m pip install "$KIVI_REPO/quant" --no-build-isolation

python - <<'PY'
import datasets
import flash_attn
import fuzzywuzzy
import jieba
import numpy
import rouge
import torch
import transformers
import kivi_gemv

assert datasets.__version__ == "2.16.1"
assert numpy.__version__ == "1.26.3"
assert torch.__version__ == "2.1.2+cu121"
assert transformers.__version__ == "4.36.2"
assert flash_attn.__version__ == "2.5.6"
assert torch.cuda.is_available()
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
print("datasets", datasets.__version__)
print("numpy", numpy.__version__)
print("fuzzywuzzy", fuzzywuzzy.__version__)
print("jieba", jieba.__version__)
print("rouge", rouge.__file__)
print("kivi_gemv", kivi_gemv.__file__)
print("evaluator dependency gate: OK")
PY
