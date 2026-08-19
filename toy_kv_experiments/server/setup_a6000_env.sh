#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
ENV_NAME="${ENV_NAME:-kvcache-py311}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"

echo "[setup] project dir: $PROJECT_DIR"
echo "[setup] env name: $ENV_NAME"

if [ ! -d "$MINIFORGE_DIR" ]; then
  echo "[setup] installing Miniforge to $MINIFORGE_DIR"
  curl -L -o /tmp/Miniforge3-Linux-x86_64.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$MINIFORGE_DIR"
else
  echo "[setup] Miniforge already exists"
fi

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] creating conda environment"
  conda create -y -n "$ENV_NAME" python=3.11 pip
else
  echo "[setup] conda environment already exists"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip wheel setuptools

echo "[setup] installing CUDA PyTorch stack"
python -m pip install --upgrade \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "[setup] installing experiment dependencies"
python -m pip install --upgrade \
  "transformers>=4.51.0" \
  "accelerate>=1.0.0" \
  "huggingface_hub>=0.24.0" \
  "datasets>=2.19.0" \
  pyarrow safetensors sentencepiece protobuf pandas tqdm

mkdir -p "$PROJECT_DIR/toy_kv_experiments/models" \
  "$PROJECT_DIR/toy_kv_experiments/results" \
  "$PROJECT_DIR/toy_kv_experiments/logs"

echo "[setup] environment check"
python - <<'PY'
import torch
import transformers
print("python ok")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

echo "[setup] done"
