#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
STRUCTEVAL_ROOT="${STRUCTEVAL_ROOT:-$PROJECT_DIR/third_party/StructEval-litellm}"
ENV_NAME="${ENV_NAME:-structeval-eval}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"

if [ ! -f "$STRUCTEVAL_ROOT/pyproject.toml" ]; then
  echo "official StructEval LiteLLM checkout not found: $STRUCTEVAL_ROOT" >&2
  exit 2
fi

echo "disk before setup"
df -h "$HOME"

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" -c conda-forge \
    python=3.12 nodejs=18 poppler graphviz imagemagick ghostscript tectonic typst
fi

conda activate "$ENV_NAME"
conda install -y -c conda-forge nodejs=18
python -m pip install --upgrade pip
python -m pip install -e "$STRUCTEVAL_ROOT[render,test]"
python -m playwright install chromium

echo
echo "StructEval checkout"
git -C "$STRUCTEVAL_ROOT" rev-parse HEAD
STRUCTEVAL_SETUP_DUMMY=doctor-only \
  python -m structeval.cli doctor --render --api-key-env STRUCTEVAL_SETUP_DUMMY
python -m pytest -q "$STRUCTEVAL_ROOT/tests"

echo
echo "disk after setup"
df -h "$HOME"
