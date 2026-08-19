#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
STRUCTEVAL_ROOT="${STRUCTEVAL_ROOT:-$PROJECT_DIR/third_party/StructEval-litellm}"
ENV_NAME="${ENV_NAME:-structeval-eval}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
INPUT_PATH="${INPUT_PATH:-}"
IMG_OUTPUT_PATH="${IMG_OUTPUT_PATH:-}"
NON_RENDERABLE_OUTPUT_DIR="${NON_RENDERABLE_OUTPUT_DIR:-}"
RENDER_CONCURRENCY="${RENDER_CONCURRENCY:-4}"

if [ -z "$INPUT_PATH" ] || [ -z "$IMG_OUTPUT_PATH" ] || [ -z "$NON_RENDERABLE_OUTPUT_DIR" ]; then
  echo "INPUT_PATH, IMG_OUTPUT_PATH, and NON_RENDERABLE_OUTPUT_DIR are required" >&2
  exit 2
fi
if ! command -v bwrap >/dev/null 2>&1; then
  echo "bubblewrap is required for isolated StructEval rendering" >&2
  exit 2
fi

RUN_ROOT="$(dirname "$INPUT_PATH")"
SANDBOX_HOME="$RUN_ROOT/sandbox_home"
REACT_SCRATCH="$RUN_ROOT/react_render"
REACT_MOUNT="$STRUCTEVAL_ROOT/structeval/render_engine/react_render"

mkdir -p "$IMG_OUTPUT_PATH" "$NON_RENDERABLE_OUTPUT_DIR" "$SANDBOX_HOME/.cache" \
  "$SANDBOX_HOME/.config/matplotlib" "$REACT_SCRATCH" "$REACT_MOUNT"

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

PYTHON_BIN="$(command -v python)"
PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"

bwrap \
  --die-with-parent \
  --new-session \
  --unshare-pid \
  --unshare-ipc \
  --unshare-uts \
  --ro-bind / / \
  --dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --bind "$RUN_ROOT" "$RUN_ROOT" \
  --bind "$REACT_SCRATCH" "$REACT_MOUNT" \
  --chdir "$PROJECT_DIR" \
  /usr/bin/env -i \
    HOME="$SANDBOX_HOME" \
    XDG_CACHE_HOME="$SANDBOX_HOME/.cache" \
    MPLCONFIGDIR="$SANDBOX_HOME/.config/matplotlib" \
    PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" \
    PATH="$PATH" \
    PYTHONPATH="$PROJECT_DIR:$STRUCTEVAL_ROOT" \
    LANG=C.UTF-8 \
    TMPDIR=/tmp \
    "$PYTHON_BIN" -m structeval.cli render \
      --input-path "$INPUT_PATH" \
      --img-output-path "$IMG_OUTPUT_PATH" \
      --non-renderable-output-dir "$NON_RENDERABLE_OUTPUT_DIR" \
      --render-concurrency "$RENDER_CONCURRENCY"
