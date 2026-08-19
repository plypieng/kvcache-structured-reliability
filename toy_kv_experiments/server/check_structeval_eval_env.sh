#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/kvcache}"
ENV_NAME="${ENV_NAME:-structeval-eval}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"

check_command() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    printf '%-16s %s\n' "$command_name" "$(command -v "$command_name")"
  else
    printf '%-16s %s\n' "$command_name" "MISSING"
  fi
}

echo "StructEval evaluator environment audit"
echo "host: $(hostname)"
echo "project: $PROJECT_DIR"
echo

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Isolation and rendering commands"
for command_name in docker podman apptainer singularity bwrap unshare node npm npx chromium \
  chromium-browser tectonic pdflatex lualatex convert magick gs pdftoppm typst dot; do
  check_command "$command_name"
done
echo

python - <<'PY'
import importlib.util
import os
import sys

modules = [
    "litellm",
    "typer",
    "playwright",
    "yaml",
    "xmltodict",
    "toml",
    "pdf2image",
    "markdown",
    "matplotlib",
    "PIL",
]

print("Python evaluator modules")
print("python", sys.executable)
for module in modules:
    state = "available" if importlib.util.find_spec(module) is not None else "MISSING"
    print(f"{module:<16} {state}")
print()
print("OPENAI_API_KEY configured:", bool(os.environ.get("OPENAI_API_KEY")))
PY
