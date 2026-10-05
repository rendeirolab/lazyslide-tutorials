#!/usr/bin/env bash
# Payload for Hugging Face Jobs: render one heavy notebook on a GPU and copy
# the executed notebook to OUT_DIR, a bucket mount. render_heavy.yaml and a
# laptop both launch it with:
#   hf jobs run ... IMAGE bash -c "$(cat scripts/hf_render.sh)"
#
# Required env:
#   CI_SHA       git commit to render
#   NOTEBOOK     notebook path in the repo, e.g. tutorials/heavy/multiple_slides.ipynb
#   OUT_DIR      where the executed notebook goes, e.g. /out/<run id>
# Optional env:
#   CI_REPO_URL  default: https://github.com/rendeirolab/lazyslide-tutorials.git
set -euo pipefail

REPO_URL="${CI_REPO_URL:-https://github.com/rendeirolab/lazyslide-tutorials.git}"
SHA="${CI_SHA:?CI_SHA is required}"
NOTEBOOK="${NOTEBOOK:?NOTEBOOK is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
WORKDIR=/tmp/lazyslide-tutorials
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# CUDA runtime images ship without git, curl, uv or Python.
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git curl ca-certificates
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
# Pin 3.12, the Python the regular render uses, so uv does not pick a newer
# one that some locked package has no wheel for. Name it in the install too:
# an older uv already in the image may not read UV_PYTHON there.
export UV_PYTHON="${UV_PYTHON:-3.12}"
uv python install "${UV_PYTHON}"

git clone --filter=blob:none "${REPO_URL}" "${WORKDIR}"
cd "${WORKDIR}"
git fetch --filter=blob:none origin "${SHA}"
git checkout --force FETCH_HEAD

# Render exactly the committed lock. The regular render keeps it current.
uv sync --locked
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"

# Keep library warnings and download bars, including those from dask
# workers, out of the rendered notebook.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

# No per-cell timeout: the job's --timeout caps the whole render.
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  "${NOTEBOOK}"

mkdir -p "${OUT_DIR}"
cp "${NOTEBOOK}" "${OUT_DIR}/"
echo "Wrote ${OUT_DIR}/$(basename "${NOTEBOOK}")"
