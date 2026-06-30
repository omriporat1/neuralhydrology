#!/usr/bin/env bash
#SBATCH --job-name=flashnh-env-install
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/env-install-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/env-install-%j.err
#
# 2K-G-C-A preflight template — NOT YET RUN.
# Installs the flashnh-moriah conda prefix env on a Moriah catfish (L4 GPU) node.
# Facts below (modules, paths, partition) were confirmed via interactive `srun`
# reconnaissance on 2026-06-30; the env install itself has not been executed.

set -euo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"

# Do not rely on /sci/home/omripo inside this job — use the lab project root.
cd "${FLASHNH_BASE}"

echo "=== preflight ==="
hostname

# Modules confirmed present on a catfish-05 GPU allocation (2026-06-30):
#   module avail showed: miniconda3/24.3.0-gcc-iqeknet
#   nvidia-smi requires:  module load nvidia/580.95.05
#   nvcc requires:        module load cuda/12.8.1
module load miniconda3/24.3.0
module load nvidia/580.95.05
module load cuda/12.8.1

echo "conda: $(which conda) ($(conda --version))"
nvidia-smi
nvcc --version

echo "=== create prefix env ==="
# Prefix env (not a named env) under the lab project root — do NOT put conda envs
# under /sci/home.
if [ ! -d "${ENV_PREFIX}" ]; then
    conda create -y -p "${ENV_PREFIX}" python=3.11
    # python=3.12 only if a later NH/PyTorch compatibility check explicitly requires it.
else
    echo "Env already exists at ${ENV_PREFIX}; skipping conda create."
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
which python
python --version

echo "=== install PyTorch (CUDA build) ==="
# CUDA 12.8.1 toolkit / driver 580.95.05 (CUDA 13.0) confirmed on catfish-05 (2026-06-30).
# PyTorch 2.7.0 ships pre-built CUDA 12.8 wheels (cu128) per upstream release notes —
# this is the closest official match to the loaded 12.8.1 toolkit. torchvision/torchaudio
# are not installed; NeuralHydrology does not require them and Smoke 0 only needs torch.
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== verify CUDA (strict — fails the job if unavailable) ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print("device0", torch.cuda.get_device_name(0))
PY

echo "=== install NeuralHydrology + minimal deps ==="
pip install neuralhydrology pandas pyarrow xarray netcdf4 tensorboard

echo "=== verify nh-run entry point ==="
which nh-run
nh-run --help | head -40

echo "=== verify ==="
python -c "
import neuralhydrology
v = getattr(neuralhydrology, '__version__', None)
print('neuralhydrology', v if v else '(no __version__ attribute; import OK)')
"

echo "=== done ==="
