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
# CUDA 12.8.1 / driver 580.95.05 confirmed on catfish-05 (2026-06-30).
# TODO(2K-G-C-A): the exact PyTorch CUDA wheel index has NOT been verified against
# this CUDA toolkit version. Do not guess a wheel tag (e.g. cu121/cu124) without
# checking https://pytorch.org/get-started/locally/ for a build compatible with
# CUDA 12.8 at install time. Conservative placeholder below is commented out.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#
echo "TODO: confirm PyTorch CUDA wheel for CUDA 12.8.1 before installing torch."
echo "Would install something equivalent to:"
echo "  pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>"

echo "=== install NeuralHydrology + minimal deps ==="
pip install neuralhydrology
pip install pandas pyarrow xarray netcdf4 tensorboard

echo "=== verify ==="
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "
import neuralhydrology
v = getattr(neuralhydrology, '__version__', None)
print('neuralhydrology', v if v else '(no __version__ attribute; import OK)')
"

echo "=== done ==="
