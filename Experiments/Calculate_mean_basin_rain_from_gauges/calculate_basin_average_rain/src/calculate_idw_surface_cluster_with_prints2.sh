#!/bin/bash -l

#SBATCH --job-name=idw_surface2
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=16:00:00
#SBATCH --output=idw_surface_%j.log
#SBATCH --error=idw_surface_error_%j.log
#SBATCH --account=efratmorin


module load spack miniconda3

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

set -e
echo "Job started on $(hostname) at $(date)"
df -h .
echo "Conda env: $(which python)"
conda list

mkdir -p logs

/usr/bin/time -v python calculate_idw_surface_cluster_parallel_with_prints2.py

conda deactivate