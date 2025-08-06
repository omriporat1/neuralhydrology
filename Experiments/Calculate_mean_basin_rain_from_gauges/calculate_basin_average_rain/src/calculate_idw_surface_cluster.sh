#!/bin/bash -l

#SBATCH --job-name=idw_surface
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=idw_surface_%j.log
#SBATCH --error=idw_surface_error_%j.log
#SBATCH --account=efratmorin

pwd
ls -l

module load spack miniconda3

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

mkdir -p logs

/usr/bin/time -v python calculate_idw_surface_cluster.py

conda deactivate