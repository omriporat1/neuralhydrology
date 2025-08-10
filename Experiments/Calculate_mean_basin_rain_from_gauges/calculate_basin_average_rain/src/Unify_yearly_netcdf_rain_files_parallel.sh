#!/bin/bash -l

#SBATCH --job-name=merge_netcdf4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=merge_netcdf_%j.log
#SBATCH --error=merge_netcdf_error_%j.log
#SBATCH --account=efratmorin

module load spack miniconda3

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

echo "Job started on $(hostname) at $(date)"
/usr/bin/time -v python Unify_yearly_netcdf_rain_files_parallel.py
echo "Job finished at $(date)"

conda deactivate