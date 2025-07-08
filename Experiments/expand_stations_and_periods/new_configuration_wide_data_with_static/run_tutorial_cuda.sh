#!/bin/bash -l

#SBATCH --job-name=gpuwcuda_job
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --output=cuda_omri_job_output.log
#SBATCH --error=cuda_omri_job_error.log
#SBATCH --account=efratmorin


module load spack miniconda3 cuda/11.7

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/haimasree/condaenvs/neuralhydrology

python /sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static//Best_HPC_omri_trial_cluster.py

conda deactivate


