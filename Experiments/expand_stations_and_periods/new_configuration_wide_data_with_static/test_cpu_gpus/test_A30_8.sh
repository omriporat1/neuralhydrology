#!/bin/bash -l

#SBATCH --job-name=test_runtime_A30_8
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mem=32G
#SBATCH --output=test_runtime_A30_8_job_output.log
#SBATCH --error=test_runtime_A30_8_error.log
#SBATCH --account=efratmorin


module load spack miniconda3 cuda/11.7

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/haimasree/condaenvs/neuralhydrology

nvidia-smi

start=$(date +%s)

python /sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static/test_cpu_gpus/test_A30_8.py

end=$(date +%s)

runtime=$((end-start))

printf 'Training time: %02d:%02d:%02d\n' $((duration/3600)) $((duration%3600/60)) $((duration%60))

conda deactivate


