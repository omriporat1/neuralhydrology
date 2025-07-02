#!/bin/bash -l

#SBATCH --job-name=N38_A30_16CPU
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --output=N38_A30_16CPU_job_output.log
#SBATCH --error=N38_A30_16CPU_error.log
#SBATCH --account=efratmorin


module load spack miniconda3 cuda/11.7

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/haimasree/condaenvs/neuralhydrology

start=$(date +%s)

python /sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static/test_cpu_gpus/N38_A30_16CPU.py

end=$(date +%s)

runtime=$((end-start))

printf 'Training time: %02d:%02d:%02d\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

conda deactivate


