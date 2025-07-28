#!/bin/bash -l

#SBATCH --job-name=validate_single
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/validation_single_014_a.log
#SBATCH --error=logs/validation_error_single_014_a.log
#SBATCH --account=efratmorin

module load spack miniconda3 cuda/11.7

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

mkdir -p logs

start=$(date +%s)

python validate_single_model.py

end=$(date +%s)

runtime=$((end-start))

printf 'Validation time: %02d:%02d:%02d\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

conda deactivate