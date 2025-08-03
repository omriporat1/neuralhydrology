#!/bin/bash -l

#SBATCH --job-name=test_val_single
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_output_%j.log
#SBATCH --error=logs/test_error_%j.log
#SBATCH --account=efratmorin

module load spack miniconda3 cuda/11.7 nvidia

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

mkdir -p logs

echo "Testing single run validation script"
echo "Start time: $(date)"

# Test with run 0 only
python Create_validation_hydrographs_multiple_epochs.py 0 41780893

echo "End time: $(date)"
conda deactivate
