#!/bin/bash -l

#SBATCH --job-name=test_val_3runs
#SBATCH --array=0-2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/test_output_%A_%a.log
#SBATCH --error=logs/test_error_%A_%a.log
#SBATCH --account=efratmorin

module load spack miniconda3 cuda/11.7 nvidia
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

mkdir -p logs

echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Testing validation script with debug output"
echo "Start time: $(date)"

# Use the debug version that only processes latest epoch and first 3 basins
python test_validation_debug.py ${SLURM_ARRAY_TASK_ID} 41780893

echo "End time: $(date)"
conda deactivate
