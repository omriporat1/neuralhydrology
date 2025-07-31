#!/bin/bash -l

#SBATCH --job-name=val_epochs_array
#SBATCH --array=0-59
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/output_val_epochs_%A_%a.log
#SBATCH --error=logs/error_val_epochs_%A_%a.log
#SBATCH --account=efratmorin

module load spack miniconda3 cuda/11.7 nvidia

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

mkdir -p logs results

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
which python
python --version
nvidia-smi

start=$(date +%s)

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used --format=csv -l 60 > logs/gpu_usage_${SLURM_ARRAY_TASK_ID}.log &
GPU_MON_PID=$!

python Create_validation_hydrographs_multiple_epochs.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_JOB_ID}

kill $GPU_MON_PID

end=$(date +%s)

runtime=$((end-start))

printf 'Validation (all epochs) time: %02d:%02d:%02d\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

conda deactivate