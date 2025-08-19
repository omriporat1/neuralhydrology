#!/bin/bash -l
set -euo pipefail

#SBATCH --job-name=lstm_avg_rain
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=logs/output_avg_rain-%x-%j.log
#SBATCH --error=logs/error_avg_rain-%x-%j.log
#SBATCH --account=efratmorin

module load spack miniconda3 cuda/11.7

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

mkdir -p logs results

nvidia-smi

start=$(date +%s)

# Ensure GPU visibility without srun (fallback if Slurm doesn’t set it for the batch step)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}}"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used --format=csv -l 60 > "logs/gpu_usage_${SLURM_JOB_ID}.log" &
GPU_MON_PID=$!
trap 'kill ${GPU_MON_PID} 2>/dev/null || true' EXIT

/usr/bin/time -v python train_single_model_with_av_rain.py

kill "$GPU_MON_PID" 2>/dev/null || true
trap - EXIT

end=$(date +%s)
runtime=$((end-start))
printf 'Training time: %02d:%02d:%02d\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

conda deactivate