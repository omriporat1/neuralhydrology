#!/bin/bash -l

#SBATCH --job-name=lstm_avg_rain
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=logs/output-%x-%j.log
#SBATCH --error=logs/error-%x-%j.log
#SBATCH --account=efratmorin

module load spack miniconda3 cuda/11.7 nvidia

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

mkdir -p logs results

nvidia-smi

start=$(date +%s)

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used --format=csv -l 60 > "logs/gpu_usage_${SLURM_JOB_ID}.log" &
GPU_MON_PID=$!

/usr/bin/time -v python train_single_model_with_av_rain.py

kill $GPU_MON_PID

end=$(date +%s)

runtime=$((end-start))

printf 'Training time: %02d:%02d:%02d\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

conda deactivate