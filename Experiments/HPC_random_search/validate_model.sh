#!/bin/bash -l

#SBATCH --job-name=validate_hydrographs
#SBATCH --array=0-59
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/validate_%A_%a.log
#SBATCH --error=logs/validate_%A_%a.log
#SBATCH --account=efratmorin

module load spack miniconda3 cuda/11.7

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/haimasree/condaenvs/neuralhydrology

mkdir -p logs results

start=$(date +%s)

python Create_validation_hydrographs_from_old_code_for_cluster.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_JOB_ID}

end=$(date +%s)

runtime=$((end-start))

printf 'Validation time: %02d:%02d:%02d\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

conda deactivate