#!/bin/bash -l

#SBATCH --job-name=nh_bayes_search
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/bayes_search.out
#SBATCH --error=logs/bayes_search.err
#SBATCH --account=efratmorin

module load spack miniconda3 cuda/11.7

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

mkdir -p logs results

# Optional: ensure optuna is available (uncomment if needed)
# python -c "import optuna" 2>/dev/null || pip install --no-input optuna

STUDY_NAME="lstm_bayes_search"
RESULTS_DIR="results/${STUDY_NAME}"
mkdir -p "$RESULTS_DIR"

STORAGE_URL="sqlite:///${RESULTS_DIR}/study.db"

echo "Storage URL: ${STORAGE_URL}"

start=$(date +%s)

/usr/bin/time -v \
python HPC_Bayesian_search.py \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE_URL" \
  --n-trials 20 \
  --device cuda:0 \
  --objective neg_nse \
  --data-dir /sci/labs/efratmorin/omripo/PhD/Data/Caravan/Caravan_winter \
  --events-csv /sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv

end=$(date +%s)

runtime=$((end-start))
printf 'Search runtime: %02d:%02d:%02d\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

conda deactivate
