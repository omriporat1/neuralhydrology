#!/bin/bash
# RD1 P17/P18/P19 recovered-original-trial qualified evaluation (Session A
# correction task, 2026-10-08). See rd1_p17p19_recovered_evaluation_runner.py
# for the full rationale and the qualified-helper composition used.
#SBATCH --job-name=rd1-p17p19-recovered-eval
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-p17p19-recovered-eval-%j.out
set -uo pipefail

REPO=/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation
OUT_DIR="$1"
PYTHON=/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python

cd "$REPO"

"$PYTHON" "$REPO/rd1_p17p19_recovered_evaluation_runner.py" "$REPO" "$OUT_DIR"
