#!/bin/zsh

#SBATCH --time=01:00:00
#SBATCH -c1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a30:1



conda --version
# conda activate /sci/labs/efratmorin/omripo/condanenvs/neuralhydrology



