import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config

#%%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU


def main():
    print("running correct file")
    start_run(config_file=Path(r"/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static/test_cpu_gpus/test_A30_8.yml"))


if __name__ == '__main__':
    main()
