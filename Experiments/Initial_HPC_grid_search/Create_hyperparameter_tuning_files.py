from pathlib import Path
from neuralhydrology.utils.configutils import create_config_files_clean as create_config_utils

baseconfigyml = Path('Feature_normalization\Best_HPC_z_score_norm.yml')
outputdir = Path('hyperparameter_calibration_flow_rain_zscore_norm')

modify_dict = {
    "hidden_size": [64, 128, 256],
    "batch_size": [512, 2048],
    "learning_rate": [1e-3, 1e-2],  # consider later how to change to a strategy learning rate
}


# Main function to call the creation script
def main():
    create_config_utils(baseconfigyml, modify_dict, outputdir)


if __name__ == "__main__":
    main()
