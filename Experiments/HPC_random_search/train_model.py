import pandas as pd
import sys
from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import start_run
import yaml


def make_yaml_safe(obj):
    if isinstance(obj, dict):
        return {k: make_yaml_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_yaml_safe(i) for i in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


def main():
    job_sub_id = int(sys.argv[1])
    job_id = int(sys.argv[2])

    df = pd.read_csv("random_search_configurations.csv", index_col="job_id")
    params = df.loc[job_sub_id]

    # Create a directory for the job
    job_dir = Path(f"results/job_{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    run_dir = job_dir / f"run_{job_sub_id:03d}"

    # run_dir = Path(f"results/run_{job_sub_id:03d}")
    run_dir.mkdir(parents=True, exist_ok=False)

    # Load and copy config as dict
    # template_config = Config(Path("template.yml")).__dict__.copy()
    template_config = Config(Path("template.yml")).as_dict()

    # Modify hyperparameters
    template_config["batch_size"] = int(params["batch_size"])
    template_config["hidden_size"] = int(params["hidden_size"])
    template_config["learning_rate"] = float(params["learning_rate"])
    template_config["output_dropout"] = float(params["output_dropout"])
    template_config["seq_length"] = int(params["seq_length"])
    template_config["statics_embedding"] = {
        "hiddens": [int(params["statics_embedding"])]
    }
    template_config["run_dir"] = str(run_dir)

    # Create the final config object
    # config = Config(template_config)

    safe_config_dict = make_yaml_safe(template_config)

    # Save to config.yml
    config_path = run_dir / "config.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(safe_config_dict, f)

    # Launch training
    start_run(config_file=config_path)


if __name__ == "__main__":
    main()
# The script is designed to run a specific configuration of a neural network model
# using a configuration file and parameters specified in a CSV file.
# It reads the job ID from command line arguments, retrieves the corresponding
# parameters from the CSV file, creates a directory for the run,
# modifies a template configuration with the parameters, saves the modified
# configuration, and then starts the run using the modified configuration file.
# It is intended to be run in a batch processing environment where multiple
# configurations can be executed in parallel by specifying different job IDs.
# The script uses the `neuralhydrology` library to handle configurations and runs.
# The parameters include batch size, hidden size, learning rate, output dropout,
# sequence length, and static embedding size, which are common hyperparameters
# for training neural network models.
# The script assumes that the necessary libraries and modules are installed
# and that the CSV file with configurations is present in the working directory.
# It also assumes that the template configuration file is named "template.yml"
# and is located in the same directory as the script.
# The results of the run will be saved in a directory named "results/run_XXX",
# where "XXX" is the job ID padded to three digits.
# The script is executed from the command line with the job ID as an argument,
# for example: `python train_model.py 0` to run the first configuration.
# The script is designed to be run in a controlled environment where the
# configurations and parameters are predefined, allowing for systematic exploration
# of hyperparameter space through random search.
