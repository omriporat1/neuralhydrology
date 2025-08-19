import pandas as pd
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
        return obj.strftime("%d/%m/%Y")
    else:
        return obj

def main():
    job_sub_id = 14  # Hardcoded for single model
    job_id = 0       # You can set this to any identifier you want

    df = pd.read_csv("random_search_configurations.csv", index_col="job_id")
    params = df.loc[job_sub_id]

    # Create a directory for the job
    job_dir = Path(f"results/job_{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    run_dir = job_dir / f"run_{job_sub_id:03d}_av_rain_all_year"  # Note the _a suffix

    run_dir.mkdir(parents=True, exist_ok=False)

    # Load and copy config as dict
    template_config = Config(Path("template.yml")).as_dict()

    # Modify hyperparameters
    template_config["batch_size"] = 512
    template_config["hidden_size"] = int(params["hidden_size"])
    template_config["learning_rate"] = 0.00005
    template_config["output_dropout"] = float(params["output_dropout"])
    template_config["seq_length"] = int(params["seq_length"])
    template_config["statics_embedding"] = {
        "hiddens": [int(params["statics_embedding"])]
    }
    template_config["num_workers"] = 16
    template_config["run_dir"] = str(run_dir)
    template_config["data_dir"] = "/sci/labs/efratmorin/omripo/PhD/Data/Caravan"  # Change to this for cluster

    safe_config_dict = make_yaml_safe(template_config)

    # Save to config.yml
    config_path = run_dir / "config.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(safe_config_dict, f)

    # Launch training using the saved config
    start_run(config_file=config_path)

if __name__ == "__main__":
    main()