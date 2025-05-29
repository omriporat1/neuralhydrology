import pandas as pd
import sys
from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import start_run

def main():
    job_id = int(sys.argv[1])
    df = pd.read_csv("random_search_configurations.csv", index_col="job_id")
    params = df.loc[job_id]

    run_dir = Path(f"results/run_{job_id:03d}")
    run_dir.mkdir(parents=True, exist_ok=False)

    # Load and copy config as dict
    template_config = Config(Path("template.yml")).__dict__.copy()

    # Modify hyperparameters
    template_config["batch_size"] = int(params["batch_size"])
    template_config["hidden_size"] = int(params["hidden_size"])
    template_config["learning_rate"] = float(params["learning_rate"])
    template_config["output_dropout"] = float(params["output_dropout"])
    template_config["seq_length"] = int(params["seq_length"])
    template_config["statics_embedding"]["hiddens"] = int(params["statics_embedding"])
    template_config["run_dir"] = str(run_dir)

    # Create the final config object
    config = Config(template_config)

    config.save(run_dir / "config.yml")
    start_run(config_file=run_dir / "config.yml")
