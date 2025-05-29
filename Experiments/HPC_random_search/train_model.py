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
    run_dir.mkdir(parents=True, exist_ok=False)  # Fail if directory exists

    # Load base config and update
    config = Config(Path("template.yml"))
    config["batch_size"] = int(params["batch_size"])
    config["hidden_size"] = int(params["hidden_size"])
    config["learning_rate"] = float(params["learning_rate"])
    config["output_dropout"] = float(params["output_dropout"])
    config["seq_length"] = int(params["seq_length"])
    config["statics_embedding"]["hiddens"] = int(params["statics_embedding"])

    config.run_dir = str(run_dir)

    config.save(run_dir / "config.yml")
    start_run(config_file=run_dir / "config.yml")


if __name__ == "__main__":
    main()
