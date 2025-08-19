import argparse
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import get_tester


def setup_logger(log_file: Path, name: str = "bayes_search") -> logging.Logger:
	"""Configure a logger that writes to both file and stdout."""
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	logger.propagate = False

	formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

	# Avoid adding duplicate handlers when re-created
	if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(log_file) for h in logger.handlers):
		fh = logging.FileHandler(log_file)
		fh.setLevel(logging.INFO)
		fh.setFormatter(formatter)
		logger.addHandler(fh)

	if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
		ch = logging.StreamHandler(sys.stdout)
		ch.setLevel(logging.INFO)
		ch.setFormatter(formatter)
		logger.addHandler(ch)

	return logger


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


class Objective:
	"""Optuna objective that trains a model and evaluates event-averaged metrics.

	The objective returns the mean MSE across all extracted events from all basins.
	"""

	def __init__(
		self,
		template_config_path: Path,
		results_root: Path,
		events_csv_path: Path,
		logger: logging.Logger,
		epochs: int | None = None,
		device: str = "cuda:0",
		data_dir_override: Path | None = None,
		delay: int = 18,
		objective_metric: str = "mse",
		model_seed: int = 42,
	) -> None:
		self.template_config_path = template_config_path
		self.results_root = results_root
		self.events_csv_path = events_csv_path
		self.logger = logger
		self.epochs = epochs
		self.device = device
		self.data_dir_override = data_dir_override
		self.delay = delay
		self.objective_metric = objective_metric  # 'mse' or 'neg_nse'
		self.model_seed = int(model_seed)

		# Preload events table
		try:
			df = pd.read_csv(self.events_csv_path)
			df = df[df.get("max_discharge", 0) > 0]
			# Ensure datetimes
			df["max_date"] = pd.to_datetime(df["max_date"])  # assume ISO-like on cluster
			df["start_date"] = df["max_date"] - pd.Timedelta(days=1)
			df["end_date"] = df["max_date"] + pd.Timedelta(days=1)
			self.events_df = df.set_index("basin")
			self.logger.info(
				f"Loaded events CSV with {len(self.events_df)} rows from {self.events_csv_path}"
			)
		except Exception:
			self.logger.exception(
				f"Failed to load events CSV from {self.events_csv_path}. Optimization cannot proceed."
			)
			raise

	def __call__(self, trial: optuna.trial.Trial) -> float:
		trial_id = trial.number
		trial_dir = self.results_root / f"trial_{trial_id:04d}"
		run_dir = trial_dir / "run"
		run_dir.mkdir(parents=True, exist_ok=False)

		# --- Sample hyperparameters ---
		params = {
			"batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
			"hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
			"learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True),
			"output_dropout": trial.suggest_categorical("output_dropout", [0.2, 0.4, 0.6]),
			"seq_length": trial.suggest_categorical("seq_length", [36, 72, 144]),
			"statics_embedding": trial.suggest_categorical("statics_embedding", [5, 10, 15, 20, 25]),
		}

		# --- Build config from template ---
		cfg_dict = Config(self.template_config_path).as_dict()
		cfg_dict["batch_size"] = int(params["batch_size"])  # type: ignore[index]
		cfg_dict["hidden_size"] = int(params["hidden_size"])  # type: ignore[index]
		cfg_dict["learning_rate"] = float(params["learning_rate"])  # type: ignore[index]
		cfg_dict["output_dropout"] = float(params["output_dropout"])  # type: ignore[index]
		cfg_dict["seq_length"] = int(params["seq_length"])  # type: ignore[index]
		cfg_dict["statics_embedding"] = {"hiddens": [int(params["statics_embedding"])]}  # type: ignore[index]
		cfg_dict["run_dir"] = str(run_dir)
		cfg_dict["device"] = self.device
		cfg_dict["seed"] = int(self.model_seed)
		if self.epochs is not None:
			cfg_dict["epochs"] = int(self.epochs)
		if self.data_dir_override is not None:
			cfg_dict["data_dir"] = str(self.data_dir_override)

		# Persist the config for traceability
		cfg_yaml_path = run_dir / "config.yml"
		with open(cfg_yaml_path, "w", encoding="utf-8") as f:
			yaml.safe_dump(make_yaml_safe(cfg_dict), f)

		self.logger.info(f"[Trial {trial_id}] Params: {params}")
		self.logger.info(f"[Trial {trial_id}] Training starting. Run dir: {run_dir}")

		# --- Train ---
		try:
			start_run(config_file=cfg_yaml_path)
		except Exception as e:
			self.logger.exception(f"[Trial {trial_id}] Training failed: {e}")
			# Mark as failed with a large loss
			return float("inf")

		# The training pipeline usually creates a subfolder with timestamp; pick the first subdir if present
		subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
		actual_run_dir = subdirs[0] if subdirs else run_dir
		self.logger.info(f"[Trial {trial_id}] Using run directory for evaluation: {actual_run_dir}")

		# --- Prepare config for validation window evaluation (replace test dates with validation dates) ---
		with open(cfg_yaml_path, "r", encoding="utf-8") as f:
			eval_cfg_dict = yaml.safe_load(f)
		if "test_start_date" in eval_cfg_dict and "validation_start_date" in eval_cfg_dict:
			eval_cfg_dict["test_start_date"] = eval_cfg_dict["validation_start_date"]
			eval_cfg_dict["test_end_date"] = eval_cfg_dict["validation_end_date"]
		eval_cfg_dict["run_dir"] = str(actual_run_dir)
		eval_cfg = Config(eval_cfg_dict)

		# --- Evaluate ---
		try:
			tester = get_tester(cfg=eval_cfg, run_dir=actual_run_dir, period="test", init_model=True)
			results = tester.evaluate(save_results=False, metrics=eval_cfg.metrics)
		except Exception as e:
			self.logger.exception(f"[Trial {trial_id}] Evaluation failed: {e}")
			return float("inf")

		# --- Compute event-averaged metrics across all basins ---
		time_scale_cache: dict[str, str] = {}
		event_rows: list[dict] = []

		for basin in results.keys():
			try:
				# Choose timescale: prefer '10min', else first available
				basin_scale_dict = results[basin]
				if basin not in time_scale_cache:
					ts_key = "10min" if "10min" in basin_scale_dict else next(iter(basin_scale_dict.keys()))
					time_scale_cache[basin] = ts_key
				ts_key = time_scale_cache[basin]

				xr_ds = basin_scale_dict[ts_key]["xr"]
				qobs = xr_ds["Flow_m3_sec_obs"]
				qsim = xr_ds["Flow_m3_sec_sim"]

				# Ensure we can slice by date
				if "time_step" in qobs.dims:
					if "date" in qobs.coords:
						qobs = qobs.swap_dims({"time_step": "date"})
					elif "time" in qobs.coords:
						qobs = qobs.swap_dims({"time_step": "time"}).rename({"time": "date"})
				if "time_step" in qsim.dims:
					if "date" in qsim.coords:
						qsim = qsim.swap_dims({"time_step": "date"})
					elif "time" in qsim.coords:
						qsim = qsim.swap_dims({"time_step": "time"}).rename({"time": "date"})

				# Shift observed by delay to account for AR inputs alignment if needed
				if "date" in qobs.dims:
					try:
						fill_value = float(qobs.isel(date=0).values)
					except Exception:
						fill_value = float(np.asarray(qobs).flatten()[0])
					qobs_shift = qobs.shift(date=self.delay, fill_value=fill_value)
				else:
					qobs_shift = qobs

				# Events for this basin
				if basin in self.events_df.index:
					basin_events = self.events_df.loc[basin]
				else:
					# If not found, try str casting fallback
					if str(basin) in self.events_df.index:
						basin_events = self.events_df.loc[str(basin)]
					else:
						continue

				if isinstance(basin_events, pd.Series):
					basin_events = basin_events.to_frame().T

				for _, ev in basin_events.iterrows():
					s = pd.to_datetime(ev["start_date"])  # type: ignore[index]
					e = pd.to_datetime(ev["end_date"])    # type: ignore[index]
					try:
						qo_seg = qobs_shift.sel(date=slice(s, e))
						qs_seg = qsim.sel(date=slice(s, e))
						obs = np.asarray(qo_seg, dtype=float).flatten()
						sim = np.asarray(qs_seg, dtype=float).flatten()
						if obs.size == 0 or sim.size == 0:
							continue
						mask = ~np.isnan(obs) & ~np.isnan(sim)
						if mask.sum() == 0:
							continue
						mse = float(np.mean((obs[mask] - sim[mask]) ** 2))
						denom = float(np.mean((obs[mask] - np.mean(obs[mask])) ** 2))
						nse = float(1.0 - (np.mean((obs[mask] - sim[mask]) ** 2) / denom)) if denom > 0 else np.nan
						event_rows.append(
							{
								"basin": str(basin),
								"start": s,
								"end": e,
								"mse": mse,
								"nse": nse,
							}
						)
					except Exception as ex:
						self.logger.warning(f"[Trial {trial_id}] Event slice failed for basin={basin}: {ex}")
						continue
			except Exception as e:
				self.logger.warning(f"[Trial {trial_id}] Skipping basin {basin} due to error: {e}")
				continue

		# Save event metrics for debugging/traceability
		events_df = pd.DataFrame(event_rows)
		events_csv = trial_dir / "event_metrics.csv"
		events_df.to_csv(events_csv, index=False)
		self.logger.info(
			f"[Trial {trial_id}] Computed event metrics for {len(events_df)} events. Saved to {events_csv}"
		)

		if events_df.empty:
			self.logger.warning(f"[Trial {trial_id}] No events found across basins; returning inf loss")
			return float("inf")

		if self.objective_metric == "mse":
			objective_value = float(events_df["mse"].mean())
		elif self.objective_metric == "neg_nse":
			# Maximize mean NSE -> minimize negative mean NSE
			# Filter NaNs first
			nse_vals = events_df["nse"].dropna()
			if nse_vals.empty:
				objective_value = float("inf")
			else:
				objective_value = float(-nse_vals.mean())
		else:
			raise ValueError(f"Unsupported objective metric: {self.objective_metric}")

		# Persist trial summary
		with open(trial_dir / "trial_summary.yaml", "w", encoding="utf-8") as f:
			yaml.safe_dump(
				{
					"params": params,
					"objective": objective_value,
					"objective_metric": self.objective_metric,
					"n_events": int(len(events_df)),
				},
				f,
			)

		self.logger.info(
			f"[Trial {trial_id}] Objective (mean MSE across events) = {objective_value:.6f}"
		)
		return objective_value


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Bayesian hyperparameter optimization for neuralhydrology LSTM with Optuna")
	p.add_argument(
		"--study-name",
		type=str,
		default="lstm_bayes_search",
		help="Name of the Optuna study",
	)
	p.add_argument(
		"--storage",
		type=str,
		default=None,
		help="Optuna storage URL (e.g., sqlite:///path/to/study.db). If omitted, in-memory storage is used.",
	)
	p.add_argument("--n-trials", type=int, default=20, help="Number of trials to run")
	p.add_argument(
		"--results-dir",
		type=str,
		default="results/bayesian_search",
		help="Directory to store trial outputs and logs",
	)
	p.add_argument(
		"--template-config",
		type=str,
		default=str(Path(__file__).resolve().parent / "template.yml"),
		help="Path to the base YAML template config",
	)
	p.add_argument(
		"--events-csv",
		type=str,
		default="/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv",
		help="Path to CSV with annual max discharge dates per basin",
	)
	p.add_argument(
		"--device",
		type=str,
		default="cuda:0",
		help="Torch device for training/evaluation (e.g., cuda:0 or cpu)",
	)
	p.add_argument(
		"--epochs",
		type=int,
		default=None,
		help="Override epochs from template (optional)",
	)
	p.add_argument(
		"--data-dir",
		type=str,
		default="/sci/labs/efratmorin/omripo/PhD/Data/Caravan/Caravan_winter",
		help="Override data_dir from template (cluster path)",
	)
	p.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for Optuna sampler",
	)
	p.add_argument(
		"--model-seed",
		type=int,
		default=42,
		help="Random seed for the NeuralHydrology run (written to config as 'seed')",
	)
	p.add_argument(
		"--objective",
		type=str,
		choices=["mse", "neg_nse"],
		default="mse",
		help="Optimization objective: mean MSE (minimize) or negative mean NSE (minimize)",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()

	# Paths
	template_cfg_path = Path(args.template_config).resolve()
	results_root = (Path(__file__).resolve().parent / "results" / args.study_name).resolve()
	events_csv_path = Path(args.events_csv)
	results_root.mkdir(parents=True, exist_ok=True)

	# Logging
	log_file = results_root / "study.log"
	logger = setup_logger(log_file)
	logger.info("Starting Bayesian hyperparameter search")
	logger.info(f"Template config: {template_cfg_path}")
	logger.info(f"Events CSV: {events_csv_path}")
	logger.info(f"Results dir: {results_root}")

	# Optuna study
	sampler = optuna.samplers.TPESampler(seed=args.seed)
	pruner = optuna.pruners.NopPruner()
	study_kwargs = {
		"study_name": args.study_name,
		"sampler": sampler,
		"pruner": pruner,
		"direction": "minimize",
	}
	if args.storage:
		study_kwargs["storage"] = args.storage
		study_kwargs["load_if_exists"] = True

	study = optuna.create_study(**study_kwargs)

	objective = Objective(
		template_config_path=template_cfg_path,
		results_root=results_root,
		events_csv_path=events_csv_path,
		logger=logger,
		epochs=args.epochs,
		device=args.device,
		data_dir_override=Path(args.data_dir) if args.data_dir else None,
		objective_metric=args.objective,
		model_seed=args.model_seed,
	)

	try:
		study.optimize(objective, n_trials=args.n_trials)
	except Exception:
		logger.exception("Optuna optimization failed with an unexpected exception")
		raise
	finally:
		# Save study summary
		try:
			best = study.best_trial
			summary = {
				"best_value": best.value,
				"best_params": best.params,
				"n_trials": len(study.trials),
			}
		except Exception:
			summary = {
				"best_value": None,
				"best_params": None,
				"n_trials": len(study.trials),
			}
		with open(results_root / "study_summary.yaml", "w", encoding="utf-8") as f:
			yaml.safe_dump(summary, f)
		logger.info(f"Saved study summary to {results_root / 'study_summary.yaml'}")


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		# Last resort logging if setup failed
		print(f"Fatal error in Bayesian search script: {e}")
		traceback.print_exc()
		sys.exit(1)
	except Exception as e:
		# Last resort logging if setup failed
		print(f"Fatal error in Bayesian search script: {e}")
		traceback.print_exc()
		sys.exit(1)

