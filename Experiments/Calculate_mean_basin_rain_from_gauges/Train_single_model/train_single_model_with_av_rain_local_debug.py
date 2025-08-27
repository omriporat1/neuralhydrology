import pandas as pd
from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import start_run
import yaml
import os
import torch
from datetime import datetime
import sys
import argparse
import logging
from typing import Optional

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

# --- new: detect debugger ---
def _in_debugger():
    try:
        return sys.gettrace() is not None
    except Exception:
        return False

# --- new: performance helpers ---
def _enable_gpu_fastpaths():
    try:
        # Keep harmless on CPU; only enable when CUDA is truly used.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('medium')  # PyTorch 2.x, TF32 matmul
            except Exception:
                pass
    except Exception:
        pass

# --- new: preflight logging helpers ---
def _warn_short_period(name: str, start: Optional[str], end: Optional[str], seq_len: int, predict_last_n: int):
    try:
        if not start or not end:
            return
        s = pd.to_datetime(start, dayfirst=True)
        e = pd.to_datetime(end, dayfirst=True)
        days = (e - s).days + 1
        need = seq_len + predict_last_n
        if days <= need:
            logging.warning(f"{name} period too short? days={days} <= seq_length+predict_last_n={need} ({start}..{end})")
    except Exception:
        pass

def _fix_windows_validation_basin_path(cfg: dict):
    # Your YAML defines validation_basin_file twice; last wins (Linux path).
    # On Windows, ensure a valid path is used instead.
    v = cfg.get("validation_basin_file")
    if isinstance(v, str) and (v.startswith("/") or v.lower().startswith("\\") or "sci" in v.replace("\\", "/")):
        tb = cfg.get("train_basin_file") or cfg.get("test_basin_file")
        if isinstance(tb, str):
            logging.warning(f"Overriding invalid validation_basin_file '{v}' with '{tb}'")
            cfg["validation_basin_file"] = tb

def _make_unique_dir(base: Path) -> Path:
    """Return a unique directory path based on 'base'. Create it on disk."""
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    # Try timestamped name first
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base.parent / f"{base.name}_{ts}"
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    # Fallback to incremental suffix
    i = 1
    while True:
        candidate = base.parent / f"{base.name}_{i:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        i += 1

def _auto_num_workers(default: int = 8) -> int:
    """Heuristic for DataLoader workers (non-debug). Keep modest on Windows."""
    try:
        cpu = os.cpu_count() or default
        # leave some headroom; cap to default
        w = max(1, min(default, cpu - 2))
        # Windows file IO often benefits little from many workers
        if os.name == "nt":
            w = min(w, 4)
        return int(w)
    except Exception:
        return int(default)

def _force_caravan_filetype(filetype: str = "csv"):
    """Monkey-patch Caravan timeseries loader to use a specific filetype ('csv' or 'netcdf')."""
    try:
        from neuralhydrology.datasetzoo import caravan as _caravan
        _orig = _caravan.load_caravan_timeseries

        def _patched(data_dir, basin, filetype_arg="netcdf"):
            # Ignore caller's filetype_arg and enforce ours
            return _orig(data_dir=data_dir, basin=basin, filetype=filetype)

        _caravan.load_caravan_timeseries = _patched
        logging.info(f"Caravan loader patched to filetype='{filetype}'.")
    except Exception:
        logging.exception("Failed to patch Caravan filetype")

def main():
    parser = argparse.ArgumentParser(description="Train single NH model (local CPU debug).")
    parser.add_argument("--job-sub-id", type=int, default=14)
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--run-suffix", type=str, default=os.getenv("RUN_SUFFIX", "").strip())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.00005")))
    parser.add_argument("--save-every", type=int, default=int(os.getenv("SAVE_WEIGHTS_EVERY", "5")))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=r"C:\PhD\Data\Caravan")
    parser.add_argument("--template", type=str, default=None,
                        help="Defaults to local_debug_template.yml next to this script")
    parser.add_argument("--csv", type=str, default=None,
                        help="Defaults to random_search_configurations.csv next to this script")
    parser.add_argument("--train-basin-file", type=str, default=None)
    parser.add_argument("--val-basin-file", type=str, default=None)
    parser.add_argument("--test-basin-file", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Turn on step-through friendly settings")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU (default).")
    parser.add_argument("--caravan-filetype", type=str, choices=["csv", "netcdf"], default="csv",
                        help="Force Caravan timeseries loader to read this filetype")
    args = parser.parse_args()

    # Force CPU by default
    force_cpu = True if not args.cpu_only is False else True  # keep True unless explicitly disabled
    is_debug = args.debug or _in_debugger()

    # Environment + logging
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""     # hide GPU from PyTorch
    if is_debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    logging.basicConfig(level=(logging.DEBUG if is_debug else logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    _enable_gpu_fastpaths()

    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    template_path = Path(args.template) if args.template else (script_dir / "local_debug_template.yml")
    csv_path = Path(args.csv) if args.csv else (script_dir / "random_search_configurations.csv")

    job_sub_id = args.job_sub_id
    job_id = args.job_id

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV with params: {csv_path}")

    df = pd.read_csv(csv_path, index_col="job_id")
    if job_sub_id not in df.index:
        raise KeyError(f"job_sub_id {job_sub_id} not found in {csv_path}")
    params = df.loc[job_sub_id]

    # Create results dir under this script
    results_root = script_dir / "results" / f"job_{job_id}"
    results_root.mkdir(parents=True, exist_ok=True)

    base_name = f"run_{job_sub_id:03d}_av_rain_local_debug"
    if args.run_suffix:
        base_name = f"{base_name}_{args.run_suffix}"
    run_dir = results_root / base_name

    if args.overwrite or os.getenv("ALLOW_OVERWRITE", "0") == "1":
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _make_unique_dir(run_dir)

    print(f"[RUN] Using run_dir: {run_dir}")

    if not template_path.exists():
        raise FileNotFoundError(f"Could not find template config: {template_path}")

    # Load and copy config as dict
    template_config = Config(template_path).as_dict()

    # Debug/CPU overrides
    template_config["device"] = "cpu"  # force CPU
    template_config["data_dir"] = str(Path(args.data_dir))
    template_config["run_dir"] = str(run_dir)

    # Fix duplicate/invalid validation_basin_file from YAML if present
    _fix_windows_validation_basin_path(template_config)

    # Optional basin file overrides
    if args.train_basin_file:
        template_config["train_basin_file"] = args.train_basin_file
    if args.val_basin_file:
        template_config["validation_basin_file"] = args.val_basin_file
    if args.test_basin_file:
        template_config["test_basin_file"] = args.test_basin_file

    # Record preferred Caravan filetype in config (harmless if unused by NH version)
    template_config["caravan_filetype"] = args.caravan_filetype

    # Throughput / determinism knobs
    if args.num_workers is not None:
        num_workers = int(args.num_workers)
    else:
        num_workers = 0 if is_debug else _auto_num_workers(default=8)

    batch_size = int(os.getenv("BATCH_SIZE", args.batch_size or int(params["batch_size"])))
    learning_rate = float(args.learning_rate)
    save_every = int(args.save_every)

    # Hyperparameters
    template_config["batch_size"] = batch_size
    template_config["hidden_size"] = int(params["hidden_size"])
    template_config["learning_rate"] = learning_rate  # honor override
    template_config["output_dropout"] = float(params["output_dropout"])
    template_config["seq_length"] = int(params["seq_length"])
    template_config["statics_embedding"] = {"hiddens": [int(params["statics_embedding"])]}
    template_config["num_workers"] = num_workers
    template_config["save_weights_every"] = save_every

    if is_debug:
        template_config["pin_memory"] = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # Quick preflight warnings that often cause empty predictions
    seq_len = int(template_config.get("seq_length", 0))
    predict_last_n = int(template_config.get("predict_last_n", 1))
    _warn_short_period("train", (template_config.get("train_start_date") or [None])[0]
                       if isinstance(template_config.get("train_start_date"), list)
                       else template_config.get("train_start_date"),
                       (template_config.get("train_end_date") or [None])[0]
                       if isinstance(template_config.get("train_end_date"), list)
                       else template_config.get("train_end_date"),
                       seq_len, predict_last_n)
    _warn_short_period("validation", template_config.get("validation_start_date"),
                       template_config.get("validation_end_date"),
                       seq_len, predict_last_n)
    _warn_short_period("test", template_config.get("test_start_date"),
                       template_config.get("test_end_date"),
                       seq_len, predict_last_n)

    safe_config_dict = make_yaml_safe(template_config)

    # Save to config.yml
    config_path = run_dir / "config.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(safe_config_dict, f, sort_keys=False)

    logging.info(f"Config written to: {config_path}")
    logging.info(f"Device: {template_config['device']}, Data dir: {template_config['data_dir']}")
    logging.info(f"Batch size: {batch_size}, Num workers: {num_workers}, LR: {learning_rate}, Save every: {save_every}")
    logging.info(f"Caravan filetype: {args.caravan_filetype}")

    # Patch Caravan loader to enforce filetype
    _force_caravan_filetype(args.caravan_filetype)

    # Launch training
    try:
        start_run(config_file=config_path)
    except Exception as e:
        logging.exception("Training failed")
        raise

if __name__ == "__main__":
    main()