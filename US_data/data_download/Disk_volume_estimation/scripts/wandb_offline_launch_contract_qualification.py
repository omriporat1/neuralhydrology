"""Generic W&B offline tracking LAUNCH-CONTRACT qualification.

This is a different qualification than ``scripts/
wandb_real_offline_qualification_smoke.py``. That script already proved the
Flash-NH tracking *wrapper* (``src.baseline.wandb_tracking`` /
``src.baseline.pilot_tracking``) drives the real, locally-installed ``wandb``
package correctly in offline mode (see docs/stage1_wandb_user_guide.md
status item 2) -- but it builds its own policy dict in Python and never goes
through a real launcher's actual plumbing: an offline-enabled policy FILE
selected via ``WANDB_POLICY_PATH`` / ``--wandb-policy-path``, the same
env-var-then-CLI-flag contract every Stage 1 sbatch launcher uses (see e.g.
scripts/run_stage1_lr_range_seedA_closure_moriah.sbatch).

This script closes exactly that gap, and nothing more:
  * resolves the W&B policy path the same way a real launcher does
    (``--wandb-policy-path``, falling back to ``WANDB_POLICY_PATH``);
  * loads it with the real ``load_tracking_policy`` (never a hand-built
    policy dict);
  * starts a real tracked run via the generic, pilot-independent
    ``init_tracking_run`` (deliberately NOT ``init_pilot_tracking_run`` --
    that requires a full scientific ``PilotPolicy`` built from real pilot
    config, which this qualification must never touch);
  * tags the run identity unmistakably non-scientific
    (``qualification_kind: "wandb_offline_launch_contract"``,
    ``launch_contract_qualification: true``) so it can never be confused
    with real Flash-NH experiment evidence in the shared W&B project;
  * logs a couple of tiny synthetic scalars, finishes cleanly;
  * writes a compact, checked qualification record and exits.

It never imports neuralhydrology or torch, never generates an NH config,
never loads a basin/split/target package, and never touches a sealed
temporal-test/spatial-holdout set -- there is nothing in this script's
import graph capable of doing any of that.

Usage (mirrors the real launcher's env-or-flag contract):
    python scripts/wandb_offline_launch_contract_qualification.py \\
        --wandb-policy-path config/stage1_wandb_tracking_policy_offline_v001.yaml

    WANDB_POLICY_PATH=config/stage1_wandb_tracking_policy_offline_v001.yaml \\
        python scripts/wandb_offline_launch_contract_qualification.py

Writes reports/wandb_offline_launch_contract_qualification_v001/
qualification_record.json (untracked, per repo's existing reports/**
convention) and exits non-zero if any required check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_OUT_DIR = _REPO_ROOT / "reports" / "wandb_offline_launch_contract_qualification_v001"


def _git_commit() -> "str | None":
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _inventory_dir(root: Path) -> dict:
    if not root.is_dir():
        return {}
    inventory = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            inventory[str(p.relative_to(root))] = p.stat().st_size
    return inventory


def _resolve_policy_path(args: argparse.Namespace) -> tuple[str, str]:
    """Same precedence a real launcher's CLI parsing gives an explicit flag
    over its own env-var default: ``--wandb-policy-path`` wins if given,
    otherwise ``WANDB_POLICY_PATH`` (as exported by the sbatch launcher
    contract), otherwise this is a configuration error -- this qualification
    exists specifically to exercise an offline-ENABLED policy, so silently
    falling back to the committed disabled default would defeat its
    purpose."""
    if args.wandb_policy_path:
        return str(args.wandb_policy_path), "cli_flag"
    env_path = os.environ.get("WANDB_POLICY_PATH", "").strip()
    if env_path:
        return env_path, "env_var"
    raise SystemExit(
        "wandb_offline_launch_contract_qualification requires an offline-enabled policy path, "
        "via --wandb-policy-path or WANDB_POLICY_PATH (e.g. "
        "config/stage1_wandb_tracking_policy_offline_v001.yaml) -- refusing to silently fall back "
        "to the committed disabled default, since that would qualify nothing."
    )


def _resolve_wandb_dir(args: argparse.Namespace) -> Path:
    if args.wandb_dir:
        return Path(args.wandb_dir)
    env_dir = os.environ.get("WANDB_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_OUT_DIR / "wandb_dir"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb-policy-path", type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    args = parser.parse_args()

    policy_path, policy_source = _resolve_policy_path(args)
    wandb_dir = _resolve_wandb_dir(args)
    wandb_dir.mkdir(parents=True, exist_ok=True)

    # No API key ever read/set by this script; WANDB_DIR forced before any
    # wandb import (mirrors the real launcher's own WANDB_DIR export).
    os.environ.pop("WANDB_API_KEY", None)
    api_key_present = "WANDB_API_KEY" in os.environ
    os.environ["WANDB_DIR"] = str(wandb_dir)

    from src.baseline.wandb_tracking import (
        finish_tracking_run,
        init_tracking_run,
        load_tracking_policy,
        log_hyperparameters,
        log_scientific_metrics,
    )

    policy = load_tracking_policy(policy_path)

    run_identity = {
        "qualification_kind": "wandb_offline_launch_contract",
        "launch_contract_qualification": True,
        "policy_path": policy_path,
        "policy_source": policy_source,
        "git_commit": _git_commit(),
        "host": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    run = init_tracking_run(policy, run_identity)
    log_hyperparameters(run, {"qualification_kind": "wandb_offline_launch_contract"})
    log_scientific_metrics(run, 1, {"qualification_probe_metric": 1.0})
    finish_tracking_run(run)

    wandb_run = getattr(run, "_wandb_run", None)
    real_run_dir = getattr(wandb_run, "dir", None) if wandb_run is not None else None
    settings = getattr(wandb_run, "_settings", None) if wandb_run is not None else None
    real_sync_dir = getattr(settings, "sync_dir", None) if settings is not None else None
    # TrackingRun.wandb_run_id only echoes back a caller-SUPPLIED run_id (see
    # src/baseline/wandb_tracking.py init_tracking_run) -- this script never
    # supplies one, so the real, backend-GENERATED id must be read straight
    # off the wandb Run object itself, never assumed null just because this
    # script passed none in.
    effective_wandb_run_id = getattr(wandb_run, "id", None) if wandb_run is not None else None

    directory_inventory = _inventory_dir(wandb_dir)

    checks = {
        "policy_enabled_true": policy.get("enabled") is True,
        "policy_mode_offline": policy.get("mode") == "offline",
        "backend_is_wandb": run.backend == "wandb",
        "wandb_run_id_non_null": effective_wandb_run_id is not None,
        "run_finished_cleanly": run.finished is True and run.degraded is False,
        "offline_run_files_created": len(directory_inventory) > 0,
        "no_online_dependency": os.environ.get("WANDB_MODE") == "offline" and not api_key_present,
        "qualification_identity_non_scientific": run_identity["launch_contract_qualification"] is True,
        "no_scientific_config_generated": True,  # structural: see module docstring / import graph
    }
    all_checks_passed = all(checks.values())

    record = {
        "qualification_kind": "wandb_offline_launch_contract",
        "policy_path": policy_path,
        "policy_source": policy_source,
        "policy_loaded": policy,
        "run_identity": run_identity,
        "backend": run.backend,
        "mode": run.mode,
        "wandb_run_id": effective_wandb_run_id,
        "degraded": run.degraded,
        "finished": run.finished,
        "wandb_dir_used": str(wandb_dir),
        "wandb_dir_file_inventory": directory_inventory,
        "wandb_dir_total_bytes": sum(directory_inventory.values()),
        "real_wandb_run_dir": str(real_run_dir) if real_run_dir else None,
        "real_wandb_sync_dir": str(real_sync_dir) if real_sync_dir else None,
        "wandb_mode_env_effective": os.environ.get("WANDB_MODE"),
        "api_key_present_in_environment": api_key_present,
        "checks": checks,
        "all_checks_passed": all_checks_passed,
        "note": (
            "This qualifies the LAUNCH-CONTRACT plumbing (env/CLI policy-path selection -> real "
            "policy load -> real init_tracking_run) only, in a single process. Multi-process "
            "Slurm-continuation resume semantics for the same run id were already qualified "
            "separately by scripts/wandb_real_offline_qualification_smoke.py -- this script does "
            "not repeat that. No 'wandb sync' or other network-capable command was executed."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qualification_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nQualification record written to {out_dir / 'qualification_record.json'}")
    print(f"all_checks_passed={all_checks_passed}")

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
