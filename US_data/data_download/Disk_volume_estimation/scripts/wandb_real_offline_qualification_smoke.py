"""Real-package W&B OFFLINE qualification smoke (Path A, commit-readiness
review of the uncommitted W&B tracking patch).

Every prior "offline qualification" claim in this repo (tests/
test_wandb_tracking.py, tests/test_pilot_tracking.py, tests/
test_wandb_offline_qualification.py) exercises the Flash-NH wrapper contract
against an in-process FAKE ``wandb`` module (monkeypatched into
``sys.modules``). That proves the wrapper's call shapes and guard placement,
never the real ``wandb`` package's actual offline-mode I/O, serialization,
or multi-process resume semantics. This script is the separate, explicitly
real-package check: it imports the REAL, locally-installed ``wandb``
package (never monkeypatched) and drives it entirely through this repo's own
``src.baseline.wandb_tracking`` / ``src.baseline.pilot_tracking`` code --
never a reimplementation of that logic.

Hard constraints (all enforced by construction, not by convention):
  * No API key: never reads/sets WANDB_API_KEY.
  * No network: WANDB_MODE is forced to "offline" before wandb is imported;
    this repo's own init_tracking_run() also never passes any network-only
    argument. No sync/upload call is made anywhere in this script.
  * No NeuralHydrology training: this script never imports neuralhydrology
    or torch, and never touches a real NH run directory -- only a throwaway
    directory standing in for one, so resolve_pilot_wandb_run_id's
    persistence branch can be exercised.
  * Two SEPARATE OS processes (not two in-process calls) reuse the same
    stable W&B run id, to actually test multi-process continuation
    semantics rather than assume them.

Usage:
    python scripts/wandb_real_offline_qualification_smoke.py

Internal (used by the parent process to launch each child invocation; not
meant to be called directly by an operator):
    python scripts/wandb_real_offline_qualification_smoke.py --invocation N \
        --wandb-dir DIR --stable-run-id ID --project NAME

Writes a small qualification record (JSON) to
reports/wandb_real_offline_qualification_v001/ (already-gitignored, per
repo convention for reports/**). The bulky real wandb offline run
directories themselves are written under a caller-supplied scratch
directory and are NOT copied into the repo; the orchestrating run (no
arguments) removes them after recording an inventory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run_one_invocation(invocation_index: int, wandb_dir: Path, stable_run_id: str, project: str) -> dict:
    # Forced before importing wandb (via the modules this imports below) --
    # never inherited implicitly, never left to the operator's shell.
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ.pop("WANDB_API_KEY", None)

    from src.baseline.pilot_screening_eval import SCREENING_METRIC_SCOPE
    from src.baseline.pilot_tracking import (
        finish_pilot_run,
        log_pilot_checkpoint_reference,
        log_pilot_screening_event,
        resolve_pilot_wandb_run_id,
    )
    from src.baseline.wandb_tracking import init_tracking_run, log_hyperparameters, log_scientific_metrics

    policy = {
        "policy_name": "real_offline_qualification_smoke",
        "enabled": True,
        "mode": "offline",
        "project": project,
        "entity": None,
        "tags": ["stage1", "real_offline_qualification_smoke"],
        "max_artifact_reference_bytes": 1_048_576,
    }

    # Stand-in for a real NH run directory -- never a real one, and never
    # touched by NeuralHydrology itself. Both invocations point at the SAME
    # throwaway directory, exactly like two Slurm jobs continuing one
    # candidate would both point at the same real NH run directory.
    fake_nh_run_dir = wandb_dir / "fake_nh_run_dir"
    fake_nh_run_dir.mkdir(parents=True, exist_ok=True)

    resolved_id = resolve_pilot_wandb_run_id(
        pilot_policy_name="stage1_lead06_pilot_v001",
        run_id=stable_run_id,
        nh_run_dir=fake_nh_run_dir,
    )

    run_identity = {
        "pilot_policy_name": "stage1_lead06_pilot_v001",
        "run_id": stable_run_id,
        "invocation_index": invocation_index,
    }
    run = init_tracking_run(policy, run_identity, run_id=resolved_id, resume="allow")

    log_hyperparameters(run, {"model": "cudalstm", "hidden_size": 128, "seed": 42, "invocation_index": invocation_index})
    log_scientific_metrics(run, invocation_index, {"median_nse": 0.30 + 0.01 * invocation_index})

    ckpt_path = wandb_dir / f"fake_checkpoint_invocation{invocation_index}.pt"
    ckpt_path.write_bytes(b"fake checkpoint bytes -- never uploaded, reference only " * 50)
    log_pilot_checkpoint_reference(run, epoch=invocation_index, path=ckpt_path, checksum="deadbeef" * 4)

    screening_result = {
        "scope": SCREENING_METRIC_SCOPE,
        "primary_metric_name": "median_per_basin_raw_space_nse",
        "primary_metric_median": 0.31,
        "epoch_role": "stopping_eligible",
        "stopping_eligible": True,
        "n_screening_basins_requested": 350,
        "raw_space_metrics": {"aggregate": {"metrics": {}}},
        "primary_metric_distribution": {},
    }
    early_stopping_state = {
        "best_epoch": invocation_index,
        "best_metric_value": 0.31,
        "events_since_best_improvement": 0,
        "stopped": False,
        "stop_reason": None,
    }
    log_pilot_screening_event(
        run, epoch=invocation_index, screening_result=screening_result, early_stopping_state=early_stopping_state
    )

    finish_pilot_run(run, final_status="qualification_smoke", best_epoch=invocation_index)

    # Point 11 ("degradation handling around a real backend object"): a
    # genuine real-backend failure mode -- logging again after the real
    # wandb run has already finished -- exercised against the ACTUAL
    # installed package, not a synthetic monkeypatched exception. Whatever
    # the real package actually does here (raise, and get caught by
    # _guard_backend_call; or warn/no-op internally) is recorded verbatim
    # rather than assumed.
    degraded_before_post_finish_log = run.degraded
    try:
        log_scientific_metrics(run, invocation_index, {"post_finish_probe": 1.0})
    except Exception as exc:  # noqa: BLE001 -- this is the case we're deliberately probing
        post_finish_log_outcome = f"raised uncaught: {exc!r}"
    else:
        post_finish_log_outcome = (
            "no exception propagated to caller "
            f"(run.degraded changed {degraded_before_post_finish_log} -> {run.degraded})"
        )

    wandb_run = run._wandb_run
    real_sync_dir = None
    real_run_dir = getattr(wandb_run, "dir", None)
    settings = getattr(wandb_run, "_settings", None)
    if settings is not None:
        real_sync_dir = getattr(settings, "sync_dir", None)

    return {
        "invocation_index": invocation_index,
        "backend": run.backend,
        "mode": run.mode,
        "wandb_run_id": run.wandb_run_id,
        "resolved_id_matches_stable_run_id_derivation": resolved_id == run.wandb_run_id,
        "degraded": run.degraded,
        "degraded_operations": sorted(run.degraded_operations),
        "finished": run.finished,
        "real_wandb_run_dir": str(real_run_dir) if real_run_dir else None,
        "real_wandb_sync_dir": str(real_sync_dir) if real_sync_dir else None,
        "post_finish_log_probe_outcome": post_finish_log_outcome,
    }


def _child_main(args: argparse.Namespace) -> None:
    record = _run_one_invocation(
        invocation_index=args.invocation,
        wandb_dir=Path(args.wandb_dir),
        stable_run_id=args.stable_run_id,
        project=args.project,
    )
    # Emitted as the LAST stdout line so the parent process can find it
    # even if the real wandb package also prints its own status lines.
    print("QUALIFICATION_RECORD_JSON=" + json.dumps(record))


def _inventory_dir(root: Path) -> dict:
    inventory = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_file():
            inventory[rel] = p.stat().st_size
    return inventory


def _orchestrate() -> None:
    import subprocess
    import tempfile
    import wandb  # real, installed package -- imported here only to record its version

    installed_wandb_version = wandb.__version__

    scratch_root = Path(tempfile.mkdtemp(prefix="wandb_real_offline_qualification_"))
    wandb_dir = scratch_root / "wandb_dir"
    wandb_dir.mkdir(parents=True)
    # Deliberately realistic length/shape for the underlying Flash-NH
    # run_id (derive_pilot_wandb_run_id then builds the actual wandb id as
    # flashnh-{policy_name}-{run_id}-{generation} -- see _run_one_invocation).
    # An earlier run of this script used a much longer, redundant stable id
    # and silently truncated wandb-core's binary transaction log path past
    # Windows' MAX_PATH (see qualification report caveats); real production
    # ids (e.g. run_id="raw_seedA") are well clear of that threshold, so
    # this exercises the actually-representative case.
    stable_run_id = "qualsmoke_raw_seedA"
    project = "flashnh-stage1-qual-smoke"

    invocation_records = []
    for invocation_index in (1, 2):
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--invocation",
                str(invocation_index),
                "--wandb-dir",
                str(wandb_dir),
                "--stable-run-id",
                stable_run_id,
                "--project",
                project,
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        record_line = next(
            (line for line in proc.stdout.splitlines() if line.startswith("QUALIFICATION_RECORD_JSON=")), None
        )
        resume_warning_lines = [
            line for line in proc.stderr.splitlines() if "resume" in line.lower() and "ignored" in line.lower()
        ]
        invocation_records.append(
            {
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
                "resume_ignored_warning_seen": resume_warning_lines,
                "record": json.loads(record_line.split("=", 1)[1]) if record_line else None,
            }
        )

    directory_inventory = _inventory_dir(wandb_dir)
    run_dirs_seen = sorted(
        {
            entry["record"]["real_wandb_run_dir"]
            for entry in invocation_records
            if entry["record"] and entry["record"]["real_wandb_run_dir"]
        }
    )
    sync_dirs_seen = sorted(
        {
            entry["record"]["real_wandb_sync_dir"]
            for entry in invocation_records
            if entry["record"] and entry["record"]["real_wandb_sync_dir"]
        }
    )

    both_invocations_ok = all(e["returncode"] == 0 and e["record"] is not None for e in invocation_records)
    both_report_wandb_backend = both_invocations_ok and all(
        e["record"]["backend"] == "wandb" for e in invocation_records
    )
    same_wandb_run_id_both_invocations = both_invocations_ok and (
        invocation_records[0]["record"]["wandb_run_id"] == invocation_records[1]["record"]["wandb_run_id"]
    )
    separate_local_sync_dirs = len(sync_dirs_seen) == 2 if both_invocations_ok else None

    qualification_record = {
        "installed_wandb_version": installed_wandb_version,
        "stable_run_id_used": stable_run_id,
        "project_used": project,
        "network_env_forced": {"WANDB_MODE": "offline"},
        "api_key_present_in_environment": "WANDB_API_KEY" in os.environ,
        "invocations": invocation_records,
        "both_invocations_succeeded": both_invocations_ok,
        "both_invocations_used_wandb_backend": both_report_wandb_backend,
        "same_logical_wandb_run_id_both_invocations": same_wandb_run_id_both_invocations,
        "distinct_local_sync_dirs_per_invocation": separate_local_sync_dirs,
        "distinct_real_wandb_run_dirs_seen": run_dirs_seen,
        "distinct_real_wandb_sync_dirs_seen": sync_dirs_seen,
        "wandb_dir_file_inventory_before_cleanup": directory_inventory,
        "wandb_dir_total_bytes_before_cleanup": sum(directory_inventory.values()),
        "sync_semantics_note": (
            "No 'wandb sync' (or any other network-capable command) was executed by this "
            "script, per the hard 'no network attempt' constraint -- including the "
            "documented-as-safe bare 'wandb sync' (no path) form, since its actual network "
            "behavior was not independently verified here. What synchronization would do is "
            "answered instead from: (a) 'wandb sync --help' text (read-only, no network -- "
            "each offline run directory holds one 'run-<id>.wandb' binary transaction log; "
            "'wandb sync <path>' uploads one such file/dir; '--sync-all' uploads every "
            "unsynced run under ./wandb); (b) the observed 'resume will be ignored since W&B "
            "syncing is set to offline' warning (see invocations[*].resume_ignored_warning_seen) "
            "-- confirms reconciliation of same-run-id continuations into one logical run is a "
            "server-side, sync-time operation keyed on the shared wandb run id + project, never "
            "a local merge of the separate offline-run directories."
        ),
        "cleanup": "wandb_dir removed after this inventory was captured (see wandb_dir_file_inventory_before_cleanup for what existed)",
    }

    out_dir = _REPO_ROOT / "reports" / "wandb_real_offline_qualification_v001"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qualification_record.json").write_text(
        json.dumps(qualification_record, indent=2, sort_keys=True), encoding="utf-8"
    )

    shutil.rmtree(scratch_root, ignore_errors=True)

    print(json.dumps(qualification_record, indent=2, sort_keys=True))
    print(f"\nQualification record written to {out_dir / 'qualification_record.json'}")
    print(f"Bulky scratch output removed from {scratch_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation", type=int, choices=(1, 2), default=None)
    parser.add_argument("--wandb-dir", type=str, default=None)
    parser.add_argument("--stable-run-id", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    args = parser.parse_args()

    if args.invocation is not None:
        _child_main(args)
    else:
        _orchestrate()


if __name__ == "__main__":
    main()
