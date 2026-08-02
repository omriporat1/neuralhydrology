"""Thin CLI entrypoint for the Stage 1 lead-6 optimization pilot (task item 6).

Wraps a single call to ``src.baseline.pilot_orchestration.run_pilot`` for one
``--run-id`` out of the closed six-run matrix declared in
``config/stage1_lead06_pilot_v001.yaml`` (see docs/stage1_lead06_pilot_v001.md).
Contains no modeling, screening, stopping, or tracking logic of its own --
every decision is made by the pilot subsystems ``run_pilot`` composes. Safe to
invoke repeatedly with the same ``--config-out-dir``/``--evidence-out-dir``
for the same ``--run-id``: already-trained chunks are not retrained and
already-logged screening epochs are not re-logged (see
``pilot_orchestration.py``'s module docstring).

Does not submit any Slurm job itself -- it is meant to be invoked inside an
already-allocated interactive or batch process, exactly like
``scripts/run_stage1_nh.py``. The paired ``.sbatch`` launcher
(``scripts/run_stage1_lead06_pilot_moriah.sbatch``, task item 7) calls this
script once per Slurm allocation and relies on this script's own idempotent
resume behavior to continue across wall-time-limited restarts.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
from pathlib import Path

# Direct execution (`python scripts/run_stage1_lead06_pilot.py ...`) puts
# scripts/ -- not the repository work directory -- at the front of
# sys.path, so the sibling top-level package `src` is otherwise unimportable
# regardless of the caller's current working directory. Insert the repo work
# directory (this file's parent's parent) before importing src.baseline.*.
_REPO_WORKDIR = Path(__file__).resolve().parent.parent
if str(_REPO_WORKDIR) not in sys.path:
    sys.path.insert(0, str(_REPO_WORKDIR))

from src.baseline.pilot_lead06_config import load_pilot_policy, pilot_run_ids
from src.baseline.pilot_orchestration import prepare_pilot_run_only, run_pilot
from src.baseline.wandb_tracking import load_tracking_policy

_DEFAULT_PILOT_POLICY_PATH = _REPO_WORKDIR / "config" / "stage1_lead06_pilot_v001.yaml"
_DEFAULT_BASELINE_POLICY_PATH = _REPO_WORKDIR / "config" / "stage1_scientific_baseline_v001.yaml"
_DEFAULT_SPLITS_DIR = _REPO_WORKDIR / "config" / "stage1_baseline_splits_v001"


def _resolve_policy_relative_paths(pilot_policy):
    """The pilot policy YAML declares its own composed-artifact paths
    (screening basin-ids file, early-stopping policy, W&B policy) relative to
    the repository work directory, matching every other committed policy file
    in this repo. Absolute-ize them here so this script behaves identically
    regardless of the caller's current working directory (a Slurm batch
    script's shell state is not this script's concern)."""

    def _abs(raw: str) -> str:
        p = Path(raw)
        return str(p) if p.is_absolute() else str(_REPO_WORKDIR / p)

    return dataclasses.replace(
        pilot_policy,
        screening_basin_ids_path=_abs(pilot_policy.screening_basin_ids_path),
        base_early_stopping_policy_path=_abs(pilot_policy.base_early_stopping_policy_path),
        wandb_policy_path=_abs(pilot_policy.wandb_policy_path),
    )


def _slurm_identity_from_env(args) -> dict:
    return {
        "job_id": args.slurm_job_id or os.environ.get("SLURM_JOB_ID"),
        "node": args.slurm_node or os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "partition": args.slurm_partition or os.environ.get("SLURM_JOB_PARTITION"),
        "gres": args.slurm_gres or os.environ.get("SLURM_JOB_GRES"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="One of the six closed pilot run_ids")
    parser.add_argument("--pilot-policy-path", type=Path, default=_DEFAULT_PILOT_POLICY_PATH)
    parser.add_argument("--baseline-policy-path", type=Path, default=_DEFAULT_BASELINE_POLICY_PATH)
    parser.add_argument(
        "--package-root", type=Path, required=True,
        help="Root of the certified full non-CA scientific package (e.g. stage1_scientific_package_v002)",
    )
    parser.add_argument("--splits-dir", type=Path, default=_DEFAULT_SPLITS_DIR)
    parser.add_argument("--config-out-dir", type=Path, required=True)
    parser.add_argument("--evidence-out-dir", type=Path, required=True)
    parser.add_argument("--static-column-manifest-path", type=Path, default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate the config even if config-out-dir already has one. Leave unset for "
        "ordinary Slurm restarts/resumes: run_pilot() always (re)writes evidence-out-dir on "
        "every call regardless of this flag, so --force is only needed to deliberately "
        "regenerate an already-generated config (rare, and never appropriate on a routine "
        "resume of a run that has already started training).",
    )
    parser.add_argument("--slurm-job-id", default=None)
    parser.add_argument("--slurm-node", default=None)
    parser.add_argument("--slurm-partition", default=None)
    parser.add_argument("--slurm-gres", default=None)
    parser.add_argument(
        "--wandb-policy-path", type=Path, default=None,
        help="Optional per-invocation override for the W&B tracking policy file. Replaces "
        "only PilotPolicy.wandb_policy_path for this one run -- the committed "
        "config/stage1_wandb_tracking_policy_v001.yaml (and its safe enabled=false/"
        "mode=disabled default) is never read, edited, or otherwise affected. Intended for "
        "an untracked, machine-local policy YAML (e.g. a copy of the committed policy with "
        "only enabled/mode flipped). Loaded through the same validator as the committed "
        "policy, so a missing or malformed file fails immediately, before any config "
        "generation or training starts. Leave unset to keep the committed disabled policy.",
    )
    parser.add_argument(
        "--tracking-generation", default="g1",
        help="W&B tracking_generation for this candidate (default 'g1', correct for every "
        "ordinary run and bounded-Slurm continuation). Only a deliberate, manual "
        "restart-from-scratch under the same --run-id (e.g. after abandoning and deleting "
        "an NH run directory) should ever pass a different value -- see "
        "src/baseline/pilot_tracking.py's module docstring.",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Generate this run_id's NH config + generation manifest (exactly the "
        "prepare_pilot_run() step run_pilot() itself calls first) and exit before any NH "
        "training call, W&B backend initialization, or offline run directory creation. "
        "Writes pilot_preparation_result.json under --evidence-out-dir. Refuses to run (see "
        "src.baseline.pilot_orchestration.prepare_pilot_run_only) if this run_id already has "
        "a real NH run directory or evidence bundle -- --prepare-only only ever prepares a "
        "brand-new, untrained candidate, never resumes or continues one.",
    )
    args = parser.parse_args()

    pilot_policy = load_pilot_policy(args.pilot_policy_path)
    if args.run_id not in pilot_run_ids(pilot_policy):
        parser.error(f"--run-id {args.run_id!r} is not one of {pilot_run_ids(pilot_policy)}")
    pilot_policy = _resolve_policy_relative_paths(pilot_policy)

    if args.wandb_policy_path is not None:
        wandb_policy_override_path = str(args.wandb_policy_path)
        # Fail loudly right here, before any config generation or training,
        # if the supplied override is missing or malformed -- reuses the
        # exact same validator the committed default policy goes through
        # (src.baseline.wandb_tracking.load_tracking_policy), not a
        # separate, weaker check.
        load_tracking_policy(wandb_policy_override_path)
        pilot_policy = dataclasses.replace(pilot_policy, wandb_policy_path=wandb_policy_override_path)

    if args.prepare_only:
        result = prepare_pilot_run_only(
            pilot_policy=pilot_policy,
            run_id=args.run_id,
            baseline_policy_path=args.baseline_policy_path,
            package_root=args.package_root,
            splits_dir=args.splits_dir,
            config_out_dir=args.config_out_dir,
            preparation_out_dir=args.evidence_out_dir,
            static_column_manifest_path=args.static_column_manifest_path,
            tracking_generation=args.tracking_generation,
            commands_used=["python scripts/run_stage1_lead06_pilot.py " + " ".join(sys.argv[1:])],
            force=args.force,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    result = run_pilot(
        pilot_policy=pilot_policy,
        run_id=args.run_id,
        baseline_policy_path=args.baseline_policy_path,
        package_root=args.package_root,
        splits_dir=args.splits_dir,
        config_out_dir=args.config_out_dir,
        evidence_out_dir=args.evidence_out_dir,
        static_column_manifest_path=args.static_column_manifest_path,
        slurm_identity=_slurm_identity_from_env(args),
        commands_used=["python scripts/run_stage1_lead06_pilot.py " + " ".join(sys.argv[1:])],
        force=args.force,
        tracking_generation=args.tracking_generation,
    )

    printable = dict(result)
    printable["nh_run_dir"] = str(printable["nh_run_dir"])
    printable["evidence_bundle_path"] = str(printable["evidence_bundle_path"])
    print(json.dumps(printable, indent=2))

    # Job 45718473: a blocked run (untrusted continuation overshoot -- see
    # pilot_orchestration.py's module docstring) is a deliberate, non-crashing
    # determination, but it is NOT an unqualified success either -- a human
    # must resolve the overshoot before this run_id can proceed. Reuse the
    # paired .sbatch launcher's own existing exit-code convention
    # (BLOCKED_MANUAL_REVIEW_REQUIRED -> 1, same as FAILED_NO_CHECKPOINT)
    # rather than always exiting 0 regardless of final_status, so a blocked
    # result is distinguishable by exit code alone even outside that launcher.
    if result["final_status"] == "blocked_continuation_overshoot_conflict":
        sys.exit(1)


if __name__ == "__main__":
    main()
