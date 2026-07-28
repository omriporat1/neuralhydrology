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
from src.baseline.pilot_orchestration import run_pilot

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
    args = parser.parse_args()

    pilot_policy = load_pilot_policy(args.pilot_policy_path)
    if args.run_id not in pilot_run_ids(pilot_policy):
        parser.error(f"--run-id {args.run_id!r} is not one of {pilot_run_ids(pilot_policy)}")
    pilot_policy = _resolve_policy_relative_paths(pilot_policy)

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
    )

    printable = dict(result)
    printable["nh_run_dir"] = str(printable["nh_run_dir"])
    printable["evidence_bundle_path"] = str(printable["evidence_bundle_path"])
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
