"""Thin CLI entrypoint for the Seed-A 50k embedding-structure closure
comparison: exactly two candidates, ``emb128x64_seedA_cap_low_cal``
(incumbent, [128,64], already trained through epoch 6) and
``emb128x32_seedA_cap_low_cal`` (challenger, [128,32], starts fresh from
original Seed-A initialization). Both are capped-update calibration-family
candidates (``max_updates_per_epoch=50000``) and are NOT part of the closed
six-run matrix declared in ``config/stage1_lead06_pilot_v001.yaml`` -- this
script never edits that file or any other committed source, and never
broadens its own two-run_id allowlist.

Before this script existed, the only working implementation of this
two-candidate phase was two untracked, near-duplicate Python drivers a prior
session staged directly under Moriah's
``/sci/labs/efratmorin/omripo/Flash-NH/tmp/operations/`` (never committed to
this repository, never covered by any test, and not reproducible from a
clean clone): ``run_pilot_cap_calibration.py`` (which additionally carried
two out-of-scope raw-pathway calibration candidates,
``raw_seedA_cap_medium_cal``/``raw_seedA_cap_low_cal``, alongside
``emb128x64_seedA_cap_low_cal``) and
``run_pilot_emb128x32_seedA_cap_low_cal.py`` (the single-candidate
challenger driver). This script consolidates exactly the two run_ids this
phase actually needs into one committed, tested entrypoint, so the launch
procedure is reproducible from git history alone. It reuses the identical
mechanism those two untracked drivers already used successfully (splicing
additional ``PilotRunSpec`` entries onto the real, validated
``PilotPolicy`` loaded from the committed six-run YAML via
``load_pilot_policy()`` -- every shared field, including
``embedding_activation``/``embedding_dropout``/seeds/early-stopping cadence/
screening subset, still comes verbatim from that one committed source) and
the identical ``pilot_policy_name`` override
(``"stage1_lead06_pilot_cap_calibration_v001"``), so continuing the
already-trained incumbent and generating the challenger's config resolve to
the exact same NH run directory / W&B identity conventions Session A's
untracked drivers already established (``experiment_name`` and
``package_type`` depend only on ``run_id``, never on which script computed
the spec -- see ``pilot_orchestration.py``).

Contains no modeling, screening, stopping, or tracking logic of its own --
every decision is made by the pilot subsystems ``run_pilot``/
``prepare_pilot_run_only``/``compute_pilot_status_fields`` compose. Mirrors
``scripts/run_stage1_lead06_pilot.py``'s CLI/call structure and docstrings
closely; the only behavioral differences are (a) the two-entry policy
splice below, (b) ``--run-id`` is restricted to exactly the two closure
run_ids (never a real six-run pilot run_id, never the out-of-scope
raw-pathway or 25k-tier calibration candidates), (c) an additional
``--status-only`` read-only mode (see below), and (d) ordinary training is
always bounded at ``CLOSURE_MAX_TARGET_EPOCH`` (12) -- this phase's approved
comparison horizon -- with no CLI flag or env var able to override it.

Does not submit any Slurm job itself. The paired
``.sbatch`` launcher (``scripts/run_stage1_cap50k_closure_moriah.sbatch``)
calls this script once per Slurm allocation, adding an explicit
commit-pin safety check (``EXPECTED_COMMIT``) this script itself does not
perform -- see that launcher's header comment for why that check lives at
the shell level, before this script (or any repo Python) is even invoked.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
from pathlib import Path

# Direct execution (`python scripts/run_stage1_cap50k_closure.py ...`) puts
# scripts/ -- not the repository work directory -- at the front of
# sys.path, so the sibling top-level package `src` is otherwise unimportable
# regardless of the caller's current working directory. Insert the repo work
# directory (this file's parent's parent) before importing src.baseline.*.
_REPO_WORKDIR = Path(__file__).resolve().parent.parent
if str(_REPO_WORKDIR) not in sys.path:
    sys.path.insert(0, str(_REPO_WORKDIR))

from src.baseline.pilot_lead06_config import PilotRunSpec, load_pilot_policy
from src.baseline.pilot_orchestration import (
    compute_pilot_status_fields,
    discover_nh_run_dir,
    prepare_pilot_run_only,
    run_pilot,
)
from src.baseline.wandb_tracking import load_tracking_policy

_DEFAULT_PILOT_POLICY_PATH = _REPO_WORKDIR / "config" / "stage1_lead06_pilot_v001.yaml"
_DEFAULT_BASELINE_POLICY_PATH = _REPO_WORKDIR / "config" / "stage1_scientific_baseline_v001.yaml"
_DEFAULT_SPLITS_DIR = _REPO_WORKDIR / "config" / "stage1_baseline_splits_v001"

_SEEDA = 967139
_CAP50K = 50_000

# This closure comparison is approved only through epoch 12 -- fixed and
# non-overridable (no CLI flag, no env var). Ordinary training always passes
# this exact value to run_pilot()'s max_target_epoch bounded-recovery
# parameter; it never changes the underlying 36-epoch early-stopping policy
# (chunk_epoch_targets/screening cadence), only where this phase pauses.
CLOSURE_MAX_TARGET_EPOCH = 12

# Exactly the two approved candidates for this phase -- both reuse an
# already-registered nh_config_generation run profile unmodified (added in
# commits 8e924cf/5aba586 respectively) and differ from each other only in
# embedding_hiddens/run_profile_name. Neither entry may ever be added to or
# removed without updating this literal mapping (there is deliberately no
# generic "any embedding shape" composition helper here -- see module
# docstring).
CLOSURE_RUN_SPECS = {
    "emb128x64_seedA_cap_low_cal": PilotRunSpec(
        run_id="emb128x64_seedA_cap_low_cal",
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 64],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x64_seedA_v001",
        max_updates_per_epoch=_CAP50K,
    ),
    "emb128x32_seedA_cap_low_cal": PilotRunSpec(
        run_id="emb128x32_seedA_cap_low_cal",
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP50K,
    ),
}

# Matches the pilot_policy_name Session A's untracked drivers already used
# for emb128x64_seedA_cap_low_cal and emb128x32_seedA_cap_low_cal -- kept
# identical so this script resolves the exact same derive_pilot_wandb_run_id
# identity and does not orphan the incumbent's already-trained NH run
# directory or the challenger's already-produced PREPARED_ONLY evidence.
_CLOSURE_POLICY_NAME = "stage1_lead06_pilot_cap_calibration_v001"


def _resolve_policy_relative_paths(pilot_policy):
    """Identical to run_stage1_lead06_pilot.py's helper of the same name --
    the pilot policy YAML declares its own composed-artifact paths relative
    to the repository work directory; absolute-ize them so this script
    behaves identically regardless of the caller's current working
    directory."""

    def _abs(raw: str) -> str:
        p = Path(raw)
        return str(p) if p.is_absolute() else str(_REPO_WORKDIR / p)

    return dataclasses.replace(
        pilot_policy,
        screening_basin_ids_path=_abs(pilot_policy.screening_basin_ids_path),
        base_early_stopping_policy_path=_abs(pilot_policy.base_early_stopping_policy_path),
        wandb_policy_path=_abs(pilot_policy.wandb_policy_path),
    )


def _build_closure_policy(base_pilot_policy):
    """Splice the two closure-only PilotRunSpec entries into a copy of the
    real, already-validated PilotPolicy. Never touches runs for any of the
    six real run_ids and never writes anything back to the committed YAML --
    this augmented object exists only in this process's memory for the
    lifetime of this one invocation."""
    augmented_runs = dict(base_pilot_policy.runs)
    for run_id, spec in CLOSURE_RUN_SPECS.items():
        if run_id in augmented_runs:
            raise RuntimeError(
                f"closure run_id {run_id!r} unexpectedly collides with an existing real "
                "pilot run_id in the committed policy -- refusing to overwrite it"
            )
        augmented_runs[run_id] = spec
    augmented_raw = dict(base_pilot_policy.raw)
    augmented_raw["policy_name"] = _CLOSURE_POLICY_NAME
    return dataclasses.replace(base_pilot_policy, runs=augmented_runs, raw=augmented_raw)


def _slurm_identity_from_env(args) -> dict:
    return {
        "job_id": args.slurm_job_id or os.environ.get("SLURM_JOB_ID"),
        "node": args.slurm_node or os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "partition": args.slurm_partition or os.environ.get("SLURM_JOB_PARTITION"),
        "gres": args.slurm_gres or os.environ.get("SLURM_JOB_GRES"),
    }


def _run_status_only(pilot_policy, run_id: str, config_out_dir: Path) -> dict:
    """Read-only continuation-safety status snapshot: no config generation,
    no NH training/evaluation call, no W&B backend, no checkpoint/
    optimizer-state/evidence file written. Calls only already-committed,
    already-tested read helpers (discover_nh_run_dir,
    compute_pilot_status_fields) against whatever NH run directory already
    exists on disk for this run_id -- exactly the mechanism Session A's
    untracked check_continuation_status.py already exercised successfully
    against the incumbent (real Moriah evidence, job 45759808: epoch 6
    trained/screened, next_intended_screening_epoch=9,
    safe_to_continue_automatically=True). For a brand-new run_id with no NH
    run directory yet (e.g. the challenger before its first chunk), reports
    a distinct 'NO_EXISTING_NH_RUN_DIRECTORY' status rather than raising."""
    experiment_name = f"stage1_lead06_pilot_{run_id}_v001"
    try:
        nh_run_dir = discover_nh_run_dir(config_out_dir, experiment_name)
    except Exception as exc:
        return {
            "status": "NO_EXISTING_NH_RUN_DIRECTORY",
            "run_id": run_id,
            "experiment_name": experiment_name,
            "config_out_dir": str(config_out_dir),
            "nh_run_dir": None,
            "detail": str(exc),
        }
    fields = compute_pilot_status_fields(nh_run_dir, pilot_policy=pilot_policy)
    return {
        "status": "EXISTING_NH_RUN_DIRECTORY_FOUND",
        "run_id": run_id,
        "experiment_name": experiment_name,
        "config_out_dir": str(config_out_dir),
        "nh_run_dir": str(nh_run_dir),
        **fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", required=True, choices=sorted(CLOSURE_RUN_SPECS),
        help="One of the two approved closure run_ids (never a real six-run pilot run_id)",
    )
    parser.add_argument("--pilot-policy-path", type=Path, default=_DEFAULT_PILOT_POLICY_PATH)
    parser.add_argument("--baseline-policy-path", type=Path, default=_DEFAULT_BASELINE_POLICY_PATH)
    parser.add_argument("--package-root", type=Path, help="Required unless --status-only")
    parser.add_argument("--splits-dir", type=Path, default=_DEFAULT_SPLITS_DIR)
    parser.add_argument("--config-out-dir", type=Path, required=True)
    parser.add_argument("--evidence-out-dir", type=Path, help="Required unless --status-only")
    parser.add_argument("--static-column-manifest-path", type=Path, default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate the config even if config-out-dir already has one. Never appropriate "
        "on a routine resume of a run that has already started training (e.g. the incumbent).",
    )
    parser.add_argument("--slurm-job-id", default=None)
    parser.add_argument("--slurm-node", default=None)
    parser.add_argument("--slurm-partition", default=None)
    parser.add_argument("--slurm-gres", default=None)
    parser.add_argument("--wandb-policy-path", type=Path, default=None)
    parser.add_argument("--tracking-generation", default="g1")
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Generate this run_id's NH config + generation manifest and exit before any NH "
        "training call or W&B backend initialization. Refuses (via prepare_pilot_run_only) if "
        "this run_id already has a real NH run directory or evidence bundle -- only ever "
        "appropriate for the challenger's first invocation, never the incumbent.",
    )
    parser.add_argument(
        "--status-only", action="store_true",
        help="Read-only: report this run_id's on-disk continuation status (highest physical "
        "checkpoint epoch, highest screened epoch, next intended screening epoch, overshoot, "
        "safe_to_continue_automatically) and exit. Never generates a config, never calls NH, "
        "never writes anything. Mutually exclusive with --prepare-only.",
    )
    args = parser.parse_args()

    if args.run_id not in CLOSURE_RUN_SPECS:
        parser.error(f"--run-id {args.run_id!r} is not one of the closure run_ids {sorted(CLOSURE_RUN_SPECS)}")
    if args.prepare_only and args.status_only:
        parser.error("--prepare-only and --status-only are mutually exclusive")

    # Load through the REAL, unmodified validator against the REAL,
    # unmodified committed YAML -- this is the only source of every shared
    # field (lead_hours, seq_length, seeds, early_stopping cadence,
    # screening subset identity/hash, base wandb_policy_path,
    # embedding_activation/embedding_dropout, etc.).
    real_pilot_policy = load_pilot_policy(args.pilot_policy_path)
    pilot_policy = _build_closure_policy(real_pilot_policy)
    pilot_policy = _resolve_policy_relative_paths(pilot_policy)

    if args.wandb_policy_path is not None:
        wandb_policy_override_path = str(args.wandb_policy_path)
        load_tracking_policy(wandb_policy_override_path)
        pilot_policy = dataclasses.replace(pilot_policy, wandb_policy_path=wandb_policy_override_path)

    if args.status_only:
        result = _run_status_only(pilot_policy, args.run_id, args.config_out_dir)
        result["closure_max_target_epoch"] = CLOSURE_MAX_TARGET_EPOCH
        print(json.dumps(result, indent=2, default=str))
        return

    if args.package_root is None:
        parser.error("--package-root is required unless --status-only")
    if args.evidence_out_dir is None:
        parser.error("--evidence-out-dir is required unless --status-only")

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
            commands_used=["python scripts/run_stage1_cap50k_closure.py " + " ".join(sys.argv[1:])],
            force=args.force,
        )
        result["closure_max_target_epoch"] = CLOSURE_MAX_TARGET_EPOCH
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
        commands_used=["python scripts/run_stage1_cap50k_closure.py " + " ".join(sys.argv[1:])],
        force=args.force,
        tracking_generation=args.tracking_generation,
        max_target_epoch=CLOSURE_MAX_TARGET_EPOCH,
    )

    printable = dict(result)
    printable["nh_run_dir"] = str(printable["nh_run_dir"])
    printable["evidence_bundle_path"] = str(printable["evidence_bundle_path"])
    printable["closure_max_target_epoch"] = CLOSURE_MAX_TARGET_EPOCH
    print(json.dumps(printable, indent=2))

    if result["final_status"] == "blocked_continuation_overshoot_conflict":
        sys.exit(1)


if __name__ == "__main__":
    main()
