"""Thin CLI entrypoint for the LR-A learning-rate range-characterization
campaign (``lr_range_seedA_25k_v001``, see docs/decision_log.md's LR-A
design-freeze entry): exactly four NEW trainable candidates, all
``[128, 32]`` learned-FC-embedding, Seed A (967139),
``max_updates_per_epoch=25000``, one uninterrupted epoch 1->6 training
segment --

    emb128x32_seedA_lr1em4_cap25k_cal   learning_rate=1e-4
    emb128x32_seedA_lr3em4_cap25k_cal   learning_rate=3e-4
    emb128x32_seedA_lr3em3_cap25k_cal   learning_rate=3e-3
    emb128x32_seedA_lr1em2_cap25k_cal   learning_rate=1e-2

-- differing from each other, and from the already-trained fifth reference
point, ``emb128x32_seedA_cap25k_cal`` (lr=1e-3, REUSE ELIGIBLE per the LR-A
design freeze), ONLY in ``learning_rate``. The reference is never retrained
or reconfigured by this script: it is not a member of ``LR_A_RUN_SPECS``
(the closed four-run_id trainable allowlist) and is reachable only through
``--status-only`` (see ``REFERENCE_RUN_ID`` below) so its already-trained,
already-evidenced state can be inspected read-only, exactly like any other
run_id, without this script ever being able to generate a config or call
``run_pilot`` for it.

Structurally a close mirror of ``scripts/run_stage1_cap50k_closure.py`` --
same splice-onto-the-real-validated-PilotPolicy mechanism (via
``load_pilot_policy()`` against the committed six-run YAML, so every shared
field -- ``embedding_activation``/``embedding_dropout``/seeds/early-stopping
cadence/screening subset -- still comes verbatim from that one committed
source), same ``--prepare-only``/``--status-only`` modes, same CLI/call
structure. The differences specific to LR-A: (a) four run_ids instead of
two, each carrying an explicit ``learning_rate`` override (see
``pilot_lead06_config.PilotRunSpec.learning_rate`` and
``nh_config_generation.validate_learning_rate_override``) rather than an
embedding-shape difference, (b) training is always bounded at
``LR_A_MAX_TARGET_EPOCH`` (6) -- this campaign's approved, frozen horizon --
with no CLI flag or env var able to override it, (c) there is deliberately
no ``--learning-rate`` CLI flag either: every candidate's learning rate is
fixed by the literal ``LR_A_RUN_SPECS`` mapping below, never a caller
-supplied value (mirrors how ``CLOSURE_MAX_TARGET_EPOCH`` has no CLI/env
override in the cap50k closure script -- this closed campaign's scientific
contract is not reconfigurable from the command line), and (d) the read-only
``REFERENCE_RUN_ID`` carve-out described above.

Contains no modeling, screening, stopping, or tracking logic of its own --
every decision is made by the pilot subsystems ``run_pilot``/
``prepare_pilot_run_only``/``compute_pilot_status_fields`` compose.

Does not submit any Slurm job itself. The paired ``.sbatch`` launcher
(``scripts/run_stage1_lr_range_seedA_closure_moriah.sbatch``) calls this
script once per Slurm allocation, adding an explicit commit-pin safety check
(``EXPECTED_COMMIT``) this script itself does not perform -- see that
launcher's header comment for why that check lives at the shell level,
before this script (or any repo Python) is even invoked. Neither this
script nor its paired launcher is ever submitted/executed against real
Slurm/NH/GPU within this task -- see docs/decision_log.md's LR-A
implementation-readiness entry.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
from pathlib import Path

# Direct execution (`python scripts/run_stage1_lr_range_seedA_closure.py ...`)
# puts scripts/ -- not the repository work directory -- at the front of
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
_CAP25K = 25_000

# LR-A's frozen, non-overridable horizon: one uninterrupted epoch 1->6
# training segment, no candidate continuing beyond epoch 6 (see Section 1 of
# the LR-A design freeze). No CLI flag, no env var can override this.
LR_A_MAX_TARGET_EPOCH = 6

# The already-trained fifth LR-A data point -- REUSE ELIGIBLE per the LR-A
# design freeze's final package-identity confirmation. Deliberately NOT a
# key of LR_A_RUN_SPECS (never trainable/configurable through this script);
# reachable only via --status-only (read-only inspection of its existing NH
# run directory), so this reused reference stays external/read-only to this
# closure script for the entirety of the LR-A campaign.
REFERENCE_RUN_ID = "emb128x32_seedA_cap25k_cal"

# Exactly the four NEW, approved LR-A candidates -- each reuses the same
# already-registered nh_config_generation run profile
# ("pilot_lead06_emb128x32_seedA_v001", i.e. the same profile the reused
# emb128x32_seedA_cap25k_cal reference itself uses) unmodified, differing
# from each other and from the reference ONLY in learning_rate. Neither this
# mapping's keys nor any entry's learning_rate may be changed without
# updating this literal mapping -- there is deliberately no generic
# "any learning rate" composition helper or CLI override here (see module
# docstring).
LR_A_RUN_SPECS = {
    "emb128x32_seedA_lr1em4_cap25k_cal": PilotRunSpec(
        run_id="emb128x32_seedA_lr1em4_cap25k_cal",
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=1e-4,
    ),
    "emb128x32_seedA_lr3em4_cap25k_cal": PilotRunSpec(
        run_id="emb128x32_seedA_lr3em4_cap25k_cal",
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=3e-4,
    ),
    "emb128x32_seedA_lr3em3_cap25k_cal": PilotRunSpec(
        run_id="emb128x32_seedA_lr3em3_cap25k_cal",
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=3e-3,
    ),
    "emb128x32_seedA_lr1em2_cap25k_cal": PilotRunSpec(
        run_id="emb128x32_seedA_lr1em2_cap25k_cal",
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=1e-2,
    ),
}

# Frozen campaign name (Section 1 of the LR-A design freeze).
_LR_A_POLICY_NAME = "lr_range_seedA_25k_v001"


def _resolve_policy_relative_paths(pilot_policy):
    """Identical to run_stage1_lead06_pilot.py's/run_stage1_cap50k_closure.py's
    helper of the same name -- the pilot policy YAML declares its own
    composed-artifact paths relative to the repository work directory;
    absolute-ize them so this script behaves identically regardless of the
    caller's current working directory."""

    def _abs(raw: str) -> str:
        p = Path(raw)
        return str(p) if p.is_absolute() else str(_REPO_WORKDIR / p)

    return dataclasses.replace(
        pilot_policy,
        screening_basin_ids_path=_abs(pilot_policy.screening_basin_ids_path),
        base_early_stopping_policy_path=_abs(pilot_policy.base_early_stopping_policy_path),
        wandb_policy_path=_abs(pilot_policy.wandb_policy_path),
    )


def _build_lr_a_policy(base_pilot_policy):
    """Splice the four LR-A-only PilotRunSpec entries into a copy of the
    real, already-validated PilotPolicy. Never touches runs for any of the
    six real run_ids or the reused reference run_id, and never writes
    anything back to the committed YAML -- this augmented object exists only
    in this process's memory for the lifetime of this one invocation."""
    augmented_runs = dict(base_pilot_policy.runs)
    if REFERENCE_RUN_ID in augmented_runs:
        raise RuntimeError(
            f"reused reference run_id {REFERENCE_RUN_ID!r} unexpectedly collides with an "
            "existing real pilot run_id in the committed policy -- refusing to proceed"
        )
    for run_id, spec in LR_A_RUN_SPECS.items():
        if run_id in augmented_runs:
            raise RuntimeError(
                f"LR-A run_id {run_id!r} unexpectedly collides with an existing real "
                "pilot run_id in the committed policy -- refusing to overwrite it"
            )
        augmented_runs[run_id] = spec
    augmented_raw = dict(base_pilot_policy.raw)
    augmented_raw["policy_name"] = _LR_A_POLICY_NAME
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
    exists on disk for this run_id. Works identically for any of the four
    LR-A run_ids or for REFERENCE_RUN_ID (the reused reference) -- the
    experiment_name naming convention depends only on run_id, never on
    whether it is a key of LR_A_RUN_SPECS, so this is the one supported way
    to inspect the reused reference's on-disk state through this script
    without ever being able to train/reconfigure it. For a brand-new run_id
    with no NH run directory yet, reports a distinct
    'NO_EXISTING_NH_RUN_DIRECTORY' status rather than raising."""
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
        "--run-id", required=True, choices=sorted(LR_A_RUN_SPECS) + [REFERENCE_RUN_ID],
        help="One of the four approved LR-A run_ids, or the reused reference run_id "
        f"({REFERENCE_RUN_ID!r}, --status-only only -- never trainable/configurable here).",
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
        "on a routine resume of a run that has already started training.",
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
        "this run_id already has a real NH run directory or evidence bundle. Never valid for "
        f"{REFERENCE_RUN_ID!r} (the reused reference).",
    )
    parser.add_argument(
        "--status-only", action="store_true",
        help="Read-only: report this run_id's on-disk continuation status (highest physical "
        "checkpoint epoch, highest screened epoch, next intended screening epoch, overshoot, "
        f"safe_to_continue_automatically) and exit. The only mode valid for {REFERENCE_RUN_ID!r} "
        "(the reused reference). Never generates a config, never calls NH, never writes "
        "anything. Mutually exclusive with --prepare-only.",
    )
    args = parser.parse_args()

    if args.run_id not in LR_A_RUN_SPECS and args.run_id != REFERENCE_RUN_ID:
        parser.error(
            f"--run-id {args.run_id!r} is not one of the LR-A run_ids {sorted(LR_A_RUN_SPECS)} "
            f"or the reused reference {REFERENCE_RUN_ID!r}"
        )
    if args.prepare_only and args.status_only:
        parser.error("--prepare-only and --status-only are mutually exclusive")
    if args.run_id == REFERENCE_RUN_ID and not args.status_only:
        parser.error(
            f"--run-id {REFERENCE_RUN_ID!r} (the reused reference) is only ever valid with "
            "--status-only -- it is never trainable or configurable through this script"
        )

    # Load through the REAL, unmodified validator against the REAL,
    # unmodified committed YAML -- this is the only source of every shared
    # field (lead_hours, seq_length, seeds, early_stopping cadence,
    # screening subset identity/hash, base wandb_policy_path,
    # embedding_activation/embedding_dropout, etc.).
    real_pilot_policy = load_pilot_policy(args.pilot_policy_path)
    pilot_policy = _build_lr_a_policy(real_pilot_policy)
    pilot_policy = _resolve_policy_relative_paths(pilot_policy)

    if args.wandb_policy_path is not None:
        wandb_policy_override_path = str(args.wandb_policy_path)
        load_tracking_policy(wandb_policy_override_path)
        pilot_policy = dataclasses.replace(pilot_policy, wandb_policy_path=wandb_policy_override_path)

    if args.status_only:
        result = _run_status_only(pilot_policy, args.run_id, args.config_out_dir)
        result["lr_a_max_target_epoch"] = LR_A_MAX_TARGET_EPOCH
        result["is_reused_reference"] = args.run_id == REFERENCE_RUN_ID
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
            commands_used=["python scripts/run_stage1_lr_range_seedA_closure.py " + " ".join(sys.argv[1:])],
            force=args.force,
        )
        result["lr_a_max_target_epoch"] = LR_A_MAX_TARGET_EPOCH
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
        commands_used=["python scripts/run_stage1_lr_range_seedA_closure.py " + " ".join(sys.argv[1:])],
        force=args.force,
        tracking_generation=args.tracking_generation,
        max_target_epoch=LR_A_MAX_TARGET_EPOCH,
    )

    printable = dict(result)
    printable["nh_run_dir"] = str(printable["nh_run_dir"])
    printable["evidence_bundle_path"] = str(printable["evidence_bundle_path"])
    printable["lr_a_max_target_epoch"] = LR_A_MAX_TARGET_EPOCH
    print(json.dumps(printable, indent=2))

    if result["final_status"] == "blocked_continuation_overshoot_conflict":
        sys.exit(1)


if __name__ == "__main__":
    main()
