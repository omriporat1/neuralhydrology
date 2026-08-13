"""Thin CLI entrypoint for the Sequence-Length-A sequence-length range-
characterization campaign (``seq_length_range_seedA_25k_v001``): exactly
four NEW trainable candidates, all ``[128, 32]`` learned-FC-embedding, Seed A
(967139), ``hidden_size=128``, ``learning_rate=3e-4`` (the LR-A-characterized
provisional anchor, held fixed), ``max_updates_per_epoch=25000``, one
uninterrupted epoch 1->6 training segment, ``statics_embedding.dropout`` left
at the profile default (the Embedding-Dropout-A-characterized 0.10
provisional anchor, held fixed, never explicitly overridden here) --

    emb128x32_seedA_seq12_h128_lr3em4_cap25k_cal   seq_length=12
    emb128x32_seedA_seq24_h128_lr3em4_cap25k_cal   seq_length=24
    emb128x32_seedA_seq48_h128_lr3em4_cap25k_cal   seq_length=48
    emb128x32_seedA_seq72_h128_lr3em4_cap25k_cal   seq_length=72

-- differing from each other ONLY in ``seq_length`` (see
``pilot_lead06_config.PilotRunSpec.seq_length`` and
``nh_config_generation.validate_seq_length``). This is Phase-A range
characterization of input sequence length at a fixed LR/hidden-size/
embedding-dropout, NOT final sequence-length selection and NOT joint
LR x hidden-size x embedding-dropout x seq-length HPO (that interaction is
deliberately out of scope). Embedding shape ``[128, 32]``, embedding
activation ``tanh``, output dropout, and every other frozen scientific
setting are unchanged from the committed six-run pilot matrix. The closed,
supported ``seq_length`` set is exactly ``{12, 24, 48, 72}`` (see
``nh_config_generation.validate_seq_length``) -- these four candidates are
that entire set, not a subsample of it.

The historical Hidden-size-A H=128 run,
``emb128x32_seedA_h128_lr3em4_cap25k_cal`` (``seq_length`` == the committed
policy's default ``24`` -- never explicitly overridden), is DELIBERATELY NOT
reused as this campaign's seq_length=24 data point -- a fresh
``emb128x32_seedA_seq24_h128_lr3em4_cap25k_cal`` candidate is trained
instead, as one of the four members of ``SEQ_LENGTH_A_RUN_SPECS`` below
(mirroring Embedding-Dropout-A's identical treatment of its own drop10 vs.
the same historical H128 run). The historical run is preserved only as a
read-only, non-pooled, non-cherry-picked reproducibility comparator: it is
not a member of ``SEQ_LENGTH_A_RUN_SPECS`` and is reachable only through
``--status-only`` (see ``REFERENCE_RUN_ID`` below) so its already-trained,
already-evidenced state can be inspected read-only, exactly like any other
run_id, without this script ever being able to retrain or reconfigure it.
The fresh-vs-historical reproducibility comparison itself is explicitly
deferred until after the fresh seq24 candidate completes -- this script does
not perform or imply that comparison.

Structurally a close mirror of
``scripts/run_stage1_embedding_dropout_range_seedA_closure.py`` -- same
splice-onto-the-real-validated-PilotPolicy mechanism (via
``load_pilot_policy()`` against the committed six-run YAML, so every shared
field still comes verbatim from that one committed source), same
``--prepare-only``/``--status-only`` modes, same CLI/call structure. The
differences specific to Sequence-Length-A: (a) four run_ids instead of five,
each carrying an explicit ``seq_length`` override (see
``pilot_lead06_config.PilotRunSpec.seq_length`` and
``nh_config_generation.validate_seq_length``) AND explicit ``hidden_size=128``
/``learning_rate=3e-4`` overrides (both held fixed across all four, at the
Hidden-size-A/LR-A provisional anchors) rather than varying hidden size, with
``embedding_dropout`` deliberately left unset (profile default, the
Embedding-Dropout-A-characterized 0.10 provisional anchor) rather than
overridden, (b) training is always bounded at
``SEQ_LENGTH_A_MAX_TARGET_EPOCH`` (6, matching every other Phase-A range-
characterization campaign's horizon) -- this campaign's approved, frozen
horizon -- with no CLI flag or env var able to override it, (c) there is
deliberately no ``--seq-length``/``--hidden-size``/``--learning-rate`` CLI
flag either: every candidate's seq_length, hidden size, and learning rate are
fixed by the literal ``SEQ_LENGTH_A_RUN_SPECS`` mapping below, never a
caller-supplied value, (d) the read-only ``REFERENCE_RUN_ID`` carve-out
described above, (e) this campaign's W&B launch contract is strict: identical
to Embedding-Dropout-A's -- ``--wandb-policy-path`` defaults to the reviewed
offline-enabled policy (``config/stage1_wandb_tracking_policy_offline_v001.yaml``),
and ``run_pilot`` is always called with ``require_tracking=True`` UNLESS the
caller passes the explicit ``--waive-tracking-requirement`` flag, and (f)
run_id reservation is delegated to the new, minimal, generic
``src.baseline.campaign_spec.CampaignSpec``/``src.baseline.campaign_registry``
machinery (built alongside this campaign) instead of a hand-maintained
``_OTHER_CAMPAIGN_RESERVED_RUN_IDS`` literal: constructing
``_SEQ_LENGTH_A_SPEC`` below reserves this campaign's four run_ids against
every historical campaign's run_ids (LR-A, Hidden-size-A, cap50k,
Embedding-Dropout-A, and the committed six-run matrix itself -- see
``campaign_registry.HISTORICAL_RESERVED_RUN_ID_GROUPS``) and raises loudly
on any collision, at import time, before any config is ever generated.

Contains no modeling, screening, stopping, or tracking logic of its own --
every decision is made by the pilot subsystems ``run_pilot``/
``prepare_pilot_run_only``/``compute_pilot_status_fields`` compose.

Does not submit any Slurm job itself. Does not itself perform or require a
paired ``.sbatch`` launcher within this task -- see the module docstring's
sibling scripts for that pattern if/when one is added. Neither this script
nor any Slurm submission is ever executed against real Moriah/NH/GPU within
this task.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
from pathlib import Path

# Direct execution (`python scripts/run_stage1_seq_length_range_seedA_closure.py ...`)
# puts scripts/ -- not the repository work directory -- at the front of
# sys.path, so the sibling top-level package `src` is otherwise unimportable
# regardless of the caller's current working directory. Insert the repo work
# directory (this file's parent's parent) before importing src.baseline.*.
_REPO_WORKDIR = Path(__file__).resolve().parent.parent
if str(_REPO_WORKDIR) not in sys.path:
    sys.path.insert(0, str(_REPO_WORKDIR))

from src.baseline.campaign_spec import CampaignSpec
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

# This campaign's strict launch contract (see module docstring, point (e)):
# offline-enabled and reviewed, never the disabled-by-default policy other
# closure scripts (e.g. LR-A) default to.
_DEFAULT_WANDB_POLICY_PATH = _REPO_WORKDIR / "config" / "stage1_wandb_tracking_policy_offline_v001.yaml"

_SEEDA = 967139
_CAP25K = 25_000
_LR3EM4 = 3e-4
_H128 = 128

# Sequence-Length-A's frozen, non-overridable horizon: one uninterrupted
# epoch 1->6 training segment, matching every other Phase-A range-
# characterization campaign (LR-A, Hidden-size-A, Embedding-Dropout-A). No
# CLI flag, no env var can override this.
SEQ_LENGTH_A_MAX_TARGET_EPOCH = 6

# The historical Hidden-size-A H=128 run -- preserved read-only-only per the
# same precedent Embedding-Dropout-A already established for this exact
# run_id (see module docstring). Deliberately NOT a key of
# SEQ_LENGTH_A_RUN_SPECS (never trainable/configurable through this script);
# reachable only via --status-only.
REFERENCE_RUN_ID = "emb128x32_seedA_h128_lr3em4_cap25k_cal"

# Exactly the four NEW, approved Sequence-Length-A candidates -- the entire
# closed {12, 24, 48, 72} seq_length set (see nh_config_generation.
# validate_seq_length), each reusing the same already-registered
# nh_config_generation run profile ("pilot_lead06_emb128x32_seedA_v001")
# unmodified, differing from each other ONLY in seq_length, with hidden_size
# and learning_rate held fixed at the Hidden-size-A/LR-A provisional anchors
# (128, 3e-4), and embedding_dropout deliberately left unset (profile
# default, the Embedding-Dropout-A-characterized 0.10 provisional anchor).
# Neither this mapping's keys nor any entry's seq_length/hidden_size/
# learning_rate may be changed without updating this literal mapping --
# there is deliberately no generic "any seq_length value" composition helper
# or CLI override here (see module docstring).
#
# Constructing this CampaignSpec immediately reserves its four run_ids in
# campaign_registry (raising campaign_registry.CampaignRegistryError loudly
# on any collision against every historical or other prospective campaign's
# run_ids) -- see module docstring point (f).
_SEQ_LENGTH_A_SPEC = CampaignSpec(
    name="Sequence-Length-A",
    version="v001",
    varied_axis="seq_length",
    candidates={
        "emb128x32_seedA_seq12_h128_lr3em4_cap25k_cal": PilotRunSpec(
            run_id="emb128x32_seedA_seq12_h128_lr3em4_cap25k_cal",
            static_pathway="learned_fc_embedding",
            embedding_hiddens=[128, 32],
            seed_name="seed_a",
            seed=_SEEDA,
            run_profile_name="pilot_lead06_emb128x32_seedA_v001",
            max_updates_per_epoch=_CAP25K,
            learning_rate=_LR3EM4,
            hidden_size=_H128,
            seq_length=12,
        ),
        "emb128x32_seedA_seq24_h128_lr3em4_cap25k_cal": PilotRunSpec(
            run_id="emb128x32_seedA_seq24_h128_lr3em4_cap25k_cal",
            static_pathway="learned_fc_embedding",
            embedding_hiddens=[128, 32],
            seed_name="seed_a",
            seed=_SEEDA,
            run_profile_name="pilot_lead06_emb128x32_seedA_v001",
            max_updates_per_epoch=_CAP25K,
            learning_rate=_LR3EM4,
            hidden_size=_H128,
            seq_length=24,
        ),
        "emb128x32_seedA_seq48_h128_lr3em4_cap25k_cal": PilotRunSpec(
            run_id="emb128x32_seedA_seq48_h128_lr3em4_cap25k_cal",
            static_pathway="learned_fc_embedding",
            embedding_hiddens=[128, 32],
            seed_name="seed_a",
            seed=_SEEDA,
            run_profile_name="pilot_lead06_emb128x32_seedA_v001",
            max_updates_per_epoch=_CAP25K,
            learning_rate=_LR3EM4,
            hidden_size=_H128,
            seq_length=48,
        ),
        "emb128x32_seedA_seq72_h128_lr3em4_cap25k_cal": PilotRunSpec(
            run_id="emb128x32_seedA_seq72_h128_lr3em4_cap25k_cal",
            static_pathway="learned_fc_embedding",
            embedding_hiddens=[128, 32],
            seed_name="seed_a",
            seed=_SEEDA,
            run_profile_name="pilot_lead06_emb128x32_seedA_v001",
            max_updates_per_epoch=_CAP25K,
            learning_rate=_LR3EM4,
            hidden_size=_H128,
            seq_length=72,
        ),
    },
    max_target_epoch=SEQ_LENGTH_A_MAX_TARGET_EPOCH,
    require_tracking=True,
    wandb_policy_path=str(_DEFAULT_WANDB_POLICY_PATH),
    comparator_run_ids=(REFERENCE_RUN_ID,),
)

SEQ_LENGTH_A_RUN_SPECS = _SEQ_LENGTH_A_SPEC.candidates

# Frozen campaign name.
_SEQ_LENGTH_A_POLICY_NAME = "seq_length_range_seedA_25k_v001"


def _resolve_policy_relative_paths(pilot_policy):
    """Identical to run_stage1_embedding_dropout_range_seedA_closure.py's
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


def _build_seq_length_a_policy(base_pilot_policy):
    """Splice the four Sequence-Length-A-only PilotRunSpec entries into a
    copy of the real, already-validated PilotPolicy. Never touches runs for
    any of the six real run_ids or the historical Hidden-size-A reference
    run_id, and never writes anything back to the committed YAML -- this
    augmented object exists only in this process's memory for the lifetime
    of this one invocation. Run-id collision against every historical/
    prospective campaign is already enforced once, earlier and more
    completely, by _SEQ_LENGTH_A_SPEC's construction (via campaign_registry)
    -- this function's own checks are narrow defense-in-depth against the
    real committed policy's runs mapping specifically."""
    augmented_runs = dict(base_pilot_policy.runs)
    if REFERENCE_RUN_ID in augmented_runs:
        raise RuntimeError(
            f"read-only reference run_id {REFERENCE_RUN_ID!r} unexpectedly collides with an "
            "existing real pilot run_id in the committed policy -- refusing to proceed"
        )
    for run_id, spec in SEQ_LENGTH_A_RUN_SPECS.items():
        if run_id in augmented_runs:
            raise RuntimeError(
                f"Sequence-Length-A run_id {run_id!r} unexpectedly collides with an existing "
                "real pilot run_id in the committed policy -- refusing to overwrite it"
            )
        augmented_runs[run_id] = spec
    augmented_raw = dict(base_pilot_policy.raw)
    augmented_raw["policy_name"] = _SEQ_LENGTH_A_POLICY_NAME
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
    Sequence-Length-A run_ids or for REFERENCE_RUN_ID (the historical
    Hidden-size-A comparator) -- the experiment_name naming convention
    depends only on run_id, never on whether it is a key of
    SEQ_LENGTH_A_RUN_SPECS, so this is the one supported way to inspect the
    historical comparator's on-disk state through this script without ever
    being able to train/reconfigure it. For a brand-new run_id with no NH
    run directory yet, reports a distinct 'NO_EXISTING_NH_RUN_DIRECTORY'
    status rather than raising."""
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
        "--run-id", required=True, choices=sorted(SEQ_LENGTH_A_RUN_SPECS) + [REFERENCE_RUN_ID],
        help="One of the four approved Sequence-Length-A run_ids, or the historical "
        f"Hidden-size-A comparator run_id ({REFERENCE_RUN_ID!r}, --status-only only -- never "
        "trainable/configurable here).",
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
    parser.add_argument(
        "--wandb-policy-path", type=Path, default=_DEFAULT_WANDB_POLICY_PATH,
        help="Defaults to this campaign's reviewed offline-enabled policy (see module "
        "docstring point (e)) -- unlike LR-A's closure script, this is never None.",
    )
    parser.add_argument("--tracking-generation", default="g1")
    parser.add_argument(
        "--waive-tracking-requirement", action="store_true",
        help="Explicit human waiver: pass run_pilot(require_tracking=False) instead of this "
        "campaign's default strict require_tracking=True, allowing a real training launch to "
        "continue untracked if W&B tracking init fails or resolves to backend null/no run id. "
        "Never pass this without deliberate, reviewed justification -- see module docstring "
        "point (e) and pilot_tracking.init_pilot_tracking_run's require_tracking parameter.",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Generate this run_id's NH config + generation manifest and exit before any NH "
        "training call or W&B backend initialization. Refuses (via prepare_pilot_run_only) if "
        "this run_id already has a real NH run directory or evidence bundle. Never valid for "
        f"{REFERENCE_RUN_ID!r} (the historical Hidden-size-A comparator).",
    )
    parser.add_argument(
        "--status-only", action="store_true",
        help="Read-only: report this run_id's on-disk continuation status (highest physical "
        "checkpoint epoch, highest screened epoch, next intended screening epoch, overshoot, "
        f"safe_to_continue_automatically) and exit. The only mode valid for {REFERENCE_RUN_ID!r} "
        "(the historical Hidden-size-A comparator). Never generates a config, never calls NH, "
        "never writes anything. Mutually exclusive with --prepare-only.",
    )
    args = parser.parse_args()

    if args.run_id not in SEQ_LENGTH_A_RUN_SPECS and args.run_id != REFERENCE_RUN_ID:
        parser.error(
            f"--run-id {args.run_id!r} is not one of the Sequence-Length-A run_ids "
            f"{sorted(SEQ_LENGTH_A_RUN_SPECS)} or the historical comparator {REFERENCE_RUN_ID!r}"
        )
    if args.prepare_only and args.status_only:
        parser.error("--prepare-only and --status-only are mutually exclusive")
    if args.run_id == REFERENCE_RUN_ID and not args.status_only:
        parser.error(
            f"--run-id {REFERENCE_RUN_ID!r} (the historical Hidden-size-A comparator) is only "
            "ever valid with --status-only -- it is never trainable or configurable through "
            "this script"
        )

    # Load through the REAL, unmodified validator against the REAL,
    # unmodified committed YAML -- this is the only source of every shared
    # field (lead_hours, seeds, early_stopping cadence, screening subset
    # identity/hash, embedding_activation, output_dropout, etc.).
    # wandb_policy_path is overridden below to this campaign's strict
    # default (or an explicit caller override).
    real_pilot_policy = load_pilot_policy(args.pilot_policy_path)
    pilot_policy = _build_seq_length_a_policy(real_pilot_policy)
    pilot_policy = _resolve_policy_relative_paths(pilot_policy)

    wandb_policy_override_path = str(args.wandb_policy_path)
    load_tracking_policy(wandb_policy_override_path)
    pilot_policy = dataclasses.replace(pilot_policy, wandb_policy_path=wandb_policy_override_path)

    require_tracking = not args.waive_tracking_requirement

    if args.status_only:
        result = _run_status_only(pilot_policy, args.run_id, args.config_out_dir)
        result["seq_length_a_max_target_epoch"] = SEQ_LENGTH_A_MAX_TARGET_EPOCH
        result["is_historical_hidden_size_a_comparator"] = args.run_id == REFERENCE_RUN_ID
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
            commands_used=[
                "python scripts/run_stage1_seq_length_range_seedA_closure.py "
                + " ".join(sys.argv[1:])
            ],
            force=args.force,
        )
        result["seq_length_a_max_target_epoch"] = SEQ_LENGTH_A_MAX_TARGET_EPOCH
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
        commands_used=[
            "python scripts/run_stage1_seq_length_range_seedA_closure.py "
            + " ".join(sys.argv[1:])
        ],
        force=args.force,
        tracking_generation=args.tracking_generation,
        max_target_epoch=SEQ_LENGTH_A_MAX_TARGET_EPOCH,
        require_tracking=require_tracking,
    )

    printable = dict(result)
    printable["nh_run_dir"] = str(printable["nh_run_dir"])
    printable["evidence_bundle_path"] = str(printable["evidence_bundle_path"])
    printable["seq_length_a_max_target_epoch"] = SEQ_LENGTH_A_MAX_TARGET_EPOCH
    printable["require_tracking_used"] = require_tracking
    print(json.dumps(printable, indent=2))

    if result["final_status"] == "blocked_continuation_overshoot_conflict":
        sys.exit(1)


if __name__ == "__main__":
    main()
