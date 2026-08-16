"""Thin CLI entrypoint for the Dynamic-Input-Family-A dynamic-input-family
range-characterization campaign (``dynamic_input_family_seedA_25k_v001``):
exactly four NEW trainable candidates, all ``[128, 32]`` learned-FC-embedding,
Seed A (967139), ``hidden_size=128``, ``learning_rate=3e-4`` (the LR-A-
characterized provisional anchor, held fixed), ``seq_length=72`` (the
Sequence-Length-A-closed working context, held fixed),
``max_updates_per_epoch=25000``, one uninterrupted epoch 1->6 training
segment, ``statics_embedding.dropout`` left at the profile default (the
Embedding-Dropout-A-characterized 0.10 provisional anchor, held fixed, never
explicitly overridden here) --

    emb128x32_seedA_dynP_seq72_h128_lr3em4_cap25k_cal      P:    mrms_qpe_1h_mm
    emb128x32_seedA_dynPT_seq72_h128_lr3em4_cap25k_cal     PT:   + rtma_2t_K
    emb128x32_seedA_dynPTM_seq72_h128_lr3em4_cap25k_cal    PTM:  + rtma_2sh_kgkg
    emb128x32_seedA_dynPTMW_seq72_h128_lr3em4_cap25k_cal   PTMW: + rtma_10u_ms, rtma_10v_ms

-- differing from each other ONLY in ``dynamic_inputs`` (see
``pilot_lead06_config.PilotRunSpec.dynamic_inputs`` and
``nh_config_generation.validate_dynamic_inputs_override``). This is Phase-A
range characterization of the dynamic-input feature set at a fixed
LR/hidden-size/embedding-dropout/seq_length anchor, NOT final dynamic-input
selection and NOT joint HPO across any other axis (that interaction is
deliberately out of scope). Embedding shape ``[128, 32]``, embedding
activation ``tanh``, output dropout, and every other frozen scientific
setting are unchanged from the committed six-run pilot matrix. See
``docs/decision_log.md``'s 2026-08-16 Dynamic-Input-Family-A design-freeze
entry for the full scientific rationale (gap-flag audit, physical-variable
audit, moisture decision, wind decision) -- none of that rationale is
repeated here.

All four families are fresh trainable candidates of this campaign; unlike
Sequence-Length-A/Embedding-Dropout-A/Hidden-size-A, this campaign does not
reuse any single historical run as one of its four data points and therefore
declares no read-only ``REFERENCE_RUN_ID`` comparator -- ``P`` (the
precipitation-only family) is this campaign's own in-campaign reference
family (see the rescue-policy wording in docs/decision_log.md: "P is
reference and is never 'rescued'"), not an external comparator reachable
only via ``--status-only``.

Every family's ``dynamic_inputs`` literal below is a strict, order-preserving
prefix/subset of the certified package's binding v001-core 8-variable list
(``config/stage1_scientific_baseline_v001.yaml``), and excludes both gap-QC
variables (``mrms_qpe_1h_mm_gap``, ``rtma_gap``) and the dewpoint variable
(``rtma_2d_K``) from all four Dynamic-Input-Family-A base definitions -- not
a permanent or global prohibition on these variables. The two gap-QC
variables are excluded only under the current hard history-gap exclusion
policy (they remain in the package for QC/provenance, and a future
gap-handling policy could reopen their modeling role); the dewpoint variable
is excluded only because this campaign chose ``rtma_2sh_kgkg`` as its primary
single moisture representation (dewpoint remains eligible for a later
targeted ablation given future evidence). Enforced defensively at this
module's import time by ``_assert_no_forbidden_dynamic_inputs`` below, in
addition to (not instead of) ``validate_dynamic_inputs_override``'s own
package-schema-membership check (see that function's docstring:
campaign-definition-layer rejection of these variables from this campaign's
base definitions is this campaign's own scientific choice, not a global
config-generation policy).

Structurally a close mirror of
``scripts/run_stage1_seq_length_range_seedA_closure.py`` -- same
splice-onto-the-real-validated-PilotPolicy mechanism (via
``load_pilot_policy()`` against the committed six-run YAML, so every shared
field still comes verbatim from that one committed source), same
``--prepare-only``/``--status-only`` modes, same CLI/call structure. The
differences specific to Dynamic-Input-Family-A: (a) four run_ids instead of
four (same count, different axis), each carrying an explicit
``dynamic_inputs`` override (see ``pilot_lead06_config.PilotRunSpec.
dynamic_inputs`` and ``nh_config_generation.validate_dynamic_inputs_
override``) AND explicit ``hidden_size=128``/``learning_rate=3e-4``/
``seq_length=72`` overrides (all three held fixed across all four, at the
Hidden-size-A/LR-A/Sequence-Length-A provisional anchors) rather than varying
any of them, with ``embedding_dropout`` deliberately left unset (profile
default, the Embedding-Dropout-A-characterized 0.10 provisional anchor)
rather than overridden, (b) training is always bounded at
``DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH`` (6, matching every other Phase-A
range-characterization campaign's horizon) -- this campaign's approved,
frozen horizon -- with no CLI flag or env var able to override it, (c) there
is deliberately no ``--dynamic-inputs``/``--hidden-size``/``--learning-rate``/
``--seq-length`` CLI flag either: every candidate's dynamic_inputs, hidden
size, learning rate, and seq_length are fixed by the literal
``DYNAMIC_INPUT_FAMILY_A_RUN_SPECS`` mapping below, never a caller-supplied
value, (d) no read-only comparator carve-out (see above -- unlike Sequence-
Length-A's ``REFERENCE_RUN_ID``, this campaign declares none), (e) this
campaign's W&B launch contract is strict: identical to Sequence-Length-A's --
``--wandb-policy-path`` defaults to the reviewed offline-enabled policy
(``config/stage1_wandb_tracking_policy_offline_v001.yaml``), and
``run_pilot`` is always called with ``require_tracking=True`` UNLESS the
caller passes the explicit ``--waive-tracking-requirement`` flag, and (f)
run_id reservation is delegated to the same generic
``src.baseline.campaign_spec.CampaignSpec``/``src.baseline.campaign_registry``
machinery Sequence-Length-A uses: constructing
``_DYNAMIC_INPUT_FAMILY_A_SPEC`` below reserves this campaign's four run_ids
against every historical campaign's run_ids (LR-A, Hidden-size-A, cap50k,
Embedding-Dropout-A, and the committed six-run matrix itself -- see
``campaign_registry.HISTORICAL_RESERVED_RUN_ID_GROUPS``) and raises loudly on
any collision, at import time, before any config is ever generated. Note
this registry does not include still-open prospective campaigns from a
different process (e.g. Sequence-Length-A's own run_ids, reserved only
within that script's own process) -- this campaign's ``dyn*``-token run_ids
do not lexically overlap any historical or Sequence-Length-A ``seq*``-token
run_id, so this is not a real collision risk in practice.

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

# Direct execution (`python scripts/run_stage1_dynamic_input_family_seedA_closure.py ...`)
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
_SEQ72 = 72

# Dynamic-Input-Family-A's frozen, non-overridable horizon: one uninterrupted
# epoch 1->6 training segment, matching every other Phase-A range-
# characterization campaign (LR-A, Hidden-size-A, Embedding-Dropout-A,
# Sequence-Length-A). No CLI flag, no env var can override this.
DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH = 6

# Forbidden in Dynamic-Input-Family-A base definitions only -- not a
# permanent or global prohibition (see docs/decision_log.md's 2026-08-16
# design-freeze entry). The two gap-QC variables are excluded under the
# current hard history-gap exclusion policy (retained in the package for
# QC/provenance; a future gap-handling policy could reopen their modeling
# role). The dewpoint variable is excluded because this campaign chose
# rtma_2sh_kgkg as its primary single moisture representation (dewpoint
# remains eligible for a later targeted ablation given future evidence).
# Used only by _assert_no_forbidden_dynamic_inputs below, as campaign-
# definition-layer defense-in-depth on top of
# validate_dynamic_inputs_override's package-schema-membership check (see
# that function's own docstring).
_FORBIDDEN_DYNAMIC_INPUTS = frozenset({"mrms_qpe_1h_mm_gap", "rtma_gap", "rtma_2d_K"})

# The frozen P/PT/PTM/PTMW family matrix (docs/decision_log.md, 2026-08-16):
# each family is the exact, order-preserving prefix of the one before it in
# this physically-nested hierarchy (precip -> +temp -> +moisture -> +winds).
_FAMILY_P = ("mrms_qpe_1h_mm",)
_FAMILY_PT = _FAMILY_P + ("rtma_2t_K",)
_FAMILY_PTM = _FAMILY_PT + ("rtma_2sh_kgkg",)
_FAMILY_PTMW = _FAMILY_PTM + ("rtma_10u_ms", "rtma_10v_ms")


def _assert_no_forbidden_dynamic_inputs(family_label: str, dynamic_inputs: "tuple[str, ...]") -> None:
    """Campaign-definition-layer defense-in-depth (see module docstring and
    ``nh_config_generation.validate_dynamic_inputs_override``'s own
    docstring): raise loudly, at import time, if any Dynamic-Input-Family-A
    family literal ever accidentally includes a variable forbidden in
    Dynamic-Input-Family-A base definitions -- a gap-QC variable (excluded
    only under the current hard history-gap exclusion policy) or the
    dewpoint variable (excluded only because this campaign chose
    ``rtma_2sh_kgkg`` as its primary moisture representation; dewpoint
    remains eligible for a later targeted ablation). Neither exclusion is
    permanent or global: validate_dynamic_inputs_override itself
    deliberately does not reject these -- rejecting them from this
    campaign's base definitions is this campaign's own scientific choice,
    enforced here."""
    forbidden_present = _FORBIDDEN_DYNAMIC_INPUTS & set(dynamic_inputs)
    if forbidden_present:
        raise RuntimeError(
            f"Dynamic-Input-Family-A family {family_label!r} unexpectedly includes "
            f"variable(s) {sorted(forbidden_present)} that are forbidden in "
            "Dynamic-Input-Family-A base definitions (gap-QC variables are excluded under "
            "the current hard history-gap exclusion policy; the dewpoint variable is "
            "excluded because this campaign chose rtma_2sh_kgkg as its primary moisture "
            "representation -- neither exclusion is permanent) -- refusing to proceed"
        )


for _family_label, _family_vars in (
    ("P", _FAMILY_P),
    ("PT", _FAMILY_PT),
    ("PTM", _FAMILY_PTM),
    ("PTMW", _FAMILY_PTMW),
):
    _assert_no_forbidden_dynamic_inputs(_family_label, _family_vars)
del _family_label, _family_vars

# Exactly the four approved Dynamic-Input-Family-A candidates -- the entire
# frozen P/PT/PTM/PTMW family matrix (see docs/decision_log.md), each reusing
# the same already-registered nh_config_generation run profile
# ("pilot_lead06_emb128x32_seedA_v001") unmodified, differing from each other
# ONLY in dynamic_inputs, with hidden_size/learning_rate/seq_length held
# fixed at the Hidden-size-A/LR-A/Sequence-Length-A provisional anchors (128,
# 3e-4, 72), and embedding_dropout deliberately left unset (profile default,
# the Embedding-Dropout-A-characterized 0.10 provisional anchor). Neither
# this mapping's keys nor any entry's dynamic_inputs/hidden_size/
# learning_rate/seq_length may be changed without updating this literal
# mapping -- there is deliberately no generic "any family" composition helper
# or CLI override here (see module docstring).
#
# Constructing this CampaignSpec immediately reserves its four run_ids in
# campaign_registry (raising campaign_registry.CampaignRegistryError loudly
# on any collision against every historical or other prospective campaign's
# run_ids) -- see module docstring point (f).
_DYNAMIC_INPUT_FAMILY_A_SPEC = CampaignSpec(
    name="Dynamic-Input-Family-A",
    version="v001",
    varied_axis="dynamic_inputs",
    candidates={
        "emb128x32_seedA_dynP_seq72_h128_lr3em4_cap25k_cal": PilotRunSpec(
            run_id="emb128x32_seedA_dynP_seq72_h128_lr3em4_cap25k_cal",
            static_pathway="learned_fc_embedding",
            embedding_hiddens=[128, 32],
            seed_name="seed_a",
            seed=_SEEDA,
            run_profile_name="pilot_lead06_emb128x32_seedA_v001",
            max_updates_per_epoch=_CAP25K,
            learning_rate=_LR3EM4,
            hidden_size=_H128,
            seq_length=_SEQ72,
            dynamic_inputs=_FAMILY_P,
        ),
        "emb128x32_seedA_dynPT_seq72_h128_lr3em4_cap25k_cal": PilotRunSpec(
            run_id="emb128x32_seedA_dynPT_seq72_h128_lr3em4_cap25k_cal",
            static_pathway="learned_fc_embedding",
            embedding_hiddens=[128, 32],
            seed_name="seed_a",
            seed=_SEEDA,
            run_profile_name="pilot_lead06_emb128x32_seedA_v001",
            max_updates_per_epoch=_CAP25K,
            learning_rate=_LR3EM4,
            hidden_size=_H128,
            seq_length=_SEQ72,
            dynamic_inputs=_FAMILY_PT,
        ),
        "emb128x32_seedA_dynPTM_seq72_h128_lr3em4_cap25k_cal": PilotRunSpec(
            run_id="emb128x32_seedA_dynPTM_seq72_h128_lr3em4_cap25k_cal",
            static_pathway="learned_fc_embedding",
            embedding_hiddens=[128, 32],
            seed_name="seed_a",
            seed=_SEEDA,
            run_profile_name="pilot_lead06_emb128x32_seedA_v001",
            max_updates_per_epoch=_CAP25K,
            learning_rate=_LR3EM4,
            hidden_size=_H128,
            seq_length=_SEQ72,
            dynamic_inputs=_FAMILY_PTM,
        ),
        "emb128x32_seedA_dynPTMW_seq72_h128_lr3em4_cap25k_cal": PilotRunSpec(
            run_id="emb128x32_seedA_dynPTMW_seq72_h128_lr3em4_cap25k_cal",
            static_pathway="learned_fc_embedding",
            embedding_hiddens=[128, 32],
            seed_name="seed_a",
            seed=_SEEDA,
            run_profile_name="pilot_lead06_emb128x32_seedA_v001",
            max_updates_per_epoch=_CAP25K,
            learning_rate=_LR3EM4,
            hidden_size=_H128,
            seq_length=_SEQ72,
            dynamic_inputs=_FAMILY_PTMW,
        ),
    },
    max_target_epoch=DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH,
    require_tracking=True,
    wandb_policy_path=str(_DEFAULT_WANDB_POLICY_PATH),
)

DYNAMIC_INPUT_FAMILY_A_RUN_SPECS = _DYNAMIC_INPUT_FAMILY_A_SPEC.candidates

# Frozen campaign name.
_DYNAMIC_INPUT_FAMILY_A_POLICY_NAME = "dynamic_input_family_seedA_25k_v001"


def _resolve_policy_relative_paths(pilot_policy):
    """Identical to run_stage1_seq_length_range_seedA_closure.py's helper of
    the same name -- the pilot policy YAML declares its own composed-artifact
    paths relative to the repository work directory; absolute-ize them so
    this script behaves identically regardless of the caller's current
    working directory."""

    def _abs(raw: str) -> str:
        p = Path(raw)
        return str(p) if p.is_absolute() else str(_REPO_WORKDIR / p)

    return dataclasses.replace(
        pilot_policy,
        screening_basin_ids_path=_abs(pilot_policy.screening_basin_ids_path),
        base_early_stopping_policy_path=_abs(pilot_policy.base_early_stopping_policy_path),
        wandb_policy_path=_abs(pilot_policy.wandb_policy_path),
    )


def _build_dynamic_input_family_a_policy(base_pilot_policy):
    """Splice the four Dynamic-Input-Family-A-only PilotRunSpec entries into
    a copy of the real, already-validated PilotPolicy. Never touches runs for
    any of the six real run_ids, and never writes anything back to the
    committed YAML -- this augmented object exists only in this process's
    memory for the lifetime of this one invocation. Run-id collision against
    every historical/prospective campaign is already enforced once, earlier
    and more completely, by _DYNAMIC_INPUT_FAMILY_A_SPEC's construction (via
    campaign_registry) -- this function's own checks are narrow defense-in-
    depth against the real committed policy's runs mapping specifically."""
    augmented_runs = dict(base_pilot_policy.runs)
    for run_id, spec in DYNAMIC_INPUT_FAMILY_A_RUN_SPECS.items():
        if run_id in augmented_runs:
            raise RuntimeError(
                f"Dynamic-Input-Family-A run_id {run_id!r} unexpectedly collides with an "
                "existing real pilot run_id in the committed policy -- refusing to overwrite it"
            )
        augmented_runs[run_id] = spec
    augmented_raw = dict(base_pilot_policy.raw)
    augmented_raw["policy_name"] = _DYNAMIC_INPUT_FAMILY_A_POLICY_NAME
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
    exists on disk for this run_id. For a brand-new run_id with no NH run
    directory yet, reports a distinct 'NO_EXISTING_NH_RUN_DIRECTORY' status
    rather than raising."""
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
        "--run-id", required=True, choices=sorted(DYNAMIC_INPUT_FAMILY_A_RUN_SPECS),
        help="One of the four approved Dynamic-Input-Family-A run_ids (P/PT/PTM/PTMW).",
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
        "this run_id already has a real NH run directory or evidence bundle.",
    )
    parser.add_argument(
        "--status-only", action="store_true",
        help="Read-only: report this run_id's on-disk continuation status (highest physical "
        "checkpoint epoch, highest screened epoch, next intended screening epoch, overshoot, "
        "safe_to_continue_automatically) and exit. Never generates a config, never calls NH, "
        "never writes anything. Mutually exclusive with --prepare-only.",
    )
    args = parser.parse_args()

    if args.run_id not in DYNAMIC_INPUT_FAMILY_A_RUN_SPECS:
        parser.error(
            f"--run-id {args.run_id!r} is not one of the Dynamic-Input-Family-A run_ids "
            f"{sorted(DYNAMIC_INPUT_FAMILY_A_RUN_SPECS)}"
        )
    if args.prepare_only and args.status_only:
        parser.error("--prepare-only and --status-only are mutually exclusive")

    # Load through the REAL, unmodified validator against the REAL,
    # unmodified committed YAML -- this is the only source of every shared
    # field (lead_hours, seeds, early_stopping cadence, screening subset
    # identity/hash, embedding_activation, output_dropout, etc.).
    # wandb_policy_path is overridden below to this campaign's strict
    # default (or an explicit caller override).
    real_pilot_policy = load_pilot_policy(args.pilot_policy_path)
    pilot_policy = _build_dynamic_input_family_a_policy(real_pilot_policy)
    pilot_policy = _resolve_policy_relative_paths(pilot_policy)

    wandb_policy_override_path = str(args.wandb_policy_path)
    load_tracking_policy(wandb_policy_override_path)
    pilot_policy = dataclasses.replace(pilot_policy, wandb_policy_path=wandb_policy_override_path)

    require_tracking = not args.waive_tracking_requirement

    if args.status_only:
        result = _run_status_only(pilot_policy, args.run_id, args.config_out_dir)
        result["dynamic_input_family_a_max_target_epoch"] = DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH
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
                "python scripts/run_stage1_dynamic_input_family_seedA_closure.py "
                + " ".join(sys.argv[1:])
            ],
            force=args.force,
        )
        result["dynamic_input_family_a_max_target_epoch"] = DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH
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
            "python scripts/run_stage1_dynamic_input_family_seedA_closure.py "
            + " ".join(sys.argv[1:])
        ],
        force=args.force,
        tracking_generation=args.tracking_generation,
        max_target_epoch=DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH,
        require_tracking=require_tracking,
    )

    printable = dict(result)
    printable["nh_run_dir"] = str(printable["nh_run_dir"])
    printable["evidence_bundle_path"] = str(printable["evidence_bundle_path"])
    printable["dynamic_input_family_a_max_target_epoch"] = DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH
    printable["require_tracking_used"] = require_tracking
    print(json.dumps(printable, indent=2))

    if result["final_status"] == "blocked_continuation_overshoot_conflict":
        sys.exit(1)


if __name__ == "__main__":
    main()
