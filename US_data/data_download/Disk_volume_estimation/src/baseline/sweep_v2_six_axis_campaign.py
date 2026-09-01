"""Additive six-axis (v2) campaign identity/canonicalization contract.

This module is a strictly additive sibling of :mod:`sweep_v1_campaign`. It
never imports anything mutable from that module and never modifies it -- it
only *reads* a handful of its frozen constants/functions (``MODEL_SEED_A``,
``CAMPAIGN_ID``, ``DOMAIN_VERSION``, ``SEARCH_DOMAIN``,
``FROZEN_FIXED_CONFIGURATION``, ``canonical_hyperparameters``,
``trial_identity_conflicts``, ``_HYPERPARAMETER_FIELDS``) so v1's five-axis
identity/hash math is reused verbatim rather than re-implemented, and any
future drift in the frozen five-axis field set is caught by
``_ASSERT_V1_AXES_UNCHANGED`` below rather than silently diverging.

Adds a sixth axis, ``seq_length``, represented on the W&B controller
boundary as ``q_uniform(min=48, max=120, q=12)`` per explicit user mandate
(superseding the categorical recommendation in the prior design gate's
Section E). Because ``q_uniform`` returns a float, every value crossing the
controller boundary MUST pass through :func:`normalize_seq_length_axis`
before it is used for canonicalization, hashing, identity construction,
durable intake, generated-config construction, or output-root construction.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from . import sweep_v1_campaign as sweep

__all__ = [
    "SweepV2CampaignError",
    "CAMPAIGN_ID_V2", "DOMAIN_VERSION_V2", "CONFIGURATION_CANONICALIZATION_VERSION_V2",
    "OBJECTIVE_ID_V2", "FIDELITY_ID_V2",
    "SEQ_LENGTH_MIN", "SEQ_LENGTH_MAX", "SEQ_LENGTH_STEP", "SEQ_LENGTH_DOMAIN_V2",
    "FORBIDDEN_V1_SWEEP_ID", "CLOSED_DISPOSABLE_REHEARSAL_SWEEP_ID",
    "FORBIDDEN_PRODUCTION_SWEEP_IDS", "SEARCH_ARMS_V2",
    "FROZEN_FIXED_CONFIGURATION_V2", "SEARCH_DOMAIN_V2",
    "TRIAL_SUMMARY_FIELDS_V2", "PROPOSAL_RECORD_FIELDS_V2",
    "normalize_seq_length_axis", "canonical_hyperparameters_v2", "configuration_id_v2",
    "proposal_id_v2", "trial_id_v2", "trial_identity_conflicts_v2",
    "validate_v2_proposal_shape", "assert_no_v1_contamination", "validate_review_record_v2",
]


class SweepV2CampaignError(ValueError):
    """Raised for any v2 six-axis identity/canonicalization/contamination violation."""


# ---------------------------------------------------------------------------
# v2 identity constants -- distinct from every v1 identifier by construction.
# ---------------------------------------------------------------------------
CAMPAIGN_ID_V2 = "stage1_phase_b_sweep_v2_six_axis_common120_v001"
DOMAIN_VERSION_V2 = "six_axis_q12_48_120_v001"
CONFIGURATION_CANONICALIZATION_VERSION_V2 = "sweep_v2_six_axis_canonical_json_v001"
OBJECTIVE_ID_V2 = "common120_raw_space_nse_v001"
# Fidelity contract (epochs/update-cap/seed) is scientifically unchanged
# between v1 and v2 -- only the search axes and objective/support contract
# differ -- so the fidelity tag is intentionally the same literal, and only
# ever appears embedded inside v2-namespaced identifiers (never bare).
FIDELITY_ID_V2 = "mf12x50000"

SEQ_LENGTH_MIN = 48
SEQ_LENGTH_MAX = 120
SEQ_LENGTH_STEP = 12
SEQ_LENGTH_DOMAIN_V2 = tuple(range(SEQ_LENGTH_MIN, SEQ_LENGTH_MAX + 1, SEQ_LENGTH_STEP))
assert SEQ_LENGTH_DOMAIN_V2 == (48, 60, 72, 84, 96, 108, 120)

FORBIDDEN_V1_SWEEP_ID = "4x3btz2s"

# The CLOSED disposable v2 rehearsal controller. It was registered once for an
# operational rehearsal, consumed its single authorized disposable proposal,
# and is not reusable for any production path. Only historical
# ``mode=rehearsal`` launch manifests may still legitimately name it.
CLOSED_DISPOSABLE_REHEARSAL_SWEEP_ID = "oz5p4csb"

# The single authoritative set of W&B sweep ids that every v2 PRODUCTION path
# must refuse: the frozen v1 production sweep and the closed disposable
# rehearsal sweep. Production controller registration, production manifest
# construction, production one-agent invocation construction, the strict
# ``mode=production`` manifest loader, the production bridge, and the
# production launcher all consult this set (the launcher mirrors the literals
# at shell level for pre-contact refusal, guarded against drift by a test).
FORBIDDEN_PRODUCTION_SWEEP_IDS = frozenset({FORBIDDEN_V1_SWEEP_ID, CLOSED_DISPOSABLE_REHEARSAL_SWEEP_ID})

# Mirrors v1's ``sweep_v1_campaign.SEARCH_ARMS`` exactly: the live W&B
# Bayesian production controller is one arm; the frozen, pre-committed IID
# ``random_control`` manifest (``sweep_v2_six_axis_random_control``) is a
# scientifically independent second arm that shares this campaign's identity
# grammar/execution/review spine but never contacts the Bayesian controller.
# Broadening this set does not widen the Bayesian search: the production
# prepare path is arm-parametrized (``prepare_bayesian_proposal_v2`` pins
# ``expected_arm="bayesian"``) and the W&B sweep config is unchanged.
SEARCH_ARMS_V2 = frozenset({"bayesian", "random_control"})

_AXES_V2 = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size", "seq_length")
# Guards against silent v1 axis-set drift: if v1's frozen five-axis field
# set ever changed, this trips immediately at import time rather than
# silently producing a wrong six-axis contract.
assert set(_AXES_V2) - {"seq_length"} == set(sweep._HYPERPARAMETER_FIELDS)

_PROVENANCE_V2 = {"proposal_order", "execution_generation", "wandb_sweep_id", "wandb_run_id",
                   "campaign_id", "domain_version", "search_arm"}
_FORBIDDEN_V2 = {"seed", "model_seed", "epochs", "target_epoch", "max_updates_per_epoch", "package_identity",
                  "screening_artifact_sha256", "screening_path", "evaluation_scope", "sealed_scope",
                  "performance_early_stopping_enabled", "qualification_kind", "launch_contract_qualification",
                  "objective_source", "support_contract_version", "support_contract_sha256"}

# v1's frozen fixed configuration minus the axis that v2 now sweeps --
# always derived from the frozen v1 dict itself, never re-typed, so it can
# never silently diverge from v1's real frozen contract.
FROZEN_FIXED_CONFIGURATION_V2 = {k: v for k, v in sweep.FROZEN_FIXED_CONFIGURATION.items() if k != "seq_length"}
assert "seq_length" not in FROZEN_FIXED_CONFIGURATION_V2

SEARCH_DOMAIN_V2 = {
    **{key: json.loads(json.dumps(value)) for key, value in sweep.SEARCH_DOMAIN.items()},
    "seq_length": {
        "kind": "continuous",
        "distribution": "q_uniform",
        "min": SEQ_LENGTH_MIN,
        "max": SEQ_LENGTH_MAX,
        "q": SEQ_LENGTH_STEP,
        "legal_values": list(SEQ_LENGTH_DOMAIN_V2),
        "requires_normalization": True,
        "normalization_function": "normalize_seq_length_axis",
    },
}


def normalize_seq_length_axis(value: Any) -> int:
    """Normalize one controller-supplied seq_length value to a legal int.

    Must run before canonicalization/hashing/identity/intake/config-
    generation/output-root construction, per the binding six-axis design.
    Accepts ``72`` or ``72.0`` (the ``q_uniform`` controller's float
    return); rejects bool, string, non-finite, fractional, off-grid, and
    out-of-range values.
    """
    if isinstance(value, bool):
        raise SweepV2CampaignError(f"seq_length must not be a bool, got {value!r}")
    if not isinstance(value, (int, float)):
        raise SweepV2CampaignError(f"seq_length must be a finite real numeric value, got {type(value).__name__}: {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SweepV2CampaignError(f"seq_length must be finite (not NaN/inf), got {value!r}")
    if numeric != math.floor(numeric):
        raise SweepV2CampaignError(f"seq_length must be an integral value, got {value!r}")
    normalized = int(numeric)
    if not (SEQ_LENGTH_MIN <= normalized <= SEQ_LENGTH_MAX):
        raise SweepV2CampaignError(
            f"seq_length {normalized} is outside the legal range [{SEQ_LENGTH_MIN}, {SEQ_LENGTH_MAX}]"
        )
    if (normalized - SEQ_LENGTH_MIN) % SEQ_LENGTH_STEP != 0:
        raise SweepV2CampaignError(
            f"seq_length {normalized} is not on the {SEQ_LENGTH_STEP}h grid from {SEQ_LENGTH_MIN}h"
        )
    return normalized


def canonical_hyperparameters_v2(hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    """Six-axis canonical coordinate: v1's five-axis canonicalization
    (reused verbatim) plus the normalized ``seq_length`` int."""
    if set(hyperparameters) != set(_AXES_V2):
        raise SweepV2CampaignError(f"hyperparameters must contain exactly {_AXES_V2}, got {sorted(hyperparameters)}")
    five_axis = sweep.canonical_hyperparameters(
        {key: hyperparameters[key] for key in sweep._HYPERPARAMETER_FIELDS}
    )
    seq_length = normalize_seq_length_axis(hyperparameters["seq_length"])
    return {**five_axis, "seq_length": seq_length}


def configuration_id_v2(hyperparameters: Mapping[str, Any], *, support_contract_version: str,
                          support_contract_sha256: str) -> str:
    """Hash the six-axis canonical coordinate together with the v2 domain/
    canonicalization version AND the frozen fixed-support contract identity
    -- per the binding requirement to "bind the fixed-support artifact
    version and checksum into the v2 scientific contract and configuration
    identity". Never collides with a v1 ``configuration_id`` (different
    prefix, different payload shape, different hashed fields)."""
    canonical = canonical_hyperparameters_v2(hyperparameters)
    if not isinstance(support_contract_version, str) or not support_contract_version:
        raise SweepV2CampaignError("support_contract_version is required and must be a non-empty string")
    if not isinstance(support_contract_sha256, str) or not support_contract_sha256:
        raise SweepV2CampaignError("support_contract_sha256 is required and must be a non-empty string")
    payload = {
        "hyperparameters": canonical,
        "campaign_id": CAMPAIGN_ID_V2,
        "domain_version": DOMAIN_VERSION_V2,
        "canonicalization_version": CONFIGURATION_CANONICALIZATION_VERSION_V2,
        "support_contract_version": support_contract_version,
        "support_contract_sha256": support_contract_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sweep_v2_cfg_" + hashlib.sha256(encoded).hexdigest()[:20]


def proposal_id_v2(search_arm: str, proposal_order: int) -> str:
    if search_arm not in SEARCH_ARMS_V2:
        raise SweepV2CampaignError(f"unknown v2 search arm: {search_arm!r}")
    if not isinstance(proposal_order, int) or isinstance(proposal_order, bool) or proposal_order < 1:
        raise SweepV2CampaignError("proposal_order must be a positive integer")
    return f"{CAMPAIGN_ID_V2}__{search_arm}__proposal{proposal_order:03d}"


def trial_id_v2(configuration_id: str, proposal_id: str, *, execution_generation: int = 1) -> str:
    """Bind one executable v2 attempt to a configuration AND a specific
    controller proposal -- unlike v1's ``trial_id`` (config-identity only),
    this always embeds ``proposal_id`` so two different controller
    proposals landing on identical six-axis coordinates never collide, per
    the binding requirement that "a v2 trial/output identity must
    distinguish two controller proposals even if W&B proposes identical
    coordinates"."""
    if not isinstance(execution_generation, int) or isinstance(execution_generation, bool) or execution_generation < 1:
        raise SweepV2CampaignError("execution_generation must be a positive integer")
    if not configuration_id.startswith("sweep_v2_cfg_"):
        raise SweepV2CampaignError(f"trial_id_v2 requires a v2 configuration_id, got {configuration_id!r}")
    if not proposal_id.startswith(CAMPAIGN_ID_V2 + "__"):
        raise SweepV2CampaignError(f"trial_id_v2 requires a v2 proposal_id, got {proposal_id!r}")
    return f"{proposal_id}__{configuration_id}__{FIDELITY_ID_V2}__seedA{sweep.MODEL_SEED_A}__attempt{execution_generation:03d}"


# Trial-identity-conflict comparison is a pure string-equality helper with
# no five-axis-specific behavior -- reused verbatim rather than
# re-implemented.
trial_identity_conflicts_v2 = sweep.trial_identity_conflicts


def assert_no_v1_contamination(*, wandb_sweep_id: "str | None" = None, campaign_id: "str | None" = None,
                                 domain_version: "str | None" = None) -> None:
    """Refuse any binding of v2 provenance to v1's frozen sweep/campaign/
    domain identity."""
    if wandb_sweep_id == FORBIDDEN_V1_SWEEP_ID:
        raise SweepV2CampaignError(
            f"refusing to bind a v2 proposal to the frozen v1 production sweep {FORBIDDEN_V1_SWEEP_ID!r}"
        )
    if campaign_id is not None and campaign_id == sweep.CAMPAIGN_ID:
        raise SweepV2CampaignError(f"refusing v1 campaign_id {sweep.CAMPAIGN_ID!r} for a v2 proposal")
    if domain_version is not None and domain_version == sweep.DOMAIN_VERSION:
        raise SweepV2CampaignError(f"refusing v1 domain_version {sweep.DOMAIN_VERSION!r} for a v2 proposal")


# Six-axis siblings of v1's TRIAL_SUMMARY_FIELDS/PROPOSAL_RECORD_FIELDS: same
# shape, but spreading the six-axis _AXES_V2 (including seq_length) instead
# of v1's five-axis _HYPERPARAMETER_FIELDS. EPOCH_TRAJECTORY_FIELDS and
# OPERATIONS_RECORD_FIELDS carry no hyperparameter fields at all, so they are
# axis-agnostic and reused verbatim from sweep_v1_campaign by
# validate_review_record_v2 below -- no v2 sibling is needed for either.
TRIAL_SUMMARY_FIELDS_V2 = frozenset({
    "campaign_id", "domain_version", "search_arm", "proposal_id", "configuration_id", "trial_id",
    "workflow_status", "objective_score", "best_epoch", "best_score", "final_epoch_score", "best_minus_final",
    "best_score_10", "best_score_12", "late_gain_10_to_12", "late_best", *_AXES_V2,
    "runtime_seconds", "gpu_hours", "execution_generation", "retry_of_trial_id", "failure_category",
    "fixed_support_metric_name", "fixed_support_epoch_trajectory",
    "natural_support_metric_name", "natural_support_epoch_trajectory",
    "support_contract_version", "support_contract_sha256", "objective_eligible", "publication_state",
})
PROPOSAL_RECORD_FIELDS_V2 = frozenset({
    "campaign_id", "domain_version", "search_arm", "proposal_id", "proposal_order", "configuration_id",
    *_AXES_V2, "valid_result_order", "boundary_review_checkpoint", "wave_id",
})


def validate_review_record_v2(record_type: str, record: Mapping[str, Any]) -> None:
    """v2 sibling of :func:`sweep_v1_campaign.validate_review_record`.

    ``sweep.validate_review_record`` hardcodes a check that
    ``record["campaign_id"] == sweep.CAMPAIGN_ID`` / ``domain_version ==
    sweep.DOMAIN_VERSION``, so it can never accept a genuine v2 record --
    this sibling applies the identical structural checks (missing-field
    schema, forbidden sealed/temporal_test/spatial_holdout/california/
    wandb_-prefixed markers, known search arm) against the v2 identity and
    six-axis field sets instead. ``epoch_trajectory``/``operations`` reuse
    v1's own field sets verbatim (both are axis-agnostic -- neither spreads
    any hyperparameter field), exactly as v1's own validator does for those
    two record types.
    """
    schemas = {
        "trial_summary": TRIAL_SUMMARY_FIELDS_V2, "epoch_trajectory": sweep.EPOCH_TRAJECTORY_FIELDS,
        "proposal": PROPOSAL_RECORD_FIELDS_V2, "operations": sweep.OPERATIONS_RECORD_FIELDS,
    }
    if record_type not in schemas:
        raise SweepV2CampaignError(f"unknown record type: {record_type!r}")
    missing = schemas[record_type] - set(record)
    if missing:
        raise SweepV2CampaignError(f"{record_type} missing required fields: {sorted(missing)}")
    forbidden_markers = ("sealed", "temporal_test", "spatial_holdout", "california")
    forbidden = {key for key in record if any(marker in key.lower() for marker in forbidden_markers)
                 or key.lower().startswith("wandb_")}
    if forbidden:
        raise SweepV2CampaignError(f"{record_type} contains forbidden non-authoritative fields: {sorted(forbidden)}")
    if record["campaign_id"] != CAMPAIGN_ID_V2 or record["domain_version"] != DOMAIN_VERSION_V2:
        raise SweepV2CampaignError("record campaign/domain identity does not match the frozen v2 six-axis wave")
    if record["search_arm"] not in SEARCH_ARMS_V2:
        raise SweepV2CampaignError("record has unknown search arm")


def validate_v2_proposal_shape(proposal: Mapping[str, Any], *, expected_arm: str = "bayesian") -> None:
    """Strict six-axis key-set/provenance validation, mirroring
    ``sweep_v1_production_adapter._validate_proposal``'s pattern: reject
    unknown/forbidden/missing fields and any v1 contamination."""
    keys = set(proposal)
    forbidden = keys & _FORBIDDEN_V2
    if forbidden:
        raise SweepV2CampaignError(f"scientific override/qualification metadata forbidden: {sorted(forbidden)}")
    unknown = keys - set(_AXES_V2) - _PROVENANCE_V2
    if unknown:
        raise SweepV2CampaignError(f"unexpected scientific/provenance fields (unknown seventh axis?): {sorted(unknown)}")
    missing = set(_AXES_V2) - keys
    if missing:
        raise SweepV2CampaignError(f"missing six-axis search fields: {sorted(missing)}")
    assert_no_v1_contamination(
        wandb_sweep_id=proposal.get("wandb_sweep_id"),
        campaign_id=proposal.get("campaign_id"),
        domain_version=proposal.get("domain_version"),
    )
    if proposal.get("campaign_id", CAMPAIGN_ID_V2) != CAMPAIGN_ID_V2:
        raise SweepV2CampaignError("proposal campaign_id does not match the v2 six-axis campaign")
    if proposal.get("domain_version", DOMAIN_VERSION_V2) != DOMAIN_VERSION_V2:
        raise SweepV2CampaignError("proposal domain_version does not match the v2 six-axis domain")
    if proposal.get("search_arm", expected_arm) != expected_arm:
        raise SweepV2CampaignError(f"preparation expects {expected_arm!r} provenance")
