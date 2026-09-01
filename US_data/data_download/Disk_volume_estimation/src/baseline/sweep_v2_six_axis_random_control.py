"""Offline static contract for the frozen Phase-B Sweep-v2 six-axis IID
random-control arm.

Strictly additive sibling of the random-control portion of
:mod:`sweep_v1_campaign`. It contains no launcher, NeuralHydrology, or W&B
integration and never contacts the live Bayesian controller. It reuses v2's
committed identity/canonicalisation grammar verbatim
(:func:`sweep_v2_six_axis_campaign.canonical_hyperparameters_v2`,
:func:`~sweep_v2_six_axis_campaign.configuration_id_v2`,
:func:`~sweep_v2_six_axis_campaign.proposal_id_v2`,
:func:`~sweep_v2_six_axis_campaign.trial_id_v2`,
:func:`~sweep_v2_six_axis_campaign.normalize_seq_length_axis`) and v2's
committed six-axis search domain (:data:`SEARCH_DOMAIN_V2`), so the frozen
random rows share exactly one scientific coordinate/identity system with the
Bayesian arm.

Scientific framing (recorded in ``docs/decision_log.md``): this manifest was
**frozen after Bayesian observation 1**. It is drawn only from the exact
frozen v2 six-axis priors and is scientifically independent of Proposal 1 --
Proposal 1's configuration, trajectory, and objective play no part in its
generation -- but it is *not* a fully pre-outcome-frozen prospective control,
because the first Bayesian observation was already recorded when it was
authorised.

One-shot realisation rule: :func:`generate_random_control_rows_v2` draws the
12 rows exactly once from the deterministic seed. There is no filtering,
stratification, balancing, space-filling, duplicate rejection, or coverage
optimisation. Natural duplicates are structurally legal.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import sweep_v1_campaign as sweep
from .sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2,
    CONFIGURATION_CANONICALIZATION_VERSION_V2,
    DOMAIN_VERSION_V2,
    FIDELITY_ID_V2,
    OBJECTIVE_ID_V2,
    SEARCH_DOMAIN_V2,
    SEQ_LENGTH_DOMAIN_V2,
    SEQ_LENGTH_MAX,
    SEQ_LENGTH_MIN,
    SEQ_LENGTH_STEP,
    _AXES_V2,
    canonical_hyperparameters_v2,
    configuration_id_v2,
    normalize_seq_length_axis,
    proposal_id_v2,
    trial_id_v2,
)

__all__ = [
    "SweepV2RandomControlError",
    "MANIFEST_RNG_NAMESPACE_V2", "MANIFEST_RNG_NAMESPACE_DIGEST_V2",
    "MANIFEST_RNG_SEED_HEX_V2", "MANIFEST_RNG_SEED_V2",
    "RANDOM_CONTROL_ARM", "RANDOM_CONTROL_COUNT_V2", "MANIFEST_SCHEMA_VERSION_V2",
    "GENERATOR_ALGORITHM_V2", "GENERATOR_VERSION_V2", "GENERATOR_RNG_IMPLEMENTATION_V2",
    "GENERATOR_DRAW_ORDER_V2", "SEQ_LENGTH_SAMPLER_V2",
    "SUPPORT_CONTRACT_VERSION_V2", "SUPPORT_CONTRACT_SHA256_V2",
    "RANDOM_CONTROL_MANIFEST_SHA256_V2",
    "derive_manifest_rng_seed", "draw_seq_length_q_uniform",
    "generate_random_control_rows_v2", "manifest_payload_v2", "render_manifest_v2",
    "validate_manifest_rows_v2", "sha256_bytes",
]


class SweepV2RandomControlError(ValueError):
    """Raised for any v2 six-axis random-control manifest identity/domain violation."""


# ---------------------------------------------------------------------------
# Deterministic RNG seed derivation (recorded in docs/decision_log.md).
#
#   namespace  -> SHA-256 over its exact UTF-8 bytes
#              -> take the first 8 lowercase hex characters
#              -> interpret as an unsigned base-16 integer
#
# The derived seed is verified distinct from model Seed A (967139) and from
# the v1 five-axis random-control manifest seed (20260822); an unexpected
# equality is a hard stop with no fallback.
# ---------------------------------------------------------------------------
MANIFEST_RNG_NAMESPACE_V2 = "stage1_phase_b_sweep_v2_six_axis_random_control_v001"
MANIFEST_RNG_NAMESPACE_DIGEST_V2 = "0979f5d60aa60db35d6f0b5c248bfdf73ac24b734b1b5fcb9753db8517299ea2"
MANIFEST_RNG_SEED_HEX_V2 = "0979f5d6"
MANIFEST_RNG_SEED_V2 = 158987734

_MODEL_SEED_A = sweep.MODEL_SEED_A            # 967139
_V1_MANIFEST_RNG_SEED = sweep.MANIFEST_RNG_SEED  # 20260822


def derive_manifest_rng_seed(namespace: str = MANIFEST_RNG_NAMESPACE_V2) -> dict[str, Any]:
    """Recompute the deterministic manifest RNG seed from ``namespace``.

    Returns the namespace, full digest, first-8-hex prefix, and resulting
    unsigned integer. Raises :class:`SweepV2RandomControlError` if the seed
    collides with model Seed A or the v1 manifest seed -- callers must STOP
    and report rather than fall back to another seed.
    """
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    prefix = digest[:8]
    seed = int(prefix, 16)
    if seed in (_MODEL_SEED_A, _V1_MANIFEST_RNG_SEED):
        raise SweepV2RandomControlError(
            f"derived manifest RNG seed {seed} collides with a reserved seed "
            f"(model Seed A {_MODEL_SEED_A!r} / v1 manifest seed {_V1_MANIFEST_RNG_SEED!r}); "
            "STOP -- no fallback seed is permitted"
        )
    return {
        "namespace": namespace,
        "digest": digest,
        "seed_hex_prefix8": prefix,
        "seed": seed,
    }


# Import-time guard: the recorded constants must match the derivation exactly.
_DERIVED = derive_manifest_rng_seed()
assert _DERIVED["digest"] == MANIFEST_RNG_NAMESPACE_DIGEST_V2, _DERIVED["digest"]
assert _DERIVED["seed_hex_prefix8"] == MANIFEST_RNG_SEED_HEX_V2, _DERIVED["seed_hex_prefix8"]
assert _DERIVED["seed"] == MANIFEST_RNG_SEED_V2, _DERIVED["seed"]


RANDOM_CONTROL_ARM = "random_control"
RANDOM_CONTROL_COUNT_V2 = 12
MANIFEST_SCHEMA_VERSION_V2 = "stage1_phase_b_sweep_v2_six_axis_random_control_manifest_v001"

# These identifiers document exactly how the frozen rows were obtained; they
# do not alter the manifest payload or bytes.
GENERATOR_ALGORITHM_V2 = (
    "python_random_mt19937_iid_log_uniform_uniform_categorical_and_q_uniform_v001"
)
GENERATOR_VERSION_V2 = "sweep_v2_six_axis_iid_random_manifest_v1"
GENERATOR_RNG_IMPLEMENTATION_V2 = "python random.Random (MT19937)"
# Per-row draw order is pinned to v2's authoritative committed axis tuple
# ``sweep_v2_six_axis_campaign._AXES_V2`` (identical to ``SEARCH_DOMAIN_V2``'s
# insertion order): learning_rate, hidden_size, embedding_dropout,
# output_dropout, batch_size, seq_length. This intentionally differs from
# v1's ``GENERATOR_DRAW_ORDER`` (continuous axes first) -- v2 has its own
# committed axis ordering and the draw order follows it.
GENERATOR_DRAW_ORDER_V2 = tuple(_AXES_V2)
assert GENERATOR_DRAW_ORDER_V2 == (
    "learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size", "seq_length",
)
# seq_length reproduces the committed W&B ``q_uniform`` wire semantics
# (config/sweep_v2 production sweep uses distribution="q_uniform",
# min=48, max=120, q=12): a value is X ~ uniform(min, max) then
# round(X / q) * q. Endpoints 48 and 120 therefore carry half the
# probability mass of an interior grid value -- this is a genuinely
# different prior from a categorical 1/7-each draw and must not be
# substituted by one. (Half-integer ties have measure zero for a
# continuous draw, so the round-half-to-even convention is immaterial.)
SEQ_LENGTH_SAMPLER_V2 = "int(round(uniform(min, max) / q) * q)"


# ---------------------------------------------------------------------------
# Frozen fixed-support (Common-120) contract identity, bound into every
# configuration_id via configuration_id_v2. Read from the committed identity
# record so it can never silently drift from the artifact descriptor.
# ---------------------------------------------------------------------------
_IDENTITY_RECORD_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "stage1_v2_common120_fixed_support_artifact_identity_v001.json"
)


def _load_support_contract_identity() -> tuple[str, str]:
    record = json.loads(_IDENTITY_RECORD_PATH.read_text(encoding="utf-8"))
    contract_id = record["fixed_support_contract"]["contract_id"]
    internal_sha = record["artifact"]["internal_canonical_contract_sha256"]
    if contract_id != OBJECTIVE_ID_V2:
        raise SweepV2RandomControlError(
            f"fixed-support identity record contract_id {contract_id!r} does not match "
            f"OBJECTIVE_ID_V2 {OBJECTIVE_ID_V2!r}"
        )
    if not (isinstance(internal_sha, str) and len(internal_sha) == 64):
        raise SweepV2RandomControlError("fixed-support identity record internal checksum is malformed")
    return contract_id, internal_sha


SUPPORT_CONTRACT_VERSION_V2, SUPPORT_CONTRACT_SHA256_V2 = _load_support_contract_identity()

# Pins exact committed manifest bytes; the manifest's internal canonical
# payload also fully determines its semantic content. Frozen once from the
# single one-shot realisation of seed 158987734 -- never re-drawn.
RANDOM_CONTROL_MANIFEST_SHA256_V2 = "59be6726b60863aeed1c25e86782bd7a5e1623434ce23c956e66eb54c527c095"

_HYPERPARAMETER_FIELDS_V2 = tuple(_AXES_V2)


sha256_bytes = sweep.sha256_bytes


def draw_seq_length_q_uniform(rng: random.Random) -> int:
    """One committed-semantics ``q_uniform(48, 120, 12)`` draw."""
    raw = rng.uniform(SEQ_LENGTH_MIN, SEQ_LENGTH_MAX)
    quantised = int(round(raw / SEQ_LENGTH_STEP) * SEQ_LENGTH_STEP)
    return normalize_seq_length_axis(quantised)


def _canonical_row(index: int, canonical: Mapping[str, Any]) -> dict[str, Any]:
    config_id = configuration_id_v2(
        canonical,
        support_contract_version=SUPPORT_CONTRACT_VERSION_V2,
        support_contract_sha256=SUPPORT_CONTRACT_SHA256_V2,
    )
    pid = proposal_id_v2(RANDOM_CONTROL_ARM, index)
    return {
        "manifest_index": index,
        "search_arm": RANDOM_CONTROL_ARM,
        "proposal_id": pid,
        "proposal_order": index,
        "configuration_id": config_id,
        "trial_id_attempt001": trial_id_v2(config_id, pid, execution_generation=1),
        **{key: canonical[key] for key in _HYPERPARAMETER_FIELDS_V2},
        "manifest_rng_seed": MANIFEST_RNG_SEED_V2,
        "campaign_id": CAMPAIGN_ID_V2,
        "domain_version": DOMAIN_VERSION_V2,
    }


def generate_random_control_rows_v2() -> list[dict[str, Any]]:
    """Generate the frozen 12-row IID six-axis control exactly once.

    Draw order per row is :data:`GENERATOR_DRAW_ORDER_V2`: learning_rate in
    log10 geometry, hidden-size category, embedding dropout, output dropout,
    batch-size category, then the ``q_uniform`` seq_length. No filtering,
    stratification, balancing, space-filling, or duplicate rejection occurs.
    """
    rng = random.Random(MANIFEST_RNG_SEED_V2)
    log_lower = math.log10(SEARCH_DOMAIN_V2["learning_rate"]["lower"])
    log_upper = math.log10(SEARCH_DOMAIN_V2["learning_rate"]["upper"])
    hidden_values = list(SEARCH_DOMAIN_V2["hidden_size"]["values"])
    batch_values = list(SEARCH_DOMAIN_V2["batch_size"]["values"])
    emb_lower = SEARCH_DOMAIN_V2["embedding_dropout"]["lower"]
    emb_upper = SEARCH_DOMAIN_V2["embedding_dropout"]["upper"]
    out_lower = SEARCH_DOMAIN_V2["output_dropout"]["lower"]
    out_upper = SEARCH_DOMAIN_V2["output_dropout"]["upper"]

    rows: list[dict[str, Any]] = []
    for index in range(1, RANDOM_CONTROL_COUNT_V2 + 1):
        hyperparameters = {
            "learning_rate": 10 ** rng.uniform(log_lower, log_upper),
            "hidden_size": rng.choice(hidden_values),
            "embedding_dropout": rng.uniform(emb_lower, emb_upper),
            "output_dropout": rng.uniform(out_lower, out_upper),
            "batch_size": rng.choice(batch_values),
            "seq_length": draw_seq_length_q_uniform(rng),
        }
        canonical = canonical_hyperparameters_v2(hyperparameters)
        rows.append(_canonical_row(index, canonical))
    return rows


def manifest_payload_v2(rows: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(generate_random_control_rows_v2() if rows is None else rows)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION_V2,
        "campaign_id": CAMPAIGN_ID_V2,
        "domain_version": DOMAIN_VERSION_V2,
        "canonicalization_version": CONFIGURATION_CANONICALIZATION_VERSION_V2,
        "objective_id": OBJECTIVE_ID_V2,
        "fidelity_id": FIDELITY_ID_V2,
        "frozen_after_bayesian_observation": 1,
        "prospective_pre_outcome_frozen": False,
        "generator_algorithm": GENERATOR_ALGORITHM_V2,
        "generator_version": GENERATOR_VERSION_V2,
        "generator_rng_implementation": GENERATOR_RNG_IMPLEMENTATION_V2,
        "per_row_draw_order": list(GENERATOR_DRAW_ORDER_V2),
        "seq_length_sampler": SEQ_LENGTH_SAMPLER_V2,
        "manifest_rng_namespace": MANIFEST_RNG_NAMESPACE_V2,
        "manifest_rng_namespace_digest": MANIFEST_RNG_NAMESPACE_DIGEST_V2,
        "manifest_rng_seed_hex_prefix8": MANIFEST_RNG_SEED_HEX_V2,
        "manifest_rng_seed": MANIFEST_RNG_SEED_V2,
        "random_control_count": RANDOM_CONTROL_COUNT_V2,
        "model_seed_a": _MODEL_SEED_A,
        "target_epoch": sweep.TARGET_EPOCH,
        "max_updates_per_epoch": sweep.MAX_UPDATES_PER_EPOCH,
        "save_weights_every": 1,
        "package_identity": sweep.PACKAGE_IDENTITY,
        "screening_policy_identity": sweep.SCREENING_POLICY_IDENTITY,
        "screening_artifact_sha256": sweep.SCREENING_ARTIFACT_SHA256,
        "support_contract_version": SUPPORT_CONTRACT_VERSION_V2,
        "support_contract_sha256": SUPPORT_CONTRACT_SHA256_V2,
        "seq_length_legal_values": list(SEQ_LENGTH_DOMAIN_V2),
        "search_domain": SEARCH_DOMAIN_V2,
        "rows": rows,
    }


def render_manifest_v2(rows: Iterable[Mapping[str, Any]] | None = None) -> bytes:
    return (json.dumps(manifest_payload_v2(rows), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def validate_manifest_rows_v2(rows: Iterable[Mapping[str, Any]]) -> None:
    """Structural/identity/domain validation only -- never uniqueness."""
    rows = list(rows)
    if len(rows) != RANDOM_CONTROL_COUNT_V2:
        raise SweepV2RandomControlError(f"expected exactly {RANDOM_CONTROL_COUNT_V2} random-control rows")
    for index, row in enumerate(rows, start=1):
        if row.get("manifest_index") != index or row.get("proposal_order") != index:
            raise SweepV2RandomControlError("manifest rows must preserve the canonical sequential index/order")
        if row.get("search_arm") != RANDOM_CONTROL_ARM or row.get("manifest_rng_seed") != MANIFEST_RNG_SEED_V2:
            raise SweepV2RandomControlError("invalid frozen random-control provenance")
        if row.get("campaign_id") != CAMPAIGN_ID_V2 or row.get("domain_version") != DOMAIN_VERSION_V2:
            raise SweepV2RandomControlError("row campaign/domain identity does not match the frozen v2 six-axis wave")
        hyperparameters = {key: row[key] for key in _HYPERPARAMETER_FIELDS_V2}
        canonical = canonical_hyperparameters_v2(hyperparameters)
        if canonical != hyperparameters:
            raise SweepV2RandomControlError("row hyperparameters are not already in v2 canonical form")
        if row["seq_length"] not in SEQ_LENGTH_DOMAIN_V2:
            raise SweepV2RandomControlError("seq_length is not on the committed 48-120h/12h grid")
        expected_config_id = configuration_id_v2(
            canonical,
            support_contract_version=SUPPORT_CONTRACT_VERSION_V2,
            support_contract_sha256=SUPPORT_CONTRACT_SHA256_V2,
        )
        if row.get("configuration_id") != expected_config_id:
            raise SweepV2RandomControlError("configuration_id does not match normalized six-axis scientific hyperparameters")
        expected_pid = proposal_id_v2(RANDOM_CONTROL_ARM, index)
        if row.get("proposal_id") != expected_pid:
            raise SweepV2RandomControlError("proposal_id does not match the frozen random-control identity grammar")
        if row.get("trial_id_attempt001") != trial_id_v2(expected_config_id, expected_pid, execution_generation=1):
            raise SweepV2RandomControlError("trial_id_attempt001 does not match the frozen v2 trial identity grammar")
