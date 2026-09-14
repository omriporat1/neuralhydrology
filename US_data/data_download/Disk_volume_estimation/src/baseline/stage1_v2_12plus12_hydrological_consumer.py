"""RD1-C4-B -- fixed-support hydrological batch consumer and frozen Q98
package for the Stage-1 v2 12+12 scientific review
(``docs/stage1_v2_12plus12_review_design_v001.md``).

This module is the sole consumer of RD1-C4-A's
``fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics(...,
return_admitted_series=True)`` producer interface. It never reopens
``validation_results.p``, never independently reconstructs support masks,
raw-space arrays, basin areas, or NSE/KGE -- every scientific number it
reports for the official objective and the full-support per-basin package
comes from the already-qualified fixed-support evaluator
(:mod:`src.baseline.fixed_support_contract_v2`) and raw-space metrics
(:mod:`src.baseline.nh_raw_space_evaluation`), and every basin-distribution
quantile/ECDF number comes from the already-qualified RD1-C3 primitive
(:func:`src.baseline.stage1_v2_12plus12_basin_analysis.analyze_basin_distribution`).

This module adds exactly three things RD1-C4-A/C3 do not provide:

1. Official-best-epoch/objective identity binding and validation
   (:class:`V2BestEpochSource` / :func:`build_v2_best_epoch_source`), reusing
   :func:`src.baseline.sweep_v2_six_axis_execution.build_v2_objective_publication_payload`
   as the sole v2 optimizer-payload validator plus a direct identity
   self-consistency recomputation via
   :func:`src.baseline.sweep_v2_six_axis_campaign.proposal_id_v2` /
   :func:`~src.baseline.sweep_v2_six_axis_campaign.trial_id_v2`.
2. A frozen, candidate-independent Q98 high-flow diagnostic package
   (:class:`CanonicalBasinQ98Facts` / :class:`Q98ConfigurationBasinDiagnostics`),
   deliberately NOT routed through the incompatible legacy
   :mod:`src.baseline.high_flow_event_metrics` (different RMSE
   normalization, different sample-size gate, an independently-searched
   simulated peak, and no NSE at all).
3. Cross-configuration observed-side identity enforcement and 24/12+12/400
   production-completeness gating
   (:func:`assemble_rd1_hydrological_review`).

Peak-tie convention (user-approved, frozen): when multiple admitted
timestamps share the maximum observed discharge within the frozen Q98
high-flow subset, the observed-peak-time diagnostic uses the EARLIEST of
those timestamps chronologically. This selection depends only on observed
discharge and timestamps -- never on simulated values -- and is computed by
sorting candidate indices by their own timestamp, so it is correct even when
the supplied admitted-series arrays are not already chronologically ordered.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .fixed_support_contract_v2 import (
    AdmittedSeries,
    FixedSupportContractError,
    SupportContractProvenance,
    build_support_contract_provenance,
    evaluate_fixed_support_raw_space_metrics,
    extract_v2_objective_from_fixed_support_result,
    validate_fixed_support_contract,
)
from .nh_raw_space_evaluation import raw_space_metrics
from .stage1_v2_12plus12_basin_analysis import BasinDistributionResult, analyze_basin_distribution
from .sweep_v1_campaign import MODEL_SEED_A, derive_trajectory_diagnostics
from .sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2,
    DOMAIN_VERSION_V2,
    FIDELITY_ID_V2,
    SEARCH_ARMS_V2,
    canonical_hyperparameters_v2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
)
from .sweep_v2_six_axis_config import V2_METRIC_NAME
from .sweep_v2_six_axis_execution import SweepV2ExecutionError, build_v2_objective_publication_payload

__all__ = [
    "HydrologicalConsumerError",
    "V2BestEpochSource",
    "build_v2_best_epoch_source",
    "CanonicalBasinQ98Facts",
    "derive_canonical_basin_q98_facts",
    "Q98ConfigurationBasinDiagnostics",
    "compute_q98_configuration_basin_diagnostics",
    "V2HydrologicalConfigurationResult",
    "evaluate_v2_configuration_hydrological_result",
    "MetricCoverage",
    "HydrologicalReviewResult",
    "assemble_rd1_hydrological_review",
    "HIGH_FLOW_QUANTILE",
    "MIN_HIGH_FLOW_NSE_SAMPLES",
]

#: The single frozen v2 evaluation-scope/sealed-scope identity every formal
#: RD1 production trial must carry (mirrors the literal already enforced by
#: the qualified producer's own :func:`sweep_v2_six_axis_execution._require_prepared_v2`
#: at preparation time -- no importable named constant exists there, so this
#: consumer re-declares the identical literal for its own independent,
#: non-caller-trusting re-validation at consumption time, per RD1-C4 review
#: Finding 2).
_EXPECTED_EVALUATION_SCOPE_V2 = "development_validation_2024_only"
_EXPECTED_SEALED_SCOPE_V2 = False


class HydrologicalConsumerError(ValueError):
    """Raised for any RD1-C4-B contract violation: a malformed/unqualified
    best-epoch source, an official-objective/epoch disagreement, a
    fixed-support/identity checksum mismatch, a cross-configuration
    observed-side mismatch, or an incomplete 24/12+12/400 production
    population. Never raised for an ordinary poor-skill outcome, and never
    silently reconciled."""


#: Frozen Q98 quantile level (design: "basin-specific, observation-derived,
#: candidate-independent" high-flow threshold).
HIGH_FLOW_QUANTILE = 0.98
#: Frozen minimum high-flow sample count for high-flow NSE to be reported
#: (secondary diagnostic; below this it is NaN/unavailable, never zero).
MIN_HIGH_FLOW_NSE_SAMPLES = 50


def _sha256_file(path: "str | Path") -> str:
    """Streaming SHA-256 of a local file -- a generic IO checksum, not
    scientific math, so it is implemented locally rather than importing the
    unrelated independent-audit module for one helper."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _frozen_array_copy(value) -> np.ndarray:
    """Return an independent, read-only copy of ``value``.

    Never aliases (shares memory with) a caller-owned array, and never
    mutates the writeability of the caller's own original array -- only the
    returned copy is marked non-writeable (RD1-C4 review: defensive
    immutability of public scientific results). Deliberately re-declared
    here rather than imported across the module boundary, matching this
    package's existing convention of not importing underscore-prefixed
    helpers across modules (see :mod:`fixed_support_contract_v2`)."""
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _frozen_scientific_value(value: Any) -> Any:
    """Recursively return an independent, immutable snapshot of one public
    scientific result value.

    A NumPy array becomes a defensively-copied, read-only array; a mapping
    becomes a :class:`~types.MappingProxyType` view over a new dict of
    recursively-frozen entries; a list/tuple becomes a tuple of
    recursively-frozen entries; every other value (str/int/float/bool/None,
    or an already-immutable object such as a frozen dataclass) is returned
    unchanged. Small and local by design -- not a repository-wide freezing
    framework."""
    if isinstance(value, np.ndarray):
        return _frozen_array_copy(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen_scientific_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_scientific_value(item) for item in value)
    return value


# --------------------------------------------------------------------------- #
# 1. Official best-epoch/objective identity source.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class V2BestEpochSource:
    """One v2 trial's qualified, receipt-content-bound official best-epoch/
    objective identity and complete per-trial provenance (RD1-C4 review
    Findings 1/2/6), bound directly from a qualified v2 execution/provenance
    record (``execution_provenance.json`` shape, as returned by
    :func:`src.baseline.sweep_v2_six_axis_execution.execute_prepared_trial_v2`)
    whose content is proven, byte-for-byte-parsed, to match
    ``source_receipt_path``. Constructed only via
    :func:`build_v2_best_epoch_source` -- never hand-built from a
    free-standing epoch integer -- and independently re-verified again at
    formal batch-assembly time by :func:`_verify_best_epoch_source_identity`
    so a hand-built/tampered instance cannot silently enter
    :func:`assemble_rd1_hydrological_review`.

    ``canonical_hyperparameters`` and ``fixed_support_epoch_trajectory`` are
    stored as :class:`types.MappingProxyType` views over defensively-copied
    dicts so a caller mutating their own original mapping after construction
    cannot mutate this immutable record.
    """

    campaign_id: str
    domain_version: str
    search_arm: str
    proposal_order: int
    proposal_id: str
    configuration_id: str
    canonical_hyperparameters: Mapping[str, Any]
    trial_id: str
    execution_generation: int
    retry_of_trial_id: Optional[str]
    official_objective: float
    best_epoch: int
    fixed_support_metric_name: str
    fixed_support_epoch_trajectory: Mapping[int, float]
    fidelity_id: str
    model_seed: int
    evaluation_scope: str
    sealed_scope: bool
    support_contract_version: str
    support_contract_sha256: str
    run_dir: str
    source_receipt_path: str
    source_receipt_sha256: str
    git_commit: Optional[str]
    wandb_sweep_id: Optional[str]
    wandb_run_id: Optional[str]
    executor_mode: Optional[str]


def _verify_best_epoch_source_identity(source: V2BestEpochSource) -> None:
    """Independently re-derive every identity/best-epoch fact a
    :class:`V2BestEpochSource` claims, from its own stored fields, via the
    same qualified helpers used at construction. Invoked both inside
    :func:`build_v2_best_epoch_source` and again for every source at formal
    batch-assembly time (:func:`_validate_production_batch_shape`) so a
    hand-constructed instance with an internally-fabricated identity cannot
    enter the formal production batch path merely by satisfying the
    dataclass's field types (RD1-C4 review Finding 2)."""
    if source.campaign_id != CAMPAIGN_ID_V2:
        raise HydrologicalConsumerError(f"trial {source.trial_id!r}: campaign_id {source.campaign_id!r} != {CAMPAIGN_ID_V2!r}")
    if source.domain_version != DOMAIN_VERSION_V2:
        raise HydrologicalConsumerError(f"trial {source.trial_id!r}: domain_version {source.domain_version!r} != {DOMAIN_VERSION_V2!r}")
    if source.fidelity_id != FIDELITY_ID_V2:
        raise HydrologicalConsumerError(f"trial {source.trial_id!r}: fidelity_id {source.fidelity_id!r} != {FIDELITY_ID_V2!r}")
    if source.model_seed != MODEL_SEED_A:
        raise HydrologicalConsumerError(f"trial {source.trial_id!r}: model_seed {source.model_seed!r} != {MODEL_SEED_A!r}")
    if source.evaluation_scope != _EXPECTED_EVALUATION_SCOPE_V2:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: evaluation_scope {source.evaluation_scope!r} != {_EXPECTED_EVALUATION_SCOPE_V2!r}"
        )
    if source.sealed_scope is not _EXPECTED_SEALED_SCOPE_V2:
        raise HydrologicalConsumerError(f"trial {source.trial_id!r}: sealed_scope {source.sealed_scope!r} != {_EXPECTED_SEALED_SCOPE_V2!r}")
    if source.search_arm not in SEARCH_ARMS_V2:
        raise HydrologicalConsumerError(f"trial {source.trial_id!r}: search_arm is not a known v2 search arm: {source.search_arm!r}")
    if not _is_strict_int(source.proposal_order) or source.proposal_order < 1:
        raise HydrologicalConsumerError(f"trial {source.trial_id!r}: proposal_order must be a positive integer, got {source.proposal_order!r}")
    if not _is_strict_int(source.execution_generation) or source.execution_generation < 1:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: execution_generation must be a positive integer, got {source.execution_generation!r}"
        )

    try:
        expected_configuration_id = configuration_id_v2(
            source.canonical_hyperparameters,
            support_contract_version=source.support_contract_version,
            support_contract_sha256=source.support_contract_sha256,
        )
    except Exception as exc:
        raise HydrologicalConsumerError(f"trial {source.trial_id!r}: could not recompute configuration_id: {exc}") from exc
    if expected_configuration_id != source.configuration_id:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: configuration_id {source.configuration_id!r} does not recompute from its "
            f"own canonical_hyperparameters/support_contract identity (recomputed {expected_configuration_id!r})"
        )

    expected_proposal_id = proposal_id_v2(source.search_arm, source.proposal_order)
    if expected_proposal_id != source.proposal_id:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: proposal_id {source.proposal_id!r} is inconsistent with search_arm/"
            f"proposal_order (recomputed {expected_proposal_id!r})"
        )
    expected_trial_id = trial_id_v2(
        source.configuration_id, expected_proposal_id, execution_generation=source.execution_generation
    )
    if expected_trial_id != source.trial_id:
        raise HydrologicalConsumerError(
            f"trial_id {source.trial_id!r} is inconsistent with configuration_id/proposal_id/execution_generation "
            f"(recomputed {expected_trial_id!r})"
        )

    try:
        diagnostics = derive_trajectory_diagnostics(source.fixed_support_epoch_trajectory)
    except ValueError as exc:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: fixed_support_epoch_trajectory is not a complete/finite 12-epoch "
            f"trajectory: {exc}"
        ) from exc
    if diagnostics["best_epoch"] != source.best_epoch:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: official best_epoch {source.best_epoch!r} != the authoritative "
            f"first-achieved-maximum epoch {diagnostics['best_epoch']!r} derived from its own trajectory"
        )
    if float(diagnostics["best_score"]) != float(source.official_objective):
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: official_objective {source.official_objective!r} != the authoritative "
            f"best trajectory score {diagnostics['best_score']!r}"
        )


def build_v2_best_epoch_source(
    *,
    execution_provenance: Mapping[str, Any],
    source_receipt_path: "str | Path",
) -> V2BestEpochSource:
    """Bind a qualified, fully receipt-content-bound v2 best-epoch/objective
    identity (RD1-C4 review Findings 1, 2, 6).

    ``source_receipt_path`` is read exactly once and its bytes are both
    hashed (SHA-256) and parsed as the authoritative JSON record. The
    caller-supplied ``execution_provenance`` mapping must compare exactly
    equal (as a plain ``dict``) to that freshly-parsed receipt content --
    any mismatch (unrelated record, stale/replaced receipt, tampered field)
    fails closed before any scientific evaluation. The evaluated ``run_dir``
    is bound from the receipt's own authoritative ``result.nh_run_dir``
    (:func:`sweep_v1_execution._summarize_receipt`'s field) -- never from an
    independently-supplied/caller-attested path -- and normalized only via
    ``str(Path(...))`` (no basename heuristics).

    Configuration identity (six-axis coordinate + frozen fixed-support
    contract identity), campaign/domain/fidelity/seed/evaluation-scope
    identity, and the official best_epoch/objective are all independently
    recomputed from the receipt's nested ``preparation_record`` snapshot via
    the same qualified v2 campaign helpers the real campaign uses to
    construct them, and via
    :func:`src.baseline.sweep_v1_campaign.derive_trajectory_diagnostics` (the
    authoritative first-achieved-best-epoch rule) -- never a parallel
    "max score" rule.

    Raises :class:`HydrologicalConsumerError` for: a missing/unreadable/
    malformed-JSON/non-object receipt; a receipt whose content does not
    exactly match ``execution_provenance``; any missing/malformed identity
    or preparation-record field; a campaign/domain/fidelity/seed/scope
    mismatch; a configuration_id that does not recompute from its own
    canonical hyperparameters; a best_epoch/objective that does not
    recompute from its own trajectory; an objective-ineligible/non-VALID
    record; or a missing/malformed ``result.nh_run_dir``.
    """
    receipt_path = Path(source_receipt_path)
    if not receipt_path.is_file():
        raise HydrologicalConsumerError(f"source_receipt_path does not exist: {receipt_path}")
    receipt_bytes = receipt_path.read_bytes()
    source_receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    try:
        receipt_record = json.loads(receipt_bytes)
    except json.JSONDecodeError as exc:
        raise HydrologicalConsumerError(f"source_receipt_path is not valid JSON: {receipt_path}: {exc}") from exc
    if not isinstance(receipt_record, dict):
        raise HydrologicalConsumerError(f"source_receipt_path does not contain a JSON object: {receipt_path}")

    if dict(execution_provenance) != receipt_record:
        raise HydrologicalConsumerError(
            f"execution_provenance does not exactly match the content of source_receipt_path ({receipt_path}) -- "
            "refusing to bind a source that is not content-identical to its own receipt"
        )
    record: Mapping[str, Any] = receipt_record

    try:
        payload = build_v2_objective_publication_payload(record)
    except SweepV2ExecutionError as exc:
        raise HydrologicalConsumerError(f"execution_provenance failed v2 objective-publication validation: {exc}") from exc

    # NOTE: the mature, frozen producer (sweep_v2_six_axis_execution.execute_prepared_trial_v2)
    # never writes "domain_version"/"proposal_order" at the top level of
    # execution_provenance.json -- only inside the nested "preparation_record"
    # snapshot (started_fields/terminal_fields there carry only campaign_id/
    # search_arm/proposal_id/configuration_id/trial_id/execution_generation at
    # top level). Requiring them at top level here would fail-closed on every
    # genuine real receipt, so they are read from preparation_record instead.
    required_fields = (
        "campaign_id", "search_arm", "proposal_id", "configuration_id",
        "trial_id", "execution_generation", "best_epoch", "objective_score", "fixed_support_metric_name",
        "fixed_support_epoch_trajectory", "support_contract_version", "support_contract_sha256",
        "preparation_record", "result", "executor_mode",
    )
    missing = [key for key in required_fields if key not in record]
    if missing:
        raise HydrologicalConsumerError(f"execution_provenance is missing required field(s): {sorted(missing)}")

    prep = record["preparation_record"]
    if not isinstance(prep, Mapping):
        raise HydrologicalConsumerError("execution_provenance.preparation_record must be a mapping")
    prep_required_fields = (
        "campaign_id", "domain_version", "fidelity_id", "evaluation_scope", "sealed_scope", "hyperparameters",
        "support_contract_version", "support_contract_sha256", "configuration_id", "proposal_id", "trial_id",
        "search_arm", "proposal_order", "execution_generation", "model_seed", "wandb_sweep_id", "wandb_run_id",
    )
    prep_missing = [key for key in prep_required_fields if key not in prep]
    if prep_missing:
        raise HydrologicalConsumerError(f"execution_provenance.preparation_record is missing required field(s): {sorted(prep_missing)}")

    for label in ("campaign_id", "search_arm", "proposal_id", "configuration_id",
                  "trial_id", "execution_generation", "support_contract_version", "support_contract_sha256"):
        if prep[label] != record[label]:
            raise HydrologicalConsumerError(
                f"execution_provenance.{label} {record[label]!r} disagrees with preparation_record.{label} {prep[label]!r}"
            )

    search_arm = record["search_arm"]
    # proposal_order has no top-level counterpart in the real producer's
    # schema (see the required_fields note above) -- sourced from
    # preparation_record and indirectly re-verified below via
    # _verify_best_epoch_source_identity's proposal_id_v2(search_arm,
    # proposal_order) recomputation against the top-level proposal_id.
    proposal_order = prep["proposal_order"]
    execution_generation = record["execution_generation"]
    configuration_id = record["configuration_id"]
    proposal_id = record["proposal_id"]
    trial_id = record["trial_id"]

    canonical_hyperparameters_raw = prep["hyperparameters"]
    if not isinstance(canonical_hyperparameters_raw, Mapping):
        raise HydrologicalConsumerError("preparation_record.hyperparameters must be a mapping")
    try:
        canonical_hyperparameters = MappingProxyType(dict(canonical_hyperparameters_v2(canonical_hyperparameters_raw)))
    except Exception as exc:
        raise HydrologicalConsumerError(
            f"preparation_record.hyperparameters do not canonicalize as a valid v2 six-axis coordinate: {exc}"
        ) from exc

    if payload["flashnh/trial_id"] != trial_id:
        raise HydrologicalConsumerError("v2 objective-publication payload trial_id disagrees with execution_provenance.trial_id")

    trajectory_raw = record["fixed_support_epoch_trajectory"]
    if not isinstance(trajectory_raw, Mapping):
        raise HydrologicalConsumerError("execution_provenance.fixed_support_epoch_trajectory must be a mapping")
    trajectory = MappingProxyType({int(key): float(value) for key, value in trajectory_raw.items()})

    if V2_METRIC_NAME not in payload:
        raise HydrologicalConsumerError(f"v2 objective-publication payload has an unexpected shape: {sorted(payload)}")
    official_objective = float(payload[V2_METRIC_NAME])

    result_summary = record["result"]
    if not isinstance(result_summary, Mapping) or "nh_run_dir" not in result_summary:
        raise HydrologicalConsumerError(
            "execution_provenance.result.nh_run_dir is missing -- cannot bind an authoritative run directory"
        )
    run_dir_raw = result_summary["nh_run_dir"]
    if not isinstance(run_dir_raw, str) or not run_dir_raw:
        raise HydrologicalConsumerError("execution_provenance.result.nh_run_dir must be a non-empty string")
    run_dir = str(Path(run_dir_raw))

    retry_of_trial_id = record.get("retry_of_trial_id")
    if retry_of_trial_id is not None and not isinstance(retry_of_trial_id, str):
        raise HydrologicalConsumerError("execution_provenance.retry_of_trial_id must be a string or absent/None")
    git_commit = record.get("git_commit")
    if git_commit is not None and not isinstance(git_commit, str):
        raise HydrologicalConsumerError("execution_provenance.git_commit must be a string or absent/None")

    # RD1-C4 review (follow-on): model_seed is a receipt-derived,
    # strictly-validated fact -- never a caller-trusted or unconditionally
    # assumed constant. The receipt's own preparation_record.model_seed is
    # required to be a strict int (never bool/float/str) and to exactly
    # equal MODEL_SEED_A; the stored field preserves the observed receipt
    # value itself (never overwritten by the expected constant).
    model_seed = prep["model_seed"]
    if not _is_strict_int(model_seed):
        raise HydrologicalConsumerError(
            f"preparation_record.model_seed must be a strict integer, got {model_seed!r}"
        )
    if model_seed != MODEL_SEED_A:
        raise HydrologicalConsumerError(
            f"preparation_record.model_seed {model_seed!r} != the frozen v2 model seed {MODEL_SEED_A!r}"
        )

    # Per-trial provenance (RD1-C4 review: preserve and expose directly from
    # the content-bound receipt when present; never infer/manufacture; a
    # null value is preserved distinctly from an absent/malformed one).
    wandb_sweep_id = prep["wandb_sweep_id"]
    if wandb_sweep_id is not None and not isinstance(wandb_sweep_id, str):
        raise HydrologicalConsumerError("preparation_record.wandb_sweep_id must be a string or null")
    wandb_run_id = prep["wandb_run_id"]
    if wandb_run_id is not None and not isinstance(wandb_run_id, str):
        raise HydrologicalConsumerError("preparation_record.wandb_run_id must be a string or null")
    executor_mode = record["executor_mode"]
    if executor_mode is not None and not isinstance(executor_mode, str):
        raise HydrologicalConsumerError("execution_provenance.executor_mode must be a string or null")

    source = V2BestEpochSource(
        campaign_id=prep["campaign_id"],
        domain_version=prep["domain_version"],
        search_arm=search_arm,
        proposal_order=proposal_order,
        proposal_id=proposal_id,
        configuration_id=configuration_id,
        canonical_hyperparameters=canonical_hyperparameters,
        trial_id=trial_id,
        execution_generation=execution_generation,
        retry_of_trial_id=retry_of_trial_id,
        official_objective=official_objective,
        best_epoch=record["best_epoch"],
        fixed_support_metric_name=record["fixed_support_metric_name"],
        fixed_support_epoch_trajectory=trajectory,
        fidelity_id=prep["fidelity_id"],
        model_seed=model_seed,
        evaluation_scope=prep["evaluation_scope"],
        sealed_scope=prep["sealed_scope"],
        support_contract_version=record["support_contract_version"],
        support_contract_sha256=record["support_contract_sha256"],
        run_dir=run_dir,
        source_receipt_path=str(receipt_path),
        source_receipt_sha256=source_receipt_sha256,
        git_commit=git_commit,
        wandb_sweep_id=wandb_sweep_id,
        wandb_run_id=wandb_run_id,
        executor_mode=executor_mode,
    )
    _verify_best_epoch_source_identity(source)
    return source


# --------------------------------------------------------------------------- #
# 2. Frozen Q98 high-flow diagnostic package.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CanonicalBasinQ98Facts:
    """One basin's candidate-independent, observation-derived Q98 facts,
    derived from exactly one canonical admitted observed-discharge series
    (design: "basin-specific, observation-derived, candidate-independent").

    ``high_flow_mask``/``canonical_date``/``canonical_obs_m3s`` are
    positionally aligned to the canonical admitted-series order (the fixed-
    support contract's own ``per_basin_support`` order). ``observed_peak_index``
    is the position, within that same order, of the earliest chronological
    timestamp among any ties at the observed maximum (user-approved
    peak-tie convention) -- computed from observed discharge and timestamps
    only, never from simulated values.
    """

    basin_id: str
    n_admitted: int
    q98_threshold: float
    high_flow_mask: np.ndarray
    n_high_flow: int
    observed_peak_value: float
    observed_peak_index: int
    canonical_date: np.ndarray
    canonical_obs_m3s: np.ndarray

    def __post_init__(self) -> None:
        # Defensively protect public scientific results (RD1-C4 review
        # follow-on): these three arrays are exposed to callers as part of
        # an already-qualified, candidate-independent record -- store an
        # independent, read-only copy of each rather than the caller-
        # supplied array itself.
        for name in ("high_flow_mask", "canonical_date", "canonical_obs_m3s"):
            object.__setattr__(self, name, _frozen_array_copy(getattr(self, name)))


def derive_canonical_basin_q98_facts(*, basin_id: str, date: np.ndarray, obs_m3s: np.ndarray) -> CanonicalBasinQ98Facts:
    """Derive one basin's frozen Q98 threshold/mask/observed-peak facts from
    its admitted timestamps and admitted raw-space observed discharge.

    ``date``/``obs_m3s`` must be 1-D, equal-length, and fully finite --
    admitted fixed-support observations are already guaranteed finite
    upstream (fail-closed elsewhere); this helper never silently drops or
    reduces the support it is given.
    """
    date_arr = np.asarray(date)
    obs_arr = np.asarray(obs_m3s, dtype=np.float64)
    if date_arr.ndim != 1 or obs_arr.ndim != 1:
        raise HydrologicalConsumerError(f"basin {basin_id!r}: Q98 date/obs inputs must be 1-D")
    if date_arr.shape != obs_arr.shape:
        raise HydrologicalConsumerError(
            f"basin {basin_id!r}: Q98 date shape {date_arr.shape} != obs shape {obs_arr.shape}"
        )
    n_admitted = int(obs_arr.size)
    if n_admitted == 0:
        raise HydrologicalConsumerError(f"basin {basin_id!r}: Q98 facts require a non-empty admitted series")
    if not np.isfinite(obs_arr).all():
        raise HydrologicalConsumerError(
            f"basin {basin_id!r}: Q98 facts require fully finite admitted observed discharge -- "
            "never silently reduce the support"
        )

    threshold = float(np.quantile(obs_arr, HIGH_FLOW_QUANTILE, method="linear"))
    mask = obs_arr >= threshold  # ties at the threshold are retained (design: inclusive of equal-to-threshold obs)
    n_high_flow = int(mask.sum())

    peak_value = float(np.max(obs_arr))
    tie_indices = np.flatnonzero(obs_arr == peak_value)
    order = np.argsort(date_arr[tie_indices], kind="stable")
    peak_index = int(tie_indices[order[0]])

    return CanonicalBasinQ98Facts(
        basin_id=basin_id,
        n_admitted=n_admitted,
        q98_threshold=threshold,
        high_flow_mask=mask,
        n_high_flow=n_high_flow,
        observed_peak_value=peak_value,
        observed_peak_index=peak_index,
        canonical_date=date_arr,
        canonical_obs_m3s=obs_arr,
    )


@dataclass(frozen=True)
class Q98ConfigurationBasinDiagnostics:
    """One trial's frozen Q98 high-flow diagnostics for one basin, computed
    against a shared :class:`CanonicalBasinQ98Facts`. Keyed by ``trial_id``
    (never ``configuration_id``) because two distinct trials may legally
    share one ``configuration_id`` (RD1-C4 review Finding 3) while still
    producing independently-simulated, independently-diagnosed results.
    Undefined results (zero/nonfinite denominator, fewer than
    :data:`MIN_HIGH_FLOW_NSE_SAMPLES` qualifying timestamps, zero/nonfinite
    observed high-flow variance) are NaN, never zero, and never invalidate
    the trial."""

    basin_id: str
    trial_id: str
    n_high_flow: int
    q98_normalized_rmse: float
    relative_volume_bias: float
    observed_peak_time_magnitude_error: float
    high_flow_nse: float
    high_flow_nse_available: bool


def compute_q98_configuration_basin_diagnostics(
    *,
    trial_id: str,
    facts: CanonicalBasinQ98Facts,
    date: np.ndarray,
    obs_m3s: np.ndarray,
    sim_m3s: np.ndarray,
) -> Q98ConfigurationBasinDiagnostics:
    """Apply one basin's canonical Q98 threshold/mask/observed-peak facts to
    one trial's admitted series. ``date``/``obs_m3s`` must equal
    ``facts.canonical_date``/``facts.canonical_obs_m3s`` exactly (candidate-
    independence enforcement at the point of use); ``sim_m3s`` must be fully
    finite (the formal production fixed-support gate already requires zero
    non-finite simulations at admitted timestamps -- this helper preserves
    that gate rather than silently pairwise-dropping non-finite values).
    """
    basin_id = facts.basin_id
    date_arr = np.asarray(date)
    obs_arr = np.asarray(obs_m3s, dtype=np.float64)
    sim_arr = np.asarray(sim_m3s, dtype=np.float64)
    if not (date_arr.ndim == obs_arr.ndim == sim_arr.ndim == 1):
        raise HydrologicalConsumerError(f"basin {basin_id!r}/{trial_id!r}: Q98 inputs must be 1-D")
    if not (date_arr.shape == obs_arr.shape == sim_arr.shape == facts.canonical_date.shape):
        raise HydrologicalConsumerError(
            f"basin {basin_id!r}/{trial_id!r}: Q98 input shape does not match the canonical admitted series"
        )
    if not np.array_equal(date_arr, facts.canonical_date):
        raise HydrologicalConsumerError(
            f"basin {basin_id!r}/{trial_id!r}: admitted timestamps differ from the canonical "
            "candidate-independent Q98 series -- Q98 threshold/mask requires identical observed-side support"
        )
    if not np.array_equal(obs_arr, facts.canonical_obs_m3s):
        raise HydrologicalConsumerError(
            f"basin {basin_id!r}/{trial_id!r}: admitted observed discharge differs from the canonical "
            "candidate-independent Q98 series"
        )
    if not np.isfinite(sim_arr).all():
        raise HydrologicalConsumerError(
            f"basin {basin_id!r}/{trial_id!r}: Q98 diagnostics require fully finite admitted simulated "
            "discharge (the formal fixed-support gate already requires zero non-finite simulations at admitted "
            "timestamps) -- refusing to silently pairwise-drop"
        )

    mask = facts.high_flow_mask
    obs_hf = obs_arr[mask]
    sim_hf = sim_arr[mask]
    hf_metrics = raw_space_metrics(obs_hf, sim_hf)

    threshold = facts.q98_threshold
    if np.isfinite(threshold) and threshold != 0.0 and np.isfinite(hf_metrics["rmse"]):
        q98_normalized_rmse = hf_metrics["rmse"] / threshold
    else:
        q98_normalized_rmse = float("nan")

    obs_sum = float(np.sum(obs_hf)) if obs_hf.size else float("nan")
    sim_sum = float(np.sum(sim_hf)) if sim_hf.size else float("nan")
    if np.isfinite(obs_sum) and obs_sum != 0.0 and np.isfinite(sim_sum):
        relative_volume_bias = (sim_sum - obs_sum) / obs_sum
    else:
        relative_volume_bias = float("nan")

    obs_peak = facts.observed_peak_value
    sim_at_peak = float(sim_arr[facts.observed_peak_index])
    if np.isfinite(obs_peak) and obs_peak != 0.0 and np.isfinite(sim_at_peak):
        observed_peak_time_magnitude_error = (sim_at_peak - obs_peak) / obs_peak
    else:
        observed_peak_time_magnitude_error = float("nan")

    obs_hf_variance = float(np.var(obs_hf)) if obs_hf.size else float("nan")
    high_flow_nse_available = (
        facts.n_high_flow >= MIN_HIGH_FLOW_NSE_SAMPLES
        and np.isfinite(hf_metrics["nse"])
        and np.isfinite(obs_hf_variance)
        and obs_hf_variance != 0.0
    )
    high_flow_nse = float(hf_metrics["nse"]) if high_flow_nse_available else float("nan")

    return Q98ConfigurationBasinDiagnostics(
        basin_id=basin_id,
        trial_id=trial_id,
        n_high_flow=facts.n_high_flow,
        q98_normalized_rmse=float(q98_normalized_rmse),
        relative_volume_bias=float(relative_volume_bias),
        observed_peak_time_magnitude_error=float(observed_peak_time_magnitude_error),
        high_flow_nse=high_flow_nse,
        high_flow_nse_available=high_flow_nse_available,
    )


# --------------------------------------------------------------------------- #
# 3. Single-configuration hydrological result.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class V2HydrologicalConfigurationResult:
    """One v2 configuration's complete RD1-C4-B hydrological review result:
    identity/best-epoch provenance, the official and deterministically
    re-scored objective, the full-support per-basin package (via RD1-C4-A),
    the RD1-C3 basin distribution built directly from those per-basin rows,
    and per-basin frozen Q98 diagnostics."""

    best_epoch_source: V2BestEpochSource
    official_objective: float
    rescored_objective: float
    fixed_support_result: Mapping[str, Any]
    admitted_series_by_basin: Mapping[str, AdmittedSeries]
    basin_distribution: BasinDistributionResult
    q98_diagnostics_by_basin: Mapping[str, Q98ConfigurationBasinDiagnostics]
    contract_id: str
    contract_checksum_sha256: str
    support_contract_provenance: SupportContractProvenance

    def __post_init__(self) -> None:
        # Defensively protect public scientific results: wrap the nested
        # result maps in a read-only view over a defensive copy so neither
        # mutating a caller's original mapping after construction, nor
        # clearing/assigning through the returned mapping itself, can alter
        # this immutable record. ``fixed_support_result`` additionally gets
        # a full recursive freeze (RD1-C4 review follow-on): it is a plain
        # dict of dicts/lists/NumPy arrays (per-basin metric rows, the
        # aggregate summary, etc.) coming straight from
        # :func:`fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics`,
        # so the outer-mapping wrap alone would still leave every nested
        # per-basin metric (e.g. NSE) mutable.
        object.__setattr__(self, "fixed_support_result", _frozen_scientific_value(self.fixed_support_result))
        object.__setattr__(self, "admitted_series_by_basin", MappingProxyType(dict(self.admitted_series_by_basin)))
        object.__setattr__(self, "q98_diagnostics_by_basin", MappingProxyType(dict(self.q98_diagnostics_by_basin)))


def evaluate_v2_configuration_hydrological_result(
    *,
    best_epoch_source: V2BestEpochSource,
    package_root: "str | Path",
    contract: Mapping[str, Any],
    basin_ids: Optional[Sequence[str]] = None,
    require_full_screening_population: bool = False,
    canonical_q98_facts_by_basin: Optional[Mapping[str, CanonicalBasinQ98Facts]] = None,
) -> V2HydrologicalConfigurationResult:
    """Evaluate one trial's complete hydrological result at its
    already-qualified official best epoch. ``run_dir`` is never accepted as
    a separate parameter -- it is always taken from
    ``best_epoch_source.run_dir``, which :func:`build_v2_best_epoch_source`
    bound directly from the source receipt's own authoritative
    ``result.nh_run_dir`` (RD1-C4 review Finding 1).

    ``canonical_q98_facts_by_basin``, if supplied, must already be
    cross-configuration-validated (see
    :func:`assemble_rd1_hydrological_review`) and is applied as-is so every
    trial shares one candidate-independent Q98 threshold/mask/observed-peak
    per basin. If omitted (the lower single-trial path), Q98 facts are
    derived from this trial's own admitted series -- valid for standalone/
    synthetic use, but candidate-independence across trials is only
    established by the batch entry point.

    Raises :class:`HydrologicalConsumerError` if
    ``best_epoch_source.support_contract_version``/``support_contract_sha256``
    disagree with ``contract``'s own identity/checksum, or if the
    deterministically re-scored fixed-support aggregate median NSE does not
    equal ``best_epoch_source.official_objective`` exactly.
    """
    contract = validate_fixed_support_contract(dict(contract))
    support_contract_provenance = build_support_contract_provenance(contract)

    if best_epoch_source.support_contract_version != contract["contract_id"]:
        raise HydrologicalConsumerError(
            f"trial {best_epoch_source.trial_id!r}: best-epoch source "
            f"support_contract_version {best_epoch_source.support_contract_version!r} != validated contract "
            f"contract_id {contract['contract_id']!r}"
        )
    if best_epoch_source.support_contract_sha256 != contract["checksum_sha256"]:
        raise HydrologicalConsumerError(
            f"trial {best_epoch_source.trial_id!r}: best-epoch source "
            f"support_contract_sha256 does not match the validated contract's checksum_sha256"
        )

    requested_basin_ids = list(basin_ids) if basin_ids is not None else (
        contract["basin_ids"] if require_full_screening_population else None
    )

    try:
        fixed_support_result = evaluate_fixed_support_raw_space_metrics(
            run_dir=best_epoch_source.run_dir,
            epoch=best_epoch_source.best_epoch,
            package_root=package_root,
            contract=contract,
            basin_ids=requested_basin_ids,
            require_full_screening_population=require_full_screening_population,
            return_admitted_series=True,
        )
    except FixedSupportContractError as exc:
        raise HydrologicalConsumerError(
            f"trial {best_epoch_source.trial_id!r}: fixed-support evaluation at official best "
            f"epoch {best_epoch_source.best_epoch} failed: {exc}"
        ) from exc

    try:
        rescored_objective = extract_v2_objective_from_fixed_support_result(fixed_support_result)
    except FixedSupportContractError as exc:
        raise HydrologicalConsumerError(
            f"trial {best_epoch_source.trial_id!r}: could not extract a re-scored v2 objective "
            f"from the fixed-support result: {exc}"
        ) from exc

    if rescored_objective != best_epoch_source.official_objective:
        raise HydrologicalConsumerError(
            f"trial {best_epoch_source.trial_id!r}: re-scored fixed-support objective "
            f"{rescored_objective!r} does not equal the official objective {best_epoch_source.official_objective!r} "
            f"recorded at best_epoch={best_epoch_source.best_epoch}"
        )

    per_basin_rows = fixed_support_result["per_basin"]
    expected_basin_ids = requested_basin_ids if requested_basin_ids is not None else [
        row["basin_id"] for row in per_basin_rows
    ]
    basin_distribution = analyze_basin_distribution(
        best_epoch_source.configuration_id, expected_basin_ids, per_basin_rows
    )

    admitted_series_by_basin: Mapping[str, AdmittedSeries] = fixed_support_result["admitted_series_by_basin"]
    q98_diagnostics_by_basin: dict[str, Q98ConfigurationBasinDiagnostics] = {}
    for basin_id, series in admitted_series_by_basin.items():
        facts = (canonical_q98_facts_by_basin or {}).get(basin_id)
        if facts is None:
            facts = derive_canonical_basin_q98_facts(basin_id=basin_id, date=series.date, obs_m3s=series.obs_m3s)
        q98_diagnostics_by_basin[basin_id] = compute_q98_configuration_basin_diagnostics(
            trial_id=best_epoch_source.trial_id,
            facts=facts,
            date=series.date,
            obs_m3s=series.obs_m3s,
            sim_m3s=series.sim_m3s,
        )

    return V2HydrologicalConfigurationResult(
        best_epoch_source=best_epoch_source,
        official_objective=best_epoch_source.official_objective,
        rescored_objective=rescored_objective,
        fixed_support_result=fixed_support_result,
        admitted_series_by_basin=admitted_series_by_basin,
        basin_distribution=basin_distribution,
        q98_diagnostics_by_basin=q98_diagnostics_by_basin,
        contract_id=contract["contract_id"],
        contract_checksum_sha256=contract["checksum_sha256"],
        support_contract_provenance=support_contract_provenance,
    )


# --------------------------------------------------------------------------- #
# 4. Top-level RD1 hydrological review result and production batch entry point.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricCoverage:
    """Finite/available coverage for one metric over its complete eligible
    configuration-basin cell population. Missing/undefined values are never
    counted as zero."""

    metric_name: str
    n_total: int
    n_finite: int

    @property
    def n_unavailable(self) -> int:
        return self.n_total - self.n_finite


_COVERAGE_METRIC_NAMES: tuple[str, ...] = (
    "nse", "kge", "kge_r", "kge_alpha", "kge_beta",
    "q98_normalized_rmse", "relative_volume_bias", "observed_peak_time_magnitude_error",
    "high_flow_nse", "high_flow_sample_count",
)


@dataclass(frozen=True)
class HydrologicalReviewResult:
    """The complete RD1 hydrological-review result: all 24 trial results
    (keyed by unique ``trial_id`` -- never ``configuration_id``, since two
    distinct trials may legally share one configuration_id, RD1-C4 review
    Finding 3), exact population/completeness accounting, the canonical
    per-basin Q98 facts, and per-metric diagnostic coverage."""

    results_by_trial_id: Mapping[str, V2HydrologicalConfigurationResult]
    n_configurations: int
    n_bayesian: int
    n_random_control: int
    basin_ids: tuple
    n_basins: int
    canonical_q98_facts_by_basin: Mapping[str, CanonicalBasinQ98Facts]
    coverage: Mapping[str, MetricCoverage]
    contract_id: str
    contract_checksum_sha256: str

    def __post_init__(self) -> None:
        # Defensively protect public scientific results: results_by_trial_id
        # (and the other nested result maps) are exposed as read-only views
        # over defensive copies -- neither mutating a caller's original
        # mapping after construction nor clearing/assigning through the
        # returned mapping can alter this immutable review result.
        object.__setattr__(self, "results_by_trial_id", MappingProxyType(dict(self.results_by_trial_id)))
        object.__setattr__(
            self, "canonical_q98_facts_by_basin", MappingProxyType(dict(self.canonical_q98_facts_by_basin))
        )
        object.__setattr__(self, "coverage", MappingProxyType(dict(self.coverage)))


def _validate_production_batch_shape(
    best_epoch_sources: Sequence[V2BestEpochSource], contract: Mapping[str, Any]
) -> None:
    if len(best_epoch_sources) != 24:
        raise HydrologicalConsumerError(
            f"RD1 hydrological review requires exactly 24 trial sources, got {len(best_epoch_sources)}"
        )
    by_arm: dict[str, list[V2BestEpochSource]] = {arm: [] for arm in SEARCH_ARMS_V2}
    proposal_ids, trial_ids = [], []
    canonical_hyperparameters_by_configuration_id: dict[str, Mapping[str, Any]] = {}
    for source in best_epoch_sources:
        # Independently re-verify every source's own identity again at formal
        # batch-entry time (RD1-C4 review Finding 2) -- a hand-constructed
        # V2BestEpochSource with a fabricated/inconsistent identity cannot
        # enter the formal production batch merely by satisfying the
        # dataclass's field types.
        _verify_best_epoch_source_identity(source)
        if source.search_arm not in by_arm:
            raise HydrologicalConsumerError(f"unknown v2 search arm: {source.search_arm!r}")
        by_arm[source.search_arm].append(source)
        proposal_ids.append(source.proposal_id)
        trial_ids.append(source.trial_id)

        # configuration_id identifies the canonical six-axis coordinate.
        # Distinct trials may legally share one configuration_id (RD1-C4
        # review Finding 3) -- but if they do, their own canonical
        # hyperparameters must agree; one configuration_id claimed by two
        # genuinely conflicting coordinates is a contract failure.
        existing = canonical_hyperparameters_by_configuration_id.get(source.configuration_id)
        if existing is None:
            canonical_hyperparameters_by_configuration_id[source.configuration_id] = source.canonical_hyperparameters
        elif dict(existing) != dict(source.canonical_hyperparameters):
            raise HydrologicalConsumerError(
                f"configuration_id {source.configuration_id!r} is shared by trials with conflicting canonical "
                f"hyperparameters ({dict(existing)!r} != {dict(source.canonical_hyperparameters)!r}) -- one "
                "configuration_id must identify exactly one canonical six-axis coordinate"
            )

    n_bayesian = len(by_arm.get("bayesian", []))
    n_random_control = len(by_arm.get("random_control", []))
    if n_bayesian != 12 or n_random_control != 12:
        raise HydrologicalConsumerError(
            f"RD1 hydrological review requires exactly 12 bayesian + 12 random_control sources, "
            f"got {n_bayesian} bayesian and {n_random_control} random_control"
        )
    for label, values in (("proposal_id", proposal_ids), ("trial_id", trial_ids)):
        if len(set(values)) != len(values):
            duplicates = sorted({v for v in values if values.count(v) > 1})
            raise HydrologicalConsumerError(f"duplicate {label} value(s) among the 24 supplied sources: {duplicates}")

    basin_ids = contract["basin_ids"]
    if len(basin_ids) != 400 or len(set(basin_ids)) != 400:
        raise HydrologicalConsumerError(
            "RD1 hydrological review requires a fixed-support contract with exactly 400 unique screening basins"
        )


def _revalidate_source_against_authoritative_receipt(source: V2BestEpochSource) -> V2BestEpochSource:
    """Prove a supplied :class:`V2BestEpochSource` is not merely internally
    self-consistent (all :func:`_verify_best_epoch_source_identity` checks)
    but genuinely derives from its own claimed, unmodified receipt (RD1-C4
    review follow-on: formal assembly must not trust dataclass
    self-consistency as receipt qualification).

    Re-reads ``source.source_receipt_path`` from scratch, re-hashes it and
    requires the digest to still equal ``source.source_receipt_sha256``,
    re-parses it as JSON, and routes it back through
    :func:`build_v2_best_epoch_source` -- the identical qualified
    construction path a real trial uses -- then requires the freshly
    rebuilt source to equal the supplied one field-for-field. A
    hand-constructed instance (even one whose own fields recompute
    consistently against each other) cannot pass unless it is exactly what
    its own claimed receipt file actually produces; a receipt that has been
    mutated or replaced since the source was bound fails on the hash check
    before any re-parse is attempted.
    """
    receipt_path = Path(source.source_receipt_path)
    if not receipt_path.is_file():
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: source_receipt_path does not exist -- cannot revalidate this source "
            f"against its authoritative receipt: {receipt_path}"
        )
    receipt_bytes = receipt_path.read_bytes()
    actual_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if actual_sha256 != source.source_receipt_sha256:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: source_receipt_path content has changed since this source was bound "
            f"(current sha256 {actual_sha256!r} != recorded {source.source_receipt_sha256!r})"
        )
    try:
        receipt_record = json.loads(receipt_bytes)
    except json.JSONDecodeError as exc:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: source_receipt_path is not valid JSON: {exc}"
        ) from exc
    rebuilt = build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
    if rebuilt != source:
        raise HydrologicalConsumerError(
            f"trial {source.trial_id!r}: the supplied V2BestEpochSource does not match the source freshly "
            "rebuilt from its own authoritative receipt -- refusing to trust dataclass self-consistency as "
            "receipt qualification"
        )
    return rebuilt


def _derive_and_validate_canonical_q98_facts(
    results_by_trial_id: Mapping[str, V2HydrologicalConfigurationResult],
) -> dict[str, CanonicalBasinQ98Facts]:
    trial_ids = sorted(results_by_trial_id)
    if not trial_ids:
        raise HydrologicalConsumerError("no trial results supplied for cross-trial Q98 derivation")

    reference_id = trial_ids[0]
    reference_basins = set(results_by_trial_id[reference_id].admitted_series_by_basin)
    for trial_id in trial_ids[1:]:
        basins = set(results_by_trial_id[trial_id].admitted_series_by_basin)
        if basins != reference_basins:
            raise HydrologicalConsumerError(
                f"trial {trial_id!r} admitted-basin population differs from {reference_id!r} "
                f"(missing: {sorted(reference_basins - basins)}, extra: {sorted(basins - reference_basins)}) -- "
                "partial/mismatched basin coverage across trials is a contract failure"
            )

    canonical: dict[str, CanonicalBasinQ98Facts] = {}
    for basin_id in sorted(reference_basins):
        reference_series = results_by_trial_id[reference_id].admitted_series_by_basin[basin_id]
        for trial_id in trial_ids[1:]:
            series = results_by_trial_id[trial_id].admitted_series_by_basin[basin_id]
            if series.date.shape != reference_series.date.shape or not np.array_equal(series.date, reference_series.date):
                raise HydrologicalConsumerError(
                    f"basin {basin_id!r}: trial {trial_id!r} admitted timestamps differ from "
                    f"{reference_id!r} -- the Q98 threshold/mask requires an identical observed-side support "
                    "across every trial"
                )
            if series.obs_m3s.shape != reference_series.obs_m3s.shape or not np.array_equal(
                series.obs_m3s, reference_series.obs_m3s
            ):
                raise HydrologicalConsumerError(
                    f"basin {basin_id!r}: trial {trial_id!r} admitted observed discharge differs "
                    f"from {reference_id!r} -- Q98 must be derived from one candidate-independent observed series"
                )
        canonical[basin_id] = derive_canonical_basin_q98_facts(
            basin_id=basin_id, date=reference_series.date, obs_m3s=reference_series.obs_m3s
        )
    return canonical


#: Metric fields every ``per_basin`` row of a qualified fixed-support result
#: must carry (RD1-C4 review Finding 7). Their absence is a schema
#: regression and fails closed; an explicit NaN present under the key is
#: "undefined/unavailable", never "missing".
_REQUIRED_PER_BASIN_METRIC_FIELDS: tuple[str, ...] = ("nse", "kge", "kge_r", "kge_alpha", "kge_beta")


def _compute_coverage(
    results_by_trial_id: Mapping[str, V2HydrologicalConfigurationResult], basin_ids: Sequence[str]
) -> dict[str, MetricCoverage]:
    """Compute finite/available coverage over the complete, already-validated
    trial x basin cell population.

    Required per-basin metric fields are validated for PRESENCE (schema
    completeness) before any finite/undefined counting: a genuinely missing
    required field fails closed with :class:`HydrologicalConsumerError`
    rather than being silently treated as an explicit NaN via ``dict.get``
    (RD1-C4 review Finding 7). Once validated complete, every cell is
    exactly one of finite/available or explicitly-undefined (NaN) -- there
    is no remaining "missing" category to track separately.
    """
    n_total = len(results_by_trial_id) * len(basin_ids)
    finite_counts = {name: 0 for name in _COVERAGE_METRIC_NAMES}
    for trial_id, result in results_by_trial_id.items():
        per_basin_by_id = {row["basin_id"]: row for row in result.fixed_support_result["per_basin"]}
        for basin_id in basin_ids:
            row = per_basin_by_id[basin_id]
            missing_fields = [name for name in _REQUIRED_PER_BASIN_METRIC_FIELDS if name not in row]
            if missing_fields:
                raise HydrologicalConsumerError(
                    f"trial {trial_id!r}/basin {basin_id!r}: per_basin row is missing required metric field(s) "
                    f"{missing_fields} -- a required field must never be silently treated as undefined"
                )
            for name in _REQUIRED_PER_BASIN_METRIC_FIELDS:
                value = row[name]
                if value is not None and not isinstance(value, (int, float)):
                    raise HydrologicalConsumerError(
                        f"trial {trial_id!r}/basin {basin_id!r}: metric field {name!r} has non-numeric value {value!r}"
                    )
                if _is_finite_number(value):
                    finite_counts[name] += 1
            q98 = result.q98_diagnostics_by_basin[basin_id]
            if np.isfinite(q98.q98_normalized_rmse):
                finite_counts["q98_normalized_rmse"] += 1
            if np.isfinite(q98.relative_volume_bias):
                finite_counts["relative_volume_bias"] += 1
            if np.isfinite(q98.observed_peak_time_magnitude_error):
                finite_counts["observed_peak_time_magnitude_error"] += 1
            if q98.high_flow_nse_available:
                finite_counts["high_flow_nse"] += 1
            finite_counts["high_flow_sample_count"] += 1  # always reported (frozen requirement E)
    return {
        name: MetricCoverage(metric_name=name, n_total=n_total, n_finite=finite_counts[name])
        for name in _COVERAGE_METRIC_NAMES
    }


def assemble_rd1_hydrological_review(
    *,
    best_epoch_sources: Sequence[V2BestEpochSource],
    package_root: "str | Path",
    contract: Mapping[str, Any],
) -> HydrologicalReviewResult:
    """The formal RD1-C4-B production batch entry point.

    Enforces: exactly 24 trial sources; exactly 12 bayesian + 12
    random_control sources; unique/internally-consistent proposal/trial
    identities (configuration_id MAY legally repeat across trials that share
    one canonical six-axis coordinate, RD1-C4 review Finding 3); exactly the
    frozen contract's 400 basin identities; no partial trial or basin
    population; one validated fixed-support contract identity shared by
    every source (every trial is evaluated against the SAME ``contract``
    object, and each source's own support-contract identity is cross-checked
    against it in :func:`evaluate_v2_configuration_hydrological_result`).
    Every source's identity is independently re-verified again here (never
    only at construction time) via :func:`_verify_best_epoch_source_identity`.

    ``run_dir`` is never supplied by the caller: each trial's evaluated run
    directory is ``best_epoch_source.run_dir``, which was bound directly
    from that trial's own source receipt's authoritative
    ``result.nh_run_dir`` (RD1-C4 review Finding 1).

    Every trial is first evaluated at its official best epoch with
    ``require_full_screening_population=True`` (RD1-C4-A's complete-
    population gate). Only after every trial's admitted observed series are
    proven exactly identical per basin (never inner-joined, never reduced to
    a common subset) is the canonical, candidate-independent Q98
    threshold/mask/observed-peak derived and applied uniformly.
    """
    contract = validate_fixed_support_contract(dict(contract))
    _validate_production_batch_shape(best_epoch_sources, contract)

    # RD1-C4 review follow-on (Finding: formal assembly must revalidate
    # receipt qualification): a directly-constructed, internally
    # self-consistent V2BestEpochSource is not sufficient to enter the
    # formal production path -- every source is independently re-proven
    # here to derive from its own claimed, unmodified receipt before any
    # evaluation is performed. Test-only lower helpers (e.g.
    # _validate_production_batch_shape exercised directly against
    # hand-built synthetic fixtures) remain usable; only this formal entry
    # point enforces receipt-backed qualification.
    best_epoch_sources = [
        _revalidate_source_against_authoritative_receipt(source) for source in best_epoch_sources
    ]

    interim_results: dict[str, V2HydrologicalConfigurationResult] = {}
    for source in best_epoch_sources:
        interim_results[source.trial_id] = evaluate_v2_configuration_hydrological_result(
            best_epoch_source=source,
            package_root=package_root,
            contract=contract,
            require_full_screening_population=True,
        )

    canonical_facts = _derive_and_validate_canonical_q98_facts(interim_results)

    final_results: dict[str, V2HydrologicalConfigurationResult] = {}
    for trial_id, result in interim_results.items():
        q98_diagnostics_by_basin = {
            basin_id: compute_q98_configuration_basin_diagnostics(
                trial_id=trial_id,
                facts=canonical_facts[basin_id],
                date=series.date,
                obs_m3s=series.obs_m3s,
                sim_m3s=series.sim_m3s,
            )
            for basin_id, series in result.admitted_series_by_basin.items()
        }
        final_results[trial_id] = replace(result, q98_diagnostics_by_basin=q98_diagnostics_by_basin)

    coverage = _compute_coverage(final_results, contract["basin_ids"])
    n_bayesian = sum(1 for source in best_epoch_sources if source.search_arm == "bayesian")
    n_random_control = sum(1 for source in best_epoch_sources if source.search_arm == "random_control")

    return HydrologicalReviewResult(
        results_by_trial_id=final_results,
        n_configurations=len(final_results),
        n_bayesian=n_bayesian,
        n_random_control=n_random_control,
        basin_ids=tuple(contract["basin_ids"]),
        n_basins=len(contract["basin_ids"]),
        canonical_q98_facts_by_basin=canonical_facts,
        coverage=coverage,
        contract_id=contract["contract_id"],
        contract_checksum_sha256=contract["checksum_sha256"],
    )
