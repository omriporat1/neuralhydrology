"""RD1 continuation evidence contract -- the smallest maintainable
producer/consumer layer that safely supports the two already-approved
future review checkpoints (36 and 48 total runs) without weakening or
mutating the frozen RD1-C4 24-run contract
(:mod:`.rd1_c4_trial_authentication`, :mod:`.rd1_c4_results_runner`,
:func:`.stage1_v2_12plus12_hydrological_consumer.assemble_rd1_hydrological_review`).

Producer / consumer contract (Interface / Consumer Contract Gate,
docs/agent_handoff_rules.md Section 5)
-------------------------------------------------------------------------
1. Producer: the same qualified v2 execution receipts and the same frozen
   fixed-support contract the RD1-C4 production path already consumes,
   selected into one of three legal staged rosters via
   :func:`~.rd1_continuation_trial_authentication.authenticate_continuation_roster`.
2. Intended consumer: the RD1 review-analysis/results tooling that produces
   the two approved future review checkpoints (36-run and 48-run) and any
   downstream reader of their evidence bundle.
3. Required inputs: a continuation trial-list JSON naming exactly which
   receipts to open (never their content), the frozen fixed-support
   contract, and the frozen observation package root.
4. Required outputs / success receipt: a :class:`ContinuationHydrologicalReviewResult`
   carrying phase identity, slice membership, and the interpretation-boundary
   flag, wrapping the exact same qualified
   :class:`~.stage1_v2_12plus12_hydrological_consumer.HydrologicalReviewResult`
   the frozen 24-run contract itself returns; ``produce_rd1_continuation_results``
   additionally writes the same evidence bundle
   :func:`~.rd1_c4_results_runner.write_rd1_c4_results` already produces,
   plus one ``phase_manifest.json`` recording phase identity and the
   interpretation boundary in machine-readable form.
5. Authority for each scientifically or operationally meaningful fact: every
   trial identity fact remains receipt-owned (authenticated by the reused
   :mod:`.rd1_c4_trial_authentication` primitive); every NSE/KGE/Q98 number
   comes from the same already-qualified consumer path the frozen contract
   uses (:func:`~.stage1_v2_12plus12_hydrological_consumer.evaluate_v2_configuration_hydrological_result`,
   :func:`~.stage1_v2_12plus12_hydrological_consumer._derive_canonical_q98_facts_from_package`,
   :func:`~.stage1_v2_12plus12_hydrological_consumer._compute_coverage`); only the
   trial-count/arm-count/proposal-order shape check is phase-aware and new.
6. Failure/incomplete semantics: fails closed with
   :class:`HydrologicalConsumerError` for any shape, identity, or population
   contradiction -- exactly the same failure class the frozen contract
   raises, never a silently narrowed population.
7. Vertical synthetic/integration test:
   ``tests/test_rd1_continuation_results_runner.py`` proves the intended
   consumer can ingest all three legal staged roster shapes and that phase
   identity / the interpretation boundary survive.

Interpretation boundary (task design constraint 5, enforced in code by
:func:`~.rd1_continuation_trial_authentication.require_two_arm_comparison_support`):
only ``initial_controlled_24`` supports the original Bayesian-versus-random
descriptive comparison. ``bayesian_continuation_36`` and ``final_bayesian_48``
support Bayesian search-progress / candidate-discovery evidence only -- no
winner, promotion, classifier, tolerance, range-change, or sealed-scope
conclusion is produced by this module.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .fixed_support_contract_v2 import load_fixed_support_contract, validate_fixed_support_contract
from .package_identity_qualification import PackageIdentityError, qualify_package_identity
from .rd1_c4_results_runner import ResultsRunnerError, _sha256_path, write_rd1_c4_results
from .rd1_continuation_trial_authentication import (
    CONTINUATION_PHASES,
    ContinuationAuthenticationError,
    EXPECTED_PROPOSAL_ORDERS_BY_PHASE,
    PHASE_INITIAL_CONTROLLED_24,
    authenticate_continuation_roster,
    require_two_arm_comparison_support,
)
from .stage1_v2_12plus12_hydrological_consumer import (
    HydrologicalConsumerError,
    HydrologicalReviewResult,
    V2BestEpochSource,
    _compute_coverage,
    _derive_canonical_q98_facts_from_package,
    _revalidate_source_against_authoritative_receipt,
    _verify_best_epoch_source_identity,
    assemble_rd1_hydrological_review,
    compute_q98_configuration_basin_diagnostics,
    evaluate_v2_configuration_hydrological_result,
)
from .sweep_v2_six_axis_campaign import SEARCH_ARMS_V2

__all__ = [
    "ContinuationResultsRunnerError",
    "ContinuationHydrologicalReviewResult",
    "assemble_rd1_continuation_hydrological_review",
    "write_rd1_continuation_results",
    "produce_rd1_continuation_results",
]

#: Continuation-scope reuse of the interpretation-boundary text (task design
#: constraint 5), written verbatim into ``phase_manifest.json`` so a
#: downstream reader never has to reconstruct it from prose.
_INTERPRETATION_BOUNDARY_TEXT = (
    "Only the initial 24-run slice (initial_controlled_24: bayesian P1-P12 + "
    "random_control R1-R12) supports the original Bayesian-versus-random "
    "descriptive comparison. The continuation blocks (bayesian_continuation_36, "
    "final_bayesian_48) support Bayesian search-progress / candidate-discovery "
    "evidence only. No winner, promotion, classifier, tolerance, range-change, "
    "or sealed-scope conclusion is produced."
)


class ContinuationResultsRunnerError(ValueError):
    """Raised for a continuation-roster/authentication failure or a
    hydrological-consumer contract violation surfaced while producing
    continuation results. Never raised for an ordinary poor-skill scientific
    outcome, and never raised to paper over a real contract violation."""


@dataclass(frozen=True)
class ContinuationHydrologicalReviewResult:
    """The continuation-scope wrapper around the frozen
    :class:`~.stage1_v2_12plus12_hydrological_consumer.HydrologicalReviewResult`.

    Every scientific number in ``review`` comes from the exact same
    qualified consumer path the frozen 24-run contract uses -- this wrapper
    adds only phase identity, slice membership, and the interpretation-
    boundary flag; it never recomputes or reinterprets a metric.
    """

    phase: str
    review: HydrologicalReviewResult
    initial_controlled_trial_ids: tuple
    continuation_only_trial_ids: tuple
    initial_slice_identity: "MappingProxyType" = MappingProxyType({})

    @property
    def two_arm_comparison_supported(self) -> bool:
        return self.phase == PHASE_INITIAL_CONTROLLED_24

    def __getattr__(self, name):
        # Transparent pass-through to the wrapped review for every field a
        # caller of the frozen contract's HydrologicalReviewResult already
        # expects (n_configurations, results_by_trial_id, coverage, ...),
        # so existing per-trial/per-basin evidence helpers
        # (rd1_c4_results_runner.*_rows) work unchanged against either.
        return getattr(self.review, name)


def _require_complete_continuation_population(
    fixed_support_result: Mapping[str, Any], *, required_basin_ids: Sequence[str], trial_id: str
) -> None:
    """Phase-aware sibling of
    :func:`~.fixed_support_contract_v2._require_complete_production_fixed_support_population`,
    generalized to the contract's actual basin population instead of a
    hardcoded 400 -- the continuation validator, unlike the frozen
    production path, must support the smallest truthful synthetic contract
    a caller supplies (task design constraint 4). Still refuses any partial
    population: every one of ``required_basin_ids`` must have been
    requested, evaluated, and admitted with zero exclusions."""
    expected = set(required_basin_ids)
    n_expected = len(expected)
    requested = fixed_support_result.get("n_basins_requested")
    evaluated = fixed_support_result.get("n_basins_evaluated")
    excluded = fixed_support_result.get("n_basins_excluded")
    per_basin = fixed_support_result.get("per_basin")
    if (requested, evaluated, excluded) != (n_expected, n_expected, 0) or not isinstance(per_basin, (list, tuple)):
        raise HydrologicalConsumerError(
            f"trial {trial_id!r}: RD1 continuation fixed-support evaluation requires {n_expected} requested, "
            f"{n_expected} evaluated, zero excluded basins (the contract's complete population); got "
            f"requested={requested}, evaluated={evaluated}, excluded={excluded}"
        )
    evaluated_ids = [row.get("basin_id") if isinstance(row, Mapping) else None for row in per_basin]
    if len(evaluated_ids) != n_expected or len(set(evaluated_ids)) != n_expected or set(evaluated_ids) != expected:
        raise HydrologicalConsumerError(
            f"trial {trial_id!r}: RD1 continuation fixed-support evaluated basin IDs must be exactly the "
            "contract's complete basin population"
        )


def _validate_continuation_batch_shape(
    best_epoch_sources: Sequence[V2BestEpochSource], contract: Mapping[str, Any], *, phase: str
) -> None:
    """Phase-aware sibling of
    :func:`~.stage1_v2_12plus12_hydrological_consumer._validate_production_batch_shape`.

    Never called by, and never called instead of, the frozen production
    validator -- the frozen 24-run contract keeps using its own hardcoded
    24/12+12/400 check unchanged. This function performs the identical
    structural checks (source identity re-verification, arm composition,
    configuration_id/canonical-hyperparameter consistency, duplicate
    proposal/trial identity) against the phase-specific expected shape from
    :data:`~.rd1_continuation_trial_authentication.EXPECTED_PROPOSAL_ORDERS_BY_PHASE`
    instead of a fixed count.
    """
    if phase not in EXPECTED_PROPOSAL_ORDERS_BY_PHASE:
        raise HydrologicalConsumerError(f"unknown RD1 continuation phase {phase!r}")
    expected_proposals = EXPECTED_PROPOSAL_ORDERS_BY_PHASE[phase]
    expected_arm_counts = {arm: len(orders) for arm, orders in expected_proposals.items()}
    expected_total = sum(expected_arm_counts.values())
    if len(best_epoch_sources) != expected_total:
        raise HydrologicalConsumerError(
            f"RD1 continuation phase {phase!r} requires exactly {expected_total} trial sources, "
            f"got {len(best_epoch_sources)}"
        )

    by_arm: dict = {arm: [] for arm in SEARCH_ARMS_V2}
    proposal_ids, trial_ids = [], []
    canonical_hyperparameters_by_configuration_id: dict = {}
    for source in best_epoch_sources:
        _verify_best_epoch_source_identity(source)
        if source.search_arm not in by_arm:
            raise HydrologicalConsumerError(f"unknown v2 search arm: {source.search_arm!r}")
        by_arm[source.search_arm].append(source)
        proposal_ids.append(source.proposal_id)
        trial_ids.append(source.trial_id)

        existing = canonical_hyperparameters_by_configuration_id.get(source.configuration_id)
        if existing is None:
            canonical_hyperparameters_by_configuration_id[source.configuration_id] = source.canonical_hyperparameters
        elif dict(existing) != dict(source.canonical_hyperparameters):
            raise HydrologicalConsumerError(
                f"configuration_id {source.configuration_id!r} is shared by trials with conflicting canonical "
                f"hyperparameters ({dict(existing)!r} != {dict(source.canonical_hyperparameters)!r}) -- one "
                "configuration_id must identify exactly one canonical six-axis coordinate"
            )

        expected_orders = expected_proposals.get(source.search_arm, ())
        if source.proposal_order not in expected_orders:
            raise HydrologicalConsumerError(
                f"RD1 continuation phase {phase!r} does not admit {source.search_arm} proposal_order "
                f"{source.proposal_order} (admits {sorted(expected_orders)})"
            )

    observed_arm_counts = {arm: len(sources) for arm, sources in by_arm.items() if sources}
    if observed_arm_counts != expected_arm_counts:
        raise HydrologicalConsumerError(
            f"RD1 continuation phase {phase!r} requires {expected_arm_counts} sources per arm, "
            f"got {observed_arm_counts}"
        )
    for label, values in (("proposal_id", proposal_ids), ("trial_id", trial_ids)):
        if len(set(values)) != len(values):
            duplicates = sorted({v for v in values if values.count(v) > 1})
            raise HydrologicalConsumerError(
                f"duplicate {label} value(s) among the supplied continuation sources: {duplicates}"
            )


def assemble_rd1_continuation_hydrological_review(
    *,
    best_epoch_sources: Sequence[V2BestEpochSource],
    package_root: "str | Path",
    contract: Mapping[str, Any],
    phase: str,
) -> ContinuationHydrologicalReviewResult:
    """The continuation-scope batch entry point for ``bayesian_continuation_36``
    and ``final_bayesian_48``.

    For ``initial_controlled_24`` this delegates unchanged to the frozen
    production entry point
    :func:`~.stage1_v2_12plus12_hydrological_consumer.assemble_rd1_hydrological_review`
    -- the immutable 24-run contract is never re-implemented here, only
    wrapped with phase/slice metadata. For the two continuation phases it
    mirrors that function's own orchestration sequence exactly (contract
    validation, package-identity qualification, per-source receipt
    revalidation, per-source hydrological evaluation, cross-trial canonical
    Q98 derivation, per-basin Q98 diagnostics, coverage), reusing every one
    of those qualified helpers unchanged and substituting only
    :func:`_validate_continuation_batch_shape` for the frozen contract's
    hardcoded 24/12+12 shape check.
    """
    if phase == PHASE_INITIAL_CONTROLLED_24:
        review = assemble_rd1_hydrological_review(
            best_epoch_sources=best_epoch_sources, package_root=package_root, contract=contract
        )
        trial_ids = tuple(sorted(review.results_by_trial_id))
        return ContinuationHydrologicalReviewResult(
            phase=phase,
            review=review,
            initial_controlled_trial_ids=trial_ids,
            continuation_only_trial_ids=(),
        )

    if phase not in EXPECTED_PROPOSAL_ORDERS_BY_PHASE:
        raise HydrologicalConsumerError(f"unknown RD1 continuation phase {phase!r}")

    contract = validate_fixed_support_contract(dict(contract))
    _validate_continuation_batch_shape(best_epoch_sources, contract, phase=phase)

    try:
        package_identity = qualify_package_identity(
            package_root=package_root, contract=contract, basin_ids=contract["basin_ids"]
        )
    except PackageIdentityError as exc:
        raise HydrologicalConsumerError(
            f"the frozen observation package at {package_root} is not qualified for contract "
            f"{contract['contract_id']!r}: {exc}"
        ) from exc

    best_epoch_sources = [
        _revalidate_source_against_authoritative_receipt(source) for source in best_epoch_sources
    ]

    interim_results: dict = {}
    for source in best_epoch_sources:
        result = evaluate_v2_configuration_hydrological_result(
            best_epoch_source=source,
            package_root=package_root,
            package_identity=package_identity,
            contract=contract,
            basin_ids=contract["basin_ids"],
            require_full_screening_population=False,
        )
        _require_complete_continuation_population(
            result.fixed_support_result, required_basin_ids=contract["basin_ids"], trial_id=source.trial_id
        )
        interim_results[source.trial_id] = result

    canonical_facts, provenance_audit_by_trial_id = _derive_canonical_q98_facts_from_package(
        interim_results, package_root=package_root, package_identity=package_identity, contract=contract
    )

    final_results: dict = {}
    for trial_id, result in interim_results.items():
        q98_diagnostics_by_basin = {
            basin_id: compute_q98_configuration_basin_diagnostics(
                trial_id=trial_id,
                facts=canonical_facts[basin_id],
                date=series.date,
                obs_m3s=canonical_facts[basin_id].canonical_obs_m3s,
                sim_m3s=series.sim_m3s,
            )
            for basin_id, series in result.admitted_series_by_basin.items()
        }
        final_results[trial_id] = replace(result, q98_diagnostics_by_basin=q98_diagnostics_by_basin)

    coverage = _compute_coverage(final_results, contract["basin_ids"])
    n_bayesian = sum(1 for source in best_epoch_sources if source.search_arm == "bayesian")
    n_random_control = sum(1 for source in best_epoch_sources if source.search_arm == "random_control")

    review = HydrologicalReviewResult(
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
        provenance_audit_by_trial_id=provenance_audit_by_trial_id,
    )

    initial_set = frozenset(
        (arm, order)
        for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[PHASE_INITIAL_CONTROLLED_24].items()
        for order in orders
    )
    initial_controlled_trial_ids = tuple(
        sorted(
            source.trial_id
            for source in best_epoch_sources
            if (source.search_arm, source.proposal_order) in initial_set
        )
    )
    continuation_only_trial_ids = tuple(
        sorted(
            source.trial_id
            for source in best_epoch_sources
            if (source.search_arm, source.proposal_order) not in initial_set
        )
    )

    return ContinuationHydrologicalReviewResult(
        phase=phase,
        review=review,
        initial_controlled_trial_ids=initial_controlled_trial_ids,
        continuation_only_trial_ids=continuation_only_trial_ids,
    )


def write_rd1_continuation_results(
    continuation_review: ContinuationHydrologicalReviewResult, out_dir: "str | Path"
) -> dict:
    """Write the same evidence bundle
    :func:`~.rd1_c4_results_runner.write_rd1_c4_results` already produces
    (reused unchanged against the wrapped review), plus one additional
    ``phase_manifest.json`` recording phase identity, slice membership, the
    immutable initial-slice identity bridge, and the interpretation boundary.

    All-or-nothing publication: the full bundle is assembled in a fresh
    temporary directory that is a sibling of ``out_dir`` (same parent, so
    the final promotion is a same-filesystem rename) and only promoted to
    the requested ``out_dir`` once every file -- including the corrected
    on-disk ``manifest.json`` -- is complete. ``write_rd1_c4_results`` is
    reused completely unchanged; it is simply pointed at the temporary
    directory instead of the caller's requested path. On any failure the
    temporary directory is removed and ``out_dir`` is left exactly as it
    was found (never created, never partially populated).
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ResultsRunnerError(f"out_dir {out_dir} already exists and is non-empty -- refusing to overwrite")

    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}.tmp-", dir=str(parent)))
    try:
        manifest = write_rd1_c4_results(continuation_review.review, tmp_dir)

        phase_manifest = {
            "phase": continuation_review.phase,
            "two_arm_comparison_supported": continuation_review.two_arm_comparison_supported,
            "initial_controlled_trial_ids": list(continuation_review.initial_controlled_trial_ids),
            "continuation_only_trial_ids": list(continuation_review.continuation_only_trial_ids),
            "initial_slice_identity": {
                key: (dict(value) if isinstance(value, MappingProxyType) else value)
                for key, value in continuation_review.initial_slice_identity.items()
            },
            "interpretation_boundary": _INTERPRETATION_BOUNDARY_TEXT,
        }
        phase_manifest_path = tmp_dir / "phase_manifest.json"
        with open(phase_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(phase_manifest, handle, indent=2, sort_keys=True)

        manifest["phase_manifest.json"] = {
            "size_bytes": phase_manifest_path.stat().st_size,
            "sha256": _sha256_path(phase_manifest_path),
        }
        # write_rd1_c4_results already wrote manifest.json before phase_manifest.json
        # existed, so the on-disk copy is stale -- rewrite it so the on-disk
        # manifest.json and the returned dict agree exactly.
        with open(tmp_dir / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    if out_dir.exists():
        # Refusal above already guarantees this is empty; clear it so the
        # promotion rename below targets a non-existent path (required on
        # Windows, where os.rename refuses an existing directory target).
        out_dir.rmdir()
    try:
        os.rename(str(tmp_dir), str(out_dir))
    except OSError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ResultsRunnerError(f"failed to atomically publish continuation results to {out_dir}: {exc}") from exc

    return manifest


def produce_rd1_continuation_results(
    *,
    trial_list_path: "str | Path",
    contract_path: "str | Path",
    package_root: "str | Path",
    phase: str,
    out_dir: "str | Path",
    initial_roster_path: "str | Path | None" = None,
) -> dict:
    """End-to-end continuation entry point: authenticate the staged roster
    for ``phase`` against the frozen fixed-support contract (binding its
    initial-24 subset to the authenticated frozen ``initial_roster_path``
    for the two continuation phases -- see
    :func:`~.rd1_continuation_trial_authentication.authenticate_continuation_roster`),
    assemble the phase-appropriate hydrological review, and write the
    results evidence bundle plus the phase manifest. Raises
    :class:`ContinuationResultsRunnerError` on any roster/authentication or
    hydrological-consumer contract violation -- never silently narrows the
    population or substitutes a partial result."""
    if phase not in CONTINUATION_PHASES:
        raise ContinuationResultsRunnerError(f"unknown RD1 continuation phase {phase!r}")
    contract = load_fixed_support_contract(contract_path)
    try:
        roster = authenticate_continuation_roster(
            trial_list_path=Path(trial_list_path),
            contract=contract,
            phase=phase,
            initial_roster_path=(None if initial_roster_path is None else Path(initial_roster_path)),
        )
    except ContinuationAuthenticationError as exc:
        raise ContinuationResultsRunnerError(
            f"{trial_list_path}: continuation trial roster is not receipt-qualified: {exc}"
        ) from exc

    best_epoch_sources = [target.best_epoch_source for target in roster]
    try:
        continuation_review = assemble_rd1_continuation_hydrological_review(
            best_epoch_sources=best_epoch_sources, package_root=package_root, contract=contract, phase=phase
        )
    except HydrologicalConsumerError as exc:
        raise ContinuationResultsRunnerError(f"RD1 continuation assembly failed for phase {phase!r}: {exc}") from exc

    continuation_review = replace(continuation_review, initial_slice_identity=roster.initial_slice_identity)

    return write_rd1_continuation_results(continuation_review, out_dir)
