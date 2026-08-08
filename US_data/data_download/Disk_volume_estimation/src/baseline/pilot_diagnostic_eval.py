"""Stage 1 lead-6 optimization pilot: all-checkpoint diagnostic evaluation
helper for the LR-A learning-rate range-characterization campaign (see
docs/decision_log.md's LR-A design-freeze entry).

LR-A's frozen contract requires evaluating EVERY saved checkpoint epoch
(1-6) for cadence-sensitivity evidence (Section 12), not only the pilot's
established 3-epoch screening cadence (epoch 3 diagnostic-only, epoch 6
stopping-eligible). Epochs 1, 2, 4, and 5 are off-cadence under
:func:`src.baseline.pilot_screening_eval.classify_screening_epoch_role`
(they classify as ``"not_a_screening_epoch"``, which
:func:`src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`
explicitly rejects -- confirmed by reading that function before writing this
module). This module therefore composes, unmodified, the same two lower-level
primitives ``evaluate_screening_checkpoint`` itself is built from --
:func:`src.baseline.pilot_orchestration.ensure_validation_results` (restart-
safe saved-checkpoint-evaluation prerequisite) and
:func:`src.baseline.nh_seed_evaluation.raw_space_metrics_for_run_period`
(the one canonical raw-space metric reader) -- to evaluate an off-cadence
epoch directly, while delegating on-cadence epochs (3 and 6) to the existing
``evaluate_screening_checkpoint`` so their already-established official
semantics are reused, never duplicated.

This module adds no new metric math, no new inference path, and never calls
:func:`src.baseline.pilot_early_stopping.record_screening_event` -- an
off-cadence retrospective evaluation must never become stopping-eligible and
must never mutate early-stopping state, regardless of its metric value. It
also never accesses the sealed temporal-test period, a spatial-holdout
population, or California basins: ``period`` is pinned to ``"validation"``,
exactly like ``pilot_screening_eval.py``.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .nh_seed_evaluation import raw_space_metrics_for_run_period
from .percentile_diagnostics import compute_percentile_table
from .pilot_lead06_config import PilotPolicy
from .pilot_orchestration import EvaluationRequest, default_evaluate_checkpoint, ensure_validation_results
from .pilot_screening_eval import (
    PRIMARY_METRIC_NAME,
    SCREENING_METRIC_SCOPE,
    PilotScreeningEvalError,
    classify_screening_epoch_role,
    evaluate_screening_checkpoint,
)

__all__ = [
    "RETROSPECTIVE_EVALUATION_ROLE",
    "OFFICIAL_EVALUATION_ROLE",
    "evaluate_diagnostic_checkpoint",
    "evaluate_all_diagnostic_checkpoints",
]

# Section 7's explicit retrospective-semantics tag for every off-cadence
# (epoch 1/2/4/5) result -- distinct from "official" so a coherent all-epoch
# table can never conflate a retrospective diagnostic with an established
# epoch-3/epoch-6 record.
RETROSPECTIVE_EVALUATION_ROLE = "retrospective_diagnostic"
OFFICIAL_EVALUATION_ROLE = "official"


def evaluate_diagnostic_checkpoint(
    *,
    nh_run_dir,
    epoch: int,
    package_root,
    target_variable: str,
    lead_hours: int,
    screening_basin_ids: Sequence[str],
    pilot_policy: PilotPolicy,
    evaluate_checkpoint_fn: "Callable[[EvaluationRequest], None]" = default_evaluate_checkpoint,
    period: str = "validation",
) -> dict:
    """Evaluate ONE checkpoint epoch's screening-subset raw-space metrics,
    whatever its cadence role -- on-cadence (3, 6) or off-cadence (1, 2, 4,
    5). Always operates on the 400-basin screening population only, via the
    same ``screening_basin_ids`` every other pilot evaluation path uses (the
    caller's orchestration is responsible for sourcing it from
    :func:`src.baseline.pilot_screening_eval.load_validated_screening_basin_ids`
    once per run).

    Restart-safe and deterministic: :func:`~src.baseline.pilot_orchestration.
    ensure_validation_results` reuses an already-saved NH validation-result
    pickle unchanged, and only runs ``evaluate_checkpoint_fn`` (real NH
    inference) when that pickle is missing -- this function never re-derives
    or overwrites an existing result.

    For an off-cadence epoch (``epoch_role == "not_a_screening_epoch"``),
    returns a payload tagged ``scope=screening_subset_provisional``,
    ``authoritative=False``, ``evaluation_role="retrospective_diagnostic"``,
    ``stopping_eligible=False`` -- unconditionally, regardless of the metric
    value, and never routed through
    :func:`~src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`
    (which structurally rejects off-cadence epochs). For an on-cadence epoch
    (3 or 6), delegates to ``evaluate_screening_checkpoint`` so its
    already-established official semantics (epoch 3 diagnostic-only,
    epoch 6+ stopping-eligible) are reused byte-for-byte, only adding the
    ``evaluation_role="official"`` tag for this module's own coherent-table
    consumers (Section 7's "no duplicate authoritative records" -- this is
    strictly additive labeling, not a second computation)."""
    if period != "validation":
        raise PilotScreeningEvalError(
            f"diagnostic checkpoint evaluation only ever uses period='validation', got {period!r} -- "
            "this pilot's evaluation path must never invoke the sealed temporal-test "
            "period or a spatial-holdout/California evaluation"
        )
    if not screening_basin_ids:
        raise PilotScreeningEvalError("screening_basin_ids must be non-empty")

    ensure_validation_results(
        nh_run_dir=nh_run_dir, epoch=epoch, evaluate_checkpoint_fn=evaluate_checkpoint_fn
    )

    role = classify_screening_epoch_role(epoch, pilot_policy)
    if role != "not_a_screening_epoch":
        result = dict(
            evaluate_screening_checkpoint(
                run_dir=nh_run_dir,
                epoch=epoch,
                package_root=package_root,
                target_variable=target_variable,
                lead_hours=lead_hours,
                screening_basin_ids=screening_basin_ids,
                pilot_policy=pilot_policy,
                period=period,
            )
        )
        result["evaluation_role"] = OFFICIAL_EVALUATION_ROLE
        return result

    metrics = raw_space_metrics_for_run_period(
        run_dir=nh_run_dir,
        period=period,
        epoch=epoch,
        package_root=package_root,
        target_variable=target_variable,
        lead_hours=lead_hours,
        basin_ids=screening_basin_ids,
        compute_pooled=True,
    )
    per_basin_primary = [r[PRIMARY_METRIC_NAME] for r in metrics["per_basin"] if PRIMARY_METRIC_NAME in r]
    distribution = compute_percentile_table(per_basin_primary, metric_name=PRIMARY_METRIC_NAME).to_dict()

    return {
        "scope": SCREENING_METRIC_SCOPE,
        "authoritative": False,
        "epoch": epoch,
        "epoch_role": role,
        "evaluation_role": RETROSPECTIVE_EVALUATION_ROLE,
        "stopping_eligible": False,
        "n_screening_basins_requested": len(list(screening_basin_ids)),
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "primary_metric_median": metrics["aggregate"]["metrics"].get(PRIMARY_METRIC_NAME, {}).get("median"),
        "primary_metric_distribution": distribution,
        "raw_space_metrics": metrics,
    }


def evaluate_all_diagnostic_checkpoints(
    *,
    nh_run_dir,
    epochs: Sequence[int],
    package_root,
    target_variable: str,
    lead_hours: int,
    screening_basin_ids: Sequence[str],
    pilot_policy: PilotPolicy,
    evaluate_checkpoint_fn: "Callable[[EvaluationRequest], None]" = default_evaluate_checkpoint,
    period: str = "validation",
) -> "list[dict]":
    """Evaluate every requested checkpoint ``epochs`` (e.g. ``[1, 2, 3, 4, 5,
    6]`` for LR-A) and return one coherent, epoch-ordered list of payloads --
    each shaped exactly like :func:`evaluate_diagnostic_checkpoint`'s single-
    epoch return value, so official (epoch 3/6) and retrospective (epoch
    1/2/4/5) records sit side by side while remaining unambiguously
    distinguishable via ``evaluation_role``/``authoritative``/
    ``stopping_eligible``. Does not sort or dedupe ``epochs`` -- callers pass
    the exact ordered saved-checkpoint list they want reported (Section 8's
    single training-then-evaluation ordering invariant means this is always
    called only after all of a candidate's checkpoints 1-6 already exist on
    disk)."""
    return [
        evaluate_diagnostic_checkpoint(
            nh_run_dir=nh_run_dir,
            epoch=epoch,
            package_root=package_root,
            target_variable=target_variable,
            lead_hours=lead_hours,
            screening_basin_ids=screening_basin_ids,
            pilot_policy=pilot_policy,
            evaluate_checkpoint_fn=evaluate_checkpoint_fn,
            period=period,
        )
        for epoch in epochs
    ]
