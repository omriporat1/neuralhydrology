"""Stage 1 lead-6 optimization pilot: provisional 400-basin screening
evaluation (task item 4).

Composes, unmodified, the existing raw-space evaluation machinery:
:func:`src.baseline.nh_seed_evaluation.raw_space_metrics_for_run_period`
(already ``basin_ids``-parameterized -- no second evaluator, no different
formulas) for the metric computation itself, and
:func:`src.baseline.percentile_diagnostics.compute_percentile_table` for the
full NSE percentile grid (p1..p99) and sign fractions the binding evaluation
policy (``docs/stage1_scientific_baseline_design.md``) requires. This module
adds no new metric math -- only:

1. **Membership provenance**: :func:`load_validated_screening_basin_ids`
   always re-derives the screening subset from the committed/generated
   foundation artifact (never a caller-supplied ad-hoc list) and re-validates
   it against the current package's own development-population membership
   (rejecting any spatial-holdout/temporal-test/CA leakage) before an
   evaluation is allowed to run.
2. **Scheduling-role tagging**: epoch 3 is diagnostic-only (recorded but
   cannot trigger early stopping); epoch 6+ on the 3-epoch cadence is
   stopping-eligible; any other epoch is rejected as off-cadence.
3. **Non-authoritative labeling**: every payload this module returns is
   tagged ``scope=SCREENING_METRIC_SCOPE`` / ``authoritative=False`` so a
   screening-subset result can never be mistaken for the eventual
   authoritative full 2,307-basin development validation (task item 9 --
   :mod:`src.baseline.pilot_full_validation`).
"""
from __future__ import annotations

from typing import Sequence

from .nh_config_generation import read_package_manifest, validate_full_population_basin_membership
from .nh_seed_evaluation import raw_space_metrics_for_run_period
from .percentile_diagnostics import compute_percentile_table
from .pilot_lead06_config import PilotPolicy, load_screening_basin_ids

__all__ = [
    "PilotScreeningEvalError",
    "SCREENING_METRIC_SCOPE",
    "PRIMARY_METRIC_NAME",
    "classify_screening_epoch_role",
    "load_validated_screening_basin_ids",
    "evaluate_screening_checkpoint",
]

SCREENING_METRIC_SCOPE = "screening_subset_provisional"
PRIMARY_METRIC_NAME = "nse"


class PilotScreeningEvalError(Exception):
    """Raised for an out-of-policy screening-evaluation request (off-cadence
    epoch, empty/unvalidated basin list), never for an ordinary poor-skill
    outcome."""


def classify_screening_epoch_role(epoch: int, pilot_policy: PilotPolicy) -> str:
    """Returns one of ``"diagnostic_only"``, ``"stopping_eligible"``, or
    ``"not_a_screening_epoch"`` for ``epoch`` under this pilot's fixed
    cadence (``screening_validation_every_n_epochs`` /
    ``diagnostic_only_epoch`` / ``stopping_eligible_from_epoch``, all read
    from the pilot policy, never redesigned here)."""
    if epoch == pilot_policy.diagnostic_only_epoch:
        return "diagnostic_only"
    if epoch % pilot_policy.screening_validation_every_n_epochs != 0:
        return "not_a_screening_epoch"
    if epoch < pilot_policy.stopping_eligible_from_epoch:
        return "diagnostic_only"
    return "stopping_eligible"


def load_validated_screening_basin_ids(
    *, pilot_policy: PilotPolicy, package_root, splits_dir
) -> list:
    """Re-derive the screening subset from the committed/generated foundation
    artifact (``pilot_policy.screening_basin_ids_path``) and re-validate it
    against the CURRENT package's own development-population membership.
    Always call this rather than reading the basin-ids file directly, so an
    evaluation can never silently run against a stale or reselected subset,
    or one containing a spatial-holdout/temporal-test/CA basin."""
    package_manifest = read_package_manifest(package_root)
    basin_membership = validate_full_population_basin_membership(package_manifest, splits_dir)
    return load_screening_basin_ids(
        pilot_policy.screening_basin_ids_path,
        development_basins=basin_membership.development_basins,
        expected_count=pilot_policy.screening_expected_count,
        expected_sha256=pilot_policy.screening_expected_sha256,
    )


def evaluate_screening_checkpoint(
    *,
    run_dir,
    epoch: int,
    package_root,
    target_variable: str,
    lead_hours: int,
    screening_basin_ids: Sequence[str],
    pilot_policy: PilotPolicy,
    period: str = "validation",
) -> dict:
    """Evaluate one training checkpoint's screening-subset raw-space metrics.

    Reads the SAME completed NH "validation"-period evaluation the pilot's
    generated config already restricts to the screening subset (see
    :func:`src.baseline.pilot_lead06_config.build_pilot_bundle`) -- never a
    second, independently triggered evaluation. ``screening_basin_ids``
    must be the output of :func:`load_validated_screening_basin_ids` (the
    caller's orchestration is responsible for calling it once per run, not
    per checkpoint, and passing the same list through).

    ``period`` must be ``"validation"``. This pilot's screening/evaluation
    path never invokes the sealed temporal-test period or a spatial-holdout
    or California evaluation -- it does not have the ability to, by
    construction, not merely by the basin list happening to be scoped (the
    generated config's ``test_start_date``/``test_end_date`` are still the
    real sealed temporal-test window; see
    :func:`src.baseline.pilot_lead06_config.build_pilot_bundle_with_validation_scope`).
    """
    if period != "validation":
        raise PilotScreeningEvalError(
            f"pilot screening evaluation only ever uses period='validation', got {period!r} -- "
            "this pilot's screening/evaluation path must never invoke the sealed temporal-test "
            "period or a spatial-holdout/California evaluation"
        )
    if not screening_basin_ids:
        raise PilotScreeningEvalError("screening_basin_ids must be non-empty")

    role = classify_screening_epoch_role(epoch, pilot_policy)
    if role == "not_a_screening_epoch":
        raise PilotScreeningEvalError(
            f"epoch {epoch} is not on this pilot's screening cadence "
            f"(every {pilot_policy.screening_validation_every_n_epochs} epochs, "
            f"diagnostic epoch {pilot_policy.diagnostic_only_epoch})"
        )

    metrics = raw_space_metrics_for_run_period(
        run_dir=run_dir,
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
        "stopping_eligible": role == "stopping_eligible",
        "n_screening_basins_requested": len(list(screening_basin_ids)),
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "primary_metric_median": metrics["aggregate"]["metrics"].get(PRIMARY_METRIC_NAME, {}).get("median"),
        "primary_metric_distribution": distribution,
        "raw_space_metrics": metrics,
    }
