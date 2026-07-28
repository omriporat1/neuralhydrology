"""Stage 1 lead-6 optimization pilot: full-population dev-validation
readiness interface (task item 9).

Prepares, but deliberately does NOT execute, the authoritative full
2,307-basin development-validation path for a promoted pilot checkpoint.
Composes, unmodified:
:func:`src.baseline.pilot_lead06_config.build_pilot_bundle_with_validation_scope`
(the same shared builder :mod:`src.baseline.pilot_screening_eval` uses for
its provisional 400-basin bundle -- only the ``validation_basin_ids``
argument and the resulting ``population_role``/``package_type`` labels
differ) and
:func:`src.baseline.nh_seed_evaluation.raw_space_metrics_for_run_period`
(the same raw-space metric implementation the screening path uses -- no new
formula, only a different basin-list scope).

This module is not invoked against the real certified package, a real
training run, or Moriah anywhere in this task -- see
``docs/stage1_lead06_pilot_v001.md``'s "known limitations" section. It
exists so that once a pilot run is promoted past screening, running the
authoritative full-population development validation of its best checkpoint
is a single documented call against already-tested code, not a redesign
under time pressure.
"""
from __future__ import annotations

from typing import Sequence

from .nh_config_generation import (
    GeneratedConfigBundle,
    read_package_manifest,
    validate_full_population_basin_membership,
)
from .nh_seed_evaluation import raw_space_metrics_for_run_period
from .percentile_diagnostics import compute_percentile_table
from .pilot_lead06_config import (
    PilotConfigError,
    PilotPolicy,
    build_pilot_bundle_with_validation_scope,
    resolve_pilot_run_spec,
)
from .pilot_screening_eval import PRIMARY_METRIC_NAME

__all__ = [
    "PilotFullValidationError",
    "FULL_VALIDATION_METRIC_SCOPE",
    "FULL_VALIDATION_POPULATION_ROLE",
    "load_validated_full_population_basin_ids",
    "build_pilot_full_validation_bundle",
    "evaluate_full_validation_checkpoint",
]

FULL_VALIDATION_METRIC_SCOPE = "development_full_population_validation"

# Distinct from pilot_lead06_config.SCREENING_VALIDATION_POPULATION_ROLE --
# a full-validation bundle's generation_manifest.json must never be
# confused with a screening-subset bundle's.
FULL_VALIDATION_POPULATION_ROLE = "development_pilot_full_validation"


class PilotFullValidationError(PilotConfigError):
    """Raised for an invalid full-validation request (empty basin list,
    etc). Subclasses ``PilotConfigError`` so callers already handling that
    exception class transparently also cover this readiness-only
    interface."""


def load_validated_full_population_basin_ids(*, package_root, splits_dir) -> list:
    """Re-derive the full development population from the CURRENT package's
    own manifest + split membership -- never a caller-supplied list, and
    never the screening subset. Rejects (via the unmodified
    ``validate_full_population_basin_membership``) any spatial-holdout /
    temporal-test / CA basin from ever entering this population."""
    package_manifest = read_package_manifest(package_root)
    basin_membership = validate_full_population_basin_membership(package_manifest, splits_dir)
    return list(basin_membership.development_basins)


def build_pilot_full_validation_bundle(
    *,
    pilot_policy: PilotPolicy,
    run_id: str,
    baseline_policy_path,
    package_root,
    splits_dir,
    static_column_manifest_path=None,
) -> GeneratedConfigBundle:
    """Build the full-population validation-readiness config bundle for one
    pilot run_id: train on and validate against the SAME full development
    population (never the spatial holdout or temporal-test period) -- the
    authoritative counterpart to
    ``pilot_lead06_config.build_pilot_bundle``'s screening-subset bundle.
    Building this bundle does not run anything; NH training/evaluation
    remains a separate, later, explicit step (see module docstring)."""
    run_spec = resolve_pilot_run_spec(pilot_policy, run_id)
    development_basins = load_validated_full_population_basin_ids(package_root=package_root, splits_dir=splits_dir)

    return build_pilot_bundle_with_validation_scope(
        baseline_policy_path=baseline_policy_path,
        package_root=package_root,
        splits_dir=splits_dir,
        lead_hours=pilot_policy.lead_hours,
        seq_length=pilot_policy.seq_length,
        run_profile_name=run_spec.run_profile_name,
        validation_basin_ids=development_basins,
        population_role=FULL_VALIDATION_POPULATION_ROLE,
        package_type=f"stage1_lead06_pilot_full_validation_{run_id}",
        static_column_manifest_path=static_column_manifest_path,
    )


def evaluate_full_validation_checkpoint(
    *,
    run_dir,
    epoch: int,
    package_root,
    target_variable: str,
    lead_hours: int,
    development_basin_ids: Sequence[str],
    promoted_from_run_id: str,
) -> dict:
    """Evaluate one promoted checkpoint's AUTHORITATIVE full-population
    raw-space metrics. ``development_basin_ids`` must be the output of
    :func:`load_validated_full_population_basin_ids` (the caller's
    orchestration is responsible for calling it once, not per checkpoint --
    the same convention ``pilot_screening_eval.evaluate_screening_checkpoint``
    uses for its screening-subset list).

    Same scientific metric implementation as
    ``pilot_screening_eval.evaluate_screening_checkpoint``
    (``raw_space_metrics_for_run_period`` + ``compute_percentile_table``, no
    new formula) -- only the basin-list scope changes, and the result is
    tagged ``authoritative=True`` / ``scope=FULL_VALIDATION_METRIC_SCOPE``
    rather than the screening path's non-authoritative tagging. NOT called
    against a real run anywhere in this task; see module docstring.
    """
    if not development_basin_ids:
        raise PilotFullValidationError("development_basin_ids must be non-empty")

    metrics = raw_space_metrics_for_run_period(
        run_dir=run_dir,
        period="validation",
        epoch=epoch,
        package_root=package_root,
        target_variable=target_variable,
        lead_hours=lead_hours,
        basin_ids=development_basin_ids,
        compute_pooled=True,
    )

    per_basin_primary = [r[PRIMARY_METRIC_NAME] for r in metrics["per_basin"] if PRIMARY_METRIC_NAME in r]
    distribution = compute_percentile_table(per_basin_primary, metric_name=PRIMARY_METRIC_NAME).to_dict()

    return {
        "scope": FULL_VALIDATION_METRIC_SCOPE,
        "authoritative": True,
        "epoch": epoch,
        "promoted_from_run_id": promoted_from_run_id,
        "n_development_basins_requested": len(list(development_basin_ids)),
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "primary_metric_median": metrics["aggregate"]["metrics"].get(PRIMARY_METRIC_NAME, {}).get("median"),
        "primary_metric_distribution": distribution,
        "raw_space_metrics": metrics,
    }
