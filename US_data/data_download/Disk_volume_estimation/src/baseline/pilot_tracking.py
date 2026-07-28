"""Stage 1 lead-6 optimization pilot: W&B tracking integration (task item 3).

Composes, unmodified, the existing optional W&B wrapper
(:mod:`src.baseline.wandb_tracking` -- ``load_tracking_policy`` /
``init_tracking_run`` / ``log_hyperparameters`` / ``log_scientific_metrics`` /
``log_resource_metrics`` / ``log_artifact_reference`` / ``finish_tracking_run``)
with this pilot's own run/bundle/screening/early-stopping objects. This module
adds no new tracking backend and no credential handling of its own -- it only
shapes pilot-specific payloads and forwards them, so every safety property
``wandb_tracking.py`` already enforces (disabled-by-default, no
credential-shaped keys, no temporal-test/spatial-holdout keys, no large
artifact files) applies unchanged here.

The one behavior this module adds on top of the underlying wrapper: a W&B
init failure (missing package, network outage in "online" mode, etc.) is
caught here and downgraded to the wrapper's own local-only null sink, with a
``warnings.warn`` -- never a raised exception that could take a real training
run down. This mirrors ``wandb_tracking.py``'s own "TRACKING ONLY, never
fatal" design (see its module docstring) one level up, at the one call
(``wandb.init``) that module does not itself guard.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

from .nh_config_generation import GeneratedConfigBundle
from .pilot_lead06_config import PilotPolicy, PilotRunSpec
from .pilot_screening_eval import SCREENING_METRIC_SCOPE
from .wandb_tracking import (
    TrackingError,
    TrackingRun,
    finish_tracking_run,
    init_tracking_run,
    load_tracking_policy,
    log_artifact_reference,
    log_hyperparameters,
    log_resource_metrics,
    log_scientific_metrics,
)

__all__ = [
    "build_pilot_run_identity",
    "build_pilot_hyperparameters",
    "init_pilot_tracking_run",
    "log_pilot_screening_event",
    "log_pilot_epoch_training_metrics",
    "log_pilot_checkpoint_reference",
    "finish_pilot_run",
]

# Config-mapping keys copied verbatim into the logged hyperparameters, all of
# which this pilot holds fixed across the closed 6-run matrix except "seed"
# and "statics_embedding" (see nh_config_generation._pilot_lead06_profile) --
# logging them anyway makes each run's evidence self-contained rather than
# relying on cross-referencing the frozen profile source.
_HYPERPARAMETER_CONFIG_KEYS = (
    "model", "hidden_size", "output_dropout", "batch_size", "optimizer",
    "learning_rate", "loss", "save_weights_every", "validate_every",
    "validate_n_random_basins", "num_workers", "seed", "statics_embedding",
)


def _git_is_dirty(cwd=None) -> "bool | None":
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd if cwd is not None else Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def build_pilot_run_identity(
    *,
    pilot_policy: PilotPolicy,
    run_spec: PilotRunSpec,
    bundle: GeneratedConfigBundle,
    effective_early_stopping_policy: dict,
    slurm_job_id: "str | None" = None,
    slurm_node: "str | None" = None,
    slurm_partition: "str | None" = None,
    slurm_gres: "str | None" = None,
) -> dict:
    """Build the compact run-identity payload passed to ``init_tracking_run``:
    pilot/run ID, architecture/static-pathway spec, seed, git commit+dirty
    state, package schema/path identity+provenance, target+seq_length, basin
    counts+period roles, effective early-stopping policy, and Slurm identity
    (filled in by the caller once the job is actually scheduled -- all
    default ``None`` here, since this function only describes the run, it
    never submits one)."""
    return {
        "pilot_policy_name": pilot_policy.raw.get("policy_name"),
        "pilot_policy_sha256": pilot_policy.sha256,
        "run_id": run_spec.run_id,
        "run_profile_name": run_spec.run_profile_name,
        "static_pathway": run_spec.static_pathway,
        "embedding_hiddens": run_spec.embedding_hiddens,
        "seed_name": run_spec.seed_name,
        "seed": run_spec.seed,
        "git_commit": bundle.git_commit,
        "git_dirty": _git_is_dirty(),
        "package_root": bundle.package_root,
        "package_manifest_identity": dict(bundle.package_manifest_identity),
        "package_type": bundle.package_type,
        "population_role": bundle.population_role,
        "target_variable": bundle.target_variable,
        "lead_hours": bundle.lead_hours,
        "seq_length": bundle.seq_length,
        "n_train_basins": len(bundle.train_basin_ids),
        "n_validation_basins": len(bundle.validation_basin_ids),
        "n_test_basins": len(bundle.test_basin_ids),
        "workflow_qualification_run_id": pilot_policy.workflow_qualification_run_id,
        "is_workflow_qualification_run": run_spec.run_id == pilot_policy.workflow_qualification_run_id,
        "effective_early_stopping_policy_name": effective_early_stopping_policy["policy_name"],
        "effective_max_epoch_budget": effective_early_stopping_policy["max_epoch_budget"],
        "min_epoch_before_stop": effective_early_stopping_policy["min_epoch_before_stop"],
        "min_delta": effective_early_stopping_policy["min_delta"],
        "patience_events": effective_early_stopping_policy["patience_events"],
        "screening_validation_every_n_epochs": pilot_policy.screening_validation_every_n_epochs,
        "diagnostic_only_epoch": pilot_policy.diagnostic_only_epoch,
        "stopping_eligible_from_epoch": pilot_policy.stopping_eligible_from_epoch,
        "slurm_job_id": slurm_job_id,
        "slurm_node": slurm_node,
        "slurm_partition": slurm_partition,
        "slurm_gres": slurm_gres,
    }


def build_pilot_hyperparameters(bundle: GeneratedConfigBundle) -> dict:
    """Extract the resolved-hyperparameters subset of ``bundle.config_mapping``
    worth logging (see ``_HYPERPARAMETER_CONFIG_KEYS``); never the full
    mapping (which also carries basin lists / paths already covered by
    ``build_pilot_run_identity``)."""
    return {k: bundle.config_mapping[k] for k in _HYPERPARAMETER_CONFIG_KEYS if k in bundle.config_mapping}


def init_pilot_tracking_run(pilot_policy: PilotPolicy, run_identity: dict) -> TrackingRun:
    """Load this pilot's W&B policy (``pilot_policy.wandb_policy_path`` --
    disabled by default, see config/stage1_wandb_tracking_policy_v001.yaml)
    and start a tracked run. Any failure to actually start a real W&B run
    (package missing, network outage, auth failure in "online" mode) is
    caught and downgraded to the wrapper's own local-only null sink with a
    warning -- tracking is never allowed to block or crash pilot training."""
    policy = load_tracking_policy(pilot_policy.wandb_policy_path)
    try:
        return init_tracking_run(policy, run_identity)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: tracking must never be fatal
        warnings.warn(
            f"W&B tracking init failed ({exc!r}); continuing with a local-only null "
            "tracking sink so pilot training is unaffected"
        )
        return TrackingRun(
            backend="null",
            max_artifact_reference_bytes=policy["max_artifact_reference_bytes"],
            run_identity=run_identity,
        )


def log_pilot_screening_event(
    run: TrackingRun,
    *,
    epoch: int,
    screening_result: dict,
    early_stopping_state: "dict | None" = None,
    epoch_training_time_s: "float | None" = None,
    screening_validation_time_s: "float | None" = None,
) -> None:
    """Log one screening checkpoint's raw-space metrics and (once available)
    the restart-safe early-stopping state, tagged non-authoritative
    (``screening_result["scope"] == SCREENING_METRIC_SCOPE`` is asserted).
    ``epoch_training_time_s``/``screening_validation_time_s`` are logged as
    resource metrics only if actually captured (``log_resource_metrics``'s
    own "nothing captured is a no-op" contract, unchanged)."""
    if screening_result["scope"] != SCREENING_METRIC_SCOPE:
        raise TrackingError(
            f"log_pilot_screening_event expects scope={SCREENING_METRIC_SCOPE!r}, "
            f"got {screening_result['scope']!r}"
        )

    aggregate_metrics = screening_result["raw_space_metrics"]["aggregate"]["metrics"]
    metrics = {
        "screening/primary_metric_name": screening_result["primary_metric_name"],
        "screening/primary_metric_median": screening_result["primary_metric_median"],
        "screening/epoch_role": screening_result["epoch_role"],
        "screening/stopping_eligible": screening_result["stopping_eligible"],
        "screening/n_basins_requested": screening_result["n_screening_basins_requested"],
        "screening/n_basins_evaluated": screening_result["raw_space_metrics"].get("n_basins_evaluated"),
        "screening/n_basins_area_excluded": screening_result["raw_space_metrics"].get("n_basins_area_excluded"),
        "screening/n_admitted_total": screening_result["raw_space_metrics"]["aggregate"].get("n_admitted_total"),
    }
    for percentile_key, percentile_value in screening_result["primary_metric_distribution"].get(
        "percentiles", {}
    ).items():
        metrics[f"screening/nse_{percentile_key}"] = percentile_value
    for frac_key in ("frac_gt_0", "frac_gt_0p5", "frac_lt_0"):
        if frac_key in screening_result["primary_metric_distribution"]:
            metrics[f"screening/{frac_key}"] = screening_result["primary_metric_distribution"][frac_key]
    for metric_name, metric_summary in aggregate_metrics.items():
        if isinstance(metric_summary, dict) and "median" in metric_summary:
            metrics[f"screening/{metric_name}_median"] = metric_summary["median"]
            metrics[f"screening/{metric_name}_mean"] = metric_summary.get("mean")
    pooled = screening_result["raw_space_metrics"].get("pooled")
    if pooled:
        for k, v in pooled.items():
            metrics[f"screening/pooled_{k}"] = v

    if early_stopping_state is not None:
        metrics["early_stopping/best_epoch"] = early_stopping_state.get("best_epoch")
        metrics["early_stopping/best_metric_value"] = early_stopping_state.get("best_metric_value")
        metrics["early_stopping/events_since_best_improvement"] = early_stopping_state.get(
            "events_since_best_improvement"
        )
        metrics["early_stopping/stopped"] = early_stopping_state.get("stopped")
        metrics["early_stopping/stop_reason"] = early_stopping_state.get("stop_reason")

    log_scientific_metrics(run, epoch, metrics)

    resource_metrics = {}
    if epoch_training_time_s is not None:
        resource_metrics["epoch_training_time_s"] = epoch_training_time_s
    if screening_validation_time_s is not None:
        resource_metrics["screening_validation_time_s"] = screening_validation_time_s
    log_resource_metrics(run, epoch, resource_metrics)


def log_pilot_epoch_training_metrics(
    run: TrackingRun,
    *,
    epoch: int,
    training_loss: "float | None" = None,
    learning_rate: "float | None" = None,
    optimizer_steps: "int | None" = None,
    admitted_training_samples: "int | None" = None,
    epoch_training_time_s: "float | None" = None,
    wall_time_s: "float | None" = None,
    slurm_job_id: "str | None" = None,
    slurm_node: "str | None" = None,
    slurm_partition: "str | None" = None,
    slurm_gres: "str | None" = None,
) -> None:
    """Log one epoch's ordinary training resource metrics (called every
    epoch, not just screening epochs). Only fields the caller actually
    passes are logged -- an all-``None`` call is a silent no-op, per
    ``log_resource_metrics``'s existing contract."""
    metrics = {
        "training_loss": training_loss,
        "learning_rate": learning_rate,
        "optimizer_steps": optimizer_steps,
        "admitted_training_samples": admitted_training_samples,
        "epoch_training_time_s": epoch_training_time_s,
        "wall_time_s": wall_time_s,
        "slurm_job_id": slurm_job_id,
        "slurm_node": slurm_node,
        "slurm_partition": slurm_partition,
        "slurm_gres": slurm_gres,
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}
    log_resource_metrics(run, epoch, metrics)


def log_pilot_checkpoint_reference(run: TrackingRun, *, epoch: int, path, checksum: str) -> None:
    """Record a compact checkpoint-file reference (path + checksum + size),
    never the checkpoint's own bytes -- ``log_artifact_reference`` already
    structurally refuses anything above its configured size ceiling."""
    log_artifact_reference(run, name=f"checkpoint_epoch_{epoch:03d}", path=path, checksum=checksum)


def finish_pilot_run(run: TrackingRun, *, final_status: str, best_epoch: "int | None" = None) -> None:
    """Record the run's terminal status (and, if known, its best checkpoint
    epoch) in the run's local mirror, then finish it."""
    run.run_identity["final_status"] = final_status
    run.run_identity["best_checkpoint_epoch"] = best_epoch
    finish_tracking_run(run)
