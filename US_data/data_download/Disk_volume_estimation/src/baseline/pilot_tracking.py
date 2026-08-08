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

Stable run identity across Slurm continuations: a Flash-NH candidate
(``pilot_policy_name`` + ``run_id``) may be trained across several separate,
bounded Slurm jobs (see ``pilot_orchestration.run_pilot``'s chunked restart
design). Without a persistent W&B run id, each job's fresh ``wandb.init()``
would start a brand-new W&B run, fragmenting one candidate's history across
several disconnected runs. :func:`derive_pilot_wandb_run_id` fixes this with
a run id that is a pure, deterministic function of
``pilot_policy_name``/``run_id``/``tracking_generation`` -- correct from the
very first call, before any NH run directory exists. :func:`resolve_pilot_wandb_run_id`
additionally persists that id (once an NH run directory exists) purely as an
auditability/contradiction-detection layer: if a later call ever resolves a
*different* deterministic id for what the persisted record claims is the
same candidate, that is a real bug (e.g. a stale run directory reused for a
different candidate) and is raised as :class:`TrackingError`, deliberately
outside :func:`init_pilot_tracking_run`'s broad init-failure downgrade --
identity contradictions must fail clearly, ordinary W&B init failures must
not. This id is never authoritative for checkpoint or scientific identity;
it only labels the wandb side of a run whose real identity is the repository
run directory.

``tracking_generation`` (default ``"g1"``, threaded through
``run_identity["tracking_generation"]``) exists to close a real collision
gap: ``(pilot_policy_name, run_id)`` alone is NOT a sufficient uniqueness
boundary, because NeuralHydrology's own run directories are timestamped
(``pilot_orchestration.discover_nh_run_dir`` matches by an
``experiment_name`` *prefix*, not an exact, permanent directory), so an
operator who deliberately abandons a candidate's NH run directory and
restarts it from scratch under the *same* ``run_id`` (a real, previously-
used Flash-NH operational pattern -- e.g. an unrecoverable crash, or a bug
found early enough to warrant a clean restart) produces a brand-new
``nh_run_dir`` with no persisted identity record to contradict against.
Without a generation marker, that fresh attempt would silently derive the
*same* W&B run id as the abandoned one and (via ``resume="allow"``) splice
its history onto the abandoned run's, corrupting the epoch axis. A genuine
Slurm-bounded *continuation* of the same in-progress attempt never needs a
new generation (the default ``"g1"`` covers the entire pilot matrix as
currently run); only a deliberate, manual restart-from-scratch decision
should ever pass a different value (e.g. ``"g2"``), and doing so is exactly
as manual/auditable as the directory abandonment that necessitated it.
Historical backfill (docs/stage1_wandb_user_guide.md, guide section 14)
would similarly use its own reserved generation (e.g. ``"backfill"``),
making it structurally impossible to collide with a live-tracked run's id.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

from .nh_config_generation import GeneratedConfigBundle
from .pilot_lead06_config import PilotPolicy, PilotRunSpec
from .pilot_screening_eval import SCREENING_METRIC_SCOPE
from .splits import sha256_of
from .wandb_tracking import (
    TrackingError,
    TrackingRun,
    finish_tracking_run,
    init_tracking_run,
    load_tracking_policy,
    log_checkpoint_reference,
    log_hyperparameters,
    log_resource_metrics,
    log_scientific_metrics,
)

__all__ = [
    "build_pilot_run_identity",
    "build_pilot_hyperparameters",
    "derive_pilot_wandb_run_id",
    "resolve_pilot_wandb_run_id",
    "init_pilot_tracking_run",
    "log_pilot_screening_event",
    "log_pilot_epoch_training_metrics",
    "log_pilot_checkpoint_reference",
    "finish_pilot_run",
]

# Name of the small JSON record persisted under the NH run directory once it
# exists, recording which deterministic W&B run id this candidate resolved
# to -- an auditability/contradiction-detection layer only (see module
# docstring); the id itself never depends on this file existing.
WANDB_RUN_ID_STATE_FILENAME = "pilot_wandb_run_identity.json"

_WANDB_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")

# Config-mapping keys copied verbatim into the logged hyperparameters, all of
# which this pilot holds fixed across the closed 6-run matrix except "seed"
# and "statics_embedding" (see nh_config_generation._pilot_lead06_profile) --
# logging them anyway makes each run's evidence self-contained rather than
# relying on cross-referencing the frozen profile source.
_HYPERPARAMETER_CONFIG_KEYS = (
    "model", "hidden_size", "output_dropout", "batch_size", "optimizer",
    "learning_rate", "loss", "save_weights_every", "validate_every",
    "validate_n_random_basins", "num_workers", "seed", "statics_embedding",
    "max_updates_per_epoch",
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


def derive_pilot_wandb_run_id(pilot_policy_name: str, run_id: str, tracking_generation: str = "g1") -> str:
    """Deterministic W&B run id for one Flash-NH candidate ATTEMPT: a pure
    function of ``(pilot_policy_name, run_id, tracking_generation)``, so it
    is already correct on the very first call -- before any NH run
    directory exists -- and identical on every subsequent call/Slurm
    continuation for the same attempt. ``tracking_generation`` defaults to
    ``"g1"`` and should stay at its default for ordinary bounded-Slurm
    continuations of one in-progress attempt; it is the caller's
    responsibility to pass a different value for a deliberate
    restart-from-scratch of the same ``run_id`` (see module docstring)."""
    raw = f"flashnh-{pilot_policy_name}-{run_id}-{tracking_generation}"
    return _WANDB_ID_SAFE_RE.sub("_", raw)


def _load_wandb_identity_record(nh_run_dir) -> "dict | None":
    p = Path(nh_run_dir) / WANDB_RUN_ID_STATE_FILENAME
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_wandb_identity_record(nh_run_dir, record: dict) -> None:
    p = Path(nh_run_dir) / WANDB_RUN_ID_STATE_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        os.replace(tmp_name, p)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def resolve_pilot_wandb_run_id(
    *,
    pilot_policy_name: str,
    run_id: str,
    tracking_generation: str = "g1",
    nh_run_dir: "str | Path | None" = None,
) -> str:
    """Resolve the W&B run id for one Flash-NH candidate attempt.

    Always returns :func:`derive_pilot_wandb_run_id`'s deterministic value
    for ``(pilot_policy_name, run_id, tracking_generation)`` -- correctness
    never depends on ``nh_run_dir``. If ``nh_run_dir`` is given and already
    exists on disk, additionally persists/cross-checks that id there (``
    WANDB_RUN_ID_STATE_FILENAME``) purely as an auditability layer: a
    genuinely contradictory reuse of the same run directory for a different
    candidate OR a different tracking generation raises :class:`TrackingError`
    rather than silently mixing two attempts' histories into one W&B run.
    """
    wandb_run_id = derive_pilot_wandb_run_id(pilot_policy_name, run_id, tracking_generation)
    if nh_run_dir is None:
        return wandb_run_id

    nh_run_dir = Path(nh_run_dir)
    if not nh_run_dir.is_dir():
        return wandb_run_id

    record = _load_wandb_identity_record(nh_run_dir)
    if record is None:
        _save_wandb_identity_record(
            nh_run_dir,
            {
                "wandb_run_id": wandb_run_id,
                "pilot_policy_name": pilot_policy_name,
                "run_id": run_id,
                "tracking_generation": tracking_generation,
            },
        )
        return wandb_run_id

    if (
        record.get("wandb_run_id") != wandb_run_id
        or record.get("run_id") != run_id
        or record.get("pilot_policy_name") != pilot_policy_name
        or record.get("tracking_generation") != tracking_generation
    ):
        raise TrackingError(
            f"NH run directory {nh_run_dir} already has a persisted W&B run identity "
            f"{record!r}, which contradicts the identity resolved for this call "
            f"(pilot_policy_name={pilot_policy_name!r}, run_id={run_id!r}, "
            f"tracking_generation={tracking_generation!r}, wandb_run_id={wandb_run_id!r}) -- "
            "refusing to reuse this run directory for a different candidate/generation "
            "under the same W&B run id"
        )
    return wandb_run_id


def build_pilot_run_identity(
    *,
    pilot_policy: PilotPolicy,
    run_spec: PilotRunSpec,
    bundle: GeneratedConfigBundle,
    effective_early_stopping_policy: dict,
    tracking_generation: str = "g1",
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
    never submits one). ``tracking_generation`` (default ``"g1"``) should be
    left at its default for ordinary runs -- see ``derive_pilot_wandb_run_id``
    for when a caller should deliberately pass a different value."""
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
        # Nullable multi-fidelity field (see docs/
        # stage1_validation_optimization_foundation.md Part L.5/L.6): None
        # (the default, and every pre-existing pilot run's value) means
        # uncapped/full-fidelity training; a positive int is this run's
        # frozen fidelity-screening cap, sourced straight from the bundle
        # that was actually used to generate this run's config.yaml -- never
        # re-derived or mutated here. Logged explicitly (never omitted) so a
        # capped run's value is always distinguishable from an uncapped run's
        # None. See pilot_orchestration.enforce_pilot_cap_identity for the
        # continuation-time safeguard that keeps this value frozen across
        # any Slurm resumption of the same run directory.
        "max_updates_per_epoch": bundle.max_updates_per_epoch,
        # Explicit per-candidate learning-rate identity (LR-A range-
        # characterization campaign; see docs/decision_log.md and
        # nh_config_generation.validate_learning_rate_override).
        # "learning_rate_override" is None for every pre-existing pilot run
        # (including the closed six-run matrix and the cap25k/cap50k closure
        # candidates) -- it is only non-None for an LR-A candidate that
        # explicitly overrode its run_profile_name's own learning_rate.
        # "resolved_learning_rate" is always the actual value this run's
        # config.yaml carries (the override if given, else the profile's own
        # value), so it is never omitted/None for a real config. Logged
        # explicitly here (not only indirectly via
        # build_pilot_hyperparameters' config-mapping dump) so a resumed run
        # directory's identity can be checked without re-parsing config.yaml
        # -- see pilot_orchestration.enforce_pilot_learning_rate_identity for
        # the continuation-time safeguard that keeps this value frozen across
        # any Slurm resumption of the same run directory.
        "learning_rate_override": bundle.learning_rate,
        "resolved_learning_rate": bundle.config_mapping.get("learning_rate"),
        "baseline_policy_sha256": bundle.policy_sha256,
        "splits_dir": bundle.splits_dir,
        # Whichever W&B policy file actually took effect for this invocation
        # -- the committed disabled default, or an explicit per-run
        # --wandb-policy-path override (see scripts/run_stage1_lead06_pilot.py
        # and docs/stage1_wandb_user_guide.md). The raw path itself is
        # machine-local and already captured verbatim in this run's
        # commands_used/evidence bundle; only the checksum belongs in the
        # portable run identity, mirroring pilot_policy_sha256/
        # baseline_policy_sha256 above.
        "wandb_policy_sha256": sha256_of(pilot_policy.wandb_policy_path),
        "tracking_generation": tracking_generation,
        "wandb_run_id": derive_pilot_wandb_run_id(
            pilot_policy.raw.get("policy_name"), run_spec.run_id, tracking_generation
        ),
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


def init_pilot_tracking_run(
    pilot_policy: PilotPolicy, run_identity: dict, nh_run_dir: "str | Path | None" = None
) -> TrackingRun:
    """Load this pilot's W&B policy (``pilot_policy.wandb_policy_path`` --
    disabled by default, see config/stage1_wandb_tracking_policy_v001.yaml)
    and start a tracked run, reusing this candidate's stable W&B run id
    (:func:`resolve_pilot_wandb_run_id`) so repeated calls -- one per Slurm
    job continuing the same candidate -- resume the same logical W&B run
    instead of fragmenting it. ``nh_run_dir``, if given and already exists
    on disk, additionally cross-checks that id against what was persisted
    there by an earlier call to this same run directory; a genuine
    contradiction is raised as :class:`TrackingError` and is NOT caught
    below (an identity contradiction is a real bug, not an ordinary
    init failure).

    Any *other* failure to actually start a real W&B run (package missing,
    network outage, auth failure in "online" mode) is caught and downgraded
    to the wrapper's own local-only null sink with a warning -- tracking is
    never allowed to block or crash pilot training."""
    policy = load_tracking_policy(pilot_policy.wandb_policy_path)
    # Disabled/null-mode policy never touches wandb at all -- don't persist
    # an identity-record file on disk for a run that never used W&B.
    policy_active = bool(policy.get("enabled", False)) and policy["mode"] != "disabled"
    wandb_run_id = resolve_pilot_wandb_run_id(
        pilot_policy_name=run_identity.get("pilot_policy_name"),
        run_id=run_identity.get("run_id"),
        tracking_generation=run_identity.get("tracking_generation", "g1"),
        nh_run_dir=nh_run_dir if policy_active else None,
    )
    try:
        return init_tracking_run(policy, run_identity, run_id=wandb_run_id, resume="allow")
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
    """Record a compact checkpoint-file reference (epoch + path + checksum +
    size + checkpoint_type), never the checkpoint's own bytes. Routed
    through :func:`wandb_tracking.log_checkpoint_reference`, which -- unlike
    the generic ``log_artifact_reference`` -- never applies a "compact
    artifact" size ceiling to the referenced file (checkpoints are always
    large) and never raises on failure; any failure degrades tracking
    instead of propagating (see Moriah job 45731908 postmortem,
    docs/stage1_lead06_pilot_v001.md)."""
    log_checkpoint_reference(
        run, epoch=epoch, path=path, checksum=checksum, checkpoint_type="nh_model_checkpoint"
    )


def finish_pilot_run(run: TrackingRun, *, final_status: str, best_epoch: "int | None" = None) -> None:
    """Record the run's terminal status (and, if known, its best checkpoint
    epoch) in the run's local mirror, then finish it."""
    run.run_identity["final_status"] = final_status
    run.run_identity["best_checkpoint_epoch"] = best_epoch
    finish_tracking_run(run)
