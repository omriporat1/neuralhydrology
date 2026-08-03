"""Optional Weights & Biases tracking for Stage 1 NeuralHydrology runs.

Implements section 10's binding decisions (docs/
stage1_validation_optimization_foundation.md, "Part F"): TRACKING ONLY. This
module never launches, configures, or participates in a hyperparameter
sweep, and never launches a training run itself -- it only records what a
training harness chooses to report about a run it is already running.

Design points, all directly from section 10:
  * Optional and disable-able: `policy["enabled"] = False` (the shipped
    default, see config/stage1_wandb_tracking_policy_v001.yaml) or
    `policy["mode"] = "disabled"` routes every call through an in-memory
    `TrackingRun` that never imports or touches the `wandb` package.
  * Offline-mode-supported: `mode: "offline"` uses wandb's own offline mode
    (no network calls, no login required).
  * No credential exposure: this module never reads, stores, or logs a
    wandb API key. `WANDB_API_KEY`, if "online" mode is ever used, must
    already be present in the operator's shell environment. Every
    dict-shaped payload passed to `log_hyperparameters`/`log_scientific_metrics`/
    `log_resource_metrics`/`init_tracking_run` is scanned for
    credential-shaped keys and rejected if any are found.
  * Compact artifacts/references only: `log_artifact_reference` refuses any
    file above `policy["max_artifact_reference_bytes"]`, so large prediction
    pickles, NetCDF, or Parquet files can never be logged by this module,
    structurally, not just by convention. Checkpoint files are the one
    reference kind that is structurally always large: `log_checkpoint_reference`
    logs epoch/path/checksum/size/checkpoint_type metadata for one without
    ever applying that size ceiling to it, and (unlike a bare
    `log_artifact_reference` call) never raises on failure -- see the next
    bullet.
  * Best-effort after init: every call that touches a real wandb backend
    (`log_hyperparameters`/`log_scientific_metrics`/`log_resource_metrics`/
    `log_artifact_reference`/`finish_tracking_run`) always updates this
    run's local in-memory mirror first, unconditionally, then attempts the
    matching wandb call through `_guard_backend_call`, which catches any
    exception, warns once per distinct failing operation (not once per
    call, so a metric log that fails every epoch cannot flood stderr), and
    records the degradation on `TrackingRun.degraded`/
    `TrackingRun.degraded_operations` -- never lets a backend failure
    propagate into the caller (training/screening/stopping/checkpoint code
    never sees or handles a wandb exception). `TrackingRun.backend` keeps
    its original "null"/"wandb" value even once degraded -- degradation is
    a separate, explicit flag, never silently reinterpreted as "tracking
    was never attempted" or "tracking completed cleanly".
  * Stable run identity across restarts: `init_tracking_run` accepts an
    optional `run_id` (the real wandb `id=`) and `resume` (defaults to
    `"allow"` whenever `run_id` is given). Passing the same `run_id` on
    every call -- across as many separate Slurm jobs/processes as a single
    Flash-NH candidate needs -- reconnects to the same logical wandb run
    instead of starting a new one each time. This module does not itself
    decide what `run_id` to use; see `pilot_tracking.py`'s
    `derive_pilot_wandb_run_id`/`resolve_pilot_wandb_run_id` for the
    Flash-NH-specific, deterministic-from-repository-identity derivation.
    This id is never authoritative for checkpoint or scientific identity --
    it only labels the wandb side of an already-decided Flash-NH run.

Every `TrackingRun` -- whether backed by real wandb or the null sink --
keeps a full local, in-memory mirror of everything logged
(`hyperparameters`, `scientific_metrics`, `resource_metrics`,
`artifact_references`). This makes the module trivially testable without a
real wandb backend, and doubles as a natural per-run provenance record.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

__all__ = [
    "TrackingError",
    "TrackingRun",
    "load_tracking_policy",
    "init_tracking_run",
    "log_hyperparameters",
    "log_scientific_metrics",
    "log_resource_metrics",
    "log_artifact_reference",
    "log_checkpoint_reference",
    "finish_tracking_run",
]

_REQUIRED_POLICY_KEYS = [
    "policy_name",
    "enabled",
    "mode",
    "project",
    "max_artifact_reference_bytes",
]

_VALID_MODES = ("disabled", "offline", "online")

# Applied to every dict this module is asked to log, regardless of field:
# refuses to forward anything shaped like a credential.
_CREDENTIAL_KEY_FRAGMENTS = ("api_key", "apikey", "secret", "password", "token", "credential")

# Applied to scientific-metric dict keys specifically: section 2.4 requires
# the temporal test and spatial holdout sets to never enter any
# stopping/tuning-adjacent record, including tracking.
_DISALLOWED_METRIC_KEY_FRAGMENTS = ("test", "holdout", "temporal", "spatial")


class TrackingError(Exception):
    """Raised for invalid W&B tracking policy/usage."""


def _reject_credential_like_keys(d: dict, context: str) -> None:
    for k in d:
        kl = str(k).lower()
        if any(frag in kl for frag in _CREDENTIAL_KEY_FRAGMENTS):
            raise TrackingError(f"{context} contains a credential-like key {k!r} -- refusing to log")


def _reject_disallowed_metric_keys(d: dict, context: str) -> None:
    for k in d:
        kl = str(k).lower()
        if any(frag in kl for frag in _DISALLOWED_METRIC_KEY_FRAGMENTS):
            raise TrackingError(
                f"{context} contains disallowed key {k!r} -- temporal-test/spatial-holdout "
                "data must never be recorded during Stage 1 optimization (section 2.4)"
            )


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_tracking_policy(path) -> dict:
    import yaml

    p = Path(path)
    if not p.is_file():
        raise TrackingError(f"W&B tracking policy file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TrackingError(f"W&B tracking policy {p} did not parse to a mapping")
    missing = [k for k in _REQUIRED_POLICY_KEYS if k not in data]
    if missing:
        raise TrackingError(f"W&B tracking policy {p} missing required key(s): {missing}")
    if data["mode"] not in _VALID_MODES:
        raise TrackingError(f"W&B tracking policy mode must be one of {_VALID_MODES}, got {data['mode']!r}")
    if data["max_artifact_reference_bytes"] <= 0:
        raise TrackingError("max_artifact_reference_bytes must be > 0")
    return data


# ---------------------------------------------------------------------------
# Run object
# ---------------------------------------------------------------------------

class TrackingRun:
    """A tracked run: either a real-wandb-backed run or an in-memory null
    sink, depending on policy. Always keeps a full local mirror of every
    call made against it, independent of backend."""

    def __init__(
        self,
        backend: str,
        max_artifact_reference_bytes: int,
        run_identity: dict,
        wandb_run: Any = None,
        mode: str = "disabled",
        wandb_run_id: str | None = None,
    ):
        self.backend = backend  # "null" or "wandb"
        self.mode = mode  # "disabled" / "offline" / "online" -- explicit backend-mode record
        self.max_artifact_reference_bytes = max_artifact_reference_bytes
        self.run_identity = dict(run_identity)
        self.wandb_run_id = wandb_run_id
        self._wandb_run = wandb_run
        self.hyperparameters: dict | None = None
        self.scientific_metrics: list[tuple[int, dict]] = []
        self.resource_metrics: list[tuple[int, dict]] = []
        self.artifact_references: list[dict] = []
        self.finished = False
        # Best-effort telemetry bookkeeping: a failure here NEVER propagates
        # to the caller and NEVER changes `finished`/scientific state -- it
        # only ever downgrades tracking completeness, which is recorded here
        # so callers/evidence bundles can report it honestly instead of
        # silently claiming tracking succeeded.
        self.degraded = False
        self.degraded_operations: set[str] = set()


def _mark_degraded(run: "TrackingRun", operation: str, exc: BaseException) -> None:
    """Record a best-effort tracking failure on ``run`` without raising.

    Warns once per distinct ``operation`` per run (not once per call) so a
    metric log failing every epoch cannot flood stderr, and records the
    degradation on ``run.degraded``/``run.degraded_operations`` so it is
    never silently mistaken for tracking having completed cleanly.
    """
    run.degraded = True
    first_for_operation = operation not in run.degraded_operations
    run.degraded_operations.add(operation)
    if first_for_operation:
        warnings.warn(
            f"W&B tracking operation {operation!r} failed and has been downgraded to "
            f"best-effort ({exc!r}); the underlying scientific operation was NOT affected. "
            "Further failures of this same operation on this run will not be re-warned.",
            RuntimeWarning,
            stacklevel=3,
        )


def _guard_backend_call(run: "TrackingRun", operation: str, fn) -> None:
    """Best-effort wrapper around a single real-wandb backend call.

    Any exception raised by ``fn`` is caught here, never re-raised: this is
    the failure-isolation boundary that keeps W&B telemetry from ever being
    able to stop training/screening/early-stopping/checkpoint selection.
    """
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- deliberate, documented catch-all
        _mark_degraded(run, operation, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_tracking_run(
    policy: dict,
    run_identity: dict,
    run_id: str | None = None,
    resume: str | None = None,
) -> TrackingRun:
    """Start a tracked run. Never launches, configures, or is part of a
    training run or a hyperparameter sweep -- ``run_identity`` is metadata
    describing a run the caller is already conducting.

    ``run_id``, if given, is passed through as wandb's own ``id=`` so that
    calling this again later (e.g. from a later Slurm job continuing the
    same Flash-NH candidate) with the SAME ``run_id`` reconnects to the same
    logical wandb run instead of starting a new one. ``resume`` defaults to
    ``"allow"`` whenever ``run_id`` is given (wandb resumes the run if it
    exists, creates it if this is the first call) and is otherwise unused.
    This function does not decide what ``run_id`` should be -- that is the
    caller's responsibility (see ``pilot_tracking.py``).
    """
    _reject_credential_like_keys(run_identity, "run_identity")

    max_bytes = policy["max_artifact_reference_bytes"]
    mode = policy["mode"]
    if not policy.get("enabled", False) or mode == "disabled":
        return TrackingRun(
            backend="null", max_artifact_reference_bytes=max_bytes, run_identity=run_identity, mode="disabled"
        )

    try:
        import wandb
    except ImportError as exc:
        raise TrackingError(
            "policy requests W&B tracking (enabled=True, mode="
            f"{policy['mode']!r}) but the wandb package is not installed; "
            "install wandb, or set enabled: false / mode: disabled in the policy"
        ) from exc

    os.environ.setdefault("WANDB_MODE", policy["mode"])
    effective_resume = resume if resume is not None else ("allow" if run_id is not None else None)
    wandb_run = wandb.init(
        project=policy["project"],
        entity=policy.get("entity"),
        tags=list(policy.get("tags", [])),
        config=dict(run_identity),
        mode=policy["mode"],
        id=run_id,
        resume=effective_resume,
    )
    return TrackingRun(
        backend="wandb",
        max_artifact_reference_bytes=max_bytes,
        run_identity=run_identity,
        wandb_run=wandb_run,
        mode=mode,
        wandb_run_id=run_id,
    )


def log_hyperparameters(run: TrackingRun, hyperparameters: dict) -> None:
    _reject_credential_like_keys(hyperparameters, "hyperparameters")
    run.hyperparameters = dict(hyperparameters)
    if run.backend == "wandb":
        _guard_backend_call(
            run, "log_hyperparameters", lambda: run._wandb_run.config.update(hyperparameters, allow_val_change=True)
        )


def log_scientific_metrics(run: TrackingRun, epoch: int, metrics: dict) -> None:
    """Log development-validation scientific metrics for one epoch (the
    section 2.2 percentile/summary contract). ``metrics`` must never contain
    temporal-test or spatial-holdout values -- see section 2.4."""
    if not isinstance(epoch, int) or epoch < 1:
        raise TrackingError(f"epoch must be a positive int, got {epoch!r}")
    _reject_disallowed_metric_keys(metrics, "scientific metrics")
    run.scientific_metrics.append((epoch, dict(metrics)))
    if run.backend == "wandb":
        _guard_backend_call(run, "log_scientific_metrics", lambda: run._wandb_run.log(dict(metrics), step=epoch))


def log_resource_metrics(run: TrackingRun, epoch: int, metrics: dict) -> None:
    """Log training/resource metrics (wall time, GPU memory, etc.) for one
    epoch, ONLY if the caller actually captured them -- an empty/falsy
    ``metrics`` is treated as "nothing captured" and is a silent no-op,
    per section 10 ("only if actually captured")."""
    if not metrics:
        return
    if not isinstance(epoch, int) or epoch < 1:
        raise TrackingError(f"epoch must be a positive int, got {epoch!r}")
    _reject_credential_like_keys(metrics, "resource metrics")
    run.resource_metrics.append((epoch, dict(metrics)))
    if run.backend == "wandb":
        _guard_backend_call(run, "log_resource_metrics", lambda: run._wandb_run.log(dict(metrics), step=epoch))


def log_artifact_reference(run: TrackingRun, name: str, path, checksum: str) -> None:
    """Record a COMPACT artifact reference (path + checksum + size) --
    never the file's contents. Refuses any file larger than
    ``run.max_artifact_reference_bytes``, which structurally blocks logging
    prediction pickles, checkpoints, NetCDF, or Parquet files by this
    module, per section 10."""
    p = Path(path)
    if not p.is_file():
        raise TrackingError(f"artifact reference path not found: {p}")
    size_bytes = p.stat().st_size
    if size_bytes > run.max_artifact_reference_bytes:
        raise TrackingError(
            f"refusing to log artifact reference {name!r}: {size_bytes} bytes exceeds "
            f"max_artifact_reference_bytes={run.max_artifact_reference_bytes} -- this module "
            "only logs compact references (config/manifest/small-JSON), never large "
            "prediction/checkpoint/NetCDF/Parquet files"
        )
    record = {"name": name, "path": str(p), "checksum": checksum, "size_bytes": size_bytes}
    run.artifact_references.append(record)
    if run.backend == "wandb":
        _guard_backend_call(
            run, "log_artifact_reference", lambda: run._wandb_run.summary.__setitem__(f"artifact_ref/{name}", record)
        )


def log_checkpoint_reference(
    run: TrackingRun, *, epoch: int, path, checksum: str, checkpoint_type: str = "nh_model_checkpoint"
) -> None:
    """Record a compact CHECKPOINT reference (epoch + path + checksum + size
    + checkpoint_type) -- never the checkpoint file's own bytes.

    Unlike :func:`log_artifact_reference`, this never applies
    ``run.max_artifact_reference_bytes`` to the referenced file's size:
    checkpoint files are structurally always large, so a "compact
    reference" to one is expected to describe a large file, not refuse to
    record it (that size ceiling exists to catch accidentally-large
    CONTENT payloads, which this function never reads -- only
    ``Path.stat()``).

    Every failure mode here (missing checkpoint file, backend error,
    formatting error) is optional telemetry, exactly like every other
    logging call in this module: it degrades tracking
    (``run.degraded``/``run.degraded_operations``) via ``_mark_degraded``
    and returns -- it NEVER raises, so a tracking failure can never abort
    training/screening/early-stopping/checkpoint selection (this is the
    exact failure mode that killed Moriah job 45731908 when the old
    generic ``log_artifact_reference`` path was used for a checkpoint
    reference instead; see docs/stage1_lead06_pilot_v001.md).
    """
    operation = "log_checkpoint_reference"
    try:
        p = Path(path)
        size_bytes = p.stat().st_size
        record = {
            "name": f"checkpoint_epoch_{epoch:03d}",
            "path": str(p),
            "checksum": checksum,
            "size_bytes": size_bytes,
            "epoch": epoch,
            "checkpoint_type": checkpoint_type,
        }
        run.artifact_references.append(record)
        if run.backend == "wandb":
            run._wandb_run.summary[f"checkpoint_ref/epoch_{epoch:03d}"] = record
    except Exception as exc:  # noqa: BLE001 -- optional telemetry, see docstring above
        _mark_degraded(run, operation, exc)


def finish_tracking_run(run: TrackingRun) -> None:
    run.finished = True
    if run.backend == "wandb":
        _guard_backend_call(run, "finish_tracking_run", lambda: run._wandb_run.finish())
