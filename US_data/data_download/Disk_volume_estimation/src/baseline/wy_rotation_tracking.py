"""W&B tracking for the water-year rotation campaign: stage A (training side).

Design v2 (adopted by Omri 2026-11-16, ``.scratch_local/wy_rotation_runs_v001/
WANDB_DESIGN_DRAFT.md``): one W&B run per Flash-NH rotation run, reconnected by
every later stage through a deterministic id. Stages open in a fixed order:

* A (this module, open): NH's per-epoch training loss (a training diagnostic in
  normalized space, never a skill score), epoch wall time, evaluation timing,
  peak memory, receipt status, the explained-missing basin list + counts, and
  the run's small evidence files.
* B (checkpoint-year spec metrics), C (assessment year at the picks) and
  D (WY2026) are closed here: their key prefixes raise at the call site. They
  get their own gated logger later.

Flash-NH evidence stays authoritative; W&B only mirrors. Every backend call is
best-effort (``wandb_tracking._guard_backend_call``): a W&B failure, including
``wandb.init`` itself, degrades tracking and is recorded in the receipt, and
never stops training or evaluation. Gate violations (a closed or unknown key
prefix, a score-like key fragment) are programming errors and raise.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

from .wandb_tracking import (
    TrackingError,
    TrackingRun,
    _guard_backend_call,
    _mark_degraded,
    _reject_credential_like_keys,
)

# Metric-key prefixes per stage. Only stage A is open in this build.
STAGE_PREFIXES = {
    "A": ("train/", "evalrun/", "resource/", "verify/", "status/"),
    "B": ("ckpt/",),
    "C": ("assess/",),
    "D": ("wy2026/",),
}
OPEN_STAGES = frozenset({"A"})

# No stage-A key may look like a score or name a sealed/evaluation scope.
# "loss" is allowed only under train/ (NH's normalized-space training loss).
FORBIDDEN_KEY_FRAGMENTS = ("nse", "kge", "mse", "mae", "mape", "pbias", "pearson", "alpha", "beta", "fhv",
                           "fms", "flv", "missed", "timing", "q98", "flood", "metric", "valid", "test",
                           "holdout", "spatial", "temporal", "skill", "score", "assess", "wy2026")

# NH logs through module loggers under "neuralhydrology"; the live loss handler
# attaches there, never to the root logger (a root handler would make NH's
# logging.basicConfig a no-op and silence output.log).
NH_LOGGER_NAME = "neuralhydrology"
_MAGNITUDE_LAMBDA_KEY = "flashnh_magnitude_weight_lambda"

# Frozen-config keys mirrored into the W&B config (hyperparameters only, no paths).
CONFIG_KEYS = ("model", "head", "hidden_size", "seq_length", "predict_last_n", "batch_size", "learning_rate",
               "optimizer", "loss", "output_dropout", "output_activation", "initial_forget_bias",
               "clip_gradient_norm", "epochs", "max_updates_per_epoch", "seed", "dynamic_inputs",
               "static_attributes", "target_variables", "statics_embedding", "dynamics_embedding",
               "flashnh_rotation_cycle", "flashnh_magnitude_weight_lambda")

# File names W&B writes itself in a run's files/ dir. ``run.save`` links an
# uploaded file into files/<relative path>; a Flash-NH file uploaded under one
# of these names gets overwritten by W&B through that link (smoke job 46762307:
# the canary's frozen config.yaml was replaced by W&B's own config at finish).
WANDB_RESERVED_FILE_NAMES = frozenset({"config.yaml", "wandb-summary.json", "wandb-metadata.json",
                                       "requirements.txt", "output.log", "diff.patch", "conda-environment.yaml"})
# Every Flash-NH file is uploaded as a private copy under this sub-path of the
# run's Files (e.g. ``flashnh/config.yaml``); the frozen original is never linked.
UPLOAD_SUBDIR = "flashnh"
UPLOAD_STAGING_DIRNAME = "flashnh_upload_staging"
# wandb.Settings of every rotation run. console="off": stdout/stderr are not
# mirrored, so only the gated keys, the run config and the uploaded small files
# reach W&B. x_disable_meta=True turns off W&B's system-metadata block: the
# program path and args, the Slurm env and the disk usage are no longer sent.
# It does NOT hide everything: in wandb 0.28.1 offline, the run record still
# carries the host name, the git remote URL + HEAD commit, the python version,
# the platform, and files/requirements.txt.
# The Slurm job/array ids are put back through the run config (slurm_ids).
# x_* settings are W&B-private API: the real-package smoke test pins what is
# dropped (program args, Slurm env) and what is kept (the Slurm ids in config).
WANDB_SETTINGS = {"console": "off", "x_disable_meta": True}

_EPOCH_LOSS_PATTERN = re.compile(r"^Epoch (\d+) average loss: (.*)$")
_LOG_LINE_PATTERN = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+: Epoch (\d+) average loss: (.*)$")
_LOSS_ITEM_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*): ([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))")
_LOSS_KEY_SAFE = re.compile(r"[^A-Za-z0-9_]")


def check_upload_name(upload_name: str) -> str:
    """Return ``upload_name`` (a relative POSIX path in the run's Files) or
    raise if W&B owns that name or the path is not a plain relative one."""
    parts = upload_name.split("/")
    if not upload_name or upload_name.startswith("/") or "\\" in upload_name or any(
            p in ("", ".", "..") for p in parts):
        raise TrackingError(f"upload name {upload_name!r} is not a plain relative POSIX path")
    if upload_name in WANDB_RESERVED_FILE_NAMES or parts[0] == "wandb" or parts[0] == "media":
        raise TrackingError(f"upload name {upload_name!r} is reserved by W&B in a run's files/ dir")
    return upload_name


def wandb_run_id(flashnh_run_id: str) -> str:
    """Deterministic W&B run id for a Flash-NH run id (stable across jobs and stages)."""
    return "wyrot-" + hashlib.sha256(flashnh_run_id.encode("utf-8")).hexdigest()[:24]


def _arm(config: Mapping[str, Any]) -> tuple[str, float]:
    unknown = sorted(k for k in config if "magnitude" in str(k).lower() and k != _MAGNITUDE_LAMBDA_KEY)
    if unknown:  # an arm-2 config under another key name would otherwise be filed silently as "base"
        raise TrackingError(f"unrecognised magnitude-weight config key(s) {unknown}; the arm name reads only "
                            f"{_MAGNITUDE_LAMBDA_KEY!r}")
    lam = config.get(_MAGNITUDE_LAMBDA_KEY)
    if lam in (None, 0, 0.0):
        return "base", 0.0
    return f"magw-l{float(lam):g}", float(lam)


def run_layout(*, cycle: int, config: Mapping[str, Any], is_canary: bool) -> dict[str, str]:
    """W&B name / group / job_type of a rotation run (design v2 layout)."""
    arm, _ = _arm(config)
    if is_canary:
        return {"name": f"canary-c{cycle}-{arm}", "group": "canary", "job_type": "canary"}
    return {"name": f"c{cycle}-{arm}", "group": arm, "job_type": "train"}


SLURM_ID_ENV = (("slurm_job_id", "SLURM_JOB_ID"), ("slurm_array_job_id", "SLURM_ARRAY_JOB_ID"),
                ("slurm_array_task_id", "SLURM_ARRAY_TASK_ID"))


def slurm_ids(environ: Mapping[str, str]) -> dict[str, str]:
    """The Slurm job id and, when present, the array job/task ids. x_disable_meta keeps
    W&B from collecting the Slurm env, so these go into the run config to show which
    Moriah job a run came from. Unset or empty variables are left out.

    The W&B config is NOT the authority on which job produced a result: on a
    ``--resume-from-job`` run the deterministic id reattaches to the earlier W&B run,
    and wandb 0.28.1 (``Run._set_run_obj``) lays the server's stored config over
    ``config=`` with allow_val_change, so these keys may still show the earlier job.
    The receipt and the summary's ``status/slurm_job_id`` are the authority; a resumed
    run also gets ``status/resumed_from_job`` and ``status/resume_slurm_job_id``
    (:meth:`RotationTracker.log_receipt_summary`)."""
    return {key: str(environ[var]) for key, var in SLURM_ID_ENV if environ.get(var)}


def run_config(*, row: Mapping[str, Any], campaign_row: Mapping[str, Any], config: Mapping[str, Any],
               campaign: Mapping[str, Any], manifest_sha256: str, provenance: Mapping[str, Any],
               slurm: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """The W&B config of a rotation run: identity, years, frozen hyperparameters and
    the Slurm ids of the job (``slurm``, from :func:`slurm_ids`)."""
    arm, lam = _arm(config)
    hyper = {k: config[k] for k in CONFIG_KEYS if k in config}
    out = {
        "flashnh_run_id": row["run_id"],
        "campaign_id": campaign.get("campaign_id"),
        "cycle": int(row["cycle"]),
        "arm": arm,
        "lambda": lam,
        "is_canary": bool(row.get("is_canary")),
        "train_wys": list(campaign_row.get("train_wys", [])),
        "checkpoint_wy": campaign_row.get("checkpoint_wy"),
        "assessment_wy": campaign_row.get("assessment_wy"),
        "split_contract_id": campaign["split_contract"].get("contract_id"),
        "split_contract_sha256": campaign["split_contract"].get("file_sha256"),
        "config_sha256": row.get("config_sha256") or campaign_row.get("config_sha256"),
        "manifest_sha256": manifest_sha256,
        "hyperparameters": json.loads(json.dumps(hyper, default=str)),
        **{k: provenance[k] for k in ("commit", "git_commit", "runtime_python") if k in provenance},
        **{key: str(slurm[key]) for key, _ in SLURM_ID_ENV if slurm and slurm.get(key)},
    }
    _reject_credential_like_keys(out, "rotation run config")
    return out


def check_metric_keys(metrics: Mapping[str, Any]) -> None:
    """Raise unless every key belongs to an OPEN stage's prefix and is score-free."""
    for key in metrics:
        if key == "epoch":
            continue
        stage = next((s for s, prefixes in STAGE_PREFIXES.items() if key.startswith(prefixes)), None)
        if stage is None:
            raise TrackingError(f"metric key {key!r} has no known stage prefix")
        if stage not in OPEN_STAGES:
            raise TrackingError(f"metric key {key!r} belongs to stage {stage}, which is closed in this logger")
        low = key.lower()
        bad = [f for f in FORBIDDEN_KEY_FRAGMENTS if f in low]
        if bad:
            raise TrackingError(f"metric key {key!r} contains forbidden fragment(s) {bad}")
        if "loss" in low and not key.startswith("train/"):
            raise TrackingError(f"loss key {key!r} outside train/")
    _reject_credential_like_keys(dict(metrics), "rotation metrics")


def parse_loss_string(loss_str: str) -> dict[str, float]:
    """``"avg_total_loss: 0.12345, avg_x: 1.0"`` -> ``{"train/avg_total_loss": 0.12345, ...}``."""
    out = {}
    for name, value in _LOSS_ITEM_PATTERN.findall(loss_str):
        out["train/" + _LOSS_KEY_SAFE.sub("_", name)] = float(value)
    return out


def parse_output_log(output_log) -> dict[int, dict[str, Any]]:
    """Per-epoch training loss + wall-clock end time from NH's ``output.log``."""
    out: dict[int, dict[str, Any]] = {}
    path = Path(output_log)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LOG_LINE_PATTERN.match(line)
        if m:
            out[int(m.group(2))] = {"end": m.group(1), "losses": parse_loss_string(m.group(3))}
    return out


class RotationTracker:
    """Stage-A tracker of one rotation run. ``run`` is a ``TrackingRun``
    (backend "null" when tracking is off or ``wandb.init`` failed)."""

    def __init__(self, run: TrackingRun, *, layout: Mapping[str, str], upload_staging_dir=None):
        self.run = run
        self.layout = dict(layout)
        self.upload_staging_dir = Path(upload_staging_dir) if upload_staging_dir is not None else None
        self.epochs_logged: set[int] = set()
        self._last_epoch_created: Optional[float] = None
        self._handler: Optional[logging.Handler] = None
        self._handler_logger: Optional[logging.Logger] = None
        self.root_logging_restored = False

    # ---- generic, gated logging -------------------------------------------------
    def log(self, metrics: Mapping[str, Any], *, epoch: Optional[int] = None) -> None:
        payload = dict(metrics)
        if epoch is not None:
            if not isinstance(epoch, int) or epoch < 1:
                raise TrackingError(f"epoch must be a positive int, got {epoch!r}")
            payload["epoch"] = epoch
        check_metric_keys(payload)
        self.run.resource_metrics.append((epoch or 0, dict(payload)))
        if self.run.backend == "wandb":
            _guard_backend_call(self.run, "rotation_log", lambda: self.run._wandb_run.log(payload))

    def set_summary(self, values: Mapping[str, Any]) -> None:
        check_metric_keys(values)
        if self.run.backend == "wandb":
            def _set():
                for k, v in values.items():
                    self.run._wandb_run.summary[k] = v
            _guard_backend_call(self.run, "rotation_summary", _set)

    def save_file(self, path, *, max_bytes: Optional[int] = None) -> None:
        """Upload one small evidence file (receipt, evidence JSON, config) to the
        run's Files as ``flashnh/<name>``.

        W&B links a saved file into its files/ dir and may later write through
        that link, so the upload is always a private byte copy in the staging
        dir, never the Flash-NH original."""
        p = Path(path)
        upload_name = check_upload_name(f"{UPLOAD_SUBDIR}/{p.name}")  # a programming error raises
        limit = max_bytes or self.run.max_artifact_reference_bytes
        if not p.is_file() or p.stat().st_size > limit:
            _mark_degraded(self.run, "rotation_save_file",
                           TrackingError(f"{p} missing or larger than {limit} bytes; not uploaded"))
            return
        self.run.artifact_references.append({"name": upload_name, "path": str(p), "size_bytes": p.stat().st_size,
                                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        if self.run.backend != "wandb":
            return
        if self.upload_staging_dir is None:
            _mark_degraded(self.run, "rotation_save_file",
                           TrackingError("no upload staging dir; refusing to link the original file"))
            return

        def _copy_and_save():
            staging = self.upload_staging_dir
            copy = staging / upload_name
            if copy.resolve() == p.resolve():
                raise TrackingError(f"staging copy {copy} would be the original file")
            copy.parent.mkdir(parents=True, exist_ok=True)
            if copy.is_symlink():
                copy.unlink()
            shutil.copyfile(p, copy)
            self.run._wandb_run.save(str(copy), base_path=str(staging), policy="now")

        _guard_backend_call(self.run, "rotation_save_file", _copy_and_save)

    # ---- training loss --------------------------------------------------------------
    def _already_logged_upto(self) -> int:
        """Last epoch a previous job logged to this (resumed) W&B run, else 0."""
        if self.run.backend != "wandb":
            return 0
        try:
            value = self.run._wandb_run.summary.get("train/last_epoch_logged")
            return int(value) if value is not None else 0
        except Exception:  # noqa: BLE001 -- best effort; worst case a duplicate point
            return 0

    def log_epoch_loss(self, epoch: int, losses: Mapping[str, float], *, epoch_wall_s: Optional[float] = None
                       ) -> None:
        if epoch in self.epochs_logged:
            return
        metrics = {}
        for k, v in losses.items():  # loss names come from NH's log text: drop (never send) a key the gate refuses
            if isinstance(v, float) and math.isnan(v):
                continue
            try:
                check_metric_keys({k: v})
            except TrackingError as exc:
                _mark_degraded(self.run, "rotation_loss_key_dropped", exc)
                continue
            metrics[k] = v
        if epoch_wall_s is not None:
            metrics["train/epoch_wall_s"] = round(epoch_wall_s, 1)
        self.log(metrics, epoch=epoch)
        self.epochs_logged.add(epoch)
        self.set_summary({"train/last_epoch_logged": epoch})

    def attach_live_loss_handler(self, logger: Optional[logging.Logger] = None) -> None:
        """Log ``Epoch N average loss`` as NH emits it (in-process, no file tailing)."""
        tracker = self

        class _Handler(logging.Handler):
            def emit(self, record):  # never raises into NH's training loop
                try:
                    m = _EPOCH_LOSS_PATTERN.match(record.getMessage())
                    if not m:
                        return
                    wall = (record.created - tracker._last_epoch_created
                            if tracker._last_epoch_created is not None else None)
                    tracker._last_epoch_created = record.created
                    tracker.log_epoch_loss(int(m.group(1)), parse_loss_string(m.group(2)), epoch_wall_s=wall)
                except Exception as exc:  # noqa: BLE001
                    _mark_degraded(tracker.run, "rotation_live_loss", exc)

        import time as _time

        self._last_epoch_created = _time.time()  # epoch 1's wall time includes NH's data loading
        self._handler = _Handler(level=logging.INFO)
        self._handler_logger = logger or logging.getLogger(NH_LOGGER_NAME)
        self._handler_logger.addHandler(self._handler)

    def detach_live_loss_handler(self) -> None:
        if self._handler is not None:
            self._handler_logger.removeHandler(self._handler)
            self._handler = None

    def backfill_from_output_log(self, output_log) -> None:
        """Log epochs found in ``output.log`` that neither this job nor an earlier
        job of the same W&B run logged (restart, or a run trained untracked)."""
        upto = self._already_logged_upto()
        parsed = parse_output_log(output_log)
        prev_end = None
        import datetime as _dt

        for epoch in sorted(parsed):
            end = _dt.datetime.strptime(parsed[epoch]["end"], "%Y-%m-%d %H:%M:%S")
            wall = (end - prev_end).total_seconds() if prev_end is not None else None
            prev_end = end
            if epoch > upto:
                self.log_epoch_loss(epoch, parsed[epoch]["losses"], epoch_wall_s=wall)

    # ---- evaluation / receipt -------------------------------------------------------
    def log_evaluation_records(self, records) -> None:
        for rec in records:
            if rec.get("skipped_existing"):
                continue
            self.log({"evalrun/setup_s": rec.get("tester_setup_s"),
                      "evalrun/evaluate_s": rec.get("evaluate_s")}, epoch=int(rec["epoch"]))

    def log_receipt_summary(self, receipt: Mapping[str, Any], evidence: Optional[Mapping[str, Any]] = None) -> None:
        summary: dict[str, Any] = {"status/receipt": receipt.get("status"),
                                   "status/trained_in_this_job": receipt.get("trained_in_this_job")}
        for key in ("train_wall_s", "evaluation_wall_s"):
            if receipt.get(key) is not None:
                summary[f"resource/{key}"] = receipt[key]
        for phase in ("train_peak_memory", "evaluation_peak_memory"):
            for k, v in (receipt.get(phase) or {}).items():
                summary[f"resource/{phase}_{k}"] = v
        if receipt.get("resumed_from_job") is not None:
            # a resume: W&B reloads the earlier job's summary, so always overwrite this
            # job's id ("none" without one) and show both the earlier and this job
            summary["status/slurm_job_id"] = str(receipt.get("slurm_job_id") or "none")
            summary["status/resumed_from_job"] = str(receipt["resumed_from_job"])
            summary["status/resume_slurm_job_id"] = str(receipt.get("slurm_job_id") or "none")
        elif receipt.get("slurm_job_id"):  # this job's id: the authority (the config may keep an earlier job's)
            summary["status/slurm_job_id"] = str(receipt["slurm_job_id"])
        if receipt.get("host"):
            summary["status/host"] = receipt["host"]
        if evidence is not None:
            summary["verify/passed"] = bool(evidence.get("passed"))
            summary["verify/n_basins_evaluated"] = evidence.get("n_basins_evaluated")
            summary["verify/n_basins_missing_explained"] = len(evidence.get("missing_basins") or [])
            summary["verify/missing_basins"] = list(evidence.get("missing_basins") or [])
            summary["verify/missing_basin_admissible_target_counts"] = dict(
                evidence.get("missing_basin_admissible_target_counts") or {})
        self.set_summary(summary)

    def receipt_record(self) -> dict[str, Any]:
        return {"mode": self.run.mode, "backend": self.run.backend, "id": self.run.wandb_run_id,
                **self.layout, "degraded": self.run.degraded, "root_logging_restored": self.root_logging_restored,
                "degraded_operations": sorted(self.run.degraded_operations), "finished": self.run.finished}

    def finish(self, *, failed: bool) -> None:
        self.detach_live_loss_handler()
        self.run.finished = True
        if self.run.backend == "wandb":
            _guard_backend_call(self.run, "rotation_finish",
                                lambda: self.run._wandb_run.finish(exit_code=1 if failed else 0))


def init_rotation_tracker(policy: Mapping[str, Any], *, flashnh_run_id: str, config: Mapping[str, Any],
                          layout: Mapping[str, str], wandb_dir) -> RotationTracker:
    """Start (or resume) the W&B run of one rotation run. Never raises for a
    W&B failure: a failed import/init returns a degraded null-backed tracker."""
    _reject_credential_like_keys(dict(config), "rotation run config")
    max_bytes = int(policy["max_artifact_reference_bytes"])
    run_id = wandb_run_id(flashnh_run_id)
    if not policy.get("enabled", False) or policy["mode"] == "disabled":
        return RotationTracker(TrackingRun(backend="null", max_artifact_reference_bytes=max_bytes,
                                           run_identity=dict(config), mode="disabled", wandb_run_id=run_id),
                               layout=layout)
    null_run = TrackingRun(backend="null", max_artifact_reference_bytes=max_bytes, run_identity=dict(config),
                           mode=policy["mode"], wandb_run_id=run_id)
    root = logging.getLogger()
    root_handlers, root_level = list(root.handlers), root.level
    wandb_run = None
    try:
        import wandb

        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(project=policy["project"], entity=policy.get("entity"),
                               tags=list(policy.get("tags", [])), config=dict(config), mode=policy["mode"],
                               id=run_id, resume="allow", dir=str(wandb_dir),
                               settings=wandb.Settings(**WANDB_SETTINGS), **dict(layout))
        wandb_run.define_metric("epoch")
        for prefix in ("train", "evalrun"):
            wandb_run.define_metric(f"{prefix}/*", step_metric="epoch")
    except Exception as exc:  # noqa: BLE001 -- tracking never stops the scientific run
        _mark_degraded(null_run, "rotation_init", exc)
        if wandb_run is not None:  # init worked, the setup after it failed: close that run
            # as failed now, instead of W&B's exit hook later closing it with exit code 0
            _guard_backend_call(null_run, "rotation_init_cleanup", lambda: wandb_run.finish(exit_code=1))
        tracker = RotationTracker(null_run, layout=layout)
    else:
        tracker = RotationTracker(TrackingRun(backend="wandb", max_artifact_reference_bytes=max_bytes,
                                              run_identity=dict(config), wandb_run=wandb_run, mode=policy["mode"],
                                              wandb_run_id=run_id), layout=layout,
                                  upload_staging_dir=Path(wandb_dir) / UPLOAD_STAGING_DIRNAME)
    # NH's setup_logging uses logging.basicConfig, a no-op once the root logger
    # has any handler: undo any root-logger change made during init.
    if list(root.handlers) != root_handlers or root.level != root_level:
        root.handlers[:] = root_handlers
        root.setLevel(root_level)
        tracker.root_logging_restored = True
    return tracker
