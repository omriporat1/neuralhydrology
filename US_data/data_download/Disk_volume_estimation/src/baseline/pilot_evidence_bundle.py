"""Stage 1 lead-6 optimization pilot: compact, transferable evidence bundle
(task item 8).

Assembles one run's evidence into a small, self-contained directory that can
be copied to a local review machine without any large training artifact:
resolved config + generation manifest (copied verbatim, both already small
text/JSON files), git commit+dirty-state, package/split/static identities,
seed+architecture spec, W&B run identity (whether or not tracking was
actually enabled), Slurm identity+accounting, an epoch timing table and
screening-event history built from
:mod:`src.baseline.pilot_tracking`'s ``TrackingRun`` local mirror (the same
object real training/screening code already populates -- no second logging
path), the restart-safe early-stopping state/history from
:mod:`src.baseline.pilot_early_stopping`, a checkpoint INVENTORY (name, size,
checksum, epoch -- never the checkpoint bytes themselves), the exact commands
used, a checksums manifest over every file this module writes, and an
explicit sealed-set non-access statement.

This module writes only small JSON/text files. It refuses to copy or hash
anything above ``MAX_COPIED_FILE_BYTES`` into the bundle body (checkpoints /
NetCDF / Parquet / prediction pickles are referenced by path + checksum only,
via :func:`_checkpoint_inventory`, never copied) -- the same "compact
reference only" discipline
:func:`src.baseline.wandb_tracking.log_artifact_reference`/
:func:`src.baseline.wandb_tracking.log_checkpoint_reference` already enforce
one layer up. The ``run_record["wandb"]`` section below is read straight off
the live ``TrackingRun`` (``degraded``/``degraded_operations``/``finished``),
so a tracking failure (e.g. a checkpoint-reference error) is always reported
honestly here, never masked as a clean finish.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .early_stopping import best_checkpoint_epoch
from .pilot_lead06_config import PilotPolicy, PilotRunSpec
from .pilot_screening_eval import SCREENING_METRIC_SCOPE
from .splits import sha256_of
from .wandb_tracking import TrackingRun

__all__ = [
    "PilotEvidenceBundleError",
    "MAX_COPIED_FILE_BYTES",
    "SEALED_SET_NON_ACCESS_STATEMENT",
    "write_pilot_evidence_bundle",
]

# Generous for config.yaml / generation_manifest.json (both are small
# text/JSON files, typically well under 100 KB); a hard structural ceiling
# so an accidental large-file path can never silently bloat a "compact"
# bundle -- mirrors wandb_tracking.py's own artifact-reference size ceiling.
MAX_COPIED_FILE_BYTES = 5 * 1024 * 1024

SEALED_SET_NON_ACCESS_STATEMENT = (
    "This evidence bundle was produced entirely from the development "
    "population's training and screening-subset/full-population VALIDATION "
    "period. This pilot task never accessed, evaluated, or logged any "
    "temporal-test-period prediction, any spatial-holdout basin, or any "
    "California basin -- see docs/stage1_lead06_pilot_v001.md's binding "
    "scope constraints. No file in this bundle contains sealed-set data."
)


class PilotEvidenceBundleError(Exception):
    """Raised for an invalid or unsafe evidence-bundle write request."""


def _checkpoint_inventory(nh_run_dir) -> list:
    """List every physical checkpoint file under ``nh_run_dir`` -- base
    directory AND any nested ``continue_training_from_epoch###`` continuation
    directory (see :mod:`src.baseline.pilot_orchestration`'s
    continuation-epoch-semantics note) -- by name/relative path/size/checksum
    only, never copying or reading their contents beyond hashing. Uses
    :func:`src.baseline.pilot_orchestration.discover_physical_checkpoints` as
    the single canonical cross-directory resolver (imported locally to avoid
    a module-level import cycle -- ``pilot_orchestration`` itself imports
    :func:`write_pilot_evidence_bundle` from this module)."""
    from .pilot_orchestration import discover_physical_checkpoints

    nh_run_dir = Path(nh_run_dir)
    inventory = []
    for epoch, ckpt in sorted(discover_physical_checkpoints(nh_run_dir).items()):
        inventory.append(
            {
                "epoch": epoch,
                "filename": ckpt.path.name,
                "relative_path": str(ckpt.path.relative_to(nh_run_dir)),
                "owning_run_dir": str(ckpt.owning_run_dir),
                "size_bytes": ckpt.path.stat().st_size,
                "sha256": sha256_of(ckpt.path),
            }
        )
    return inventory


def _copy_small_file(src, dest_dir) -> "dict | None":
    src = Path(src)
    if not src.is_file():
        return None
    size_bytes = src.stat().st_size
    if size_bytes > MAX_COPIED_FILE_BYTES:
        raise PilotEvidenceBundleError(
            f"refusing to copy {src} into the evidence bundle: {size_bytes} bytes exceeds "
            f"MAX_COPIED_FILE_BYTES={MAX_COPIED_FILE_BYTES} -- this bundle only carries compact "
            "config/manifest files, never large training artifacts"
        )
    dest = Path(dest_dir) / src.name
    shutil.copyfile(src, dest)
    return {"filename": src.name, "size_bytes": size_bytes, "sha256": sha256_of(src)}


def write_pilot_evidence_bundle(
    *,
    out_dir,
    config_dir,
    nh_run_dir,
    pilot_policy: PilotPolicy,
    run_spec: PilotRunSpec,
    tracking_run: TrackingRun,
    early_stopping_state: dict,
    screening_events: list,
    run_status: str,
    commands_used: "list[str]",
    slurm_identity: "dict | None" = None,
    continuation_status: "dict | None" = None,
    actual_optimizer_updates_by_epoch: "dict[int, int] | None" = None,
    force: bool = False,
) -> Path:
    """Write this run's compact evidence bundle to ``out_dir`` and return it.

    ``screening_events`` must be the list of
    :func:`src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`
    return dicts accumulated over the run (every entry's ``scope`` is
    checked here and must equal ``SCREENING_METRIC_SCOPE`` -- an
    authoritative full-validation result belongs in a separate, later
    artifact, never mixed into this pilot-screening bundle).

    ``continuation_status``, when given, is
    :func:`src.baseline.pilot_orchestration.compute_pilot_status_fields`'s
    return dict, recorded verbatim so this bundle separately and explicitly
    distinguishes the highest PHYSICAL checkpoint epoch from the highest
    logically-screened epoch and any untrusted overshoot epochs -- never
    letting ``checkpoint_inventory`` containing epoch 15 be misread as
    "screening reached epoch 15" (see ``pilot_orchestration.py``'s
    continuation-epoch-semantics note).

    ``actual_optimizer_updates_by_epoch``, when given, is
    :func:`src.baseline.pilot_orchestration.actual_optimizer_updates_by_epoch`'s
    return dict -- the exact cumulative ``optimizer.step()`` count read
    straight from each epoch's real, unconditionally-saved PyTorch optimizer
    state (see that function's docstring for why this is exact evidence, not
    an inference). Left at its default ``None`` for every existing
    local/test caller (all of which use byte-content fake checkpoint files,
    never real torch state); a real Moriah launch computes it explicitly
    against the real run directory and passes it in. Recorded verbatim
    (never re-derived here) so a capped-fidelity candidate's actual
    updates-per-epoch can be verified against its configured
    ``max_updates_per_epoch`` without re-reading any checkpoint file a
    second time.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise PilotEvidenceBundleError(f"evidence bundle output directory already exists and is non-empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for event in screening_events:
        if event["scope"] != SCREENING_METRIC_SCOPE:
            raise PilotEvidenceBundleError(
                f"screening_events contains a non-screening-scope entry (scope={event['scope']!r}) -- "
                "this bundle only carries provisional screening-subset evidence"
            )

    config_dir = Path(config_dir)
    copied_files = []
    for name in ("config.yaml", "generation_manifest.json"):
        record = _copy_small_file(config_dir / name, out_dir)
        if record is not None:
            copied_files.append(record)

    checkpoint_inventory = _checkpoint_inventory(nh_run_dir)
    best_epoch = best_checkpoint_epoch(early_stopping_state)

    epoch_timing_table = [
        {"epoch": epoch, **metrics} for epoch, metrics in tracking_run.resource_metrics
    ]

    run_record = {
        "schema_name": "stage1_lead06_pilot_evidence_bundle",
        "schema_version": 1,
        "pilot_policy_name": pilot_policy.raw.get("policy_name"),
        "pilot_policy_sha256": pilot_policy.sha256,
        "run_id": run_spec.run_id,
        "run_profile_name": run_spec.run_profile_name,
        "static_pathway": run_spec.static_pathway,
        "embedding_hiddens": run_spec.embedding_hiddens,
        "seed_name": run_spec.seed_name,
        "seed": run_spec.seed,
        "max_updates_per_epoch": run_spec.max_updates_per_epoch,
        "run_status": run_status,
        "run_identity": dict(tracking_run.run_identity),
        "hyperparameters": tracking_run.hyperparameters,
        "wandb": {
            "backend": tracking_run.backend,
            "mode": tracking_run.mode,
            "wandb_run_id": tracking_run.wandb_run_id,
            "finished": tracking_run.finished,
            "degraded": tracking_run.degraded,
            "degraded_operations": sorted(tracking_run.degraded_operations),
        },
        "slurm_identity": dict(slurm_identity) if slurm_identity else None,
        "epoch_timing_table": epoch_timing_table,
        "screening_events": screening_events,
        "early_stopping_state": early_stopping_state,
        "best_checkpoint_epoch": best_epoch,
        "checkpoint_inventory": checkpoint_inventory,
        "continuation_status": continuation_status,
        "actual_optimizer_updates_by_epoch": actual_optimizer_updates_by_epoch,
        "artifact_references": tracking_run.artifact_references,
        "commands_used": list(commands_used),
        "sealed_set_non_access_statement": SEALED_SET_NON_ACCESS_STATEMENT,
    }

    run_record_path = out_dir / "pilot_run_evidence.json"
    run_record_path.write_text(json.dumps(run_record, indent=2, default=str), encoding="utf-8")

    checksums = {rec["filename"]: rec["sha256"] for rec in copied_files}
    checksums["pilot_run_evidence.json"] = sha256_of(run_record_path)
    checksums_path = out_dir / "checksums.json"
    checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")

    return out_dir
