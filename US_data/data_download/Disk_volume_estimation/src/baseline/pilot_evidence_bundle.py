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
:func:`src.baseline.wandb_tracking.log_artifact_reference` already enforces
one layer up.
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
    """List checkpoint files under ``nh_run_dir`` by name/size/checksum only
    -- never copies or reads their contents beyond hashing. Matches NH's own
    ``model_epoch###.pt`` naming convention."""
    nh_run_dir = Path(nh_run_dir)
    inventory = []
    for ckpt_path in sorted(nh_run_dir.glob("model_epoch*.pt")):
        inventory.append(
            {
                "filename": ckpt_path.name,
                "size_bytes": ckpt_path.stat().st_size,
                "sha256": sha256_of(ckpt_path),
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
    force: bool = False,
) -> Path:
    """Write this run's compact evidence bundle to ``out_dir`` and return it.

    ``screening_events`` must be the list of
    :func:`src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`
    return dicts accumulated over the run (every entry's ``scope`` is
    checked here and must equal ``SCREENING_METRIC_SCOPE`` -- an
    authoritative full-validation result belongs in a separate, later
    artifact, never mixed into this pilot-screening bundle).
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
        "run_status": run_status,
        "run_identity": dict(tracking_run.run_identity),
        "hyperparameters": tracking_run.hyperparameters,
        "wandb": {
            "backend": tracking_run.backend,
            "finished": tracking_run.finished,
        },
        "slurm_identity": dict(slurm_identity) if slurm_identity else None,
        "epoch_timing_table": epoch_timing_table,
        "screening_events": screening_events,
        "early_stopping_state": early_stopping_state,
        "best_checkpoint_epoch": best_epoch,
        "checkpoint_inventory": checkpoint_inventory,
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
