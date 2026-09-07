"""SHARED-A5 Workstream C: tracked canonical scoring / orchestration path.

The minimum reusable tracked path from

    one completed audit NH evaluation run (staged + evaluated by
    :func:`devpop_audit_eval_run_producer.prepare_devpop_audit_eval_run_dir`,
    then run through NeuralHydrology out of band)

to

    one canonical development-population Common-120 audit row
    (:func:`devpop_common120_audit_evaluator.evaluate_devpop_common120_audit_row`)
    plus a small decision-relevant driver receipt.

This module adds **no** scientific math and **no** second evaluator.  It is a
thin binding layer that:

* re-validates the frozen seven-entry selection manifest and selects exactly
  one entry by ``trial_id`` -- the scoring call is bound to that entry, never
  to P1 constants or a caller-chosen epoch;
* (optionally) runs the Workstream-A per-entry preflight against the staged
  run's generated config + eval-run manifest;
* reuses the already-qualified authoritative helpers verbatim:
  :func:`selection_manifest_entry_to_checkpoint_identity`,
  :func:`build_devpop_audit_provenance_receipt`,
  :func:`evaluate_devpop_common120_audit_row` (which itself routes canonical
  contract validation, canonical population validation, provenance-receipt
  construction/verification, optimized single-row evaluation and canonical
  completeness validation);
* fails closed if the emitted row does not correspond *exactly* to the
  intended frozen entry / population / epoch / checkpoint.

It runs no inference and contacts no remote service.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from .devpop_audit_eval_run_producer import AUDIT_EVAL_RUN_MANIFEST_FILENAME
from .devpop_audit_preflight import (
    DevpopAuditPreflightError,
    _select_manifest_entry,
    preflight_devpop_audit_entry,
)
from .devpop_audit_selection_manifest import (
    DevpopAuditSelectionManifestError,
    selection_manifest_entry_to_checkpoint_identity,
)
from .devpop_common120_audit_contract import (
    AUDIT_PERIOD_NAME,
    ExpectedPopulationSpec,
)
from .devpop_common120_audit_evaluator import (
    DevpopAuditEvaluatorError,
    build_devpop_audit_provenance_receipt,
    evaluate_devpop_common120_audit_row,
)

__all__ = [
    "DevpopAuditScoringError",
    "score_devpop_audit_entry",
]


class DevpopAuditScoringError(ValueError):
    """Raised when the scoring driver cannot bind the intended frozen entry to
    an audit row, or when the emitted row does not correspond exactly to that
    entry / population / epoch / checkpoint.  Never raised for an ordinary
    poor-skill outcome (that surfaces as the row's metrics)."""


def _load_eval_run_manifest(run_dir: Path) -> dict:
    import json

    p = run_dir / AUDIT_EVAL_RUN_MANIFEST_FILENAME
    if not p.is_file():
        raise DevpopAuditScoringError(f"eval-run manifest not found under run_dir: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def score_devpop_audit_entry(
    *,
    selection_manifest: Mapping,
    entry_trial_id: str,
    run_dir,
    package_root,
    contract: Mapping,
    population: ExpectedPopulationSpec,
    checkpoint_path=None,
    eval_run_manifest: "Mapping | None" = None,
    generated_config: "Mapping | str | Path | None" = None,
    require_canonical: bool = True,
    run_preflight: bool = True,
    min_area_samples: "int | None" = None,
    max_relative_mad: "float | None" = None,
) -> dict:
    """Score one already-completed development-population audit evaluation run
    into a canonical Common-120 audit row.

    ``selection_manifest`` is the full validated seven-entry frozen selection
    manifest; ``entry_trial_id`` selects exactly one member.  ``run_dir`` is
    the staged + NH-evaluated audit run directory (it must already contain the
    ``{period}/model_epoch{epoch:03d}/{period}_results.p`` pickle -- this
    module does not run NeuralHydrology).  ``contract`` / ``population`` are the
    SHARED-A1 audit contract and expected-population spec; with
    ``require_canonical=True`` (the default) they must be the canonical ones or
    the underlying evaluator fails closed.

    Returns ``{"audit_row": <row>, "driver_receipt": {...},
    "preflight_report": <report|None>}``.

    Raises :class:`DevpopAuditScoringError` on any binding / correspondence
    contradiction; propagates ``DevpopAuditEvaluatorError`` /
    ``DevpopAuditPreflightError`` / SHARED-A1 errors unchanged.
    """
    try:
        validated_manifest, entry = _select_manifest_entry(selection_manifest, entry_trial_id)
    except DevpopAuditPreflightError as exc:
        raise DevpopAuditScoringError(str(exc)) from exc

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise DevpopAuditScoringError(f"run_dir does not exist: {run_dir}")

    if eval_run_manifest is None:
        eval_run_manifest = _load_eval_run_manifest(run_dir)
    if not isinstance(eval_run_manifest, Mapping):
        raise DevpopAuditScoringError("eval_run_manifest must be a mapping when supplied")

    # The eval-run manifest must itself be the one staged for this exact frozen
    # entry (identity + selection-manifest SHA), before we trust anything under
    # run_dir.
    for key, expected in (
        ("trial_id", entry["trial_id"]),
        ("configuration_id", entry["configuration_id"]),
        ("checkpoint_epoch", entry["screening_best_epoch"]),
        ("checkpoint_sha256", entry["checkpoint_sha256"]),
        ("selection_manifest_sha256", validated_manifest["manifest_sha256"]),
        ("period", AUDIT_PERIOD_NAME),
    ):
        if eval_run_manifest.get(key) != expected:
            raise DevpopAuditScoringError(
                f"staged eval-run manifest {key}={eval_run_manifest.get(key)!r} does not match the "
                f"intended frozen selection-manifest entry ({expected!r})"
            )

    preflight_report: Optional[dict] = None
    if run_preflight:
        if generated_config is None:
            generated_config = run_dir / "config.yml"
        try:
            preflight_report = preflight_devpop_audit_entry(
                selection_manifest=selection_manifest,
                entry_trial_id=entry_trial_id,
                eval_run_manifest=eval_run_manifest,
                generated_config=generated_config,
                run_dir=run_dir,
                strict=True,
            )
        except DevpopAuditPreflightError:
            raise
        except DevpopAuditSelectionManifestError as exc:
            raise DevpopAuditScoringError(f"preflight rejected the selection manifest: {exc}") from exc

    checkpoint_identity = selection_manifest_entry_to_checkpoint_identity(entry)
    if checkpoint_path is None:
        checkpoint_path = run_dir / entry["checkpoint_filename"]
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != entry["checkpoint_filename"]:
        raise DevpopAuditScoringError(
            f"checkpoint_path basename {checkpoint_path.name!r} does not match the frozen entry "
            f"checkpoint_filename {entry['checkpoint_filename']!r}"
        )

    provenance_receipt = build_devpop_audit_provenance_receipt(
        trial_id=entry["trial_id"],
        configuration_id=entry["configuration_id"],
        run_dir=run_dir,
        period=AUDIT_PERIOD_NAME,
        checkpoint_epoch=entry["screening_best_epoch"],
        checkpoint_path=checkpoint_path,
    )

    eval_kwargs: dict = {
        "checkpoint_identity": checkpoint_identity,
        "run_dir": run_dir,
        "package_root": package_root,
        "population": population,
        "contract": contract,
        "provenance_receipt": provenance_receipt,
        "checkpoint_path": checkpoint_path,
        "require_canonical": require_canonical,
    }
    if min_area_samples is not None:
        eval_kwargs["min_area_samples"] = min_area_samples
    if max_relative_mad is not None:
        eval_kwargs["max_relative_mad"] = max_relative_mad

    row = evaluate_devpop_common120_audit_row(**eval_kwargs)

    # -- fail closed unless the row corresponds EXACTLY to the intended entry - #
    contract_checksum = dict(contract).get("checksum_sha256")
    correspondence = (
        ("trial_id", row.get("trial_id"), entry["trial_id"]),
        ("configuration_id", row.get("configuration_id"), entry["configuration_id"]),
        ("checkpoint_epoch", row.get("checkpoint_epoch"), entry["screening_best_epoch"]),
        ("checkpoint_sha256", row.get("checkpoint_sha256"), entry["checkpoint_sha256"]),
        ("membership_ids_sha256", row.get("membership_ids_sha256"), population.membership_ids_sha256),
        ("contract_checksum_sha256", row.get("contract_checksum_sha256"), contract_checksum),
    )
    for label, observed, expected in correspondence:
        if observed != expected:
            raise DevpopAuditScoringError(
                f"emitted audit row {label}={observed!r} does not correspond to the intended frozen "
                f"entry / population / contract ({expected!r})"
            )
    if row.get("provenance_verified") is not True:
        raise DevpopAuditScoringError("emitted audit row is not provenance-verified")
    if require_canonical:
        if row.get("canonical_completeness") is not True:
            raise DevpopAuditScoringError(
                "require_canonical=True but the emitted row did not pass the canonical completeness gate"
            )
        if row.get("canonical_population_verified") is not True:
            raise DevpopAuditScoringError(
                "require_canonical=True but the emitted row's canonical population was not verified"
            )

    driver_receipt = {
        "schema": "flashnh_devpop_audit_scoring_driver_receipt_v001",
        "entry_trial_id": entry["trial_id"],
        "configuration_id": entry["configuration_id"],
        "search_arm": entry["search_arm"],
        "proposal_order": entry["proposal_order"],
        "selection_manifest_sha256": validated_manifest["manifest_sha256"],
        "checkpoint_epoch": entry["screening_best_epoch"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "checkpoint_filename": entry["checkpoint_filename"],
        "run_dir": str(run_dir),
        "require_canonical": bool(require_canonical),
        "preflight_ran": preflight_report is not None,
        "preflight_ok": None if preflight_report is None else bool(preflight_report.get("ok")),
        "provenance_receipt": provenance_receipt.to_mapping(),
        "row_contract_checksum_sha256": row.get("contract_checksum_sha256"),
        "row_membership_ids_sha256": row.get("membership_ids_sha256"),
        "row_nse_median": row.get("nse_median"),
        "row_canonical_completeness": row.get("canonical_completeness"),
        "row_fixture_completeness": row.get("fixture_completeness"),
    }

    return {
        "audit_row": row,
        "driver_receipt": driver_receipt,
        "preflight_report": preflight_report,
    }
