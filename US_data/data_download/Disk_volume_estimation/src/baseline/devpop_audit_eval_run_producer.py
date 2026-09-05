"""SHARED-A4: audit-specific full-development-population evaluation-run
producer.

Prepares, for ONE entry -- selected by ``trial_id`` -- of a VALIDATED frozen
seven-checkpoint selection manifest (see
:mod:`devpop_audit_selection_manifest`), a NH-Tester-compatible run directory
that re-evaluates that already-screened checkpoint against the full
2,307-basin development population instead of the 400-basin screening
subset. This is the smallest additive path from a manifest entry to
something the SHARED-A2 evaluator (:mod:`devpop_common120_audit_evaluator`)
can consume -- it does not run NeuralHydrology, select an epoch, or touch the
real seven-checkpoint set.

Deliberately NOT a thin call to
:func:`nh_seed_evaluation.prepare_development_population_eval_run_dir`: that
function's own docstring and on-disk marker declare its results "NOT
authoritative for full-population validation, NOT usable for checkpoint or
architecture selection" -- exactly the opposite of what this audit exists to
produce. This module reuses its lower-level primitives (byte-copy-and-verify,
:func:`weight_stem`, :func:`sha256_file`, holdout-bundle rejection) but writes
its own, differently-labelled marker and manifest.
"""
from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

from .devpop_audit_selection_manifest import validate_devpop_audit_selection_manifest
from .devpop_common120_audit_contract import (
    AUDIT_DATE_MAX,
    AUDIT_DATE_MIN,
    AUDIT_PERIOD_NAME,
    CANONICAL_LEAD_HOURS,
    CANONICAL_TARGET_VARIABLE,
)
from .fixed_support_contract_v2 import load_fixed_support_contract
from .nh_config_generation import raise_if_holdout_bundle, write_generated_config
from .nh_seed_evaluation import weight_stem
from .package_audit import sha256_file
from .pilot_full_validation import load_validated_full_population_basin_ids
from .pilot_lead06_config import build_pilot_bundle_with_validation_scope
from .policy_v2_six_axis import load_stage1_baseline_policy_v2_six_axis
from .sweep_v2_six_axis_campaign import configuration_id_v2

__all__ = [
    "DevpopAuditEvalRunProducerError",
    "AUDIT_EVAL_RUN_MARKER_FILENAME",
    "AUDIT_EVAL_RUN_MANIFEST_FILENAME",
    "AUDIT_EVAL_RUN_POPULATION_ROLE",
    "prepare_devpop_audit_eval_run_dir",
]


class DevpopAuditEvalRunProducerError(ValueError):
    """Raised for a setup/provenance/identity problem while preparing an
    audit-specific evaluation run directory. Never raised for an ordinary
    poor-skill outcome."""


#: Distinct from ``nh_seed_evaluation.EVALUATION_ONLY_MARKER_FILENAME`` --
#: that marker's text disclaims full-population authority; this run
#: directory's whole purpose IS the full-population audit.
AUDIT_EVAL_RUN_MARKER_FILENAME = "DEVPOP_AUDIT_EVAL_ONLY_DO_NOT_TRAIN.txt"
AUDIT_EVAL_RUN_MANIFEST_FILENAME = "DEVPOP_AUDIT_EVAL_RUN_MANIFEST.json"

#: Distinct from ``pilot_lead06_config.SCREENING_VALIDATION_POPULATION_ROLE``
#: and ``pilot_full_validation.FULL_VALIDATION_POPULATION_ROLE`` -- neither
#: names this audit's own diagnostic scope, and "distinct output identity for
#: changed identity" is a hard project rule (never reuse another path's role
#: label so a downstream reader could mistake one run for the other).
AUDIT_EVAL_RUN_POPULATION_ROLE = "devpop_audit_full_population_v001"


def _copy_and_verify(src: Path, dst: Path, *, label: str) -> str:
    shutil.copy2(src, dst)
    src_sha256 = sha256_file(src)
    dst_sha256 = sha256_file(dst)
    if src_sha256 != dst_sha256:
        raise DevpopAuditEvalRunProducerError(
            f"{label} copy corrupted: source sha256 {src_sha256} != dest sha256 {dst_sha256}"
        )
    return dst_sha256


def prepare_devpop_audit_eval_run_dir(
    *,
    selection_manifest: Mapping,
    entry_trial_id: str,
    baseline_policy_path,
    policy_overlay_path,
    package_root,
    splits_dir,
    fixed_support_contract_path,
    checkpoint_src_path,
    scaler_src_path,
    run_profile_name: str,
    out_generated_dir,
    out_run_dir,
    force: bool = False,
) -> dict:
    """Prepare one audit-specific, full-development-population NH validation
    run directory for a single checkpoint of the frozen SHARED-A3 comparison
    set.

    ``selection_manifest`` must be the full seven-entry selection manifest
    (the mapping returned by
    :func:`devpop_audit_selection_manifest.validate_devpop_audit_selection_manifest`
    or loaded by
    :func:`devpop_audit_selection_manifest.load_devpop_audit_selection_manifest`);
    it is re-validated here as an atomic seven-entry, hash-pinned set.
    ``entry_trial_id`` selects exactly one member of that manifest -- an
    arbitrary standalone entry is NOT accepted, so every staged run can later
    prove it came from entry X of this exact frozen manifest SHA. The
    validated ``manifest_sha256`` is carried into the persisted eval-run
    manifest as ``selection_manifest_sha256``.

    Reuses :func:`build_pilot_bundle_with_validation_scope` (the "cleanest
    existing shared primitive", per the SHARED-A4 design doc) with
    ``validation_basin_ids`` re-derived from the CURRENT package via
    :func:`load_validated_full_population_basin_ids` -- never a caller-
    supplied list -- and a distinct, non-holdout, non-screening
    ``population_role``. Explicitly verifies target variable, lead hours,
    the frozen audit date window, and that the copied checkpoint's bytes
    match the manifest entry's frozen ``checkpoint_sha256`` before anything
    is written to ``out_run_dir``.

    Returns a manifest dict (also persisted as
    ``out_run_dir/DEVPOP_AUDIT_EVAL_RUN_MANIFEST.json``) carrying exactly the
    facts :func:`devpop_common120_audit_evaluator.build_devpop_audit_provenance_receipt`
    needs (``trial_id``, ``configuration_id``, ``run_dir``, ``period``,
    ``checkpoint_epoch``, ``checkpoint_path``) plus the identity/provenance
    fields this producer itself is responsible for.
    """
    if not isinstance(selection_manifest, Mapping):
        raise DevpopAuditEvalRunProducerError(
            "selection_manifest must be the validated seven-entry selection-manifest mapping "
            f"(got {type(selection_manifest).__name__})"
        )
    validated_manifest = validate_devpop_audit_selection_manifest(selection_manifest.get("entries"))
    selection_manifest_sha256 = validated_manifest["manifest_sha256"]
    recorded_sha256 = selection_manifest.get("manifest_sha256")
    if recorded_sha256 is not None and recorded_sha256 != selection_manifest_sha256:
        raise DevpopAuditEvalRunProducerError(
            f"selection_manifest manifest_sha256 {recorded_sha256!r} does not match the recomputed "
            f"identity of its own seven entries ({selection_manifest_sha256!r}) -- possible tampering"
        )
    selected = [e for e in validated_manifest["entries"] if e["trial_id"] == entry_trial_id]
    if len(selected) != 1:
        raise DevpopAuditEvalRunProducerError(
            f"entry_trial_id {entry_trial_id!r} does not identify exactly one entry of the validated "
            f"seven-entry selection manifest (matched {len(selected)})"
        )
    entry = selected[0]

    checkpoint_src_path = Path(checkpoint_src_path)
    scaler_src_path = Path(scaler_src_path)
    out_generated_dir = Path(out_generated_dir)
    out_run_dir = Path(out_run_dir)

    if not checkpoint_src_path.is_file():
        raise DevpopAuditEvalRunProducerError(f"checkpoint source does not exist: {checkpoint_src_path}")
    if checkpoint_src_path.name != entry["checkpoint_filename"]:
        raise DevpopAuditEvalRunProducerError(
            f"checkpoint source filename {checkpoint_src_path.name!r} does not match manifest entry "
            f"checkpoint_filename {entry['checkpoint_filename']!r}"
        )
    checkpoint_src_sha256 = sha256_file(checkpoint_src_path)
    if checkpoint_src_sha256 != entry["checkpoint_sha256"]:
        raise DevpopAuditEvalRunProducerError(
            f"checkpoint source sha256 {checkpoint_src_sha256} does not match the frozen manifest entry "
            f"checkpoint_sha256 {entry['checkpoint_sha256']} -- refusing to prepare an audit run from an "
            "unverified checkpoint"
        )
    if not scaler_src_path.is_file():
        raise DevpopAuditEvalRunProducerError(f"scaler source does not exist: {scaler_src_path}")

    # Identity cross-check: the loaded fixed-support contract must be the
    # SAME one the manifest entry's configuration_id was computed under --
    # otherwise this producer would silently stage a run for a
    # scientifically different configuration than the one screened.
    contract = load_fixed_support_contract(fixed_support_contract_path)
    support_contract_version = contract["contract_id"]
    support_contract_sha256 = contract["checksum_sha256"]
    recomputed_configuration_id = configuration_id_v2(
        entry["hyperparameters"],
        support_contract_version=support_contract_version,
        support_contract_sha256=support_contract_sha256,
    )
    if recomputed_configuration_id != entry["configuration_id"]:
        raise DevpopAuditEvalRunProducerError(
            f"loaded fixed-support contract ({fixed_support_contract_path}) yields configuration_id "
            f"{recomputed_configuration_id!r}, which does not match the manifest entry's "
            f"configuration_id {entry['configuration_id']!r} -- refusing to prepare a run under a "
            "different support-contract identity than the one this configuration was screened under"
        )

    development_basins = load_validated_full_population_basin_ids(package_root=package_root, splits_dir=splits_dir)
    policy_v2 = load_stage1_baseline_policy_v2_six_axis(baseline_policy_path, policy_overlay_path)
    axes = entry["hyperparameters"]

    bundle = build_pilot_bundle_with_validation_scope(
        baseline_policy_path=baseline_policy_path,
        package_root=package_root,
        splits_dir=splits_dir,
        lead_hours=CANONICAL_LEAD_HOURS,
        seq_length=axes["seq_length"],
        run_profile_name=run_profile_name,
        validation_basin_ids=development_basins,
        population_role=AUDIT_EVAL_RUN_POPULATION_ROLE,
        package_type=f"devpop_audit_v001_{entry['configuration_id']}",
        learning_rate=float(axes["learning_rate"]),
        hidden_size=axes["hidden_size"],
        embedding_dropout=float(axes["embedding_dropout"]),
        output_dropout=float(axes["output_dropout"]),
        batch_size=axes["batch_size"],
        policy_override=policy_v2,
    )

    # Explicit target/lead/period/population verification -- the exact
    # checks the SHARED-A4 design doc calls out by name.
    if bundle.target_variable != CANONICAL_TARGET_VARIABLE:
        raise DevpopAuditEvalRunProducerError(
            f"generated bundle target_variable {bundle.target_variable!r} != canonical "
            f"{CANONICAL_TARGET_VARIABLE!r}"
        )
    if bundle.lead_hours != CANONICAL_LEAD_HOURS:
        raise DevpopAuditEvalRunProducerError(
            f"generated bundle lead_hours {bundle.lead_hours} != canonical {CANONICAL_LEAD_HOURS}"
        )
    if bundle.population_role != AUDIT_EVAL_RUN_POPULATION_ROLE:
        raise DevpopAuditEvalRunProducerError("generated bundle population_role drifted from the audit role")
    if sorted(bundle.validation_basin_ids) != sorted(development_basins):
        raise DevpopAuditEvalRunProducerError(
            "generated bundle validation_basin_ids do not match the freshly re-derived development population"
        )
    val_start = bundle.config_mapping.get("validation_start_date")
    val_end = bundle.config_mapping.get("validation_end_date")
    expected_start = date.fromisoformat(AUDIT_DATE_MIN).strftime("%d/%m/%Y")
    expected_end = date.fromisoformat(AUDIT_DATE_MAX).strftime("%d/%m/%Y")
    if val_start != expected_start or val_end != expected_end:
        raise DevpopAuditEvalRunProducerError(
            f"generated bundle validation window {val_start!r}..{val_end!r} != the frozen audit window "
            f"{expected_start!r}..{expected_end!r}"
        )

    write_generated_config(bundle, out_generated_dir, force=force)
    raise_if_holdout_bundle(out_generated_dir)

    if out_run_dir.exists():
        if not force:
            raise DevpopAuditEvalRunProducerError(f"out_run_dir already exists (pass force=True to overwrite): {out_run_dir}")
        shutil.rmtree(out_run_dir)
    out_run_dir.mkdir(parents=True)
    (out_run_dir / "train_data").mkdir()

    epoch = entry["screening_best_epoch"]
    config_dst = out_run_dir / "config.yml"
    checkpoint_dst = out_run_dir / f"{weight_stem(epoch)}.pt"
    scaler_dst = out_run_dir / "train_data" / "train_data_scaler.yml"

    config_yaml_sha256 = _copy_and_verify(out_generated_dir / "config.yaml", config_dst, label="config.yaml")
    checkpoint_dst_sha256 = _copy_and_verify(checkpoint_src_path, checkpoint_dst, label="checkpoint")
    scaler_dst_sha256 = _copy_and_verify(scaler_src_path, scaler_dst, label="scaler")
    if checkpoint_dst_sha256 != entry["checkpoint_sha256"]:
        raise DevpopAuditEvalRunProducerError(
            "copied checkpoint sha256 does not match the frozen manifest entry checkpoint_sha256 -- "
            "should be unreachable given the pre-copy check above"
        )

    with open(out_run_dir / AUDIT_EVAL_RUN_MARKER_FILENAME, "w", encoding="utf-8") as fh:
        fh.write(
            "This run directory stages a DEVELOPMENT-POPULATION COMMON-120 AUDIT\n"
            "evaluation (SHARED-A4) for one already-screened v2 six-axis\n"
            f"configuration (trial_id={entry['trial_id']}).\n"
            "\n"
            f"Its checkpoint ({checkpoint_dst.name}) and scaler are copied byte-for-byte\n"
            "from an already-completed screening training run; the recorded\n"
            f"screening_best_epoch ({epoch}) is RECORDED here, not recomputed.\n"
            "\n"
            "Do NOT run a trainer against this config.yml -- refitting from here\n"
            "would silently discard the original training and invalidate the\n"
            "checkpoint identity this audit row is bound to.\n"
            "\n"
            "This run's validation basin scope is the FULL 2,307-basin\n"
            "development population (not the 400-basin screening subset) --\n"
            "results from it ARE the intended input to the development-population\n"
            "Common-120 audit, once evaluated via\n"
            "devpop_common120_audit_evaluator.evaluate_devpop_common120_audit_row.\n"
        )

    manifest = {
        "schema_name": "devpop_audit_eval_run_manifest",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_manifest_sha256": selection_manifest_sha256,
        "selection_manifest_schema": validated_manifest["schema"],
        "trial_id": entry["trial_id"],
        "configuration_id": entry["configuration_id"],
        "search_arm": entry["search_arm"],
        "proposal_order": entry["proposal_order"],
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": checkpoint_dst_sha256,
        "checkpoint_path": str(checkpoint_dst),
        "scaler_sha256": scaler_dst_sha256,
        "config_yaml_sha256": config_yaml_sha256,
        "run_dir": str(out_run_dir),
        "period": AUDIT_PERIOD_NAME,
        "target_variable": bundle.target_variable,
        "lead_hours": bundle.lead_hours,
        "population_role": bundle.population_role,
        "validation_basin_count": len(bundle.validation_basin_ids),
        "date_window": [AUDIT_DATE_MIN, AUDIT_DATE_MAX],
        "support_contract_version": support_contract_version,
        "support_contract_sha256": support_contract_sha256,
        "evaluation_only_marker": AUDIT_EVAL_RUN_MARKER_FILENAME,
    }
    with open(out_run_dir / AUDIT_EVAL_RUN_MANIFEST_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
