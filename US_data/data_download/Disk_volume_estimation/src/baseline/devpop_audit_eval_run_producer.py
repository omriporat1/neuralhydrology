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
import re
import shutil
from dataclasses import replace
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
from .nh_config_generation import (
    EXPECTED_DEVELOPMENT_BASIN_COUNT,
    raise_if_holdout_bundle,
    write_generated_config,
)
from .nh_seed_evaluation import weight_stem
from .package_audit import sha256_file
from .pilot_full_validation import load_validated_full_population_basin_ids
from .pilot_lead06_config import build_pilot_bundle_with_validation_scope
from .policy_v2_six_axis import load_stage1_baseline_policy_v2_six_axis
from .sweep_v2_six_axis_campaign import FROZEN_FIXED_CONFIGURATION_V2, configuration_id_v2

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

_SHA256_HEX_RE = re.compile(r"\A[0-9a-f]{64}\Z")


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
    expected_scaler_sha256: "str | None" = None,
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

    Also overrides the reused pilot profile's screening-era
    ``validate_n_random_basins`` (1000) with the exact canonical
    development-population count, so NH evaluates the FULL population rather
    than a shuffled 1,000-basin subsample, and fails closed if that
    evaluation count does not equal the canonical 2,307-basin population.

    Returns a manifest dict (also persisted as
    ``out_run_dir/DEVPOP_AUDIT_EVAL_RUN_MANIFEST.json``) carrying exactly the
    facts :func:`devpop_common120_audit_evaluator.build_devpop_audit_provenance_receipt`
    needs (``trial_id``, ``configuration_id``, ``run_dir``, ``period``,
    ``checkpoint_epoch``, ``checkpoint_path``) plus the identity/provenance
    fields this producer itself is responsible for.

    Scaler provenance (SHARED-A5 Workstream B). The frozen seven-entry
    selection manifest carries NO authoritative historical scaler
    identity/hash -- the screening runs never recorded one -- so this producer
    cannot and does not claim historical scaler identity. What it *can* do is
    record, at audit-execution time, the exact bytes of the scaler it stages:
    the source scaler's sha256 is always computed and both the source and the
    byte-verified staged sha256 are written into a dedicated
    ``audit_execution_time_scaler_provenance`` manifest block, clearly
    labelled as execution-time (not screening-time) provenance. If the caller
    already knows the scaler hash it expects (e.g. from an out-of-band
    screening-run inventory), it may pass ``expected_scaler_sha256`` and the
    producer fails closed on any mismatch. This is additive: the existing
    ``scaler_sha256`` key and all checkpoint provenance are unchanged.
    """
    if expected_scaler_sha256 is not None and not _SHA256_HEX_RE.match(str(expected_scaler_sha256)):
        raise DevpopAuditEvalRunProducerError(
            f"expected_scaler_sha256 must be 64 lowercase hex chars or None, got {expected_scaler_sha256!r}"
        )
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
    # SHARED-A5 Workstream B: execution-time scaler byte provenance. There is
    # no authoritative historical scaler hash in the frozen selection manifest;
    # this records the actual staged bytes, and fails closed only against a
    # hash the caller explicitly supplies.
    scaler_src_sha256 = sha256_file(scaler_src_path)
    if expected_scaler_sha256 is not None and scaler_src_sha256 != expected_scaler_sha256:
        raise DevpopAuditEvalRunProducerError(
            f"scaler source sha256 {scaler_src_sha256} does not match the caller-supplied "
            f"expected_scaler_sha256 {expected_scaler_sha256} -- refusing to stage an audit run from an "
            "unexpected scaler"
        )

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

    # The v2 six-axis campaign sweeps only learning_rate/hidden_size/
    # embedding_dropout/output_dropout/batch_size/seq_length; the
    # dynamic-input family is a FROZEN fixed-campaign property (PT:
    # mrms_qpe_1h_mm, rtma_2t_K), pinned identically for every screened
    # configuration via FROZEN_FIXED_CONFIGURATION_V2. The base scientific
    # baseline policy still carries the older, wider dynamic-input set, so
    # this must be passed explicitly -- otherwise the regenerated audit
    # config silently inherits the baseline default and the resulting LSTM
    # input dimension no longer matches the frozen screening checkpoint.
    frozen_dynamic_inputs = list(FROZEN_FIXED_CONFIGURATION_V2["dynamic_inputs"])

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
        dynamic_inputs=frozen_dynamic_inputs,
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
    # Fail closed on dynamic-input-family drift: the regenerated audit config
    # MUST carry exactly the frozen v2 six-axis campaign's dynamic inputs, in
    # order -- if it silently inherited the wider base-baseline set the LSTM
    # input dimension would no longer match the frozen screening checkpoint
    # and NH would abort loading the state dict.
    generated_dynamic_inputs = list(bundle.dynamic_inputs)
    if generated_dynamic_inputs != frozen_dynamic_inputs:
        raise DevpopAuditEvalRunProducerError(
            f"generated bundle dynamic_inputs {generated_dynamic_inputs!r} != the frozen v2 six-axis "
            f"campaign dynamic-input family {frozen_dynamic_inputs!r} -- refusing to stage an audit run "
            "whose input dimension would not match the frozen screening checkpoint"
        )
    config_dynamic_inputs = list(bundle.config_mapping.get("dynamic_inputs", []))
    if config_dynamic_inputs != frozen_dynamic_inputs:
        raise DevpopAuditEvalRunProducerError(
            f"generated config_mapping dynamic_inputs {config_dynamic_inputs!r} != the frozen v2 six-axis "
            f"campaign dynamic-input family {frozen_dynamic_inputs!r}"
        )

    # ---- audit-specific evaluation-population carry-through ----------------
    # The reused pilot run profile (_PILOT_LEAD06_BASE_PROFILE) carries
    # ``validate_n_random_basins: 1000`` -- historically correct for the
    # <=500-basin screening workflow (it covered all 400 screening basins),
    # but for this FULL-development-population audit that same setting silently
    # makes NH ``BaseTester.evaluate()`` shuffle/truncate the 2,307-basin
    # validation population down to 1,000 (exactly the defect that made P1
    # job 46098997 scientifically invalid for the audit population). This
    # audit exists to re-evaluate the screened checkpoint against the ENTIRE
    # canonical development population, so the generated audit config must
    # instruct NH to score exactly that population -- never a subsample.
    #
    # This is an audit-local override of the produced config only: the shared
    # ``_PILOT_LEAD06_BASE_PROFILE`` (and its ``validate_n_random_basins:
    # 1000``) is deliberately left untouched, per the SHARED baseline policy.
    audit_eval_population_count = len(development_basins)
    audit_config_mapping = dict(bundle.config_mapping)
    audit_config_mapping["validate_n_random_basins"] = audit_eval_population_count
    bundle = replace(bundle, config_mapping=audit_config_mapping)

    # Fail closed, before anything is staged: the NH evaluation count MUST be
    # the exact canonical audit validation-population count, and that
    # population MUST be the frozen canonical 2,307-basin development set.
    if audit_eval_population_count != EXPECTED_DEVELOPMENT_BASIN_COUNT:
        raise DevpopAuditEvalRunProducerError(
            f"canonical development population resolved to {audit_eval_population_count} basins, "
            f"expected exactly {EXPECTED_DEVELOPMENT_BASIN_COUNT} -- refusing to stage a devpop audit run"
        )
    if sorted(bundle.validation_basin_ids) != sorted(development_basins):
        raise DevpopAuditEvalRunProducerError(
            "generated bundle validation_basin_ids do not exactly match the canonical development "
            "population -- refusing to stage a devpop audit run"
        )
    if bundle.config_mapping.get("validate_n_random_basins") != audit_eval_population_count:
        raise DevpopAuditEvalRunProducerError(
            f"generated audit config validate_n_random_basins "
            f"{bundle.config_mapping.get('validate_n_random_basins')!r} != the canonical audit "
            f"validation-population count {audit_eval_population_count} -- NH would not evaluate the "
            "full development population"
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
        "audit_execution_time_scaler_provenance": {
            "provenance_kind": "audit_execution_time",
            "note": (
                "Execution-time scaler byte provenance ONLY. The frozen seven-entry devpop-audit "
                "selection manifest records no historical screening-time scaler identity, so this "
                "block does not and cannot attest that these bytes are the scaler the screening run "
                "trained under -- it attests only that the bytes staged into this audit run's "
                "train_data/train_data_scaler.yml are exactly the bytes read from scaler_src_path."
            ),
            "scaler_src_path": str(scaler_src_path),
            "scaler_src_sha256": scaler_src_sha256,
            "scaler_staged_relpath": "train_data/train_data_scaler.yml",
            "scaler_staged_sha256": scaler_dst_sha256,
            "scaler_bytes_verified_equal": scaler_src_sha256 == scaler_dst_sha256,
            "expected_scaler_sha256": expected_scaler_sha256,
            "expected_scaler_sha256_verified": (
                None if expected_scaler_sha256 is None else scaler_src_sha256 == expected_scaler_sha256
            ),
        },
        "config_yaml_sha256": config_yaml_sha256,
        "run_dir": str(out_run_dir),
        "period": AUDIT_PERIOD_NAME,
        "target_variable": bundle.target_variable,
        "lead_hours": bundle.lead_hours,
        "population_role": bundle.population_role,
        "validation_basin_count": len(bundle.validation_basin_ids),
        "nh_evaluation_population_count": bundle.config_mapping["validate_n_random_basins"],
        "date_window": [AUDIT_DATE_MIN, AUDIT_DATE_MAX],
        "support_contract_version": support_contract_version,
        "support_contract_sha256": support_contract_sha256,
        "evaluation_only_marker": AUDIT_EVAL_RUN_MARKER_FILENAME,
    }
    with open(out_run_dir / AUDIT_EVAL_RUN_MANIFEST_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
