"""SHARED-A5 Workstream C: the tracked canonical scoring / orchestration path
(:func:`src.baseline.devpop_audit_scoring.score_devpop_audit_entry`).

Proves the driver

* binds the scoring call to exactly one frozen selection-manifest entry --
  never to a caller-chosen epoch or P1 constants;
* fails closed when the staged eval-run manifest, the checkpoint path, or the
  emitted row does not correspond exactly to that entry;
* reuses the already-qualified SHARED-A2 evaluator verbatim (no second
  evaluator, no duplicated scientific math);
* treats the canonical completeness gate as mandatory when
  ``require_canonical=True`` and never forges it.

Uses the SAME real committed config-generation fixtures as
``tests/test_devpop_audit_eval_run_producer.py`` for the producer step, but a
small synthetic NH result / evaluator population + contract local to that
sibling module.  Nothing here runs NeuralHydrology or contacts a remote host.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baseline.devpop_audit_eval_run_producer import (
    AUDIT_EVAL_RUN_MANIFEST_FILENAME,
    prepare_devpop_audit_eval_run_dir,
)
from src.baseline.devpop_audit_preflight import DevpopAuditPreflightError
from src.baseline.devpop_audit_scoring import (
    DevpopAuditScoringError,
    score_devpop_audit_entry,
)
from src.baseline.devpop_common120_audit_evaluator import DevpopAuditEvaluatorError
from tests.test_devpop_audit_eval_run_producer import (
    EVAL_BASINS,
    _eval_population_and_contract,
    _producer_paths,
    _stage_sources,
    _write_eval_package,
    _write_run_pickle,
)

_CKPT_BYTES = b"synthetic-screened-checkpoint-weights"


def _stage_run(tmp_path):
    manifest, entry, contract_path, ckpt_src, scaler_src = _stage_sources(tmp_path, _CKPT_BYTES)
    prepare_devpop_audit_eval_run_dir(
        selection_manifest=manifest,
        entry_trial_id=entry["trial_id"],
        fixed_support_contract_path=contract_path,
        checkpoint_src_path=ckpt_src,
        scaler_src_path=scaler_src,
        out_generated_dir=tmp_path / "generated",
        out_run_dir=tmp_path / "run",
        **_producer_paths(tmp_path),
    )
    return manifest, entry


def _score_kwargs(tmp_path, manifest, entry, **overrides):
    package_root, population, contract = _eval_context(tmp_path)
    kwargs = dict(
        selection_manifest=manifest,
        entry_trial_id=entry["trial_id"],
        run_dir=tmp_path / "run",
        package_root=package_root,
        contract=contract,
        population=population,
        require_canonical=False,
    )
    kwargs.update(overrides)
    return kwargs


def _eval_context(tmp_path):
    package_root = tmp_path / "eval_pkg"
    _write_eval_package(package_root, EVAL_BASINS)
    population, contract = _eval_population_and_contract()
    return package_root, population, contract


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #

def test_scores_the_intended_frozen_entry(tmp_path):
    manifest, entry = _stage_run(tmp_path)
    _write_run_pickle(tmp_path / "run", EVAL_BASINS, epoch=entry["screening_best_epoch"])

    out = score_devpop_audit_entry(**_score_kwargs(tmp_path, manifest, entry))

    row = out["audit_row"]
    assert row["trial_id"] == entry["trial_id"]
    assert row["configuration_id"] == entry["configuration_id"]
    assert row["checkpoint_epoch"] == entry["screening_best_epoch"]
    assert row["checkpoint_sha256"] == entry["checkpoint_sha256"]
    assert row["provenance_verified"] is True

    receipt = out["driver_receipt"]
    assert receipt["schema"] == "flashnh_devpop_audit_scoring_driver_receipt_v001"
    assert receipt["entry_trial_id"] == entry["trial_id"]
    assert receipt["selection_manifest_sha256"] == manifest["manifest_sha256"]
    assert receipt["checkpoint_epoch"] == entry["screening_best_epoch"]
    assert receipt["require_canonical"] is False
    # the driver ran the Workstream-A preflight and it passed
    assert out["preflight_report"] is not None
    assert out["preflight_report"]["ok"] is True
    assert receipt["preflight_ran"] is True
    assert receipt["preflight_ok"] is True
    # honest completeness labelling: synthetic fixtures are fixture-complete,
    # NOT canonical-complete, and the driver does not claim otherwise
    assert receipt["row_fixture_completeness"] is True
    assert receipt["row_canonical_completeness"] is False


# --------------------------------------------------------------------------- #
# fail-closed binding / correspondence
# --------------------------------------------------------------------------- #

def test_rejects_trial_id_not_in_seven_entry_manifest(tmp_path):
    manifest, entry = _stage_run(tmp_path)
    _write_run_pickle(tmp_path / "run", EVAL_BASINS, epoch=entry["screening_best_epoch"])

    with pytest.raises(DevpopAuditScoringError, match="does not identify exactly one entry"):
        score_devpop_audit_entry(
            **_score_kwargs(
                tmp_path, manifest, entry,
                entry_trial_id="sweep_v2_campaign__bayesian__proposal001__not_a_real_trial",
            )
        )


def test_rejects_tampered_eval_run_manifest_epoch(tmp_path):
    manifest, entry = _stage_run(tmp_path)
    _write_run_pickle(tmp_path / "run", EVAL_BASINS, epoch=entry["screening_best_epoch"])

    mpath = tmp_path / "run" / AUDIT_EVAL_RUN_MANIFEST_FILENAME
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["checkpoint_epoch"] = m["checkpoint_epoch"] + 1
    mpath.write_text(json.dumps(m), encoding="utf-8")

    with pytest.raises(DevpopAuditScoringError, match="does not match the intended frozen"):
        score_devpop_audit_entry(**_score_kwargs(tmp_path, manifest, entry))


def test_rejects_checkpoint_path_basename_mismatch(tmp_path):
    manifest, entry = _stage_run(tmp_path)
    _write_run_pickle(tmp_path / "run", EVAL_BASINS, epoch=entry["screening_best_epoch"])

    wrong = tmp_path / "run" / "model_epoch999.pt"
    wrong.write_bytes(_CKPT_BYTES)

    with pytest.raises(DevpopAuditScoringError, match="checkpoint_filename"):
        score_devpop_audit_entry(
            **_score_kwargs(tmp_path, manifest, entry, checkpoint_path=wrong)
        )


def test_missing_results_pickle_fails_closed(tmp_path):
    manifest, entry = _stage_run(tmp_path)
    # deliberately do NOT write the NH results pickle

    with pytest.raises(DevpopAuditEvaluatorError):
        score_devpop_audit_entry(**_score_kwargs(tmp_path, manifest, entry))


def test_tampered_staged_checkpoint_fails_closed(tmp_path):
    manifest, entry = _stage_run(tmp_path)
    _write_run_pickle(tmp_path / "run", EVAL_BASINS, epoch=entry["screening_best_epoch"])
    (tmp_path / "run" / entry["checkpoint_filename"]).write_bytes(b"corrupted-weights")

    with pytest.raises((DevpopAuditPreflightError, DevpopAuditEvaluatorError)):
        score_devpop_audit_entry(**_score_kwargs(tmp_path, manifest, entry))


def test_require_canonical_true_rejects_synthetic_fixtures(tmp_path):
    manifest, entry = _stage_run(tmp_path)
    _write_run_pickle(tmp_path / "run", EVAL_BASINS, epoch=entry["screening_best_epoch"])

    # synthetic population + generic contract cannot pass SHARED-A1's canonical
    # gate -- the driver must not silently emit a row claiming canonical status
    with pytest.raises((DevpopAuditScoringError, DevpopAuditEvaluatorError)):
        score_devpop_audit_entry(
            **_score_kwargs(tmp_path, manifest, entry, require_canonical=True)
        )
