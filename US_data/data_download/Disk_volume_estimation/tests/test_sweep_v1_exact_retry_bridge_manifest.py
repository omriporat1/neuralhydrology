"""Tests for the manifest-driven exact-retry bridge entry point,
``scripts.run_sweep_v1_exact_retry_bridge.main_from_manifest`` -- the
disposable exact-retry startup rehearsal's Binding Design Decisions 1/2/4/5:
one positional launch-manifest JSON replaces the long
``sbatch --export=ALL,VAR=value,...`` interface, and production/rehearsal
share the identical runtime-contract/durable-intake/W&B-init/executor-
selection core (``_execute_retry``) that ``tests/test_sweep_v1_exact_retry_bridge.py``
already exercises end to end via the legacy CLI entry point (``main``).

Covers, in order:
  1. ``main_from_manifest`` runs the shared runtime contract BEFORE any
     durable intake write or W&B import; a runtime-contract failure leaves no
     output directory behind and never imports wandb.
  2. Manifest provenance (``launch_manifest_path``/``launch_manifest_sha256``/
     ``launch_manifest_label``) is merged onto the SAME durable
     ``proposal_intake``-stage record, still strictly before any W&B call.
  3. A rehearsal manifest (``mode="rehearsal"``, ``stop_before_training=True``)
     records the pure executor-mode selection and finishes the W&B run
     cleanly, but NEVER calls ``run_prepared_trial_in_production`` -- no
     training/NH-orchestration entry point is ever reached.
  4. A production manifest (``mode="production"``, ``stop_before_training=False``)
     runs the full path end-to-end, including
     ``run_prepared_trial_in_production`` -- it is never coerced into
     stopping early.
  5. A W&B ``init()`` failure under manifest mode still produces the same
     durable terminal evidence (``wandb_init_failed``) as the CLI path.
  6. A successful W&B association under manifest mode is durably recorded
     (stage ``wandb_associated``) before preparation starts.

The real commit/interpreter/HOME/netrc runtime contract itself is already
fully covered in isolation by ``tests/test_sweep_v1_runtime_contract.py``;
here it is monkeypatched (to either a no-op or a controlled failure) so these
tests exercise the manifest-driven WIRING, not the runtime contract's own
git/filesystem logic a second time.

Never imports the real ``wandb`` package. Never touches the network. Never
targets the real production sweep (``4x3btz2s``) or any real Moriah path.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from src.baseline import pilot_orchestration as orchestration
from src.baseline import sweep_v1_campaign as sweep
from src.baseline import sweep_v1_runtime_contract as runtime_contract
from src.baseline.sweep_v1_launch_manifest import MODE_PRODUCTION, MODE_REHEARSAL, write_launch_manifest
from src.baseline.sweep_v1_production_adapter import PreparationPaths
from src.baseline.sweep_v1_runtime_contract import RuntimeContractError
from tests._pilot_support import PILOT_POLICY_PATH
from tests.test_sweep_v1_exact_retry_bridge import (
    _AXES, _fake_result, _no_real_wandb_module, _patch_real_canonical_split_shas_for_local_checkout,
    _paths, _pinned_identity_path, _write_frozen_record, _write_real_checkpoints, fake_wandb_module,
)

import scripts.run_sweep_v1_exact_retry_bridge as retry_bridge

__all__ = ["fake_wandb_module", "_no_real_wandb_module"]  # re-exported fixtures


def _noop_runtime_contract(monkeypatch):
    monkeypatch.setattr(runtime_contract, "run_full_runtime_contract", lambda **kwargs: None)


def _write_manifest(tmp_path, *, mode, wandb_sweep_id, stop_before_training, frozen_record_path,
                     identity_path, output_root, paths: PreparationPaths, execution_generation=2,
                     manifest_name="manifest.json") -> Path:
    manifest_path = tmp_path / manifest_name
    write_launch_manifest(
        manifest_path,
        manifest_label="sweep_v1_exact_retry_test_manifest_v001",
        created_at_utc="2026-08-24T00:00:00Z",
        mode=mode,
        expected_commit="0" * 40,
        expected_runtime_python=sys.executable,
        frozen_proposal_record_path=str(frozen_record_path),
        expected_identity=json.loads(identity_path.read_text(encoding="utf-8")),
        execution_generation=execution_generation,
        package_root=str(paths.package_root),
        screening_basin_ids_path=str(paths.screening_basin_ids_path),
        output_root=str(output_root),
        baseline_policy_path=str(paths.baseline_policy_path),
        base_pilot_policy_path=str(PILOT_POLICY_PATH),
        wandb_project="flashnh-stage1-test",
        wandb_sweep_id=wandb_sweep_id,
        stop_before_training=stop_before_training,
    )
    return manifest_path


# --- 1: runtime contract runs first, before any durable intake or wandb -----

def test_runtime_contract_failure_leaves_no_output_directory_and_no_wandb_import(tmp_path, monkeypatch):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "o1"
    paths = _paths(tmp_path / "p", monkeypatch)

    def _raiser(**kwargs):
        raise RuntimeContractError("simulated runtime-contract failure")

    monkeypatch.setattr(runtime_contract, "run_full_runtime_contract", _raiser)

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        frozen_record_path=record_path, identity_path=identity_path, output_root=output_root, paths=paths,
    )

    with pytest.raises(RuntimeContractError, match="simulated runtime-contract failure"):
        retry_bridge.main_from_manifest(manifest_path)

    assert not output_root.exists()
    assert "wandb" not in sys.modules


# --- 2: manifest provenance merged onto the SAME proposal_intake record -----

def test_manifest_provenance_merged_onto_durable_intake_before_wandb_init(tmp_path, monkeypatch):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "o1"
    paths = _paths(tmp_path / "p", monkeypatch)
    retry_trial_id = sweep.trial_id(written["configuration_id"], execution_generation=2)
    provenance_path = output_root / retry_trial_id / "execution_provenance.json"

    _noop_runtime_contract(monkeypatch)

    observed: "dict[str, object]" = {}

    class _ObservingWandbModule(types.ModuleType):
        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def init(self, **kwargs):
            observed["provenance_at_init_time"] = json.loads(provenance_path.read_text(encoding="utf-8"))
            from tests.test_sweep_v1_exact_retry_bridge import _FakeWandbRun
            return _FakeWandbRun("fake-run", written["wandb_sweep_id"], kwargs["config"])

    monkeypatch.setitem(sys.modules, "wandb", _ObservingWandbModule("wandb"))
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id=written["wandb_sweep_id"], stop_before_training=True,
        frozen_record_path=record_path, identity_path=identity_path, output_root=output_root, paths=paths,
    )
    manifest_sha256 = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_sha256"]

    retry_bridge.main_from_manifest(manifest_path)

    at_init = observed["provenance_at_init_time"]
    assert at_init["provenance_stage"] == "proposal_intake"
    assert at_init["wandb_run_id"] is None
    assert at_init["launch_manifest_path"] == str(manifest_path.resolve())
    assert at_init["launch_manifest_sha256"] == manifest_sha256
    assert at_init["launch_manifest_label"] == "sweep_v1_exact_retry_test_manifest_v001"


# --- 3: rehearsal never calls run_prepared_trial_in_production ---------------

def test_rehearsal_manifest_selects_executor_and_finishes_run_but_never_starts_training(
    tmp_path, monkeypatch, fake_wandb_module,
):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "o1"
    paths = _paths(tmp_path / "p", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)

    fake = fake_wandb_module(_AXES, sweep_id="rehearsal-sweep-abc")

    def _must_not_be_called(**kwargs):
        raise AssertionError("run_prepared_trial_in_production must never be called during a rehearsal")

    monkeypatch.setattr(retry_bridge, "run_prepared_trial_in_production", _must_not_be_called)
    monkeypatch.setattr(
        orchestration, "execute_prepared_pilot_run_monolithic",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("NH orchestration must never be invoked")),
    )

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        frozen_record_path=record_path, identity_path=identity_path, output_root=output_root, paths=paths,
    )

    exit_code = retry_bridge.main_from_manifest(manifest_path)

    assert exit_code == 0
    assert fake.run.finished is True
    assert fake.run.logged == []
    assert fake.run.summary["flashnh/rehearsal_stopped_before_training"] is True
    assert fake.run.summary["flashnh/executor_mode_selected"] == "monolithic"
    assert fake.run.summary["flashnh/retry_of_trial_id"] == written["trial_id"]

    retry_trial_id = fake.run.summary["flashnh/trial_id"]
    provenance = json.loads((output_root / retry_trial_id / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["provenance_stage"] == "executor_mode_selected"
    assert provenance["executor_mode"] == "monolithic"
    assert "execution_status" not in provenance  # never reached a terminal training outcome


# --- 4: production manifest runs the full path, never stops early -----------

def test_production_manifest_runs_full_path_and_calls_run_prepared_trial_in_production(
    tmp_path, monkeypatch, fake_wandb_module,
):
    torch = pytest.importorskip("torch")
    written = _write_frozen_record(tmp_path, wandb_sweep_id="4x3btz2s")
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "o2"
    paths = _paths(tmp_path / "p", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)

    fake = fake_wandb_module(_AXES, sweep_id="4x3btz2s")

    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.22, 5: 0.25, 6: 0.28, 7: 0.30, 8: 0.32,
              9: 0.40, 10: 0.38, 11: 0.36, 12: 0.35}

    def fake_execute(**kwargs):
        return _fake_result(nh_run_dir, checkpoint_epochs=epochs, screening_scores=scores, n_basins=kwargs["screening_basin_ids"].__len__())

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_PRODUCTION, wandb_sweep_id="4x3btz2s", stop_before_training=False,
        frozen_record_path=record_path, identity_path=identity_path, output_root=output_root, paths=paths,
    )

    exit_code = retry_bridge.main_from_manifest(manifest_path)

    assert exit_code == 0
    assert fake.run.finished is True
    assert fake.run.logged == [{"flashnh/best_score": pytest.approx(0.40)}]
    assert fake.run.summary["flashnh/valid"] is True
    assert "flashnh/rehearsal_stopped_before_training" not in fake.run.summary


# --- 5: wandb.init() failure durable evidence, manifest mode ----------------

def test_manifest_mode_wandb_init_exception_produces_durable_terminal_evidence(tmp_path, monkeypatch):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "o1"
    paths = _paths(tmp_path / "p", monkeypatch)
    retry_trial_id = sweep.trial_id(written["configuration_id"], execution_generation=2)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)

    class _RaisingWandbModule(types.ModuleType):
        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def init(self, **kwargs):
            raise RuntimeError("simulated manifest-mode wandb.init() failure")

    monkeypatch.setitem(sys.modules, "wandb", _RaisingWandbModule("wandb"))

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        frozen_record_path=record_path, identity_path=identity_path, output_root=output_root, paths=paths,
    )

    with pytest.raises(SystemExit, match="wandb.init"):
        retry_bridge.main_from_manifest(manifest_path)

    provenance = json.loads((output_root / retry_trial_id / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["provenance_stage"] == "wandb_init_failed"
    assert provenance["retry_of_trial_id"] == written["trial_id"]
    assert "simulated manifest-mode wandb.init()" in provenance["error"]
    # Manifest provenance must have survived onto this terminal-evidence record too.
    assert provenance["launch_manifest_label"] == "sweep_v1_exact_retry_test_manifest_v001"


# --- 6: successful association recorded before preparation, manifest mode ---

def test_manifest_mode_wandb_association_recorded_before_preparation_starts(
    tmp_path, monkeypatch, fake_wandb_module,
):
    torch = pytest.importorskip("torch")
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "o1"
    paths = _paths(tmp_path / "p", monkeypatch)
    retry_trial_id = sweep.trial_id(written["configuration_id"], execution_generation=2)
    provenance_path = output_root / retry_trial_id / "execution_provenance.json"
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)

    fake = fake_wandb_module(_AXES, sweep_id="rehearsal-sweep-abc")

    observed: "dict[str, object]" = {}
    real_prepare = retry_bridge.prepare_bayesian_proposal

    def _spying_prepare(*, proposal, paths):
        observed["provenance_at_prepare_time"] = json.loads(provenance_path.read_text(encoding="utf-8"))
        return real_prepare(proposal=proposal, paths=paths)

    monkeypatch.setattr(retry_bridge, "prepare_bayesian_proposal", _spying_prepare)

    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {e: 0.10 + 0.01 * e for e in epochs}

    def fake_execute(**kwargs):
        return _fake_result(nh_run_dir, checkpoint_epochs=epochs, screening_scores=scores, n_basins=kwargs["screening_basin_ids"].__len__())

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        frozen_record_path=record_path, identity_path=identity_path, output_root=output_root, paths=paths,
    )

    retry_bridge.main_from_manifest(manifest_path)

    at_prepare = observed["provenance_at_prepare_time"]
    assert at_prepare["provenance_stage"] == "wandb_associated"
    assert at_prepare["wandb_run_id"] == fake.run.id
    assert at_prepare["wandb_sweep_id"] == fake.run.sweep_id
