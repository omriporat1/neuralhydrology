"""Tests for the manifest-driven fresh-proposal bridge entry point,
``scripts.run_sweep_v1_wandb_bridge.main_from_manifest`` -- the fresh-bridge
sibling of ``tests/test_sweep_v1_exact_retry_bridge_manifest.py``. One
positional launch-manifest JSON replaces the CLI/environment-variable
interface ``main()`` uses; both entry points delegate to the identical shared
core ``_execute_fresh_proposal``, which ``tests/test_sweep_v1_wandb_bridge_provenance.py``
already exercises end to end via the legacy CLI entry point.

Unlike the exact-retry bridge, the fresh bridge has no frozen proposal record
to recover -- ``wandb.init()`` happens FIRST, and durable proposal-intake
provenance can only be written once the real controller-assigned ``run.config``
has passed identity/shape/objective-contract validation. So "manifest
provenance merged before any W&B call" (the retry bridge's ordering) becomes
here "manifest provenance merged onto the durable intake record strictly
before preparation begins" -- intake cannot predate controller assignment.

Covers, in order:
  1. ``main_from_manifest`` runs the shared runtime contract BEFORE any W&B
     import; a runtime-contract failure leaves no output directory behind and
     never imports wandb.
  2. Manifest provenance (``launch_manifest_path``/``launch_manifest_sha256``/
     ``launch_manifest_label``) is merged onto the SAME durable
     ``proposal_intake``-stage record before preparation starts.
  3. A rehearsal manifest (``mode="rehearsal"``, ``stop_before_training=True``)
     records the pure executor-mode selection and finishes the W&B run
     cleanly, but NEVER calls ``run_prepared_trial_in_production`` -- no
     training/NH-orchestration entry point is ever reached.
  4. A production manifest (``mode="production"``, ``stop_before_training=False``)
     runs the full path end-to-end, including
     ``run_prepared_trial_in_production`` -- it is never coerced into
     stopping early.
  5. A ``wandb.init()`` failure under manifest mode propagates directly with
     NO durable evidence fabricated -- there is no real run identity yet to
     key an incident record on, and intake can never predate controller
     assignment.
  6. A post-init controller-assignment rejection (sweep-identity mismatch)
     under manifest mode leaves a durable, identity-safe bootstrap incident
     record, and still finishes the W&B run cleanly.

Never imports the real ``wandb`` package. Never touches the network. Never
targets the real production sweep (``4x3btz2s``) except via the fully faked
in-process ``wandb`` module, and never starts real NH training (all training
call sites are monkeypatched to fakes, same as
``tests/test_sweep_v1_wandb_bridge_provenance.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.baseline import pilot_orchestration as orchestration
from src.baseline import sweep_v1_runtime_contract as runtime_contract
from src.baseline.sweep_v1_launch_manifest import MODE_PRODUCTION, MODE_REHEARSAL
from src.baseline.sweep_v1_production_adapter import PreparationPaths
from src.baseline.sweep_v1_runtime_contract import RuntimeContractError
from src.baseline.sweep_v1_wandb_bridge_manifest import (
    PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER, REHEARSAL_RESERVED_EXECUTION_GENERATION,
    REHEARSAL_RESERVED_PROPOSAL_ORDER, write_wandb_bridge_manifest,
)
from tests._pilot_support import PILOT_POLICY_PATH
from tests.test_sweep_v1_wandb_bridge_provenance import (
    _AXES, _fake_result, _no_real_wandb_module, _patch_real_canonical_split_shas_for_local_checkout,
    _paths, _write_real_checkpoints, fake_wandb_module,
)

import scripts.run_sweep_v1_wandb_bridge as bridge

__all__ = ["fake_wandb_module", "_no_real_wandb_module"]  # re-exported fixtures


def _noop_runtime_contract(monkeypatch):
    monkeypatch.setattr(runtime_contract, "run_full_runtime_contract", lambda **kwargs: None)


def _write_manifest(tmp_path, *, mode, wandb_sweep_id, stop_before_training, output_root,
                     paths: PreparationPaths, proposal_order=None, execution_generation=None,
                     manifest_name="manifest.json") -> Path:
    if proposal_order is None:
        proposal_order = (
            PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER if mode == MODE_PRODUCTION else REHEARSAL_RESERVED_PROPOSAL_ORDER
        )
    if execution_generation is None:
        execution_generation = 1 if mode == MODE_PRODUCTION else REHEARSAL_RESERVED_EXECUTION_GENERATION
    manifest_path = tmp_path / manifest_name
    write_wandb_bridge_manifest(
        manifest_path,
        manifest_label="sweep_v1_wandb_bridge_test_manifest_v001",
        created_at_utc="2026-08-26T00:00:00Z",
        mode=mode,
        expected_commit="0" * 40,
        expected_runtime_python=sys.executable,
        package_root=str(paths.package_root),
        screening_basin_ids_path=str(paths.screening_basin_ids_path),
        output_root=str(output_root),
        baseline_policy_path=str(paths.baseline_policy_path),
        base_pilot_policy_path=str(PILOT_POLICY_PATH),
        wandb_project="flashnh-stage1-test",
        wandb_sweep_id=wandb_sweep_id,
        proposal_order=proposal_order,
        execution_generation=execution_generation,
        stop_before_training=stop_before_training,
    )
    return manifest_path


# --- 1: runtime contract runs first, before any wandb import ----------------

def test_runtime_contract_failure_leaves_no_output_directory_and_no_wandb_import(tmp_path, monkeypatch):
    paths = _paths(tmp_path / "p", monkeypatch)
    output_root = tmp_path / "o1"

    def _raiser(**kwargs):
        raise RuntimeContractError("simulated runtime-contract failure")

    monkeypatch.setattr(runtime_contract, "run_full_runtime_contract", _raiser)

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        output_root=output_root, paths=paths,
    )

    with pytest.raises(RuntimeContractError, match="simulated runtime-contract failure"):
        bridge.main_from_manifest(manifest_path)

    assert not output_root.exists()
    assert "wandb" not in sys.modules


# --- 2: manifest provenance merged onto the durable intake record -----------

def test_manifest_provenance_merged_onto_durable_intake_before_preparation_starts(
    tmp_path, monkeypatch, fake_wandb_module,
):
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)
    output_root = tmp_path / "o1"
    fake_wandb_module(_AXES, sweep_id="rehearsal-sweep-abc")

    observed: "dict[str, object]" = {}
    real_prepare = bridge.prepare_bayesian_proposal

    def _spying_prepare(*, proposal, paths):
        trial_dirs = [entry for entry in output_root.iterdir() if entry.is_dir()]
        assert len(trial_dirs) == 1
        provenance_path = trial_dirs[0] / "execution_provenance.json"
        observed["at_prepare_time"] = json.loads(provenance_path.read_text(encoding="utf-8"))
        return real_prepare(proposal=proposal, paths=paths)

    monkeypatch.setattr(bridge, "prepare_bayesian_proposal", _spying_prepare)

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        output_root=output_root, paths=paths,
    )
    manifest_sha256 = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_sha256"]

    bridge.main_from_manifest(manifest_path)

    at_prepare = observed["at_prepare_time"]
    assert at_prepare["provenance_stage"] == "proposal_intake"
    assert at_prepare["wandb_run_id"] == "fake-run-0001"
    assert at_prepare["wandb_sweep_id"] == "rehearsal-sweep-abc"
    assert at_prepare["launch_manifest_path"] == str(manifest_path.resolve())
    assert at_prepare["launch_manifest_sha256"] == manifest_sha256
    assert at_prepare["launch_manifest_label"] == "sweep_v1_wandb_bridge_test_manifest_v001"


# --- 3: rehearsal never calls run_prepared_trial_in_production ---------------

def test_rehearsal_manifest_selects_executor_and_finishes_run_but_never_starts_training(
    tmp_path, monkeypatch, fake_wandb_module,
):
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)
    output_root = tmp_path / "o1"
    fake = fake_wandb_module(_AXES, sweep_id="rehearsal-sweep-abc")

    def _must_not_be_called(**kwargs):
        raise AssertionError("run_prepared_trial_in_production must never be called during a rehearsal")

    monkeypatch.setattr(bridge, "run_prepared_trial_in_production", _must_not_be_called)
    monkeypatch.setattr(
        orchestration, "execute_prepared_pilot_run_monolithic",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("NH orchestration must never be invoked")),
    )

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        output_root=output_root, paths=paths,
    )

    exit_code = bridge.main_from_manifest(manifest_path)

    assert exit_code == 0
    assert fake.run.finished is True
    assert fake.run.logged == []
    assert fake.run.summary["flashnh/rehearsal_stopped_before_training"] is True
    assert fake.run.summary["flashnh/executor_mode_selected"] == "monolithic"
    assert fake.run.summary["flashnh/proposal_order"] == REHEARSAL_RESERVED_PROPOSAL_ORDER
    assert fake.run.summary["flashnh/execution_generation"] == REHEARSAL_RESERVED_EXECUTION_GENERATION

    trial_id = fake.run.summary["flashnh/trial_id"]
    provenance = json.loads((output_root / trial_id / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["provenance_stage"] == "executor_mode_selected"
    assert provenance["executor_mode"] == "monolithic"
    assert "execution_status" not in provenance  # never reached a terminal training outcome


# --- 4: production manifest runs the full path, never stops early -----------

def test_production_manifest_runs_full_path_and_calls_run_prepared_trial_in_production(
    tmp_path, monkeypatch, fake_wandb_module,
):
    torch = pytest.importorskip("torch")
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)
    output_root = tmp_path / "o2"
    fake = fake_wandb_module(_AXES, sweep_id=bridge.PRODUCTION_WANDB_SWEEP_ID)

    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.22, 5: 0.25, 6: 0.28, 7: 0.30, 8: 0.32,
              9: 0.40, 10: 0.38, 11: 0.36, 12: 0.35}

    def fake_execute(**kwargs):
        return _fake_result(nh_run_dir, checkpoint_epochs=epochs, screening_scores=scores, n_basins=kwargs["screening_basin_ids"].__len__())

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_PRODUCTION, wandb_sweep_id=bridge.PRODUCTION_WANDB_SWEEP_ID, stop_before_training=False,
        output_root=output_root, paths=paths,
    )

    exit_code = bridge.main_from_manifest(manifest_path)

    assert exit_code == 0
    assert fake.run.finished is True
    assert fake.run.logged == [{"flashnh/best_score": pytest.approx(0.40)}]
    assert fake.run.summary["flashnh/valid"] is True
    assert "flashnh/rehearsal_stopped_before_training" not in fake.run.summary


# --- 5: wandb.init() failure -- no identity to fabricate evidence around ----

def test_manifest_mode_wandb_init_failure_propagates_without_fabricating_identity(tmp_path, monkeypatch):
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)
    output_root = tmp_path / "o1"

    import types

    class _RaisingWandbModule(types.ModuleType):
        def init(self, **kwargs):
            raise RuntimeError("simulated manifest-mode wandb.init() failure")

    monkeypatch.setitem(sys.modules, "wandb", _RaisingWandbModule("wandb"))

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        output_root=output_root, paths=paths,
    )

    with pytest.raises(RuntimeError, match="simulated manifest-mode wandb.init"):
        bridge.main_from_manifest(manifest_path)

    assert not output_root.exists()


# --- 6: post-init controller-assignment rejection leaves durable evidence ---

def test_manifest_mode_sweep_identity_mismatch_leaves_durable_bootstrap_incident_and_finishes_run(
    tmp_path, monkeypatch, fake_wandb_module,
):
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    _noop_runtime_contract(monkeypatch)
    output_root = tmp_path / "o1"
    # The real controller joined a DIFFERENT sweep than this manifest expects
    # -- must be refused before any durable proposal-intake write, but the
    # W&B run itself must still finish cleanly (it is a real, joined run).
    fake = fake_wandb_module(_AXES, sweep_id="some-other-sweep-not-in-manifest")

    manifest_path = _write_manifest(
        tmp_path, mode=MODE_REHEARSAL, wandb_sweep_id="rehearsal-sweep-abc", stop_before_training=True,
        output_root=output_root, paths=paths,
    )

    with pytest.raises(SystemExit, match="sweep_id"):
        bridge.main_from_manifest(manifest_path)

    assert fake.run.finished is True

    incident_dirs = [entry for entry in output_root.iterdir() if entry.is_dir()]
    assert len(incident_dirs) == 1
    assert incident_dirs[0].name.startswith("bootstrap_assignment_rejected__wandb_run_")
    record = json.loads((incident_dirs[0] / "execution_provenance.json").read_text(encoding="utf-8"))
    assert record["provenance_stage"] == "sweep_identity_rejected"
    assert record["expected_sweep_id"] == "rehearsal-sweep-abc"
    assert record["actual_sweep_id"] == "some-other-sweep-not-in-manifest"
    # No trial/proposal identity is ever fabricated for a pre-intake failure.
    assert "trial_id" not in record
    assert "proposal_id" not in record
