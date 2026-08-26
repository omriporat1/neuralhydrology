"""Tests for the Layer-B pre-execution provenance repair: a real W&B-assigned
proposal must be durably, locally recoverable before any failure-prone
preparation/config-write/execution step, per docs/decision_log.md's
"Repair Pre-Execution Layer-B Provenance" entry.

Covers, in order:
  1. proposal acquired -> intake provenance written -> prepare_bayesian_proposal
     raises -> provenance remains inspectable with the exact proposal + W&B IDs.
  2. proposal acquired -> preparation succeeds -> intake enriched -> config
     write raises -> same proposal/config linkage remains durable.
  3. preparation/write succeed -> execution raises -> the pre-existing
     execute_prepared_trial STARTED/INVALID behavior remains correct on top
     of the new intake/enrichment writes.
  4. full VALID path via scripts/run_sweep_v1_wandb_bridge.py::main() with a
     fake in-process wandb module -- finite objective logged, full
     provenance progression on disk.
  5. INVALID path via the same full bridge main() -- no finite
     flashnh/best_score is ever logged.
  6. retry semantics -- the durable intake record alone (no new W&B
     proposal) reproduces an identical configuration_id/proposal_id, with
     only the attempt-specific trial_id/execution_generation changing.

Never imports the real ``wandb`` package; never starts real NH training for
cases 1/2/6 (execute_prepared_trial/run_prepared_trial_in_production are not
reached). Cases 3/4/5 use the same real, torch-backed checkpoint/optimizer-
state fixture as tests/test_sweep_v1_execution.py, with
pilot_orchestration.execute_prepared_pilot_run_monolithic monkeypatched to a
fake receipt -- exactly test_sweep_v1_execution.py's established pattern --
so no real NH/torch training ever starts.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from src.baseline import pilot_orchestration as orchestration
from src.baseline import sweep_v1_campaign as sweep
from src.baseline.pilot_screening_eval import PRIMARY_METRIC_NAME, SCREENING_METRIC_SCOPE
from src.baseline.sweep_v1_execution import (
    SweepV1ExecutionError, enrich_layer_b_provenance, execute_prepared_trial, write_proposal_intake_provenance,
)
from src.baseline.sweep_v1_production_adapter import (
    PreparationPaths, SweepV1PreparationError, canonicalize_wandb_proposal, prepare_bayesian_proposal,
    write_prepared_proposal,
)
from tests._pilot_support import (
    BASELINE_POLICY_PATH, PILOT_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR,
    build_full_union_package, write_screening_basin_ids_file,
)

import scripts.run_sweep_v1_wandb_bridge as bridge


# --- fixture plumbing (duplicated from tests/test_sweep_v1_production_adapter.py's
# private helpers of the same shape -- repo convention for private test
# helpers, see test_prepared_execution_core.py's module docstring) ----------

def _paths(tmp_path, monkeypatch):
    package = build_full_union_package(tmp_path / "package")
    manifests = package / "manifests"
    (manifests / "file_checksums.csv").write_text("relative_path,sha256,size_bytes,artifact_role\n", encoding="utf-8")
    (package / "run_provenance.json").write_text('{"fixture":true}\n', encoding="utf-8")
    import src.baseline.sweep_v1_production_adapter as adapter
    monkeypatch.setattr(adapter, "PACKAGE_MANIFEST_SHA256", hashlib.sha256((manifests / "package_manifest.json").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_FILE_CHECKSUMS_SHA256", hashlib.sha256((manifests / "file_checksums.csv").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_RUN_PROVENANCE_SHA256", hashlib.sha256((package / "run_provenance.json").read_bytes()).hexdigest())
    splits = tmp_path / "canonical_splits"; splits.mkdir()
    for source in Path(SPLITS_DIR).glob("*.txt"):
        (splits / source.name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:400])
    monkeypatch.setattr("src.baseline.pilot_lead06_config.sha256_of", lambda _: sweep.SCREENING_ARTIFACT_SHA256)
    return PreparationPaths(BASELINE_POLICY_PATH, package, splits, screening)


_AXES = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10, "output_dropout": 0.25, "batch_size": 256}


def _write_real_checkpoints(nh_run_dir: Path, epochs, torch, *, updates_per_epoch: int = 100) -> None:
    nh_run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
        state_dict = {"state": {0: {"step": torch.tensor(epoch * updates_per_epoch)}}, "param_groups": []}
        torch.save(state_dict, nh_run_dir / f"optimizer_state_epoch{epoch:03d}.pt")


def _screening_event(epoch: int, score, *, n_basins: int) -> dict:
    return {
        "scope": SCREENING_METRIC_SCOPE, "authoritative": False, "epoch": epoch,
        "epoch_role": "stopping_eligible", "stopping_eligible": True,
        "n_screening_basins_requested": n_basins,
        "primary_metric_name": PRIMARY_METRIC_NAME, "primary_metric_median": score,
        "primary_metric_distribution": {},
        "raw_space_metrics": {
            "n_basins_requested": n_basins, "n_basins_evaluated": n_basins,
            "n_basins_area_excluded": 0, "area_derivation_excluded": [], "per_basin": [],
        },
    }


def _fake_result(nh_run_dir: Path, *, checkpoint_epochs, screening_scores: "dict[int, float]", n_basins: int
                 ) -> "orchestration.PreparedPilotExecutionResult":
    checkpoint_inventory = {
        epoch: orchestration.PhysicalCheckpoint(
            epoch=epoch, path=nh_run_dir / f"model_epoch{epoch:03d}.pt", owning_run_dir=nh_run_dir
        )
        for epoch in checkpoint_epochs
    }
    screening_events = [_screening_event(epoch, score, n_basins=n_basins) for epoch, score in sorted(screening_scores.items())]
    return orchestration.PreparedPilotExecutionResult(
        final_status="completed_at_full_budget", blocked_reason=None,
        effective_policy={"max_epoch_budget": 12, "performance_early_stopping_enabled": False},
        nh_run_dir=nh_run_dir, blocked=False, stopped=False, stop_reason=None,
        checkpoint_inventory=checkpoint_inventory, early_stopping_state={}, screening_events=screening_events,
    )


# --- fake in-process wandb module for full bridge main() tests --------------

class _FakeWandbRun:
    def __init__(self, run_id: str, sweep_id: str, config: dict, *, entity: str = "test-entity", project: str = "test-project"):
        self.id = run_id
        self.sweep_id = sweep_id
        self.config = dict(config)
        self.entity = entity
        self.project = project
        self.summary: "dict[str, object]" = {}
        self.logged: "list[dict]" = []
        self.finished = False

    def log(self, data):
        self.logged.append(dict(data))

    def finish(self):
        self.finished = True


class _FakeWandbSweepHandle:
    def __init__(self, config: dict):
        self.config = config


class _FakeWandbApi:
    def __init__(self, sweep_config: dict):
        self._sweep_config = sweep_config

    def sweep(self, path):
        return _FakeWandbSweepHandle(self._sweep_config)


_PRODUCTION_METRIC_CONTRACT = {"method": "bayes", "metric": {"name": "flashnh/best_score", "goal": "maximize"}}


class _FakeWandbModule(types.ModuleType):
    def __init__(self, *, config: dict, run_id: str = "fake-run-0001", sweep_id: str = bridge.PRODUCTION_WANDB_SWEEP_ID,
                 sweep_config: "dict | None" = None):
        super().__init__("wandb")
        self._config = config
        self._run_id = run_id
        self._sweep_id = sweep_id
        self._sweep_config = sweep_config if sweep_config is not None else dict(_PRODUCTION_METRIC_CONTRACT)
        self.run: "_FakeWandbRun | None" = None

    def init(self, **kwargs):
        self.run = _FakeWandbRun(self._run_id, self._sweep_id, self._config)
        return self.run

    def Api(self):
        return _FakeWandbApi(self._sweep_config)


@pytest.fixture
def fake_wandb_module(monkeypatch):
    def _make(config, *, sweep_id: str = bridge.PRODUCTION_WANDB_SWEEP_ID, sweep_config: "dict | None" = None):
        fake = _FakeWandbModule(config=config, sweep_id=sweep_id, sweep_config=sweep_config)
        monkeypatch.setitem(sys.modules, "wandb", fake)
        return fake
    return _make


@pytest.fixture(autouse=True)
def _no_real_wandb_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


def _bridge_argv(*, paths: PreparationPaths, output_root: Path, proposal_order: int) -> list:
    return [
        "run_sweep_v1_wandb_bridge.py",
        "--package-root", str(paths.package_root),
        "--screening-basin-ids", str(paths.screening_basin_ids_path),
        "--output-root", str(output_root),
        "--proposal-order", str(proposal_order),
        "--baseline-policy-path", str(paths.baseline_policy_path),
        "--base-pilot-policy-path", str(PILOT_POLICY_PATH),
    ]


def _patch_real_canonical_split_shas_for_local_checkout(monkeypatch) -> None:
    """``run_sweep_v1_wandb_bridge.main`` always reads the real committed
    ``ROOT/config/stage1_baseline_splits_v001`` splits (by design -- a
    production bridge must never accept a caller-supplied splits directory
    for sealed-set-adjacent inputs), never the tmp-copied splits ``_paths``
    builds for the other tests in this file. On a local Windows checkout with
    ``core.autocrlf`` those real files are on-disk as CRLF, so their raw bytes
    do not match ``DEVELOPMENT_SPLIT_SHA256``/``SPATIAL_HOLDOUT_SPLIT_SHA256``
    (both frozen from the LF-normalized committed content) even though the
    committed content itself is correct -- a pre-existing local-checkout
    artifact, unrelated to this repair. Pin the two constants to whatever is
    actually on disk right now so cases 4/5 can exercise the real bridge
    end-to-end on any checkout without weakening the pins for any other test.
    """
    import src.baseline.sweep_v1_production_adapter as adapter_module
    real_splits = bridge.ROOT / "config" / "stage1_baseline_splits_v001"
    monkeypatch.setattr(
        adapter_module, "DEVELOPMENT_SPLIT_SHA256",
        hashlib.sha256((real_splits / "development_train.txt").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        adapter_module, "SPATIAL_HOLDOUT_SPLIT_SHA256",
        hashlib.sha256((real_splits / "spatial_holdout_nonca.txt").read_bytes()).hexdigest(),
    )


# --- case 1: prepare_bayesian_proposal raises after intake -------------------

def test_prepare_failure_after_intake_leaves_durable_recoverable_provenance(tmp_path):
    output_root = tmp_path / "out"
    intake = write_proposal_intake_provenance(
        output_root=output_root, axes=_AXES, search_arm="bayesian",
        proposal_order=7, wandb_sweep_id="prod-sweep", wandb_run_id="run-7",
    )
    output_dir = output_root / intake["trial_id"]
    assert (output_dir / "execution_provenance.json").is_file()

    # A structurally plausible but non-production package: the frozen
    # production SHA pins are NOT monkeypatched, so preparation must fail at
    # artifact-identity verification -- the same real failure exercised by
    # test_structurally_plausible_fake_package_fails_the_production_v002_pin.
    package = build_full_union_package(tmp_path / "fabricated_v002")
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:400])
    paths = PreparationPaths(BASELINE_POLICY_PATH, package, SPLITS_DIR, screening)
    proposal = canonicalize_wandb_proposal(
        _AXES, metadata={"proposal_order": 7, "wandb_sweep_id": "prod-sweep", "wandb_run_id": "run-7"},
    )
    with pytest.raises(SweepV1PreparationError):
        prepare_bayesian_proposal(proposal=proposal, paths=paths)

    recovered = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert recovered["provenance_stage"] == "proposal_intake"
    assert recovered["hyperparameters"] == intake["hyperparameters"]
    assert recovered["wandb_sweep_id"] == "prod-sweep"
    assert recovered["wandb_run_id"] == "run-7"
    assert recovered["proposal_order"] == 7
    assert recovered["configuration_id"] == intake["configuration_id"]
    assert recovered["proposal_id"] == intake["proposal_id"]
    assert recovered["trial_id"] == intake["trial_id"]
    assert recovered["objective_score"] is None


# --- case 2: write_prepared_proposal raises after prepared-stage enrichment -

def test_config_write_failure_after_prepared_enrichment_leaves_durable_linkage(tmp_path, monkeypatch):
    paths = _paths(tmp_path / "prep", monkeypatch)
    output_root = tmp_path / "out"
    intake = write_proposal_intake_provenance(
        output_root=output_root, axes=_AXES, search_arm="bayesian",
        proposal_order=7, wandb_sweep_id="prod-sweep", wandb_run_id="run-7",
    )
    output_dir = output_root / intake["trial_id"]

    proposal = canonicalize_wandb_proposal(
        _AXES, metadata={"proposal_order": 7, "wandb_sweep_id": "prod-sweep", "wandb_run_id": "run-7"},
    )
    prepared = prepare_bayesian_proposal(proposal=proposal, paths=paths)
    enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields=prepared.evidence)

    import src.baseline.sweep_v1_production_adapter as adapter_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure during NH config write")

    monkeypatch.setattr(adapter_module, "write_generated_config", _boom)
    with pytest.raises(RuntimeError, match="simulated failure"):
        write_prepared_proposal(prepared, output_dir)

    recovered = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert recovered["provenance_stage"] == "prepared"
    assert recovered["configuration_id"] == intake["configuration_id"]
    assert recovered["proposal_id"] == intake["proposal_id"]
    assert recovered["trial_id"] == intake["trial_id"]
    assert recovered["hyperparameters"] == intake["hyperparameters"]
    assert "generated_nh_config_sha256" not in recovered
    assert recovered["objective_score"] is None


def test_enrich_rejects_trial_id_mismatch(tmp_path):
    output_root = tmp_path / "out"
    intake = write_proposal_intake_provenance(
        output_root=output_root, axes=_AXES, search_arm="bayesian",
        proposal_order=7, wandb_sweep_id="prod-sweep", wandb_run_id="run-7",
    )
    output_dir = output_root / intake["trial_id"]
    with pytest.raises(SweepV1ExecutionError):
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields={"trial_id": "some_other_trial_id"})


# --- case 3: execution raises after prepared_with_config enrichment ---------

def test_full_prepare_enrich_then_execution_failure_preserves_existing_behavior(tmp_path, monkeypatch):
    paths = _paths(tmp_path / "prep", monkeypatch)
    output_root = tmp_path / "out"
    intake = write_proposal_intake_provenance(
        output_root=output_root, axes=_AXES, search_arm="bayesian",
        proposal_order=7, wandb_sweep_id="prod-sweep", wandb_run_id="run-7",
    )
    output_dir = output_root / intake["trial_id"]

    proposal = canonicalize_wandb_proposal(
        _AXES, metadata={"proposal_order": 7, "wandb_sweep_id": "prod-sweep", "wandb_run_id": "run-7"},
    )
    prepared = prepare_bayesian_proposal(proposal=proposal, paths=paths)
    enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields=prepared.evidence)
    # allow_layer_b_provenance=True: output_dir already durably holds the
    # proposal_intake/prepared execution_provenance.json written above;
    # write_generated_config's empty-directory guard must tolerate exactly
    # that one file, the same way the real bridge does.
    record = write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)
    enrich_layer_b_provenance(output_dir=output_dir, stage="prepared_with_config", fields=record)

    before = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert before["provenance_stage"] == "prepared_with_config"
    assert before["trial_id"] == intake["trial_id"]

    def _boom():
        raise RuntimeError("simulated NH execution failure")

    outcome = execute_prepared_trial(prepared_record=record, output_dir=output_dir, execute_prepared_run_fn=_boom)
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["failure_category"] == "technical_execution_failure"
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None

    final = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert final["execution_status"] == "INVALID"
    assert final["trial_id"] == intake["trial_id"]
    assert final["preparation_record"]["trial_id"] == intake["trial_id"]
    assert final["preparation_record"]["wandb_sweep_id"] == "prod-sweep"


# --- narrow allow_layer_b_provenance safety-boundary tests -------------------
# (repairs the Codex-flagged force=True blocker: write_generated_config must
# never silently replace an already-generated scientific artifact, even when
# a durable Layer-B execution_provenance.json legitimately pre-exists.)

def _intake_and_prepare(tmp_path, monkeypatch, *, proposal_order: int = 7):
    paths = _paths(tmp_path / "prep", monkeypatch)
    output_root = tmp_path / "out"
    intake = write_proposal_intake_provenance(
        output_root=output_root, axes=_AXES, search_arm="bayesian",
        proposal_order=proposal_order, wandb_sweep_id="prod-sweep", wandb_run_id="run-7",
    )
    output_dir = output_root / intake["trial_id"]
    proposal = canonicalize_wandb_proposal(
        _AXES, metadata={"proposal_order": proposal_order, "wandb_sweep_id": "prod-sweep", "wandb_run_id": "run-7"},
    )
    prepared = prepare_bayesian_proposal(proposal=proposal, paths=paths)
    return paths, output_dir, prepared, intake


def test_provenance_only_directory_succeeds_and_preserves_provenance_bytes(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    provenance_path = output_dir / "execution_provenance.json"
    before_bytes = provenance_path.read_bytes()

    record = write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    assert provenance_path.read_bytes() == before_bytes
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "generation_manifest.json").is_file()
    assert record["generated_nh_config_sha256"] == hashlib.sha256((output_dir / "config.yaml").read_bytes()).hexdigest()


def test_existing_config_yaml_blocks_generation_before_mutation(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    provenance_path = output_dir / "execution_provenance.json"
    provenance_before = provenance_path.read_bytes()
    config_path = output_dir / "config.yaml"
    config_path.write_bytes(b"not a real generated config\n")
    config_before = config_path.read_bytes()

    with pytest.raises(Exception):
        write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    assert config_path.read_bytes() == config_before
    assert provenance_path.read_bytes() == provenance_before
    assert not (output_dir / "generation_manifest.json").exists()
    assert not (output_dir / "train_basins.txt").exists()


@pytest.mark.parametrize("basin_file", ["train_basins.txt", "validation_basins.txt", "test_basins.txt"])
def test_existing_basin_file_blocks_generation_before_mutation(tmp_path, monkeypatch, basin_file):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    provenance_path = output_dir / "execution_provenance.json"
    provenance_before = provenance_path.read_bytes()
    basin_path = output_dir / basin_file
    basin_path.write_bytes(b"01010101\n")
    basin_before = basin_path.read_bytes()

    with pytest.raises(Exception):
        write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    assert basin_path.read_bytes() == basin_before
    assert provenance_path.read_bytes() == provenance_before
    assert not (output_dir / "config.yaml").exists()
    assert not (output_dir / "generation_manifest.json").exists()


def test_existing_generation_manifest_blocks_generation_before_mutation(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    provenance_path = output_dir / "execution_provenance.json"
    provenance_before = provenance_path.read_bytes()
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_bytes(b'{"fake": true}')
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(Exception):
        write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    assert manifest_path.read_bytes() == manifest_before
    assert provenance_path.read_bytes() == provenance_before
    assert not (output_dir / "config.yaml").exists()
    assert not (output_dir / "train_basins.txt").exists()


def test_unrelated_pre_existing_file_blocks_generation_before_mutation(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    provenance_path = output_dir / "execution_provenance.json"
    provenance_before = provenance_path.read_bytes()
    stray_path = output_dir / "some_unrelated_file.txt"
    stray_path.write_text("unrelated", encoding="utf-8")

    with pytest.raises(Exception):
        write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    assert provenance_path.read_bytes() == provenance_before
    assert not (output_dir / "config.yaml").exists()
    assert not (output_dir / "generation_manifest.json").exists()
    assert not (output_dir / "train_basins.txt").exists()


def test_wrong_trial_provenance_blocks_generation_before_any_write(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    provenance_path = output_dir / "execution_provenance.json"
    foreign = json.loads(provenance_path.read_text(encoding="utf-8"))
    foreign["trial_id"] = "some_completely_different_trial_id"
    provenance_path.write_text(json.dumps(foreign), encoding="utf-8")
    provenance_before = provenance_path.read_bytes()

    with pytest.raises(SweepV1PreparationError):
        write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    assert provenance_path.read_bytes() == provenance_before
    assert not (output_dir / "config.yaml").exists()
    assert not (output_dir / "generation_manifest.json").exists()
    assert not (output_dir / "train_basins.txt").exists()


def test_missing_trial_id_provenance_blocks_generation_before_mutation(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    provenance_path = output_dir / "execution_provenance.json"
    malformed = json.loads(provenance_path.read_text(encoding="utf-8"))
    del malformed["trial_id"]
    provenance_path.write_text(json.dumps(malformed), encoding="utf-8")
    provenance_before = provenance_path.read_bytes()

    with pytest.raises(SweepV1PreparationError):
        write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    assert provenance_path.read_bytes() == provenance_before
    assert not (output_dir / "config.yaml").exists()
    assert not (output_dir / "generation_manifest.json").exists()
    assert not (output_dir / "train_basins.txt").exists()


def test_null_trial_id_provenance_blocks_generation_before_mutation(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    provenance_path = output_dir / "execution_provenance.json"
    malformed = json.loads(provenance_path.read_text(encoding="utf-8"))
    malformed["trial_id"] = None
    provenance_path.write_text(json.dumps(malformed), encoding="utf-8")
    provenance_before = provenance_path.read_bytes()

    with pytest.raises(SweepV1PreparationError):
        write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    assert provenance_path.read_bytes() == provenance_before
    assert not (output_dir / "config.yaml").exists()
    assert not (output_dir / "generation_manifest.json").exists()
    assert not (output_dir / "train_basins.txt").exists()


def test_repeated_invocation_against_generated_directory_does_not_overwrite(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    hashes_before = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in ("config.yaml", "generation_manifest.json", "train_basins.txt",
                     "validation_basins.txt", "test_basins.txt", "execution_provenance.json")
    }

    with pytest.raises(Exception):
        write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)

    hashes_after = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in hashes_before
    }
    assert hashes_after == hashes_before


def test_default_call_site_still_requires_strict_empty_directory(tmp_path, monkeypatch):
    _, output_dir, prepared, intake = _intake_and_prepare(tmp_path, monkeypatch)
    # No allow_layer_b_provenance kwarg at all -- the pre-existing durable
    # execution_provenance.json alone must still be rejected by default,
    # exactly as it always was before this repair.
    with pytest.raises(Exception):
        write_prepared_proposal(prepared, output_dir)


# --- cases 4/5: full bridge main() -------------------------------------------

def test_bridge_main_valid_trial_logs_finite_objective_and_full_provenance_progression(tmp_path, monkeypatch, fake_wandb_module):
    torch = pytest.importorskip("torch")
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    fake = fake_wandb_module(_AXES)
    output_root = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", _bridge_argv(
        paths=paths, output_root=output_root, proposal_order=bridge.PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER,
    ))

    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.22, 5: 0.25, 6: 0.28, 7: 0.30, 8: 0.32,
              9: 0.40, 10: 0.38, 11: 0.36, 12: 0.35}

    def fake_execute(**kwargs):
        return _fake_result(nh_run_dir, checkpoint_epochs=epochs, screening_scores=scores, n_basins=kwargs["screening_basin_ids"].__len__())

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    exit_code = bridge.main()

    assert exit_code == 0
    assert fake.run.finished is True
    assert fake.run.logged == [{"flashnh/best_score": pytest.approx(0.40)}]
    assert fake.run.summary["flashnh/valid"] is True

    trial_id = fake.run.summary["flashnh/trial_id"]
    output_dir = output_root / trial_id
    provenance = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["execution_status"] == "VALID"
    assert provenance["objective_score"] == pytest.approx(0.40)
    assert provenance["preparation_record"]["wandb_sweep_id"] == fake._sweep_id
    assert provenance["preparation_record"]["wandb_run_id"] == "fake-run-0001"


def test_bridge_main_invalid_trial_never_logs_a_finite_objective(tmp_path, monkeypatch, fake_wandb_module):
    torch = pytest.importorskip("torch")
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)
    fake = fake_wandb_module(_AXES)
    output_root = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", _bridge_argv(
        paths=paths, output_root=output_root, proposal_order=bridge.PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER,
    ))

    nh_run_dir = tmp_path / "nh_run"
    incomplete_epochs = [e for e in range(1, 13) if e != 7]
    _write_real_checkpoints(nh_run_dir, incomplete_epochs, torch)
    scores = {e: 0.30 + 0.01 * e for e in incomplete_epochs}

    def fake_execute(**kwargs):
        return _fake_result(nh_run_dir, checkpoint_epochs=incomplete_epochs, screening_scores=scores,
                            n_basins=kwargs["screening_basin_ids"].__len__())

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    exit_code = bridge.main()

    assert exit_code == 1
    assert fake.run.logged == []
    assert fake.run.summary["flashnh/valid"] is False

    trial_id = fake.run.summary["flashnh/trial_id"]
    output_dir = output_root / trial_id
    provenance = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["execution_status"] == "INVALID"
    assert provenance["objective_score"] is None


# --- case 6: retry recovers identical scientific identity without a new W&B proposal

def test_retry_recovers_identical_scientific_identity_from_durable_intake_record(tmp_path, monkeypatch):
    first = write_proposal_intake_provenance(
        output_root=tmp_path / "attempt1", axes=_AXES, search_arm="bayesian",
        proposal_order=7, wandb_sweep_id="prod-sweep", wandb_run_id="run-7",
    )
    # Recover the exact proposed axes from the durable record alone -- no
    # new W&B proposal is requested.
    recovered_axes = first["hyperparameters"]
    retry = write_proposal_intake_provenance(
        output_root=tmp_path / "attempt2", axes=recovered_axes, search_arm=first["search_arm"],
        proposal_order=first["proposal_order"], wandb_sweep_id=first["wandb_sweep_id"],
        wandb_run_id=first["wandb_run_id"], execution_generation=2, retry_of_trial_id=first["trial_id"],
    )
    assert retry["configuration_id"] == first["configuration_id"]
    assert retry["proposal_id"] == first["proposal_id"]
    assert retry["trial_id"] != first["trial_id"]
    assert retry["retry_of_trial_id"] == first["trial_id"]
    assert retry["execution_generation"] == 2

    # And the recovered proposal still prepares to the identical
    # configuration via the real, unmodified production adapter -- proving
    # the durable record alone suffices to retry the SAME scientific
    # configuration.
    paths = _paths(tmp_path / "prep", monkeypatch)
    proposal_retry = canonicalize_wandb_proposal(
        recovered_axes,
        metadata={"proposal_order": first["proposal_order"], "execution_generation": 2,
                  "wandb_sweep_id": first["wandb_sweep_id"], "wandb_run_id": first["wandb_run_id"]},
    )
    prepared_retry = prepare_bayesian_proposal(proposal=proposal_retry, paths=paths)
    assert prepared_retry.configuration_id == first["configuration_id"]
    assert prepared_retry.proposal_id == first["proposal_id"]
    assert prepared_retry.trial_id == retry["trial_id"]


# --- case: canonical validation failure at intake is still durably recorded -

def test_domain_invalid_proposal_is_rejected_with_a_durable_local_record(tmp_path):
    bad_axes = {**_AXES, "learning_rate": 9e-5}  # below the frozen [1e-4, 1e-3] domain
    with pytest.raises(ValueError):
        write_proposal_intake_provenance(
            output_root=tmp_path / "out", axes=bad_axes, search_arm="bayesian",
            proposal_order=7, wandb_sweep_id="prod-sweep", wandb_run_id="run-99",
        )
    rejected_path = tmp_path / "out" / "proposal_intake_rejected__wandb_run_run-99" / "execution_provenance.json"
    assert rejected_path.is_file()
    record = json.loads(rejected_path.read_text(encoding="utf-8"))
    assert record["provenance_stage"] == "proposal_intake_rejected"
    assert record["wandb_run_id"] == "run-99"
    assert record["raw_proposed_axes"]["learning_rate"] == 9e-5
    assert "rejection_reason" in record
