"""Tests for the operator-facing exact-retry launcher,
``scripts/run_sweep_v1_exact_retry_bridge.py``.

Covers, in order:
  1. The ``FLASHNH_SWEEP_V1_RETRY_BRIDGE_SELFTEST=resolve_only`` hook resolves
     the frozen record + pinned identity + retry identity and lands on a
     fresh output directory, importing no wandb and starting no execution.
  2. Refuses (before any wandb import or call) when the derived retry output
     directory already exists.
  3. Refuses (before any wandb import or call) when the pinned expected
     identity contradicts the frozen record.
  4. Full VALID path via a fake in-process wandb module: the run associates
     with the requested sweep, a finite objective is logged, provenance
     records ``retry_of_trial_id``/``execution_generation``, and the
     ORIGINAL attempt's directory is provably untouched byte-for-byte.
  5. Refuses (no silent fallback) when the fake run's ``sweep_id`` does not
     match the requested production sweep id.
  6. INVALID path: no finite objective is ever logged; ``flashnh/valid`` is
     False; ``retry_of_trial_id`` is still recorded.

Never imports the real ``wandb`` package. Cases 4-6 use the same real,
torch-backed checkpoint fixture as
tests/test_sweep_v1_wandb_bridge_provenance.py, with
``pilot_orchestration.execute_prepared_pilot_run_monolithic`` monkeypatched
to a fake receipt, so no real NH/torch training ever starts.
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
from src.baseline.sweep_v1_execution import write_proposal_intake_provenance
from src.baseline.sweep_v1_production_adapter import PreparationPaths
from src.baseline.sweep_v1_retry import SweepV1RetryError
from tests._pilot_support import (
    BASELINE_POLICY_PATH, PILOT_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR,
    build_full_union_package, write_screening_basin_ids_file,
)

import scripts.run_sweep_v1_exact_retry_bridge as retry_bridge

_AXES = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10, "output_dropout": 0.25, "batch_size": 256}

# Sanitized golden fixture matching the REAL Sweep-v1 production attempt001
# ``execution_provenance.json`` executed-attempt envelope shape (fetched
# directly off Moriah; baseline sha256
# 4da96c34dcaee4cad83c9371c62fccbf609fad05985f8b56e4d69f3633afb052). Real
# non-secret identity/hyperparameter values and evidence-tree paths -- no
# credentials or machine-private secrets are present. See the identical
# fixture (with commentary) in tests/test_sweep_v1_retry.py.
_REAL_ATTEMPT001_PREPARATION_RECORD = {
    "campaign_id": "stage1_phase_b_sweep_v1_original_domain_v001",
    "configuration_id": "sweep_v1_cfg_5731e180d1bf9d582afc",
    "domain_version": "original_domain_v001",
    "execution_generation": 1,
    "hyperparameters": {
        "batch_size": 512,
        "embedding_dropout": "0.08637149503762416",
        "hidden_size": 64,
        "learning_rate": "0.00024474898782657741",
        "output_dropout": "0.20096018948154892",
    },
    "model_seed": 967139,
    "objective_score": None,
    "proposal_id": "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001",
    "proposal_order": 1,
    "search_arm": "bayesian",
    "trial_id": (
        "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc__mf12x50000__seedA967139__attempt001"
    ),
    "wandb_run_id": "7nxaim79",
    "wandb_sweep_id": "4x3btz2s",
}

_REAL_ATTEMPT001_ENVELOPE = {
    "campaign_id": "stage1_phase_b_sweep_v1_original_domain_v001",
    "configuration_id": "sweep_v1_cfg_5731e180d1bf9d582afc",
    "execution_generation": 1,
    "execution_status": "INVALID",
    "objective_score": None,
    "preparation_record": _REAL_ATTEMPT001_PREPARATION_RECORD,
    "proposal_id": "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001",
    "result": {"blocked": True, "final_status": "blocked_continuation_overshoot_conflict"},
    "retry_of_trial_id": None,
    "search_arm": "bayesian",
    "trial_id": _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"],
}


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


def _patch_real_canonical_split_shas_for_local_checkout(monkeypatch) -> None:
    """See tests/test_sweep_v1_wandb_bridge_provenance.py's identical helper
    docstring -- the retry bridge, like the production bridge, always reads
    the real committed ``ROOT/config/stage1_baseline_splits_v001`` splits,
    never a caller-supplied splits directory."""
    import src.baseline.sweep_v1_production_adapter as adapter_module
    real_splits = retry_bridge.ROOT / "config" / "stage1_baseline_splits_v001"
    monkeypatch.setattr(
        adapter_module, "DEVELOPMENT_SPLIT_SHA256",
        hashlib.sha256((real_splits / "development_train.txt").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        adapter_module, "SPATIAL_HOLDOUT_SPLIT_SHA256",
        hashlib.sha256((real_splits / "spatial_holdout_nonca.txt").read_bytes()).hexdigest(),
    )


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


def _fake_result(nh_run_dir: Path, *, checkpoint_epochs, screening_scores, n_basins: int):
    checkpoint_inventory = {
        epoch: orchestration.PhysicalCheckpoint(epoch=epoch, path=nh_run_dir / f"model_epoch{epoch:03d}.pt", owning_run_dir=nh_run_dir)
        for epoch in checkpoint_epochs
    }
    screening_events = [_screening_event(epoch, score, n_basins=n_basins) for epoch, score in sorted(screening_scores.items())]
    return orchestration.PreparedPilotExecutionResult(
        final_status="completed_at_full_budget", blocked_reason=None,
        effective_policy={"max_epoch_budget": 12, "performance_early_stopping_enabled": False},
        nh_run_dir=nh_run_dir, blocked=False, stopped=False, stop_reason=None,
        checkpoint_inventory=checkpoint_inventory, early_stopping_state={}, screening_events=screening_events,
    )


class _FakeWandbRun:
    def __init__(self, run_id: str, sweep_id: str, config: dict):
        self.id = run_id
        self.sweep_id = sweep_id
        self.config = dict(config)
        self.summary: "dict[str, object]" = {}
        self.logged: "list[dict]" = []
        self.finished = False

    def log(self, data):
        self.logged.append(dict(data))

    def finish(self):
        self.finished = True


class _FakeWandbModule(types.ModuleType):
    def __init__(self, *, config: dict, run_id: str = "fake-retry-run-0001", sweep_id: str = "prod-sweep-xyz"):
        super().__init__("wandb")
        self._config = config
        self._run_id = run_id
        self._sweep_id = sweep_id
        self.run: "_FakeWandbRun | None" = None
        self.captured_init_kwargs: "dict | None" = None

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def init(self, **kwargs):
        self.captured_init_kwargs = kwargs
        self.run = _FakeWandbRun(self._run_id, self._sweep_id, self._config)
        return self.run


@pytest.fixture
def fake_wandb_module(monkeypatch):
    def _make(config, **kwargs):
        fake = _FakeWandbModule(config=config, **kwargs)
        monkeypatch.setitem(sys.modules, "wandb", fake)
        return fake
    return _make


@pytest.fixture(autouse=True)
def _no_real_wandb_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


def _write_frozen_record(tmp_path, *, wandb_sweep_id="prod-sweep-xyz"):
    return write_proposal_intake_provenance(
        output_root=tmp_path / "attempt1", axes=_AXES, search_arm="bayesian",
        proposal_order=7, wandb_sweep_id=wandb_sweep_id, wandb_run_id="run-7",
    )


def _pinned_identity_path(tmp_path, written, *, mismatched_field=None, mismatched_value=None):
    # Pin against the record's own already-canonicalized hyperparameters
    # (continuous axes are stored as .17g strings, not raw floats -- see
    # sweep_v1_campaign.canonical_hyperparameters), not the raw _AXES floats.
    pinned = {
        "proposal_order": written["proposal_order"], "configuration_id": written["configuration_id"],
        "search_arm": written["search_arm"], "wandb_sweep_id": written["wandb_sweep_id"],
        "model_seed": sweep.MODEL_SEED_A, **written["hyperparameters"],
    }
    if mismatched_field is not None:
        pinned[mismatched_field] = mismatched_value
    path = tmp_path / "expected_identity.json"
    path.write_text(json.dumps(pinned), encoding="utf-8")
    return path


def _retry_argv(*, tmp_path, paths: PreparationPaths, frozen_record_path, expected_identity_path,
                output_root: Path, execution_generation: int = 2) -> list:
    return [
        "run_sweep_v1_exact_retry_bridge.py",
        "--frozen-proposal-record", str(frozen_record_path),
        "--expected-identity", str(expected_identity_path),
        "--execution-generation", str(execution_generation),
        "--package-root", str(paths.package_root),
        "--screening-basin-ids", str(paths.screening_basin_ids_path),
        "--output-root", str(output_root),
        "--baseline-policy-path", str(paths.baseline_policy_path),
        "--base-pilot-policy-path", str(PILOT_POLICY_PATH),
    ]


# --- case 1: resolve_only selftest hook --------------------------------------

def test_selftest_resolve_only_hook_never_imports_wandb(tmp_path, monkeypatch, capsys):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"

    monkeypatch.setenv(retry_bridge.ENV_SELFTEST, "resolve_only")
    monkeypatch.setattr(sys, "argv", [
        "run_sweep_v1_exact_retry_bridge.py",
        "--frozen-proposal-record", str(record_path),
        "--expected-identity", str(identity_path),
        "--execution-generation", "2",
        "--package-root", str(tmp_path / "unused_package"),
        "--screening-basin-ids", str(tmp_path / "unused_screening.txt"),
        "--output-root", str(output_root),
    ])

    exit_code = retry_bridge.main()

    assert exit_code == 0
    assert "wandb" not in sys.modules
    printed = json.loads(capsys.readouterr().out)
    assert printed["retry_identity"]["retry_of_trial_id"] == written["trial_id"]
    assert printed["retry_identity"]["execution_generation"] == 2
    assert printed["retry_identity"]["configuration_id"] == written["configuration_id"]
    assert not Path(printed["output_dir"]).exists()


# --- case 1b: resolve_only accepts the real attempt001 executed envelope ----

def test_selftest_resolve_only_accepts_real_attempt001_executed_envelope(tmp_path, monkeypatch, capsys):
    import copy
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    record_path = tmp_path / "execution_provenance.json"
    record_path.write_text(json.dumps(envelope), encoding="utf-8")

    pinned = {
        "proposal_order": 1, "configuration_id": "sweep_v1_cfg_5731e180d1bf9d582afc",
        "search_arm": "bayesian", "wandb_sweep_id": "4x3btz2s", "model_seed": 967139,
        **_REAL_ATTEMPT001_PREPARATION_RECORD["hyperparameters"],
    }
    identity_path = tmp_path / "expected_identity.json"
    identity_path.write_text(json.dumps(pinned), encoding="utf-8")
    output_root = tmp_path / "out_attempt002"

    monkeypatch.setenv(retry_bridge.ENV_SELFTEST, "resolve_only")
    monkeypatch.setattr(sys, "argv", [
        "run_sweep_v1_exact_retry_bridge.py",
        "--frozen-proposal-record", str(record_path),
        "--expected-identity", str(identity_path),
        "--execution-generation", "2",
        "--package-root", str(tmp_path / "unused_package"),
        "--screening-basin-ids", str(tmp_path / "unused_screening.txt"),
        "--output-root", str(output_root),
    ])

    exit_code = retry_bridge.main()

    assert exit_code == 0
    assert "wandb" not in sys.modules
    printed = json.loads(capsys.readouterr().out)
    assert printed["retry_identity"]["retry_of_trial_id"] == _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
    assert printed["retry_identity"]["execution_generation"] == 2
    assert printed["retry_identity"]["configuration_id"] == "sweep_v1_cfg_5731e180d1bf9d582afc"
    assert printed["retry_identity"]["proposal_id"] == "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001"
    assert printed["retry_identity"]["trial_id"] != _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
    assert not Path(printed["output_dir"]).exists()


# --- case 2: refuses on a pre-existing retry output directory ---------------

def test_refuses_when_retry_output_directory_already_exists(tmp_path, monkeypatch):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"

    retry_trial_id = sweep.trial_id(written["configuration_id"], execution_generation=2)
    (output_root / retry_trial_id).mkdir(parents=True)

    monkeypatch.setattr(sys, "argv", [
        "run_sweep_v1_exact_retry_bridge.py",
        "--frozen-proposal-record", str(record_path),
        "--expected-identity", str(identity_path),
        "--execution-generation", "2",
        "--package-root", str(tmp_path / "unused_package"),
        "--screening-basin-ids", str(tmp_path / "unused_screening.txt"),
        "--output-root", str(output_root),
    ])

    with pytest.raises(SystemExit, match="already exists"):
        retry_bridge.main()
    assert "wandb" not in sys.modules


# --- case 3: refuses on a pinned-identity contradiction ---------------------

def test_refuses_on_pinned_identity_contradiction(tmp_path, monkeypatch):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written, mismatched_field="hidden_size", mismatched_value=256)
    output_root = tmp_path / "out_retry"

    monkeypatch.setattr(sys, "argv", [
        "run_sweep_v1_exact_retry_bridge.py",
        "--frozen-proposal-record", str(record_path),
        "--expected-identity", str(identity_path),
        "--execution-generation", "2",
        "--package-root", str(tmp_path / "unused_package"),
        "--screening-basin-ids", str(tmp_path / "unused_screening.txt"),
        "--output-root", str(output_root),
    ])

    with pytest.raises(SweepV1RetryError):
        retry_bridge.main()
    assert "wandb" not in sys.modules
    assert not output_root.exists()


# --- case 4: full VALID retry path -------------------------------------------

def test_valid_retry_logs_objective_and_leaves_original_attempt_untouched(tmp_path, monkeypatch, fake_wandb_module):
    torch = pytest.importorskip("torch")
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)

    written = _write_frozen_record(tmp_path)
    original_dir = tmp_path / "attempt1" / written["trial_id"]
    original_bytes_before = (original_dir / "execution_provenance.json").read_bytes()

    record_path = original_dir / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"

    fake = fake_wandb_module(_AXES, sweep_id=written["wandb_sweep_id"])
    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path, paths=paths, frozen_record_path=record_path,
        expected_identity_path=identity_path, output_root=output_root,
    ))

    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.22, 5: 0.25, 6: 0.28, 7: 0.30, 8: 0.32,
              9: 0.40, 10: 0.38, 11: 0.36, 12: 0.35}

    def fake_execute(**kwargs):
        return _fake_result(nh_run_dir, checkpoint_epochs=epochs, screening_scores=scores, n_basins=kwargs["screening_basin_ids"].__len__())

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    exit_code = retry_bridge.main()

    assert exit_code == 0
    assert fake.run.finished is True
    assert fake.run.logged == [{"flashnh/best_score": pytest.approx(0.40)}]
    assert fake.run.summary["flashnh/valid"] is True
    assert fake.run.summary["flashnh/retry_of_trial_id"] == written["trial_id"]
    assert fake.run.summary["flashnh/execution_generation"] == 2
    assert fake.run.summary["flashnh/exact_retry"] is True

    retry_trial_id = fake.run.summary["flashnh/trial_id"]
    assert retry_trial_id != written["trial_id"]
    retry_provenance = json.loads((output_root / retry_trial_id / "execution_provenance.json").read_text(encoding="utf-8"))
    assert retry_provenance["execution_status"] == "VALID"
    assert retry_provenance["retry_of_trial_id"] == written["trial_id"]

    # The ORIGINAL attempt's own durable record is never opened for writing.
    assert (original_dir / "execution_provenance.json").read_bytes() == original_bytes_before


# --- case 5: refuses on a sweep_id association mismatch ----------------------

def test_refuses_when_run_sweep_id_does_not_match_requested_sweep(tmp_path, monkeypatch, fake_wandb_module):
    torch = pytest.importorskip("torch")
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)

    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"

    # The fake run reports a DIFFERENT sweep_id than the one requested --
    # simulates a server-side failure to associate the run with the sweep.
    fake = fake_wandb_module(_AXES, sweep_id="some-unrelated-sweep")
    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path, paths=paths, frozen_record_path=record_path,
        expected_identity_path=identity_path, output_root=output_root,
    ))

    with pytest.raises(SystemExit, match="did not associate"):
        retry_bridge.main()
    assert fake.run.finished is True  # finally: run.finish() still runs
    assert fake.run.logged == []


# --- case 6: INVALID retry path never logs a finite objective ---------------

def test_invalid_retry_never_logs_a_finite_objective(tmp_path, monkeypatch, fake_wandb_module):
    torch = pytest.importorskip("torch")
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)

    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"

    fake = fake_wandb_module(_AXES, sweep_id=written["wandb_sweep_id"])
    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path, paths=paths, frozen_record_path=record_path,
        expected_identity_path=identity_path, output_root=output_root,
    ))

    nh_run_dir = tmp_path / "nh_run"
    incomplete_epochs = [e for e in range(1, 13) if e != 7]
    _write_real_checkpoints(nh_run_dir, incomplete_epochs, torch)
    scores = {e: 0.30 + 0.01 * e for e in incomplete_epochs}

    def fake_execute(**kwargs):
        return _fake_result(nh_run_dir, checkpoint_epochs=incomplete_epochs, screening_scores=scores,
                            n_basins=kwargs["screening_basin_ids"].__len__())

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    exit_code = retry_bridge.main()

    assert exit_code == 1
    assert fake.run.logged == []
    assert fake.run.summary["flashnh/valid"] is False
    assert fake.run.summary["flashnh/retry_of_trial_id"] == written["trial_id"]

    retry_trial_id = fake.run.summary["flashnh/trial_id"]
    retry_provenance = json.loads((output_root / retry_trial_id / "execution_provenance.json").read_text(encoding="utf-8"))
    assert retry_provenance["execution_status"] == "INVALID"
    assert retry_provenance["objective_score"] is None


# --- Repair 2: durable intake exists BEFORE any W&B call --------------------

def test_durable_intake_exists_before_wandb_init_is_called(tmp_path, monkeypatch):
    """Direct proof of the required ordering: by the time wandb.init() is
    invoked, the durable Layer-B intake record is already on disk, at stage
    'proposal_intake', with no fabricated wandb_run_id."""
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"
    retry_trial_id = sweep.trial_id(written["configuration_id"], execution_generation=2)
    provenance_path = output_root / retry_trial_id / "execution_provenance.json"

    observed: "dict[str, object]" = {}

    class _ObservingWandbModule(types.ModuleType):
        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def init(self, **kwargs):
            observed["provenance_at_init_time"] = json.loads(provenance_path.read_text(encoding="utf-8"))
            return _FakeWandbRun("fake-run", written["wandb_sweep_id"], kwargs["config"])

    monkeypatch.setitem(sys.modules, "wandb", _ObservingWandbModule("wandb"))
    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path, paths=_paths(tmp_path / "prep", monkeypatch), frozen_record_path=record_path,
        expected_identity_path=identity_path, output_root=output_root,
    ))
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)

    # We only need to observe ordering up to wandb.init(); whatever happens to
    # the (unpatched, real) execution path afterwards -- typically an INVALID
    # trial, since no real training/checkpoints are wired into this fixture --
    # is irrelevant to this test.
    retry_bridge.main()

    assert provenance_path.is_file()  # written before the call we observed inside
    at_init = observed["provenance_at_init_time"]
    assert at_init["provenance_stage"] == "proposal_intake"
    assert at_init["wandb_run_id"] is None
    assert at_init["retry_of_trial_id"] == written["trial_id"]
    assert at_init["execution_generation"] == 2


# --- Repair 1: tag rejection is recorded durably and precedes any W&B call ---

def test_overlong_tags_are_rejected_before_wandb_import_with_durable_provenance_retained(tmp_path, monkeypatch):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"
    retry_trial_id = sweep.trial_id(written["configuration_id"], execution_generation=2)

    # Force an overlong tag through, simulating a future regression in
    # build_bounded_wandb_tags -- validate_wandb_tags must still catch it.
    monkeypatch.setattr(retry_bridge, "build_bounded_wandb_tags", lambda **kwargs: ["x" * 70])
    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path, paths=_paths(tmp_path / "prep", monkeypatch), frozen_record_path=record_path,
        expected_identity_path=identity_path, output_root=output_root,
    ))

    with pytest.raises(SystemExit, match="tag"):
        retry_bridge.main()

    assert "wandb" not in sys.modules
    provenance = json.loads((output_root / retry_trial_id / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["provenance_stage"] == "wandb_tags_rejected"
    assert provenance["retry_of_trial_id"] == written["trial_id"]
    assert "execution_status" not in provenance  # never reached training


# --- Repair 2: wandb.init() exception yields durable terminal evidence, no objective ---

def test_wandb_init_exception_produces_durable_terminal_evidence_and_no_objective(tmp_path, monkeypatch):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"
    retry_trial_id = sweep.trial_id(written["configuration_id"], execution_generation=2)

    class _RaisingWandbModule(types.ModuleType):
        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def init(self, **kwargs):
            raise RuntimeError("simulated real wandb.init() network/validation failure")

    monkeypatch.setitem(sys.modules, "wandb", _RaisingWandbModule("wandb"))
    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path, paths=_paths(tmp_path / "prep", monkeypatch), frozen_record_path=record_path,
        expected_identity_path=identity_path, output_root=output_root,
    ))

    with pytest.raises(SystemExit, match="wandb.init"):
        retry_bridge.main()

    provenance = json.loads((output_root / retry_trial_id / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["provenance_stage"] == "wandb_init_failed"
    assert provenance["retry_of_trial_id"] == written["trial_id"]
    assert provenance["objective_score"] is None
    assert "execution_status" not in provenance  # training never started
    assert "simulated real wandb.init()" in provenance["error"]


# --- Repair 2: successful association is durably recorded before preparation ---

def test_wandb_association_recorded_before_preparation_starts(tmp_path, monkeypatch, fake_wandb_module):
    torch = pytest.importorskip("torch")
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)

    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"
    retry_trial_id = sweep.trial_id(written["configuration_id"], execution_generation=2)
    provenance_path = output_root / retry_trial_id / "execution_provenance.json"

    fake = fake_wandb_module(_AXES, sweep_id=written["wandb_sweep_id"])
    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path, paths=paths, frozen_record_path=record_path,
        expected_identity_path=identity_path, output_root=output_root,
    ))

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

    retry_bridge.main()

    at_prepare = observed["provenance_at_prepare_time"]
    assert at_prepare["provenance_stage"] == "wandb_associated"
    assert at_prepare["wandb_run_id"] == fake.run.id
    assert at_prepare["wandb_sweep_id"] == fake.run.sweep_id


# --- Repair 2/1: full identity present in the W&B run config, not only tags --

def test_full_retry_identity_is_present_in_wandb_config(tmp_path, monkeypatch, fake_wandb_module):
    torch = pytest.importorskip("torch")
    paths = _paths(tmp_path / "prep", monkeypatch)
    _patch_real_canonical_split_shas_for_local_checkout(monkeypatch)

    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"

    fake = fake_wandb_module(_AXES, sweep_id=written["wandb_sweep_id"])
    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path, paths=paths, frozen_record_path=record_path,
        expected_identity_path=identity_path, output_root=output_root,
    ))

    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {e: 0.10 + 0.01 * e for e in epochs}

    def fake_execute(**kwargs):
        return _fake_result(nh_run_dir, checkpoint_epochs=epochs, screening_scores=scores, n_basins=kwargs["screening_basin_ids"].__len__())

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    retry_bridge.main()

    init_kwargs = fake.captured_init_kwargs
    assert all(1 <= len(tag) <= 64 for tag in init_kwargs["tags"])
    identity = init_kwargs["config"]["retry_identity"]
    assert identity["retry_of_trial_id"] == written["trial_id"]
    assert identity["proposal_id"] == written["proposal_id"]
    assert identity["configuration_id"] == written["configuration_id"]
    assert identity["execution_generation"] == 2


# --- Repair 2 + generation-reuse guard: --prior-attempts CLI wiring ----------

def test_resolve_only_refuses_reusing_a_generation_recorded_in_prior_attempts_file(tmp_path, monkeypatch):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"
    prior_attempts_path = tmp_path / "prior_attempts.json"
    prior_attempts_path.write_text(
        json.dumps([{"execution_generation": 2, "slurm_job_id": "45939764"}]), encoding="utf-8",
    )

    monkeypatch.setenv(retry_bridge.ENV_SELFTEST, "resolve_only")
    monkeypatch.setattr(sys, "argv", [
        "run_sweep_v1_exact_retry_bridge.py",
        "--frozen-proposal-record", str(record_path),
        "--expected-identity", str(identity_path),
        "--execution-generation", "2",
        "--prior-attempts", str(prior_attempts_path),
        "--package-root", str(tmp_path / "unused_package"),
        "--screening-basin-ids", str(tmp_path / "unused_screening.txt"),
        "--output-root", str(output_root),
    ])

    with pytest.raises(SweepV1RetryError, match="already reserved"):
        retry_bridge.main()
    assert "wandb" not in sys.modules
    assert not output_root.exists()


def test_resolve_only_allows_a_fresh_generation_with_prior_attempts_file_present(tmp_path, monkeypatch, capsys):
    written = _write_frozen_record(tmp_path)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"
    prior_attempts_path = tmp_path / "prior_attempts.json"
    prior_attempts_path.write_text(
        json.dumps([{"execution_generation": 2, "slurm_job_id": "45939764"}]), encoding="utf-8",
    )

    monkeypatch.setenv(retry_bridge.ENV_SELFTEST, "resolve_only")
    monkeypatch.setattr(sys, "argv", [
        "run_sweep_v1_exact_retry_bridge.py",
        "--frozen-proposal-record", str(record_path),
        "--expected-identity", str(identity_path),
        "--execution-generation", "3",
        "--prior-attempts", str(prior_attempts_path),
        "--package-root", str(tmp_path / "unused_package"),
        "--screening-basin-ids", str(tmp_path / "unused_screening.txt"),
        "--output-root", str(output_root),
    ])

    exit_code = retry_bridge.main()

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["retry_identity"]["execution_generation"] == 3
    assert printed["retry_identity"]["retry_of_trial_id"] == written["trial_id"]


# --- production-launcher migration: legacy main() cannot bypass the runtime contract ----

def test_legacy_cli_main_cannot_reach_the_real_production_sweep_without_runtime_contract(tmp_path, monkeypatch):
    """The legacy multi-flag CLI route (main -> _execute_retry with the
    default runtime_contract_verified=False) must hard-fail, BEFORE any
    durable-intake write or wandb import, the moment it would resolve to the
    real production sweep (PRODUCTION_WANDB_SWEEP_ID). Only main_from_manifest
    (which runs run_full_runtime_contract first) is allowed to reach it. This
    is deliberately NOT the ENV_SELFTEST=resolve_only path (which returns
    before target_sweep_id is ever computed and so never exercises this
    guard) -- this test goes through the real target_sweep_id resolution.
    """
    written = _write_frozen_record(tmp_path, wandb_sweep_id=retry_bridge.PRODUCTION_WANDB_SWEEP_ID)
    record_path = tmp_path / "attempt1" / written["trial_id"] / "execution_provenance.json"
    identity_path = _pinned_identity_path(tmp_path, written)
    output_root = tmp_path / "out_retry"

    monkeypatch.setattr(sys, "argv", _retry_argv(
        tmp_path=tmp_path,
        paths=PreparationPaths(BASELINE_POLICY_PATH, tmp_path / "unused_package", tmp_path / "unused_splits", tmp_path / "unused_screening.txt"),
        frozen_record_path=record_path, expected_identity_path=identity_path, output_root=output_root,
    ))

    with pytest.raises(SystemExit, match="runtime contract"):
        retry_bridge.main()

    assert "wandb" not in sys.modules
    assert not output_root.exists()


# NOTE: that the legacy CLI route still functions end-to-end for a
# non-production sweep id is already proven by
# test_valid_retry_logs_objective_and_leaves_original_attempt_untouched
# and test_invalid_retry_never_logs_a_finite_objective above, both of which
# go through main() -> _execute_retry with the default
# runtime_contract_verified=False and a non-production
# wandb_sweep_id="prod-sweep-xyz" fixture default, and both passed
# unchanged after this guard was added -- confirming the hard-fail is
# scoped exactly to PRODUCTION_WANDB_SWEEP_ID, not to main() as a whole.
