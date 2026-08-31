"""Fake-only / local vertical tests for the v2 PRODUCTION controller-to-agent
launch seam and offline production-registration readiness.

Never contacts the real W&B service and never creates a real sweep, run, or
proposal. Two idioms, matching this suite's existing conventions:

* subprocess + ``PYTHONPATH``-poisoned ``wandb.py`` to prove the offline
  serializer and the pre-W&B manifest-resolution boundary never import
  ``wandb``;
* a directly injected ``register_fn`` plus
  ``monkeypatch.setattr(P, "run_full_runtime_contract", ...)`` for the full
  registration lifecycle (the real runtime contract pins the canonical
  Moriah interpreter and cannot pass on a local dev machine).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.baseline.sweep_v2_six_axis_campaign import (
    CLOSED_DISPOSABLE_REHEARSAL_SWEEP_ID,
    FORBIDDEN_PRODUCTION_SWEEP_IDS,
    FORBIDDEN_V1_SWEEP_ID,
    OBJECTIVE_ID_V2,
)
from src.baseline.sweep_v2_six_axis_config import V2_METRIC_NAME, build_production_sweep_config_v2
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import (
    SweepV2BridgeManifestError,
    load_v2_wandb_bridge_manifest,
    write_v2_wandb_bridge_manifest,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "create_sweep_v2_six_axis_wandb_bridge_production_sweep.py"
SERIALIZER = ROOT / "scripts" / "build_sweep_v2_six_axis_production_sweep_config.py"
BRIDGE = ROOT / "scripts" / "run_sweep_v2_six_axis_wandb_bridge.py"
REAL_DESCRIPTOR_PATH = ROOT / "config" / "stage1_v2_common120_fixed_support_artifact_identity_v001.json"

BRIDGE_PROGRAM = "scripts/run_sweep_v2_six_axis_wandb_bridge.py"
COMMIT_A = "a" * 40
SIX_AXES = {"learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size", "seq_length"}
MORIAH_BASE = "/sci/labs/efratmorin/omripo/Flash-NH"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(f"prod_seam_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _module(SCRIPT)


def _poison_dir(tmp_path: Path) -> Path:
    poison = tmp_path / "poison"
    poison.mkdir(exist_ok=True)
    (poison / "wandb.py").write_text("raise AssertionError('wandb import is forbidden')\n", encoding="utf-8")
    return poison


def _run(script: Path, *arguments: str, env_extra: dict | None = None, poison: Path | None = None):
    environment = dict(os.environ)
    python_path = [str(ROOT)]
    if poison is not None:
        python_path.insert(0, str(poison))
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment.pop("FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST", None)
    environment.pop("FLASHNH_SWEEP_V2_BRIDGE_SELFTEST", None)
    if env_extra:
        environment.update(env_extra)
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
    )


# ---------------------------------------------------------------------------
# AT1 -- production serialization equals the authoritative builder, no W&B.
# ---------------------------------------------------------------------------

def test_production_serializer_writes_exact_authoritative_config_and_never_imports_wandb(tmp_path):
    output = tmp_path / "nested" / "production_sweep_config.json"
    result = _run(SERIALIZER, "--output", str(output), poison=_poison_dir(tmp_path))

    assert result.returncode == 0, result.stderr
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert serialized == build_production_sweep_config_v2(program=BRIDGE_PROGRAM)
    assert output.read_bytes().endswith(b"\n")

    source = SERIALIZER.read_text(encoding="utf-8")
    assert "import wandb" not in source and "wandb.sweep" not in source


# ---------------------------------------------------------------------------
# AT2 -- static controller command, frozen six-axis domain, production metric.
# ---------------------------------------------------------------------------

def test_production_controller_config_is_static_and_preserves_frozen_domain_and_objective():
    cfg = P.build_production_controller_config()

    assert cfg["command"] == ["${interpreter}", "${program}"]
    assert "${args}" not in cfg["command"]
    assert not any(str(part).endswith(".json") for part in cfg["command"])

    assert cfg["metric"] == {"name": V2_METRIC_NAME, "goal": "maximize"}
    assert cfg["method"] == "bayes"
    assert set(cfg["parameters"]) == SIX_AXES
    assert cfg["parameters"]["seq_length"] == {"distribution": "q_uniform", "min": 48, "max": 120, "q": 12}

    assert "4x3btz2s" not in json.dumps(cfg)
    assert "flashnh/best_score" not in json.dumps(cfg)
    assert cfg == build_production_sweep_config_v2(program=BRIDGE_PROGRAM)


# ---------------------------------------------------------------------------
# AT3 -- a real subprocess shaped from the serialized production command
#        resolves the manifest through the env seam, with no swept CLI args.
# ---------------------------------------------------------------------------

def test_serialized_production_command_subprocess_resolves_manifest_via_env_seam(tmp_path):
    output = tmp_path / "production_sweep_config.json"
    assert _run(SERIALIZER, "--output", str(output), poison=_poison_dir(tmp_path)).returncode == 0
    config = json.loads(output.read_text(encoding="utf-8"))

    # Emulate exactly what W&B does with a command of ["${interpreter}", "${program}"]:
    # substitute the macros, append NOTHING (no ${args} -> no --key=value flags).
    assert config["command"] == ["${interpreter}", "${program}"]
    argv = [
        sys.executable if part == "${interpreter}" else (str(ROOT / config["program"]) if part == "${program}" else part)
        for part in config["command"]
    ]
    manifest_path = f"{MORIAH_BASE}/evidence/sweep_v2_prod/proposal_000012/execution_generation_003/launch_manifest.json"
    result = subprocess.run(
        argv, cwd=ROOT, text=True, capture_output=True, check=False,
        env={
            **{k: v for k, v in os.environ.items() if k not in {"FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST"}},
            "PYTHONPATH": os.pathsep.join([str(_poison_dir(tmp_path)), str(ROOT)]),
            "FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST": manifest_path,
            "FLASHNH_SWEEP_V2_BRIDGE_SELFTEST": "resolve_only",
        },
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["resolved_manifest_path"] == manifest_path
    assert payload["argv_tail"] == []
    assert "wandb import is forbidden" not in result.stderr


# ---------------------------------------------------------------------------
# AT4 -- the CLOSED rehearsal positional invocation still resolves unchanged.
# ---------------------------------------------------------------------------

def test_rehearsal_positional_manifest_invocation_still_resolves(tmp_path):
    manifest_path = f"{MORIAH_BASE}/evidence/rehearsal/launch_manifest.json"
    result = _run(
        BRIDGE, manifest_path,
        env_extra={"FLASHNH_SWEEP_V2_BRIDGE_SELFTEST": "resolve_only"},
        poison=_poison_dir(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["resolved_manifest_path"] == manifest_path
    assert payload["argv_tail"] == [manifest_path]
    assert "wandb import is forbidden" not in result.stderr


# ---------------------------------------------------------------------------
# AT5 -- missing / contradictory manifest inputs fail before W&B and intake.
# ---------------------------------------------------------------------------

def test_missing_manifest_source_refuses_before_wandb(tmp_path):
    result = _run(BRIDGE, poison=_poison_dir(tmp_path))
    assert result.returncode != 0
    assert "REFUSING" in result.stderr and "no launch manifest source" in result.stderr
    assert "wandb import is forbidden" not in result.stderr


def test_contradictory_manifest_sources_refuse_before_wandb(tmp_path):
    result = _run(
        BRIDGE, f"{MORIAH_BASE}/a/positional.json",
        env_extra={"FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST": f"{MORIAH_BASE}/b/env.json"},
        poison=_poison_dir(tmp_path),
    )
    assert result.returncode != 0
    assert "REFUSING" in result.stderr and "contradicts" in result.stderr
    assert "wandb import is forbidden" not in result.stderr


def test_agreeing_positional_and_env_manifest_sources_resolve(tmp_path):
    manifest_path = f"{MORIAH_BASE}/evidence/agree/launch_manifest.json"
    result = _run(
        BRIDGE, manifest_path,
        env_extra={
            "FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST": manifest_path,
            "FLASHNH_SWEEP_V2_BRIDGE_SELFTEST": "resolve_only",
        },
        poison=_poison_dir(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["resolved_manifest_path"] == manifest_path


# ---------------------------------------------------------------------------
# AT6 -- strict mode=production manifest builder.
# ---------------------------------------------------------------------------

def _descriptor() -> dict:
    return json.loads(REAL_DESCRIPTOR_PATH.read_text(encoding="utf-8"))


def test_write_production_manifest_shape_bindings_and_no_clobber(tmp_path):
    manifest_path = tmp_path / "launch_manifest.json"
    receipt = P.write_production_manifest(
        manifest_path=manifest_path, wandb_sweep_id="prodsweep01", expected_commit=COMMIT_A,
        proposal_order=12, execution_generation=3,
        output_root_base=f"{MORIAH_BASE}/evidence/sweep_v2_prod",
    )
    assert receipt["status"] == "success"

    manifest = load_v2_wandb_bridge_manifest(manifest_path)
    assert manifest["mode"] == "production"
    assert manifest["stop_before_training"] is False
    assert manifest["max_agents"] == 1
    assert manifest["objective_id"] == OBJECTIVE_ID_V2
    assert manifest["wandb_sweep_id"] == "prodsweep01"
    assert manifest["proposal_order"] == 12 and manifest["execution_generation"] == 3
    assert manifest["output_root"].endswith("/proposal_000012/execution_generation_003")

    descriptor = _descriptor()
    assert manifest["fixed_support_contract_sha256"] == descriptor["artifact"]["internal_canonical_contract_sha256"]
    assert manifest["fixed_support_contract_sha256"] != descriptor["artifact"]["serialized_file_sha256"]
    assert manifest["fixed_support_contract_path"] == descriptor["artifact"]["deployment_provenance_moriah_absolute_path"]
    assert manifest["screening_basin_ids_sha256"] == descriptor["bindings"]["screening_population"]["basin_ids_sha256"]
    assert manifest["manifest_sha256"] == receipt["manifest_sha256"]

    with pytest.raises(SweepV2BridgeManifestError):
        P.write_production_manifest(
            manifest_path=manifest_path, wandb_sweep_id="prodsweep01", expected_commit=COMMIT_A,
            proposal_order=12, execution_generation=4,
            output_root_base=f"{MORIAH_BASE}/evidence/sweep_v2_prod",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"wandb_sweep_id": FORBIDDEN_V1_SWEEP_ID},
        {"wandb_sweep_id": CLOSED_DISPOSABLE_REHEARSAL_SWEEP_ID},
        {"wandb_sweep_id": "PREFLIGHT-PLACEHOLDER-NOT-A-REAL-SWEEP-ID"},
        {"expected_commit": "not-a-sha"},
        {"proposal_order": 0},
        {"execution_generation": -1},
        {"output_root_base": "relative/output"},
    ],
)
def test_write_production_manifest_rejects_bad_inputs(tmp_path, overrides):
    kwargs = dict(
        manifest_path=tmp_path / "m.json", wandb_sweep_id="prodsweep01", expected_commit=COMMIT_A,
        proposal_order=5, execution_generation=2, output_root_base=f"{MORIAH_BASE}/evidence/x",
    )
    kwargs.update(overrides)
    with pytest.raises(P.RegistrationSeamError):
        P.write_production_manifest(**kwargs)
    assert not (tmp_path / "m.json").exists()


# ---------------------------------------------------------------------------
# AT7 -- injected registration: one call, id preserved, no retry.
# ---------------------------------------------------------------------------

def _fake_register_factory(calls, sweep_id="prodsweep01"):
    def fake_register(config, *, project, entity):
        calls.append({"config": config, "project": project, "entity": entity})
        return sweep_id
    return fake_register


def test_register_v2_production_controller_fake_success(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "run_full_runtime_contract", lambda **_: None)
    receipt_path = tmp_path / "controller_receipt.json"
    calls = []

    receipt = P.register_v2_production_controller(
        expected_commit=COMMIT_A, receipt_path=receipt_path,
        wandb_project="fixture-project", wandb_entity="fixture-entity",
        register_fn=_fake_register_factory(calls),
    )

    assert len(calls) == 1
    assert calls[0]["project"] == "fixture-project" and calls[0]["entity"] == "fixture-entity"
    assert calls[0]["config"] == P.build_production_controller_config()
    assert calls[0]["config"]["command"] == ["${interpreter}", "${program}"]
    assert not any(str(part).endswith(".json") for part in calls[0]["config"]["command"])

    assert receipt["status"] == "success"
    assert receipt["wandb_sweep_id"] == "prodsweep01"
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["wandb_sweep_id"] == "prodsweep01"
    assert persisted["sweep_config_metric"] == {"name": V2_METRIC_NAME, "goal": "maximize"}


def test_register_v2_production_controller_never_retries_after_post_creation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "run_full_runtime_contract", lambda **_: None)

    def _boom(_receipt_path, _receipt):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(P, "_persist_controller_receipt", _boom)
    calls = []

    with pytest.raises(P.RegistrationPartialFailure) as excinfo:
        P.register_v2_production_controller(
            expected_commit=COMMIT_A, receipt_path=tmp_path / "r.json",
            register_fn=_fake_register_factory(calls),
        )

    assert len(calls) == 1, "must never retry controller creation after a partial failure"
    receipt = excinfo.value.receipt
    assert receipt["status"] == "partial_failure"
    assert receipt["reason"] == "controller_receipt_write_failed"
    assert receipt["returned_wandb_sweep_id"] == "prodsweep01"
    assert "already exist" in receipt["note"]


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_PRODUCTION_SWEEP_IDS))
def test_register_v2_production_controller_refuses_unsafe_returned_id(tmp_path, monkeypatch, forbidden):
    monkeypatch.setattr(P, "run_full_runtime_contract", lambda **_: None)
    calls = []

    with pytest.raises(P.RegistrationPartialFailure) as excinfo:
        P.register_v2_production_controller(
            expected_commit=COMMIT_A, receipt_path=tmp_path / "r.json",
            register_fn=_fake_register_factory(calls, sweep_id=forbidden),
        )

    assert len(calls) == 1
    assert not (tmp_path / "r.json").exists()
    assert excinfo.value.receipt["reason"] == "unsafe_sweep_id"
    assert excinfo.value.receipt["returned_wandb_sweep_id"] == forbidden


def test_register_v2_production_controller_runtime_refusal_blocks_registration(tmp_path, monkeypatch):
    from src.baseline.sweep_v1_runtime_contract import RuntimeContractError

    def _raise(**_):
        raise RuntimeContractError("simulated: interpreter/commit pin failed")

    monkeypatch.setattr(P, "run_full_runtime_contract", _raise)
    calls = []

    with pytest.raises(RuntimeContractError):
        P.register_v2_production_controller(
            expected_commit=COMMIT_A, receipt_path=tmp_path / "r.json",
            register_fn=_fake_register_factory(calls),
        )

    assert calls == []
    assert not (tmp_path / "r.json").exists()


def test_register_v2_production_controller_refuses_existing_receipt_before_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "run_full_runtime_contract", lambda **_: None)
    receipt_path = tmp_path / "r.json"
    receipt_path.write_text("{}", encoding="utf-8")
    calls = []

    with pytest.raises(P.RegistrationSeamError):
        P.register_v2_production_controller(
            expected_commit=COMMIT_A, receipt_path=receipt_path, register_fn=_fake_register_factory(calls),
        )
    assert calls == []
    assert receipt_path.read_text(encoding="utf-8") == "{}"


# ---------------------------------------------------------------------------
# AT8 -- one-agent launch helper + static W&B boundary in source.
# ---------------------------------------------------------------------------

def _write_valid_production_manifest(tmp_path, *, sweep_id="prodsweep01", name="launch_manifest.json"):
    manifest_path = tmp_path / name
    P.write_production_manifest(
        manifest_path=manifest_path, wandb_sweep_id=sweep_id, expected_commit=COMMIT_A,
        proposal_order=12, execution_generation=3,
        output_root_base=f"{MORIAH_BASE}/evidence/sweep_v2_prod",
    )
    return manifest_path


def test_build_one_agent_invocation_is_count_one_and_derives_id_from_validated_manifest(tmp_path):
    manifest_path = _write_valid_production_manifest(tmp_path)

    invocation = P.build_one_agent_invocation(manifest_path=manifest_path)
    assert invocation["argv"] == ["wandb", "agent", "--count", "1", "prodsweep01"]
    assert invocation["env"]["FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST"] == str(manifest_path)
    assert invocation["env"]["WANDB_PROJECT"] == "flashnh-stage1"

    # A matching redundant sweep id is accepted; a contradictory one is refused.
    assert P.build_one_agent_invocation(
        manifest_path=manifest_path, wandb_sweep_id="prodsweep01",
    )["argv"][-1] == "prodsweep01"
    with pytest.raises(P.RegistrationSeamError):
        P.build_one_agent_invocation(manifest_path=manifest_path, wandb_sweep_id="somethingelse")

    # A relative path, a rehearsal-mode manifest, a malformed manifest, and a
    # forbidden-id manifest are all rejected before an invocation is produced.
    with pytest.raises(P.RegistrationSeamError):
        P.build_one_agent_invocation(manifest_path="relative/manifest.json")

    rehearsal_manifest = tmp_path / "rehearsal.json"
    write_v2_wandb_bridge_manifest(rehearsal_manifest, **_rehearsal_fields(
        wandb_sweep_id=CLOSED_DISPOSABLE_REHEARSAL_SWEEP_ID,
    ))
    with pytest.raises(P.RegistrationSeamError):
        P.build_one_agent_invocation(manifest_path=rehearsal_manifest)

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"mode": "production"}', encoding="utf-8")
    with pytest.raises(SweepV2BridgeManifestError):
        P.build_one_agent_invocation(manifest_path=malformed)


ROOT_CONFIG = ROOT / "config"


def _rehearsal_fields(**changes):
    value = dict(
        manifest_label="fixture", created_at_utc="2026-08-26T00:00:00Z", mode="rehearsal",
        expected_commit="a" * 40, repository_root=str(ROOT), expected_runtime_python="/canonical/python",
        wandb_project="fixture", wandb_sweep_id=CLOSED_DISPOSABLE_REHEARSAL_SWEEP_ID,
        output_root=str(ROOT / "tmp/out"), package_root=str(ROOT / "tmp/pkg"),
        screening_basin_ids_path=str(ROOT / "tmp/screening.txt"), screening_basin_ids_sha256="b" * 64,
        fixed_support_contract_path=str(ROOT / "tmp/support.json"), fixed_support_contract_version=OBJECTIVE_ID_V2,
        fixed_support_contract_sha256="c" * 64,
        baseline_policy_path=str(ROOT_CONFIG / "stage1_scientific_baseline_v001.yaml"),
        policy_overlay_path=str(ROOT_CONFIG / "stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml"),
        base_pilot_policy_path=str(ROOT_CONFIG / "stage1_lead06_pilot_v001.yaml"),
        proposal_order=900001, execution_generation=900001, stop_before_training=True, max_agents=1,
        campaign_id=_campaign_ids()["campaign"], domain_version=_campaign_ids()["domain"],
        canonicalization_version=_campaign_ids()["canon"], objective_id=OBJECTIVE_ID_V2,
    )
    value.update(changes)
    return value


def _campaign_ids():
    from src.baseline.sweep_v2_six_axis_campaign import (
        CAMPAIGN_ID_V2, CONFIGURATION_CANONICALIZATION_VERSION_V2, DOMAIN_VERSION_V2,
    )
    return {"campaign": CAMPAIGN_ID_V2, "domain": DOMAIN_VERSION_V2, "canon": CONFIGURATION_CANONICALIZATION_VERSION_V2}


# ---------------------------------------------------------------------------
# Pre-agent validator (resolve_validated_production_launch / validate-launch).
# ---------------------------------------------------------------------------

def test_resolve_validated_production_launch_returns_authoritative_id(tmp_path):
    manifest_path = _write_valid_production_manifest(tmp_path)
    resolved = P.resolve_validated_production_launch(manifest_path=manifest_path)
    assert resolved["wandb_sweep_id"] == "prodsweep01"
    assert resolved["mode"] == "production"
    assert resolved["manifest_path"] == str(manifest_path)

    # Redundant matching id accepted; contradiction refused before any W&B.
    assert P.resolve_validated_production_launch(
        manifest_path=manifest_path, expected_sweep_id="prodsweep01",
    )["wandb_sweep_id"] == "prodsweep01"
    with pytest.raises(P.RegistrationSeamError):
        P.resolve_validated_production_launch(manifest_path=manifest_path, expected_sweep_id="mismatch99")


def test_resolve_validated_production_launch_rejects_rehearsal_and_forbidden_and_corruption(tmp_path):
    rehearsal_manifest = tmp_path / "rehearsal.json"
    write_v2_wandb_bridge_manifest(rehearsal_manifest, **_rehearsal_fields())
    with pytest.raises(P.RegistrationSeamError):
        P.resolve_validated_production_launch(manifest_path=rehearsal_manifest)

    corrupt = tmp_path / "corrupt.json"
    data = json.loads(_write_valid_production_manifest(tmp_path, name="ok.json").read_text(encoding="utf-8"))
    data["proposal_order"] = data["proposal_order"] + 1  # break the checksum
    corrupt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SweepV2BridgeManifestError):
        P.resolve_validated_production_launch(manifest_path=corrupt)


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_PRODUCTION_SWEEP_IDS))
def test_validate_launch_cli_prints_bare_id_and_never_imports_wandb(tmp_path, forbidden):
    manifest_path = _write_valid_production_manifest(tmp_path)
    ok = _run(SCRIPT, "validate-launch", "--manifest-path", str(manifest_path), poison=_poison_dir(tmp_path))
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.strip() == "prodsweep01"
    assert "wandb import is forbidden" not in ok.stderr

    # A contradictory --expect-sweep-id fails nonzero.
    bad = _run(SCRIPT, "validate-launch", "--manifest-path", str(manifest_path),
               "--expect-sweep-id", "mismatch99", poison=_poison_dir(tmp_path))
    assert bad.returncode != 0 and "REFUSING" in bad.stderr

    # A mode=production manifest that names a forbidden id cannot even be
    # loaded (the strict loader rejects it); the CLI exits nonzero, no W&B.
    forbidden_manifest = tmp_path / "forbidden.json"
    fields = _rehearsal_fields(mode="production", stop_before_training=False, wandb_sweep_id=forbidden)
    with pytest.raises(SweepV2BridgeManifestError):
        write_v2_wandb_bridge_manifest(forbidden_manifest, **fields)


def test_production_seam_source_imports_wandb_only_inside_call_boundary():
    source = SCRIPT.read_text(encoding="utf-8")
    lines = source.splitlines()
    import_lines = [i for i, line in enumerate(lines) if line.strip() == "import wandb"]
    assert len(import_lines) == 1
    assert lines[import_lines[0]].startswith("    "), "wandb must not be imported at module level"

    marker = '"""'
    body = source[source.index(marker, source.index(marker) + 3) + 3:]
    assert "wandb.init(" not in body and "wandb.agent(" not in body
    assert body.count("wandb.sweep(") == 1
    for forbidden in FORBIDDEN_PRODUCTION_SWEEP_IDS:
        assert f'"{forbidden}"' not in source and f"'{forbidden}'" not in source
    real = _descriptor()
    assert real["artifact"]["internal_canonical_contract_sha256"] not in source
    assert real["artifact"]["serialized_file_sha256"] not in source


# ---------------------------------------------------------------------------
# AT9 -- an inherited bridge self-test hook cannot reach a real production
#        agent invocation (static ordering + a real bash subprocess).
# ---------------------------------------------------------------------------

PRODUCTION_LAUNCHER = ROOT / "scripts" / "run_sweep_v2_six_axis_wandb_agent_moriah.sbatch"


def test_inherited_selftest_cannot_reach_production_agent_invocation(tmp_path):
    lines = PRODUCTION_LAUNCHER.read_text(encoding="utf-8").splitlines()
    refuse_idx = next(
        i for i, l in enumerate(lines)
        if l.strip().startswith("if [ -n ") and "FLASHNH_SWEEP_V2_BRIDGE_SELFTEST" in l
    )
    agent_idx = next(i for i, l in enumerate(lines) if l.startswith("wandb agent "))
    validate_idx = next(i for i, l in enumerate(lines) if "validate-launch" in l and not l.lstrip().startswith("#"))
    assert refuse_idx < validate_idx < agent_idx
    assert any("exit 1" in lines[j] for j in range(refuse_idx, refuse_idx + 4))

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable for the subprocess demonstration")
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [bash, str(PRODUCTION_LAUNCHER)], cwd=str(ROOT), text=True, capture_output=True, check=False,
        env={
            **os.environ,
            "WANDB_PROJECT": "p", "WANDB_ENTITY": "e",
            "FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST": str(manifest),
            "FLASHNH_SWEEP_V2_BRIDGE_SELFTEST": "resolve_only",
        },
    )
    assert result.returncode == 1
    assert "REFUSING" in result.stderr and "FLASHNH_SWEEP_V2_BRIDGE_SELFTEST" in result.stderr
    assert "wandb agent" not in result.stdout
