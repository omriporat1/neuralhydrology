"""Fake-only/local vertical tests for the v2 rehearsal registration entry
point, ``scripts/create_sweep_v2_six_axis_wandb_bridge_rehearsal_sweep.py``.

Never contacts the real W&B service and never creates a real sweep or run.
Two fake-injection idioms are used, matching existing conventions elsewhere
in this suite:

* subprocess + ``PYTHONPATH``-poisoned ``wandb.py`` (mirrors
  ``tests/test_sweep_v2_wandb_bridge_rehearsal_sweep_config_serializer.py``)
  to prove the real CLI entry point (argument parsing, ``--preflight-only``)
  never imports ``wandb``.
* direct injected ``register_fn`` callables plus
  ``monkeypatch.setattr(..., "run_full_runtime_contract", lambda **_: None)``
  (mirrors ``tests/test_sweep_v2_six_axis_wandb_bridge_foundation.py``) for
  the full registration lifecycle, since the real runtime contract pins the
  canonical Moriah interpreter and cannot pass on a local dev machine
  regardless of fake W&B injection -- exercising the *full* lifecycle
  (including a real ``wandb.sweep`` call site) via subprocess would require
  either faking HOME/.netrc (irrelevant to what these tests verify) or a
  test-only bypass in production code, which the task forbids. The injected
  ``register_fn`` seam gives equivalent -- and more precise -- coverage of
  call count, arguments, and return-value handling without either.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.baseline.sweep_v2_six_axis_wandb_bridge_manifest as manifest_module
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import (
    SweepV2BridgeManifestError as ManifestError,
    write_v2_wandb_bridge_manifest,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "create_sweep_v2_six_axis_wandb_bridge_rehearsal_sweep.py"
REAL_DESCRIPTOR_PATH = ROOT / "config" / "stage1_v2_common120_fixed_support_artifact_identity_v001.json"


def _module():
    spec = importlib.util.spec_from_file_location("create_sweep_v2_rehearsal_registration_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _module()


def _real_descriptor() -> dict:
    return json.loads(REAL_DESCRIPTOR_PATH.read_text(encoding="utf-8"))


def _write_descriptor(tmp_path, data: dict) -> Path:
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


COMMIT_A = "a" * 40


def _kwargs(tmp_path, **overrides) -> dict:
    value = dict(
        expected_commit=COMMIT_A,
        manifest_path=tmp_path / "manifest.json",
        output_root="/sci/labs/efratmorin/omripo/Flash-NH/evidence/tmp_out",
        package_root="/sci/labs/efratmorin/omripo/Flash-NH/data/tmp_pkg",
        screening_basin_ids_path="/sci/labs/efratmorin/omripo/Flash-NH/data/tmp_screening.txt",
        wandb_project="fixture-project",
        wandb_entity="fixture-entity",
        proposal_order=900001,
        execution_generation=900001,
        descriptor_path=REAL_DESCRIPTOR_PATH,
    )
    value.update(overrides)
    return value


def _noop_runtime_contract(**_):
    return None


# ---------------------------------------------------------------------------
# (1) Descriptor loading and strict validation.
# ---------------------------------------------------------------------------

def test_descriptor_loading_accepts_the_real_committed_descriptor():
    descriptor = M.load_and_validate_descriptor(REAL_DESCRIPTOR_PATH)
    assert descriptor["record_id"] == "stage1_v2_common120_fixed_support_artifact_identity_v001"
    assert descriptor["bindings"]["screening_population"]["basin_count"] == 400


@pytest.mark.parametrize(
    "mutate,description",
    [
        (lambda d: d.__setitem__("schema_name", "wrong"), "schema_name"),
        (lambda d: d.__setitem__("schema_version", 99), "schema_version"),
        (lambda d: d.__setitem__("record_id", "wrong"), "record_id"),
        (lambda d: d["artifact"].__setitem__("tracking_status", "tracked"), "tracking_status"),
        (lambda d: d["artifact"].__setitem__("deployment_provenance_moriah_absolute_path", "relative/path.json"), "non_absolute_deployment_path"),
        (lambda d: d["fixed_support_contract"].__setitem__("schema_name", "wrong"), "contract_schema_name"),
        (lambda d: d["fixed_support_contract"].__setitem__("schema_version", 99), "contract_schema_version"),
        (lambda d: d["fixed_support_contract"].__setitem__("contract_id", "wrong"), "contract_id"),
        (lambda d: d["fixed_support_contract"].__setitem__("optimizer_metric", "wrong"), "optimizer_metric"),
        (lambda d: d["bindings"]["screening_population"].__setitem__("policy_identity", "wrong"), "screening_policy_identity"),
        (lambda d: d["bindings"]["screening_population"].__setitem__("basin_count", 399), "screening_basin_count"),
        (lambda d: d["bindings"]["screening_population"].__setitem__("basin_ids_sha256", "0" * 64), "screening_checksum"),
        (lambda d: d.__delitem__("artifact"), "missing_artifact_section"),
        (lambda d: d.__delitem__("bindings"), "missing_bindings_section"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_descriptor_loading_rejects_corrupted_descriptors(tmp_path, mutate, description):
    data = copy.deepcopy(_real_descriptor())
    mutate(data)
    path = _write_descriptor(tmp_path, data)
    with pytest.raises(M.RegistrationSeamError):
        M.load_and_validate_descriptor(path)


def test_descriptor_loading_rejects_missing_file(tmp_path):
    with pytest.raises(M.RegistrationSeamError):
        M.load_and_validate_descriptor(tmp_path / "does-not-exist.json")


def test_descriptor_loading_rejects_non_hex_checksum(tmp_path):
    data = copy.deepcopy(_real_descriptor())
    data["artifact"]["internal_canonical_contract_sha256"] = "not-hex" * 8
    path = _write_descriptor(tmp_path, data)
    with pytest.raises(M.RegistrationSeamError):
        M.load_and_validate_descriptor(path)


# ---------------------------------------------------------------------------
# (2) Exact descriptor-to-manifest mapping.
# ---------------------------------------------------------------------------

def test_descriptor_manifest_bindings_exact_mapping():
    descriptor = M.load_and_validate_descriptor(REAL_DESCRIPTOR_PATH)
    bindings = M._descriptor_manifest_bindings(descriptor)

    assert bindings["fixed_support_contract_path"] == descriptor["artifact"]["deployment_provenance_moriah_absolute_path"]
    assert bindings["fixed_support_contract_version"] == descriptor["fixed_support_contract"]["contract_id"]
    assert bindings["fixed_support_contract_sha256"] == descriptor["artifact"]["internal_canonical_contract_sha256"]
    assert bindings["screening_basin_ids_sha256"] == descriptor["bindings"]["screening_population"]["basin_ids_sha256"]

    # For the real committed descriptor these two identities currently
    # differ; this is an observation about the fixture, not a schema rule
    # the loader enforces (equality is a legal descriptor state -- see
    # ``test_descriptor_with_equal_well_formed_checksums_is_not_rejected``
    # below). Substitution-prevention comes from mapping always reading
    # ``internal_canonical_contract_sha256``, not from forced inequality.
    assert bindings["fixed_support_contract_sha256"] != descriptor["artifact"]["serialized_file_sha256"]


def test_descriptor_with_equal_well_formed_checksums_is_not_rejected(tmp_path):
    """Repair 4: equality between the two checksum identities is not, by
    itself, an invalid descriptor state -- only field authority/mapping
    (never numerical inequality) must prevent the external serialized-file
    checksum from being substituted for the internal canonical one. Uses a
    synthetic well-formed hex digest, never a real committed checksum
    literal."""
    data = copy.deepcopy(_real_descriptor())
    synthetic_sha256 = "ab" * 32
    data["artifact"]["serialized_file_sha256"] = synthetic_sha256
    data["artifact"]["internal_canonical_contract_sha256"] = synthetic_sha256
    path = _write_descriptor(tmp_path, data)

    descriptor = M.load_and_validate_descriptor(path)
    bindings = M._descriptor_manifest_bindings(descriptor)

    assert descriptor["artifact"]["serialized_file_sha256"] == descriptor["artifact"]["internal_canonical_contract_sha256"]
    assert bindings["fixed_support_contract_sha256"] == synthetic_sha256
    assert bindings["fixed_support_contract_sha256"] == descriptor["artifact"]["internal_canonical_contract_sha256"]


def test_descriptor_manifest_bindings_treats_artifact_path_as_opaque_string(tmp_path):
    """Mapping the descriptor must never open/dereference the external
    Common-120 artifact file itself -- only copy its recorded path/checksum
    strings through from the small committed descriptor. Proven by pointing
    ``deployment_provenance_moriah_absolute_path`` at a file that does not
    exist anywhere and confirming the mapping still succeeds."""
    data = copy.deepcopy(_real_descriptor())
    nonexistent = "/sci/labs/efratmorin/omripo/Flash-NH/data/does-not-exist-anywhere.json"
    data["artifact"]["deployment_provenance_moriah_absolute_path"] = nonexistent
    path = _write_descriptor(tmp_path, data)

    descriptor = M.load_and_validate_descriptor(path)
    bindings = M._descriptor_manifest_bindings(descriptor)

    assert bindings["fixed_support_contract_path"] == nonexistent
    assert isinstance(bindings["fixed_support_contract_path"], str)


# ---------------------------------------------------------------------------
# (3) Offline preflight.
# ---------------------------------------------------------------------------

def test_run_preflight_writes_no_manifest_and_returns_placeholder_receipt(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    receipt = M.run_preflight(**_kwargs(tmp_path, manifest_path=manifest_path))

    assert receipt["preflight"] is True
    assert receipt["placeholder_sweep_id"] == M._PREFLIGHT_PLACEHOLDER_SWEEP_ID
    assert not manifest_path.exists()
    assert receipt["proposal_order"] == 900001
    assert receipt["execution_generation"] == 900001
    assert receipt["stop_before_training"] is True
    assert receipt["max_agents"] == 1


def test_run_preflight_refuses_when_manifest_already_exists(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(M.RegistrationSeamError):
        M.run_preflight(**_kwargs(tmp_path, manifest_path=manifest_path))


def test_run_preflight_refuses_relative_paths(tmp_path):
    with pytest.raises(M.RegistrationSeamError):
        M.run_preflight(**_kwargs(tmp_path, output_root="relative/output"))


def test_run_preflight_refuses_non_positive_proposal_order(tmp_path):
    with pytest.raises(M.RegistrationSeamError):
        M.run_preflight(**_kwargs(tmp_path, proposal_order=0))


def test_run_preflight_subprocess_never_imports_wandb(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "wandb.py").write_text("raise AssertionError('wandb import is forbidden')\n", encoding="utf-8")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(poison), str(ROOT)])
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--preflight-only",
            "--expected-commit", COMMIT_A,
            "--manifest-path", str(manifest_path),
            "--proposal-order", "900001",
            "--execution-generation", "900001",
        ],
        cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["preflight"] is True
    assert not manifest_path.exists()


# ---------------------------------------------------------------------------
# (3b) Prospective strict-manifest validation before registration (repair 2).
# ---------------------------------------------------------------------------

def test_prospective_manifest_validation_catches_corrupted_fields_before_registration(tmp_path, monkeypatch):
    """Repair 2: the shared prospective-manifest construction/validation
    (:func:`_assemble_and_validate_prospective_manifest`, reused from
    :func:`run_preflight`) must catch a local manifest-field-assembly defect
    before ``register_fn`` is ever called. Corrupts ``max_agents`` -- a
    field this script's own CLI/descriptor-level validation never checks,
    but the manifest schema does -- by wrapping the real ``_manifest_fields``
    helper, simulating a bug in that shared assembly step."""
    monkeypatch.setattr(M, "run_full_runtime_contract", _noop_runtime_contract)
    real_manifest_fields = M._manifest_fields

    def corrupted_manifest_fields(**kwargs):
        fields = real_manifest_fields(**kwargs)
        fields["max_agents"] = 2
        return fields

    monkeypatch.setattr(M, "_manifest_fields", corrupted_manifest_fields)
    manifest_path = tmp_path / "manifest.json"
    calls = []

    def fake_register(config, *, project, entity):
        calls.append(1)
        return "abcd1234"

    with pytest.raises(ManifestError, match="exactly one agent"):
        M.register_v2_rehearsal_sweep(
            **_kwargs(tmp_path, manifest_path=manifest_path), register_fn=fake_register,
        )

    assert calls == [], "register_fn must not be called when prospective manifest validation fails"
    assert not manifest_path.exists()


def test_register_v2_rehearsal_sweep_runtime_refusal_blocks_registration(tmp_path, monkeypatch):
    """Repair 3: uses the real production entry point
    (``register_v2_rehearsal_sweep``) with only the runtime-contract helper
    and the registration callable substituted -- the runtime helper is made
    to RAISE (not no-op), proving refusal ordering: ``register_fn`` is never
    called, no manifest is written, no success receipt is emitted, and the
    runtime failure surfaces as its own controlled exception rather than
    being converted into a post-registration ``RegistrationPartialFailure``."""
    from src.baseline.sweep_v1_runtime_contract import RuntimeContractError

    def raising_runtime_contract(**_):
        raise RuntimeContractError("simulated: commit/dirty-tree/interpreter pin failed")

    monkeypatch.setattr(M, "run_full_runtime_contract", raising_runtime_contract)
    manifest_path = tmp_path / "manifest.json"
    calls = []

    def fake_register(config, *, project, entity):
        calls.append(1)
        return "abcd1234"

    with pytest.raises(RuntimeContractError):
        M.register_v2_rehearsal_sweep(
            **_kwargs(tmp_path, manifest_path=manifest_path), register_fn=fake_register,
        )

    assert calls == [], "register_fn must not be called when the runtime contract refuses"
    assert not manifest_path.exists()


# ---------------------------------------------------------------------------
# (4) Fake registration success.
# ---------------------------------------------------------------------------

def test_register_v2_rehearsal_sweep_fake_success(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "run_full_runtime_contract", _noop_runtime_contract)
    manifest_path = tmp_path / "manifest.json"
    calls = []

    def fake_register(config, *, project, entity):
        calls.append({"config": config, "project": project, "entity": entity})
        return "abcd1234"

    receipt = M.register_v2_rehearsal_sweep(
        **_kwargs(tmp_path, manifest_path=manifest_path), register_fn=fake_register,
    )

    assert len(calls) == 1
    assert calls[0]["project"] == "fixture-project"
    assert calls[0]["entity"] == "fixture-entity"
    expected_config, _ = M._build_config_and_bindings(
        descriptor=M.load_and_validate_descriptor(REAL_DESCRIPTOR_PATH), manifest_path=manifest_path,
    )
    assert calls[0]["config"] == expected_config

    assert receipt["status"] == "success"
    assert receipt["wandb_sweep_id"] == "abcd1234"
    assert manifest_path.exists()

    from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import load_v2_wandb_bridge_manifest

    manifest = load_v2_wandb_bridge_manifest(manifest_path)
    assert manifest["wandb_sweep_id"] == "abcd1234"
    assert manifest["mode"] == "rehearsal"
    assert manifest["stop_before_training"] is True
    assert manifest["max_agents"] == 1
    assert manifest["proposal_order"] == 900001
    assert manifest["execution_generation"] == 900001
    descriptor = M.load_and_validate_descriptor(REAL_DESCRIPTOR_PATH)
    assert manifest["fixed_support_contract_sha256"] == descriptor["artifact"]["internal_canonical_contract_sha256"]
    assert manifest["screening_basin_ids_sha256"] == descriptor["bindings"]["screening_population"]["basin_ids_sha256"]
    assert manifest["manifest_sha256"] == receipt["manifest_sha256"]


def test_register_v2_rehearsal_sweep_never_calls_wandb_init_or_agent(tmp_path, monkeypatch):
    """Even on the success path, no wandb.init/agent access is possible: the
    fake register_fn is a plain callable, never a fake wandb module, so
    there is no ``wandb.init``/``wandb.agent`` attribute for this lifecycle
    to reach even accidentally."""
    monkeypatch.setattr(M, "run_full_runtime_contract", _noop_runtime_contract)

    def fake_register(config, *, project, entity):
        assert not hasattr(fake_register, "init") and not hasattr(fake_register, "agent")
        return "abcd1234"

    receipt = M.register_v2_rehearsal_sweep(
        **_kwargs(tmp_path, manifest_path=tmp_path / "manifest.json"), register_fn=fake_register,
    )
    assert receipt["status"] == "success"


# ---------------------------------------------------------------------------
# (5) Unsafe returned sweep IDs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "returned_id",
    ["4x3btz2s", "", None, 12345, "  abcd1234  ", "abcd 1234", "abcd*1234", "ab"],
    ids=["frozen_v1_id", "empty", "none", "non_string", "whitespace_padded", "internal_whitespace", "malformed_chars", "too_short"],
)
def test_register_v2_rehearsal_sweep_refuses_unsafe_returned_ids(tmp_path, monkeypatch, returned_id):
    monkeypatch.setattr(M, "run_full_runtime_contract", _noop_runtime_contract)
    manifest_path = tmp_path / "manifest.json"
    calls = []

    def fake_register(config, *, project, entity):
        calls.append(1)
        return returned_id

    with pytest.raises(M.RegistrationPartialFailure) as excinfo:
        M.register_v2_rehearsal_sweep(
            **_kwargs(tmp_path, manifest_path=manifest_path), register_fn=fake_register,
        )

    assert len(calls) == 1
    assert not manifest_path.exists()
    receipt = excinfo.value.receipt
    assert receipt["reason"] == "unsafe_sweep_id"
    assert receipt["returned_wandb_sweep_id"] == returned_id
    assert receipt["wandb_project"] == "fixture-project"
    assert receipt["wandb_entity"] == "fixture-entity"
    assert receipt["intended_manifest_path"] == str(manifest_path)


# ---------------------------------------------------------------------------
# (6) Partial failure after a valid sweep ID is returned.
# ---------------------------------------------------------------------------

def test_register_v2_rehearsal_sweep_partial_failure_on_manifest_write_error(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "run_full_runtime_contract", _noop_runtime_contract)
    manifest_path = tmp_path / "manifest.json"
    calls = []

    def fake_register(config, *, project, entity):
        calls.append(1)
        return "abcd1234"

    def _boom(path, **fields):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(M, "write_v2_wandb_bridge_manifest", _boom)

    with pytest.raises(M.RegistrationPartialFailure) as excinfo:
        M.register_v2_rehearsal_sweep(
            **_kwargs(tmp_path, manifest_path=manifest_path), register_fn=fake_register,
        )

    assert len(calls) == 1, "must never retry registration after a partial failure"
    assert not manifest_path.exists()
    receipt = excinfo.value.receipt
    assert receipt["status"] == "partial_failure"
    assert receipt["reason"] == "manifest_write_failed"
    assert receipt["returned_wandb_sweep_id"] == "abcd1234"
    assert receipt["wandb_project"] == "fixture-project"
    assert receipt["wandb_entity"] == "fixture-entity"
    assert receipt["intended_manifest_path"] == str(manifest_path)
    assert "note" in receipt and "already exist" in receipt["note"]


# ---------------------------------------------------------------------------
# (7) No-overwrite / race behavior.
# ---------------------------------------------------------------------------

def test_register_v2_rehearsal_sweep_refuses_before_registration_when_manifest_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "run_full_runtime_contract", _noop_runtime_contract)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    calls = []

    def fake_register(config, *, project, entity):
        calls.append(1)
        return "abcd1234"

    with pytest.raises(M.RegistrationSeamError):
        M.register_v2_rehearsal_sweep(
            **_kwargs(tmp_path, manifest_path=manifest_path), register_fn=fake_register,
        )
    assert len(calls) == 0, "must refuse before ever calling the registration boundary"
    assert manifest_path.read_text(encoding="utf-8") == "{}"


def test_register_v2_rehearsal_sweep_race_manifest_created_after_local_check(tmp_path, monkeypatch):
    """Simulates a race where the manifest path comes into existence between
    this script's own pre-check and the underlying writer's own internal
    re-check -- the underlying writer (already independently tested in
    ``sweep_v2_six_axis_wandb_bridge_manifest``) must still refuse, and that
    refusal must surface here as a partial failure, not a silent overwrite.
    """
    monkeypatch.setattr(M, "run_full_runtime_contract", _noop_runtime_contract)
    manifest_path = tmp_path / "manifest.json"

    def fake_register(config, *, project, entity):
        manifest_path.write_text("raced-in", encoding="utf-8")
        return "abcd1234"

    with pytest.raises(M.RegistrationPartialFailure) as excinfo:
        M.register_v2_rehearsal_sweep(
            **_kwargs(tmp_path, manifest_path=manifest_path), register_fn=fake_register,
        )
    assert manifest_path.read_text(encoding="utf-8") == "raced-in"
    assert excinfo.value.receipt["reason"] == "manifest_write_failed"
    assert excinfo.value.receipt["returned_wandb_sweep_id"] == "abcd1234"


# ---------------------------------------------------------------------------
# (7b) Manifest writer's own atomic no-clobber publication (repair 1).
# ---------------------------------------------------------------------------

def _real_manifest_fields(tmp_path, *, name="target.json", sweep_id="abcd1234"):
    descriptor = M.load_and_validate_descriptor(REAL_DESCRIPTOR_PATH)
    manifest_path = tmp_path / name
    _config, bindings = M._build_config_and_bindings(descriptor=descriptor, manifest_path=manifest_path)
    fields = M._manifest_fields(
        expected_commit=COMMIT_A,
        output_root="/sci/labs/efratmorin/omripo/Flash-NH/evidence/tmp_out",
        package_root="/sci/labs/efratmorin/omripo/Flash-NH/data/tmp_pkg",
        screening_basin_ids_path="/sci/labs/efratmorin/omripo/Flash-NH/data/tmp_screening.txt",
        wandb_project="fixture-project", wandb_entity="fixture-entity",
        sweep_id=sweep_id, proposal_order=900001, execution_generation=900001, bindings=bindings,
    )
    return manifest_path, fields


def test_write_v2_wandb_bridge_manifest_publication_race_and_temp_cleanup(tmp_path):
    """Repair 1 deterministic race-boundary regression test: intercepts the
    writer's real atomic no-clobber publication call (``os.link``) and
    creates the destination with known sentinel bytes immediately before
    delegating to the real operation -- simulating a concurrent writer
    racing in exactly at the publication boundary. Proves: a controlled
    ``SweepV2BridgeManifestError`` is raised, the sentinel destination is
    byte-for-byte unchanged, no temp writer file remains, and the writer
    never falls back to a clobbering operation. Also proves an ordinary
    successful publish leaves no leftover temp file behind."""
    manifest_path, fields = _real_manifest_fields(tmp_path, name="raced.json")
    sentinel = b"raced-in-sentinel-bytes-must-survive-byte-for-byte"
    real_link = os.link

    def racy_link(source, destination, *args, **kwargs):
        Path(destination).write_bytes(sentinel)
        return real_link(source, destination, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifest_module.os, "link", racy_link)
        with pytest.raises(ManifestError):
            write_v2_wandb_bridge_manifest(manifest_path, **fields)

    assert manifest_path.read_bytes() == sentinel
    leftovers = {p.name for p in manifest_path.parent.iterdir()} - {manifest_path.name}
    assert leftovers == set(), f"temp writer file(s) left behind after a failed publish: {leftovers}"

    source_text = Path(manifest_module.__file__).read_text(encoding="utf-8")
    assert "os.replace(" not in source_text, "writer must never fall back to a clobbering replace call"

    other_path, other_fields = _real_manifest_fields(tmp_path, name="ordinary.json", sweep_id="efgh5678")
    write_v2_wandb_bridge_manifest(other_path, **other_fields)
    leftovers_after_success = {p.name for p in other_path.parent.iterdir()} - {manifest_path.name, other_path.name}
    assert leftovers_after_success == set(), "temp writer file(s) left behind after a successful publish"


# ---------------------------------------------------------------------------
# (8) Source / static W&B boundary.
# ---------------------------------------------------------------------------

def _source_without_module_docstring() -> str:
    source = SCRIPT.read_text(encoding="utf-8")
    marker = '"""'
    first = source.index(marker)
    second = source.index(marker, first + 3) + 3
    return source[second:]


def test_source_imports_wandb_only_inside_call_boundary():
    source = SCRIPT.read_text(encoding="utf-8")
    lines = source.splitlines()
    import_lines = [i for i, line in enumerate(lines) if line.strip() == "import wandb"]
    assert len(import_lines) == 1, "expected exactly one 'import wandb' occurrence in source"
    (line_index,) = import_lines
    # Must be indented (i.e. nested inside a function), never at module level.
    assert lines[line_index].startswith("    "), "wandb must not be imported at module level"

    code = _source_without_module_docstring()
    assert "wandb.init(" not in code
    assert "wandb.agent(" not in code
    assert code.count("wandb.sweep(") == 1


def test_source_forbidden_v1_sweep_id_appears_only_as_symbolic_import():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"4x3btz2s"' not in source
    assert "'4x3btz2s'" not in source


def test_source_never_hardcodes_descriptor_checksum_literals():
    source = SCRIPT.read_text(encoding="utf-8")
    real = _real_descriptor()
    assert real["artifact"]["internal_canonical_contract_sha256"] not in source
    assert real["artifact"]["serialized_file_sha256"] not in source


# ---------------------------------------------------------------------------
# (9) Vertical CLI test.
# ---------------------------------------------------------------------------

def test_cli_preflight_only_vertical_subprocess(tmp_path):
    manifest_path = tmp_path / "cli-manifest.json"
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "wandb.py").write_text("raise AssertionError('wandb import is forbidden')\n", encoding="utf-8")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(poison), str(ROOT)])
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--preflight-only",
            "--expected-commit", COMMIT_A,
            "--manifest-path", str(manifest_path),
            "--output-root", "/sci/labs/efratmorin/omripo/Flash-NH/evidence/tmp_out",
            "--package-root", "/sci/labs/efratmorin/omripo/Flash-NH/data/tmp_pkg",
            "--screening-basin-ids-path", "/sci/labs/efratmorin/omripo/Flash-NH/data/tmp_screening.txt",
            "--wandb-project", "fixture-project",
            "--wandb-entity", "fixture-entity",
            "--proposal-order", "900001",
            "--execution-generation", "900001",
        ],
        cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["preflight"] is True
    assert payload["placeholder_sweep_id"] == "PREFLIGHT-PLACEHOLDER-NOT-A-REAL-SWEEP-ID"
    assert not manifest_path.exists()


def test_cli_missing_required_arguments_refused_before_any_import(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower()
