"""Tests for the strict, versioned, checksummed Sweep-v1 exact-retry launch
manifest, ``src.baseline.sweep_v1_launch_manifest`` -- the disposable exact-
retry startup rehearsal's Design Decision 1/2 replacement for the long
``sbatch --export=ALL,VAR=value,...`` interface.

Covers, in order:
  1. A minimal valid rehearsal manifest round-trips through
     ``build_launch_manifest`` with a stable, recomputable checksum.
  2. ``write_launch_manifest`` + ``load_launch_manifest`` round-trip on disk.
  3. Strict no-overwrite: a second write to the same path is refused.
  4. An unknown field is rejected.
  5. A missing required field is rejected.
  6. An unsupported ``schema_version`` is rejected.
  7. No credential-shaped field NAME is ever accepted.
  8. No bare hex-token-shaped VALUE is ever accepted (except the two
     hex-exempt fields).
  9. ``mode="production"`` requires the real production sweep id and
     ``stop_before_training=False``; any contradiction is rejected.
  10. ``mode="rehearsal"`` must never target the production sweep id, and
      requires ``stop_before_training=True``; any contradiction is rejected.
  11. A tampered on-disk manifest (checksum mismatch) is rejected on load.

Never imports wandb; never touches the network; never starts NH training.
"""
from __future__ import annotations

import pytest

from src.baseline.sweep_v1_launch_manifest import (
    LaunchManifestError, MODE_PRODUCTION, MODE_REHEARSAL, PRODUCTION_WANDB_SWEEP_ID,
    build_launch_manifest, compute_manifest_checksum, load_launch_manifest, write_launch_manifest,
)


def _rehearsal_fields(**overrides):
    fields = dict(
        manifest_label="sweep_v1_exact_retry_rehearsal_v001",
        created_at_utc="2026-08-24T00:00:00Z",
        mode=MODE_REHEARSAL,
        expected_commit="3d72d14" + "0" * 33,
        expected_runtime_python="/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python",
        frozen_proposal_record_path="/tmp/attempt1/execution_provenance.json",
        expected_identity={"proposal_order": 1, "configuration_id": "sweep_v1_cfg_5731e180d1bf9d582afc"},
        execution_generation=4,
        package_root="/tmp/package",
        screening_basin_ids_path="/tmp/screening.txt",
        output_root="/tmp/out_rehearsal",
        baseline_policy_path="/tmp/baseline_policy.yaml",
        base_pilot_policy_path="/tmp/base_pilot_policy.yaml",
        wandb_project="flashnh-stage1-rehearsal",
        wandb_sweep_id="rehearsal-sweep-abc123",
        stop_before_training=True,
    )
    fields.update(overrides)
    return fields


def _production_fields(**overrides):
    fields = _rehearsal_fields(
        mode=MODE_PRODUCTION, wandb_sweep_id=PRODUCTION_WANDB_SWEEP_ID, stop_before_training=False,
        manifest_label="sweep_v1_exact_retry_production_v001",
    )
    fields.update(overrides)
    return fields


# --- 1-2: build/write/load round-trip -----------------------------------------

def test_build_launch_manifest_round_trips_with_stable_checksum():
    manifest = build_launch_manifest(**_rehearsal_fields())
    assert manifest["schema_version"] == 1
    assert manifest["manifest_sha256"] == compute_manifest_checksum(manifest)


def test_write_then_load_launch_manifest_round_trips(tmp_path):
    path = tmp_path / "manifest.json"
    written = write_launch_manifest(path, **_rehearsal_fields())
    loaded = load_launch_manifest(path)
    assert loaded == written
    assert loaded["mode"] == MODE_REHEARSAL


# --- 3: strict no-overwrite ----------------------------------------------------

def test_write_launch_manifest_refuses_to_overwrite_an_existing_file(tmp_path):
    path = tmp_path / "manifest.json"
    write_launch_manifest(path, **_rehearsal_fields())
    with pytest.raises(LaunchManifestError, match="REFUSING to overwrite"):
        write_launch_manifest(path, **_rehearsal_fields())


# --- 4-6: schema rejection -----------------------------------------------------

def test_build_launch_manifest_rejects_unknown_field():
    with pytest.raises(LaunchManifestError, match="unknown field"):
        build_launch_manifest(**_rehearsal_fields(), some_unexpected_field="x")


def test_build_launch_manifest_rejects_missing_required_field():
    fields = _rehearsal_fields()
    del fields["expected_commit"]
    with pytest.raises(LaunchManifestError, match="missing required field"):
        build_launch_manifest(**fields)


def test_build_launch_manifest_rejects_unsupported_schema_version():
    with pytest.raises(LaunchManifestError, match="schema_version"):
        build_launch_manifest(**_rehearsal_fields(), schema_version=2)


# --- 7-8: no credentials --------------------------------------------------------

@pytest.mark.parametrize("bad_key", ["wandb_api_key", "auth_token", "netrc_contents", "password", "secret_value"])
def test_build_launch_manifest_rejects_credential_shaped_field_names(bad_key):
    with pytest.raises(LaunchManifestError, match="credential-shaped"):
        build_launch_manifest(**_rehearsal_fields(), **{bad_key: "whatever"})


@pytest.mark.parametrize("hex_len", [32, 40, 64])
def test_build_launch_manifest_rejects_bare_hex_token_shaped_values(hex_len):
    with pytest.raises(LaunchManifestError, match="hex-token-shaped"):
        build_launch_manifest(**_rehearsal_fields(manifest_label="a" * hex_len))


def test_build_launch_manifest_allows_hex_shaped_expected_commit():
    # expected_commit is explicitly hex-exempt -- a real 40-char git SHA must
    # still be accepted.
    manifest = build_launch_manifest(**_rehearsal_fields(expected_commit="3d72d14" + "0" * 33))
    assert len(manifest["expected_commit"]) == 40


# --- 9-10: production/rehearsal sweep-id separation ----------------------------

def test_production_mode_requires_the_real_production_sweep_id():
    with pytest.raises(LaunchManifestError, match="requires wandb_sweep_id"):
        build_launch_manifest(**_production_fields(wandb_sweep_id="not-the-real-sweep"))


def test_production_mode_requires_stop_before_training_false():
    with pytest.raises(LaunchManifestError, match="stop_before_training=False"):
        build_launch_manifest(**_production_fields(stop_before_training=True))


def test_rehearsal_mode_refuses_the_production_sweep_id():
    with pytest.raises(LaunchManifestError, match="must never target the production sweep"):
        build_launch_manifest(**_rehearsal_fields(wandb_sweep_id=PRODUCTION_WANDB_SWEEP_ID))


def test_rehearsal_mode_requires_stop_before_training_true():
    with pytest.raises(LaunchManifestError, match="stop_before_training=True"):
        build_launch_manifest(**_rehearsal_fields(stop_before_training=False))


def test_production_manifest_builds_cleanly_with_correct_fields():
    manifest = build_launch_manifest(**_production_fields())
    assert manifest["mode"] == MODE_PRODUCTION
    assert manifest["wandb_sweep_id"] == PRODUCTION_WANDB_SWEEP_ID
    assert manifest["stop_before_training"] is False


# --- 11: tamper detection --------------------------------------------------------

def test_load_launch_manifest_rejects_a_tampered_checksum(tmp_path):
    path = tmp_path / "manifest.json"
    write_launch_manifest(path, **_rehearsal_fields())
    tampered = path.read_text(encoding="utf-8").replace('"execution_generation": 4', '"execution_generation": 5')
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(LaunchManifestError, match="checksum mismatch"):
        load_launch_manifest(path)
