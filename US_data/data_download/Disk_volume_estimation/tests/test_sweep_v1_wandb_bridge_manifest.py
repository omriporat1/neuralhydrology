"""Tests for the strict, versioned, checksummed Sweep-v1 FRESH-PROPOSAL
W&B-agent bridge launch manifest, ``src.baseline.sweep_v1_wandb_bridge_manifest``
-- the fresh-proposal sibling of ``test_sweep_v1_launch_manifest.py`` (built
for the exact-retry bridge).

Covers, in order:
  1. A minimal valid rehearsal manifest round-trips through
     ``build_wandb_bridge_manifest`` with a stable, recomputable checksum.
  2. ``write_wandb_bridge_manifest`` + ``load_wandb_bridge_manifest``
     round-trip on disk.
  3. Strict no-overwrite: a second write to the same path is refused.
  4. An unknown field is rejected.
  5. A missing required field is rejected.
  6. An unsupported ``schema_version`` is rejected.
  7. No credential-shaped field NAME is ever accepted.
  8. ``mode="production"`` requires the real production sweep id,
     ``stop_before_training=False``, ``proposal_order ==
     PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER``, and
     ``execution_generation == 1``; any contradiction is rejected.
  9. ``mode="rehearsal"`` must never target the production sweep id,
     requires ``stop_before_training=True``, and requires both
     ``proposal_order``/``execution_generation`` to sit in the explicit
     rehearsal namespace (>= ``REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR``);
     any contradiction is rejected.
  10. A tampered on-disk manifest (checksum mismatch) is rejected on load.
  11. The reserved rehearsal constants can never collide with the production
      ledger's next permissible order.

Never imports wandb; never touches the network; never starts NH training.
"""
from __future__ import annotations

import pytest

from src.baseline.sweep_v1_launch_manifest import MODE_PRODUCTION, MODE_REHEARSAL, PRODUCTION_WANDB_SWEEP_ID
from src.baseline.sweep_v1_wandb_bridge_manifest import (
    PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER, REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR,
    REHEARSAL_RESERVED_EXECUTION_GENERATION, REHEARSAL_RESERVED_PROPOSAL_ORDER, WandbBridgeManifestError,
    build_wandb_bridge_manifest, compute_manifest_checksum, load_wandb_bridge_manifest, write_wandb_bridge_manifest,
)


def _rehearsal_fields(**overrides):
    fields = dict(
        manifest_label="sweep_v1_wandb_bridge_rehearsal_v001",
        created_at_utc="2026-08-26T00:00:00Z",
        mode=MODE_REHEARSAL,
        expected_commit="3d72d14" + "0" * 33,
        expected_runtime_python="/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python",
        package_root="/tmp/package",
        screening_basin_ids_path="/tmp/screening.txt",
        output_root="/tmp/out_rehearsal",
        baseline_policy_path="/tmp/baseline_policy.yaml",
        base_pilot_policy_path="/tmp/base_pilot_policy.yaml",
        wandb_project="flashnh-stage1-rehearsal",
        wandb_sweep_id="rehearsal-sweep-abc123",
        proposal_order=REHEARSAL_RESERVED_PROPOSAL_ORDER,
        execution_generation=REHEARSAL_RESERVED_EXECUTION_GENERATION,
        stop_before_training=True,
    )
    fields.update(overrides)
    return fields


def _production_fields(**overrides):
    fields = _rehearsal_fields(
        mode=MODE_PRODUCTION, wandb_sweep_id=PRODUCTION_WANDB_SWEEP_ID, stop_before_training=False,
        manifest_label="sweep_v1_wandb_bridge_production_v001",
        proposal_order=PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER, execution_generation=1,
    )
    fields.update(overrides)
    return fields


# --- 1-2: build/write/load round-trip -----------------------------------------

def test_build_wandb_bridge_manifest_round_trips_with_stable_checksum():
    manifest = build_wandb_bridge_manifest(**_rehearsal_fields())
    assert manifest["schema_version"] == 1
    assert manifest["manifest_sha256"] == compute_manifest_checksum(manifest)


def test_write_then_load_wandb_bridge_manifest_round_trips(tmp_path):
    path = tmp_path / "manifest.json"
    written = write_wandb_bridge_manifest(path, **_rehearsal_fields())
    loaded = load_wandb_bridge_manifest(path)
    assert loaded == written
    assert loaded["mode"] == MODE_REHEARSAL


# --- 3: strict no-overwrite ----------------------------------------------------

def test_write_wandb_bridge_manifest_refuses_to_overwrite_an_existing_file(tmp_path):
    path = tmp_path / "manifest.json"
    write_wandb_bridge_manifest(path, **_rehearsal_fields())
    with pytest.raises(WandbBridgeManifestError, match="REFUSING to overwrite"):
        write_wandb_bridge_manifest(path, **_rehearsal_fields())


# --- 4-6: schema rejection -----------------------------------------------------

def test_build_wandb_bridge_manifest_rejects_unknown_field():
    with pytest.raises(WandbBridgeManifestError, match="unknown field"):
        build_wandb_bridge_manifest(**_rehearsal_fields(), some_unexpected_field="x")


def test_build_wandb_bridge_manifest_rejects_missing_required_field():
    fields = _rehearsal_fields()
    del fields["expected_commit"]
    with pytest.raises(WandbBridgeManifestError, match="missing required field"):
        build_wandb_bridge_manifest(**fields)


def test_build_wandb_bridge_manifest_rejects_unsupported_schema_version():
    with pytest.raises(WandbBridgeManifestError, match="schema_version"):
        build_wandb_bridge_manifest(**_rehearsal_fields(), schema_version=2)


# --- 7: no credentials ----------------------------------------------------------

@pytest.mark.parametrize("bad_key", ["wandb_api_key", "auth_token", "netrc_contents", "password", "secret_value"])
def test_build_wandb_bridge_manifest_rejects_credential_shaped_field_names(bad_key):
    with pytest.raises(WandbBridgeManifestError, match="credential-shaped"):
        build_wandb_bridge_manifest(**_rehearsal_fields(), **{bad_key: "whatever"})


# --- 8: production mode contract ------------------------------------------------

def test_production_mode_requires_the_real_production_sweep_id():
    with pytest.raises(WandbBridgeManifestError, match="requires wandb_sweep_id"):
        build_wandb_bridge_manifest(**_production_fields(wandb_sweep_id="not-the-real-sweep"))


def test_production_mode_requires_stop_before_training_false():
    with pytest.raises(WandbBridgeManifestError, match="stop_before_training=False"):
        build_wandb_bridge_manifest(**_production_fields(stop_before_training=True))


def test_production_mode_requires_next_permissible_proposal_order():
    with pytest.raises(WandbBridgeManifestError, match="requires proposal_order =="):
        build_wandb_bridge_manifest(**_production_fields(proposal_order=900001))


def test_production_mode_requires_execution_generation_one():
    with pytest.raises(WandbBridgeManifestError, match="requires execution_generation == 1"):
        build_wandb_bridge_manifest(**_production_fields(execution_generation=2))


def test_production_manifest_builds_cleanly_with_correct_fields():
    manifest = build_wandb_bridge_manifest(**_production_fields())
    assert manifest["mode"] == MODE_PRODUCTION
    assert manifest["wandb_sweep_id"] == PRODUCTION_WANDB_SWEEP_ID
    assert manifest["stop_before_training"] is False
    assert manifest["proposal_order"] == PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER


# --- 9: rehearsal mode contract --------------------------------------------------

def test_rehearsal_mode_refuses_the_production_sweep_id():
    with pytest.raises(WandbBridgeManifestError, match="must never target the production sweep"):
        build_wandb_bridge_manifest(**_rehearsal_fields(wandb_sweep_id=PRODUCTION_WANDB_SWEEP_ID))


def test_rehearsal_mode_requires_stop_before_training_true():
    with pytest.raises(WandbBridgeManifestError, match="stop_before_training=True"):
        build_wandb_bridge_manifest(**_rehearsal_fields(stop_before_training=False))


def test_rehearsal_mode_requires_proposal_order_in_reserved_namespace():
    with pytest.raises(WandbBridgeManifestError, match="explicit rehearsal namespace"):
        build_wandb_bridge_manifest(**_rehearsal_fields(proposal_order=1))


def test_rehearsal_mode_requires_execution_generation_in_reserved_namespace():
    with pytest.raises(WandbBridgeManifestError, match="explicit rehearsal namespace"):
        build_wandb_bridge_manifest(**_rehearsal_fields(execution_generation=1))


def test_rehearsal_manifest_builds_cleanly_with_correct_fields():
    manifest = build_wandb_bridge_manifest(**_rehearsal_fields())
    assert manifest["mode"] == MODE_REHEARSAL
    assert manifest["proposal_order"] >= REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR
    assert manifest["execution_generation"] >= REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR


# --- 10: tamper detection --------------------------------------------------------

def test_load_wandb_bridge_manifest_rejects_a_tampered_checksum(tmp_path):
    path = tmp_path / "manifest.json"
    write_wandb_bridge_manifest(path, **_rehearsal_fields())
    tampered = path.read_text(encoding="utf-8").replace(
        f'"wandb_sweep_id": "rehearsal-sweep-abc123"', '"wandb_sweep_id": "tampered-sweep-xyz"'
    )
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(WandbBridgeManifestError, match="checksum mismatch"):
        load_wandb_bridge_manifest(path)


# --- 11: namespace non-collision --------------------------------------------------

def test_reserved_rehearsal_namespace_cannot_collide_with_production_ledger():
    assert REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR > PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER
    assert REHEARSAL_RESERVED_PROPOSAL_ORDER >= REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR
    assert REHEARSAL_RESERVED_EXECUTION_GENERATION >= REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR
