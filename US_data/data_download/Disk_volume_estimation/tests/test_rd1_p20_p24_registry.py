"""Tests for the RD1 P20-P24 registry lock/append/verify contract in
``src.baseline.rd1_p20_p24_registry``.

Covers, in order:
  1. ``append_row_verified`` happy path: appends one new row, returns a
     backup path that exists on disk, and the on-disk registry afterward
     contains exactly the prior rows plus the new one, in order.
  2. ``append_row_verified`` refuses (``duplicate_append_refused``) an order
     that is already present, leaving the on-disk registry byte-identical.
  3. ``append_row_verified`` refuses (``registry_not_at_expected_prior_state``)
     when the caller's ``expected_prior_orders`` disagrees with what is
     actually on disk, leaving the on-disk registry byte-identical.
  4. ``registry_orders`` returns a sorted list of int orders from a registry
     mapping, tolerating an empty ``runs`` list.
  5. ``RegistryLock`` raises ``RegistryError("fcntl_unavailable", ...)`` on a
     platform without ``fcntl`` (exercised directly on this Windows dev
     machine via monkeypatching the module's ``fcntl`` reference to
     ``None``, rather than skipped -- the guard itself is what is tested).

Never touches Moriah, W&B, or Slurm.
"""
from __future__ import annotations

import json

import pytest

from src.baseline.rd1_p20_p24_registry import (
    RegistryError,
    RegistryLock,
    append_row_verified,
    registry_orders,
)
import src.baseline.rd1_p20_p24_registry as registry_module


def _write_registry(path, orders):
    payload = {"runs": [{"order": o, "wandb_run_id": f"run-{o}"} for o in orders]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def test_append_row_verified_happy_path(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, [1, 2, 3])
    backup_dir = tmp_path / "backups"

    result = append_row_verified(
        registry_path,
        {"order": 4, "wandb_run_id": "run-4"},
        expected_prior_orders=[1, 2, 3],
        backup_dir=backup_dir,
    )

    assert result["orders_after"] == [1, 2, 3, 4]
    backup_path = result["registry_backup_path"]
    assert backup_path
    from pathlib import Path

    assert Path(backup_path).is_file()

    reread = json.loads(registry_path.read_text(encoding="utf-8"))
    assert [r["order"] for r in reread["runs"]] == [1, 2, 3, 4]
    assert reread["runs"][-1] == {"order": 4, "wandb_run_id": "run-4"}


def test_append_row_verified_refuses_duplicate_order(tmp_path):
    registry_path = tmp_path / "registry.json"
    original = _write_registry(registry_path, [1, 2, 3])
    backup_dir = tmp_path / "backups"

    with pytest.raises(RegistryError) as excinfo:
        append_row_verified(
            registry_path,
            {"order": 3, "wandb_run_id": "run-3-again"},
            expected_prior_orders=[1, 2],
            backup_dir=backup_dir,
        )

    assert excinfo.value.reason == "duplicate_append_refused"
    assert json.loads(registry_path.read_text(encoding="utf-8")) == original
    assert not backup_dir.exists() or not list(backup_dir.iterdir())


def test_append_row_verified_refuses_unexpected_prior_state(tmp_path):
    registry_path = tmp_path / "registry.json"
    original = _write_registry(registry_path, [1, 2, 3])
    backup_dir = tmp_path / "backups"

    with pytest.raises(RegistryError) as excinfo:
        append_row_verified(
            registry_path,
            {"order": 4, "wandb_run_id": "run-4"},
            expected_prior_orders=[1, 2],
            backup_dir=backup_dir,
        )

    assert excinfo.value.reason == "registry_not_at_expected_prior_state"
    assert json.loads(registry_path.read_text(encoding="utf-8")) == original


def test_registry_orders_sorted_and_tolerates_empty():
    assert registry_orders({"runs": [{"order": 3}, {"order": 1}, {"order": 2}]}) == [1, 2, 3]
    assert registry_orders({"runs": []}) == []
    assert registry_orders({}) == []


def test_registry_lock_raises_when_fcntl_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "fcntl", None)
    lock = RegistryLock(tmp_path / "registry.lock", exclusive=True)
    with pytest.raises(RegistryError) as excinfo:
        with lock:
            pass
    assert excinfo.value.reason == "fcntl_unavailable"
