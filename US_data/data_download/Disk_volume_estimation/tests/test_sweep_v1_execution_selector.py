"""Tests for the pure, side-effect-free Sweep-v1 executor-selection function,
``select_executor_mode`` in ``src.baseline.sweep_v1_execution`` -- Binding
Design Decision 5 of the accepted exact-retry startup rehearsal task: no
general ``dry_run`` executor bypass, only one selector shared by production
(which dispatches on its result) and the disposable rehearsal (which calls
it and records the result, but never invokes the executor).

Covers:
  1. A real, valid prepared record (built via the same
     ``prepare_bayesian_proposal`` -> ``write_prepared_proposal`` path used
     throughout ``tests/test_sweep_v1_execution.py``) selects
     ``EXECUTOR_MODE_MONOLITHIC``.
  2. The selector refuses (reusing ``_require_prepared``'s own contract) a
     record that fails the frozen Sweep-v1 legality contract -- it adds no
     separate/weaker legality rule of its own.
  3. The selector never imports ``pilot_orchestration``/NH/torch as a
     consequence of being called (no training side effect is possible).
  4. The selector does not mutate its input.

Reuses ``tests/test_sweep_v1_execution.py``'s private fixture helpers
directly (repo convention for private test helpers of the same shape, see
that module's own docstring) rather than duplicating prepared-record
construction.
"""
from __future__ import annotations

import copy
import sys

import pytest

from src.baseline.sweep_v1_execution import EXECUTOR_MODE_MONOLITHIC, SweepV1ExecutionError, select_executor_mode
from tests.test_sweep_v1_execution import _prepared_record


def test_select_executor_mode_returns_monolithic_for_a_real_valid_prepared_record(tmp_path, monkeypatch):
    record, _paths = _prepared_record(tmp_path, monkeypatch)
    assert select_executor_mode(record) == EXECUTOR_MODE_MONOLITHIC


def test_select_executor_mode_refuses_a_record_that_fails_the_prepared_contract(tmp_path, monkeypatch):
    record, _paths = _prepared_record(tmp_path, monkeypatch)
    broken = dict(record)
    broken["performance_early_stopping_enabled"] = True  # violates the frozen Sweep-v1 contract
    with pytest.raises(SweepV1ExecutionError, match="prepared-trial contract mismatch"):
        select_executor_mode(broken)


def test_select_executor_mode_does_not_mutate_its_input(tmp_path, monkeypatch):
    record, _paths = _prepared_record(tmp_path, monkeypatch)
    before = copy.deepcopy(record)
    select_executor_mode(record)
    assert record == before


def test_select_executor_mode_never_imports_pilot_orchestration_or_torch(tmp_path, monkeypatch):
    # Repo convention (see tests/test_wandb_offline_qualification.py's
    # "never imports wandb" scenario): poison the import so any attempt by
    # select_executor_mode itself to import either module raises ImportError
    # immediately, rather than merely observing sys.modules afterward.
    record, _paths = _prepared_record(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "src.baseline.pilot_orchestration", None)

    assert select_executor_mode(record) == EXECUTOR_MODE_MONOLITHIC
