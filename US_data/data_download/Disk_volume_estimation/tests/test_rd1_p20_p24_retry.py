"""Tests for bounded same-order retry eligibility in
``src.baseline.rd1_p20_p24_retry``.

Covers, in order:
  1. ``is_socket_failure_retry_eligible`` is eligible when every condition
     holds: budget remains, exact marker present, no run id resolved, audit
     confirms no new run.
  2. Ineligible (``retry_budget_exhausted``) once ``attempt_index`` reaches
     ``max_attempts``.
  3. Ineligible (``failure_output_missing_exact_socket_marker``) for any
     other failure text, even one that mentions W&B or sockets loosely.
  4. Ineligible (``wandb_run_already_exists_retry_forbidden``) when a run id
     was resolved from the failed attempt, regardless of the marker.
  5. Ineligible (``sweep_audit_could_not_confirm_no_new_run``) when the
     audit could not confirm the sweep is unchanged.
  6. ``no_new_run_appeared`` true for identical before/after id sets, false
     when a new id appears, true for two empty sets.
  7. ``fetch_with_backoff`` returns the first successful result without
     retrying when the first call succeeds.
  8. ``fetch_with_backoff`` retries through transient failures and returns
     the eventual success, sleeping with the expected exponential schedule.
  9. ``fetch_with_backoff`` raises ``RetryAuditError`` (fail-closed, never a
     silent "assume ok") after exhausting every attempt, chaining the last
     underlying exception.
  10. ``fetch_with_backoff`` rejects ``max_attempts < 1`` outright.

Never contacts W&B; every W&B-shaped call is a plain injected callable.
"""
from __future__ import annotations

import pytest

from src.baseline.rd1_p20_p24_retry import (
    WANDB_SERVICE_STARTUP_FAILURE_MARKER,
    RetryAuditError,
    fetch_with_backoff,
    is_socket_failure_retry_eligible,
    no_new_run_appeared,
)


def test_eligible_when_all_conditions_hold():
    eligible, reason = is_socket_failure_retry_eligible(
        attempt_index=1,
        max_attempts=2,
        attempt_output_text=f"wandb: ERROR {WANDB_SERVICE_STARTUP_FAILURE_MARKER} unix:///tmp/x.sock",
        run_id_resolved=None,
        no_new_run_confirmed=True,
    )
    assert eligible is True
    assert reason == "all_conditions_met"


def test_ineligible_when_budget_exhausted():
    eligible, reason = is_socket_failure_retry_eligible(
        attempt_index=2,
        max_attempts=2,
        attempt_output_text=WANDB_SERVICE_STARTUP_FAILURE_MARKER,
        run_id_resolved=None,
        no_new_run_confirmed=True,
    )
    assert eligible is False
    assert reason == "retry_budget_exhausted"


def test_ineligible_when_marker_absent():
    eligible, reason = is_socket_failure_retry_eligible(
        attempt_index=1,
        max_attempts=2,
        attempt_output_text="wandb: ERROR some unrelated socket timeout while syncing",
        run_id_resolved=None,
        no_new_run_confirmed=True,
    )
    assert eligible is False
    assert reason == "failure_output_missing_exact_socket_marker"


def test_ineligible_when_run_id_already_resolved():
    eligible, reason = is_socket_failure_retry_eligible(
        attempt_index=1,
        max_attempts=2,
        attempt_output_text=WANDB_SERVICE_STARTUP_FAILURE_MARKER,
        run_id_resolved="abc123",
        no_new_run_confirmed=True,
    )
    assert eligible is False
    assert reason == "wandb_run_already_exists_retry_forbidden"


def test_ineligible_when_audit_cannot_confirm_no_new_run():
    eligible, reason = is_socket_failure_retry_eligible(
        attempt_index=1,
        max_attempts=2,
        attempt_output_text=WANDB_SERVICE_STARTUP_FAILURE_MARKER,
        run_id_resolved=None,
        no_new_run_confirmed=False,
    )
    assert eligible is False
    assert reason == "sweep_audit_could_not_confirm_no_new_run"


def test_no_new_run_appeared_variants():
    assert no_new_run_appeared(["a", "b"], ["a", "b"]) is True
    assert no_new_run_appeared(["a"], ["a", "b"]) is False
    assert no_new_run_appeared([], []) is True


def test_fetch_with_backoff_returns_first_success_without_sleeping():
    sleeps = []
    calls = {"n": 0}

    def fetch_fn():
        calls["n"] += 1
        return "ok"

    result = fetch_with_backoff(fetch_fn, max_attempts=3, sleep_fn=sleeps.append)
    assert result == "ok"
    assert calls["n"] == 1
    assert sleeps == []


def test_fetch_with_backoff_retries_then_succeeds_with_expected_schedule():
    sleeps = []
    attempts = {"n": 0}

    def fetch_fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError(f"transient failure {attempts['n']}")
        return "eventually ok"

    result = fetch_with_backoff(fetch_fn, max_attempts=3, base_delay_sec=2.0, sleep_fn=sleeps.append)
    assert result == "eventually ok"
    assert attempts["n"] == 3
    assert sleeps == [2.0, 4.0]


def test_fetch_with_backoff_raises_retry_audit_error_after_exhausting_attempts():
    def always_fails():
        raise RuntimeError("permanent failure")

    with pytest.raises(RetryAuditError) as excinfo:
        fetch_with_backoff(always_fails, max_attempts=2, sleep_fn=lambda _: None)
    assert "2 read-only audit attempts" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_fetch_with_backoff_rejects_nonpositive_max_attempts():
    with pytest.raises(ValueError):
        fetch_with_backoff(lambda: "x", max_attempts=0)
