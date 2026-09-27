"""Tests for the RD1 P20-P24 per-proposal driver in ``scripts.rd1_p20_p24_job``.

Covers, in order:
  1. ``check_preflight`` passes silently for P20 when the registry holds
     exactly P1-P19 contiguous.
  2. ``check_preflight`` refuses (``registry_not_contiguous_for_order``) for
     P20 when the registry is missing an intermediate proposal (ambiguous/
     incomplete state) -- no submission-worthy state, no W&B contact.
  3. ``check_preflight`` refuses (``duplicate_order_already_registered``)
     when the requested order is already present in the registry.
  4. ``check_preflight`` refuses (``hard_stop_exceeded``) for order 25,
     proving there is no way to advance past P24 through this gate.
  5. ``check_preflight`` accepts exactly order 24 when P1-P23 are present
     (the last legal order under the hard stop).
  6. ``check_preflight`` refuses a non-positive order.
  7. ``compute_expected_prior_orders`` returns the full 1..order-1 range.
  8. ``verify_pinned_manifest_checksum`` passes for a file matching the
     expected sha256.
  9. ``verify_pinned_manifest_checksum`` raises ``ManifestIdentityError`` on
     a checksum mismatch (byte-for-byte reuse violated).
  10. ``verify_pinned_manifest_checksum`` raises ``ManifestIdentityError``
      when the pinned manifest file does not exist.
  11. ``run_agent_with_retry`` returns immediately on a first-attempt
      success, having taken exactly one "before" audit snapshot up front
      (needed for a possible retry-eligibility diff) and no more.
  12. ``run_agent_with_retry`` retries exactly once and succeeds on the
      exact socket-failure signature with no run id and a confirmed-empty
      audit diff, then stops (does not exceed the retry budget).
  13. ``run_agent_with_retry`` does NOT retry (returns the failed attempt)
      when the failure text does not match the exact socket-failure marker.
  14. ``run_agent_with_retry`` does NOT retry when a run id was already
      resolved on the failed attempt, even with the exact marker present.
  15. ``run_agent_with_retry`` respects ``max_attempts`` and stops retrying
      once the budget is exhausted, even if every condition would otherwise
      be eligible.

Every W&B/subprocess-shaped input is a plain injected callable; never
touches Moriah, W&B, or Slurm.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rd1_p20_p24_job import (  # noqa: E402
    AgentAttemptResult,
    ManifestIdentityError,
    PreflightError,
    check_preflight,
    compute_expected_prior_orders,
    run_agent_with_retry,
    verify_pinned_manifest_checksum,
)
from src.baseline.rd1_p20_p24_retry import WANDB_SERVICE_STARTUP_FAILURE_MARKER  # noqa: E402


def test_check_preflight_accepts_p20_when_p1_p19_present():
    check_preflight(20, list(range(1, 20)))  # must not raise


def test_check_preflight_refuses_incomplete_prior_history():
    with pytest.raises(PreflightError) as excinfo:
        check_preflight(20, [o for o in range(1, 20) if o != 15])
    assert excinfo.value.reason == "registry_not_contiguous_for_order"


def test_check_preflight_refuses_duplicate_order():
    with pytest.raises(PreflightError) as excinfo:
        check_preflight(20, list(range(1, 21)))
    assert excinfo.value.reason == "duplicate_order_already_registered"


def test_check_preflight_hard_stop_refuses_order_25():
    with pytest.raises(PreflightError) as excinfo:
        check_preflight(25, list(range(1, 25)))
    assert excinfo.value.reason == "hard_stop_exceeded"


def test_check_preflight_accepts_exactly_p24():
    check_preflight(24, list(range(1, 24)))  # must not raise


def test_check_preflight_refuses_nonpositive_order():
    with pytest.raises(PreflightError) as excinfo:
        check_preflight(0, [])
    assert excinfo.value.reason == "invalid_order"


def test_compute_expected_prior_orders():
    assert compute_expected_prior_orders(20) == list(range(1, 20))
    assert compute_expected_prior_orders(1) == []


def test_verify_pinned_manifest_checksum_passes_on_match(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b'{"proposal_order": 20}')
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    verify_pinned_manifest_checksum(manifest_path, expected)  # must not raise


def test_verify_pinned_manifest_checksum_raises_on_mismatch(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b'{"proposal_order": 20}')
    with pytest.raises(ManifestIdentityError):
        verify_pinned_manifest_checksum(manifest_path, "0" * 64)


def test_verify_pinned_manifest_checksum_raises_when_missing(tmp_path):
    with pytest.raises(ManifestIdentityError):
        verify_pinned_manifest_checksum(tmp_path / "does_not_exist.json", "0" * 64)


def test_run_agent_with_retry_returns_immediately_on_first_success():
    audit_calls = {"n": 0}

    def audit_fn():
        audit_calls["n"] += 1
        return ["existing-run"]

    def attempt_fn():
        return AgentAttemptResult(success=True, returncode=0, output_text="ok", run_id="new-run")

    result = run_agent_with_retry(attempt_fn=attempt_fn, audit_run_ids_fn=audit_fn)
    assert result.success is True
    assert audit_calls["n"] == 1


def test_run_agent_with_retry_retries_once_on_exact_socket_failure_then_succeeds():
    attempts = {"n": 0}
    audit_snapshots = iter([["r1"], ["r1"]])  # before first attempt, after failed attempt: unchanged

    def audit_fn():
        return next(audit_snapshots)

    def attempt_fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            return AgentAttemptResult(
                success=False,
                returncode=1,
                output_text=f"wandb: ERROR {WANDB_SERVICE_STARTUP_FAILURE_MARKER}",
                run_id=None,
            )
        return AgentAttemptResult(success=True, returncode=0, output_text="ok", run_id="r2")

    result = run_agent_with_retry(attempt_fn=attempt_fn, audit_run_ids_fn=audit_fn, max_attempts=2)
    assert result.success is True
    assert attempts["n"] == 2


def test_run_agent_with_retry_does_not_retry_without_exact_marker():
    attempts = {"n": 0}

    def audit_fn():
        return ["r1"]

    def attempt_fn():
        attempts["n"] += 1
        return AgentAttemptResult(success=False, returncode=1, output_text="unrelated failure", run_id=None)

    result = run_agent_with_retry(attempt_fn=attempt_fn, audit_run_ids_fn=audit_fn, max_attempts=2)
    assert result.success is False
    assert attempts["n"] == 1


def test_run_agent_with_retry_does_not_retry_when_run_id_already_resolved():
    attempts = {"n": 0}

    def audit_fn():
        return ["r1"]

    def attempt_fn():
        attempts["n"] += 1
        return AgentAttemptResult(
            success=False,
            returncode=1,
            output_text=WANDB_SERVICE_STARTUP_FAILURE_MARKER,
            run_id="already-created",
        )

    result = run_agent_with_retry(attempt_fn=attempt_fn, audit_run_ids_fn=audit_fn, max_attempts=2)
    assert result.success is False
    assert attempts["n"] == 1


def test_run_agent_with_retry_respects_max_attempts_budget():
    attempts = {"n": 0}

    def audit_fn():
        return ["r1"]

    def attempt_fn():
        attempts["n"] += 1
        return AgentAttemptResult(
            success=False,
            returncode=1,
            output_text=WANDB_SERVICE_STARTUP_FAILURE_MARKER,
            run_id=None,
        )

    result = run_agent_with_retry(attempt_fn=attempt_fn, audit_run_ids_fn=audit_fn, max_attempts=1)
    assert result.success is False
    assert attempts["n"] == 1  # never retried: budget was 1 from the start
