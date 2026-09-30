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
  16. ``run_agent_with_retry`` invokes ``on_attempt`` for every single
      attempt (success and failure alike), never influencing the retry
      decision itself.
  17. ``redact_secrets`` redacts every known credential shape (key=value,
      Authorization: Bearer, .netrc password line, bare hex token) while
      leaving ordinary log text untouched.
  18. ``find_local_execution_provenance`` resolves a unique match, refuses
      (``local_execution_provenance_not_found``) on zero matches, and
      refuses (``local_execution_provenance_ambiguous``) on more than one.
  19. ``validate_local_provenance_record`` accepts a fully valid record
      (independently re-derived identity, ``VALID``/eligible/finite),
      refuses a non-``VALID`` status, refuses a non-finite objective, and
      refuses an identity mismatch against the caller's own expectations.
  20. ``validate_run_for_registry_acceptance`` refuses whenever the
      independent read-only W&B lookup disagrees with local state (not
      finished, not flagged valid, non-finite objective) or local identity
      re-derivation disagrees, and accepts only when both independent
      sources fully agree.
  21. ``_cmd_append_registry_row`` never appends -- and leaves the on-disk
      registry byte-identical -- when the acceptance gate refuses (e.g. an
      agent that exited 0 but whose dispatched W&B run never reached a
      finished/valid state); it appends a row with every required field
      populated only when the full gate passes.
  22. ``_persist_attempt_evidence`` persists exactly one redacted evidence
      file per attempt, for both a successful and a failed attempt, and
      writes nothing outside the caller-supplied attempts directory.
  23. ``find_local_execution_provenance_by_order`` resolves a unique
      order-only match, refuses (``local_execution_provenance_not_found``)
      on zero matches, and refuses (``local_execution_provenance_ambiguous``)
      on more than one. ``resolve_run_id_from_local_provenance`` resolves
      the run id this proposal's own attempt created from that local
      record -- selecting it over a later-created, differently-ordered
      run elsewhere in the same sweep (the root-cause fix for a
      mis-registered run on 2026-09-30: the removed
      ``_wandb_resolve_latest_run_id`` picked the sweep-wide "most
      recently created" run instead, with no guarantee it was this
      attempt's own run) -- refuses on missing/ambiguous local provenance
      for this order, refuses on a record with no usable ``wandb_run_id``,
      and never falls back to any sweep-wide listing. A resolved run id is
      additionally run through the full acceptance gate before being
      returned, so it also refuses whenever the independent W&B lookup
      disagrees with local state (not finished, not flagged valid).
  24. ``validate_terminal_wandb_run`` likewise transparently retries a
      transient W&B API error and still enforces its full acceptance
      contract (state/``flashnh/valid``/finite-objective) once the call
      succeeds, and raises ``RunAcceptanceError("wandb_run_lookup_failed",
      ...)`` -- unchanged from before this fix -- after exhausting backoff
      on a persistent failure.

Every W&B/subprocess-shaped input is a plain injected callable; never
touches Moriah, W&B, or Slurm. The registry-acceptance/append tests
monkeypatch only ``validate_terminal_wandb_run`` (the true W&B network
boundary) and ``RegistryLock`` (POSIX ``fcntl``-only, already separately
covered by ``tests/test_rd1_p20_p24_registry.py`` on this Windows dev
machine) -- the real ``configuration_id_v2``/``proposal_id_v2``/
``trial_id_v2``/``append_row_verified`` implementations are exercised as-is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.rd1_p20_p24_job as rd1_job  # noqa: E402
from scripts.rd1_p20_p24_job import (  # noqa: E402
    AgentAttemptResult,
    ManifestIdentityError,
    PreflightError,
    RunAcceptanceError,
    V2_METRIC_NAME,
    _cmd_append_registry_row,
    _persist_attempt_evidence,
    check_preflight,
    compute_expected_prior_orders,
    find_local_execution_provenance,
    find_local_execution_provenance_by_order,
    redact_secrets,
    resolve_run_id_from_local_provenance,
    run_agent_with_retry,
    validate_local_provenance_record,
    validate_run_for_registry_acceptance,
    verify_pinned_manifest_checksum,
)
from src.baseline.rd1_p20_p24_registry import read_registry  # noqa: E402
from src.baseline.rd1_p20_p24_retry import WANDB_SERVICE_STARTUP_FAILURE_MARKER  # noqa: E402
from src.baseline.sweep_v2_six_axis_campaign import (  # noqa: E402
    CAMPAIGN_ID_V2,
    CONFIGURATION_CANONICALIZATION_VERSION_V2,
    DOMAIN_VERSION_V2,
    OBJECTIVE_ID_V2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
)
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import (  # noqa: E402
    write_v2_wandb_bridge_manifest,
)


class _NullLock:
    """Stand-in for ``RegistryLock`` in tests that exercise
    ``_cmd_append_registry_row``'s acceptance-gate wiring on this Windows
    dev machine, where ``RegistryLock`` unconditionally raises
    (``fcntl_unavailable``) before ever reaching the gate. The real lock's
    own behavior (including that exact refusal) is already independently
    covered by ``tests/test_rd1_p20_p24_registry.py``; this class contains
    no lock semantics of its own."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_TEST_HYPERPARAMETERS = {
    "learning_rate": 5e-4,
    "hidden_size": 128,
    "embedding_dropout": 0.1,
    "output_dropout": 0.1,
    "batch_size": 256,
    "seq_length": 72,
}
_TEST_SUPPORT_CONTRACT_VERSION = OBJECTIVE_ID_V2
_TEST_SUPPORT_CONTRACT_SHA256 = "c" * 64


def _build_provenance_record(
    *,
    order: int,
    sweep_id: str,
    run_id: str,
    search_arm: str = "bayesian",
    execution_generation: int = 1,
    execution_status: str = "VALID",
    objective_eligible: bool = True,
    objective_score: float = 0.42,
    fixed_support_metric_name: "str | None" = None,
) -> dict:
    """A fully valid ``execution_provenance.json`` record, built from the
    real v2 identity-derivation primitives (never a hand-typed identity
    string), so identity-mismatch tests are corrupting a genuinely
    independently-re-derivable value rather than an arbitrary fixture."""
    configuration_id = configuration_id_v2(
        _TEST_HYPERPARAMETERS,
        support_contract_version=_TEST_SUPPORT_CONTRACT_VERSION,
        support_contract_sha256=_TEST_SUPPORT_CONTRACT_SHA256,
    )
    proposal_id = proposal_id_v2(search_arm, order)
    trial_id = trial_id_v2(configuration_id, proposal_id, execution_generation=execution_generation)
    return {
        "hyperparameters": dict(_TEST_HYPERPARAMETERS),
        "search_arm": search_arm,
        "proposal_order": order,
        "execution_generation": execution_generation,
        "configuration_id": configuration_id,
        "proposal_id": proposal_id,
        "trial_id": trial_id,
        "campaign_id": CAMPAIGN_ID_V2,
        "domain_version": DOMAIN_VERSION_V2,
        "wandb_sweep_id": sweep_id,
        "wandb_run_id": run_id,
        "support_contract_version": _TEST_SUPPORT_CONTRACT_VERSION,
        "support_contract_sha256": _TEST_SUPPORT_CONTRACT_SHA256,
        "execution_status": execution_status,
        "objective_eligible": objective_eligible,
        "fixed_support_metric_name": fixed_support_metric_name or V2_METRIC_NAME,
        "objective_score": objective_score,
    }


def _write_provenance_file(output_root: Path, record: dict, *, trial_dir_name: "str | None" = None) -> Path:
    trial_dir = output_root / (trial_dir_name or record["trial_id"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = trial_dir / "execution_provenance.json"
    provenance_path.write_text(json.dumps(record), encoding="utf-8")
    return provenance_path


def _write_test_manifest(path: Path, *, output_root: Path, wandb_sweep_id: str) -> None:
    """The one launch-identity manifest ``validate_run_for_registry_acceptance``
    reads ``output_root``/``wandb_sweep_id`` from -- built with the same real,
    loader-validated manifest writer used everywhere else in this codebase."""
    write_v2_wandb_bridge_manifest(
        path,
        manifest_label="rd1-p20p24-registry-acceptance-test",
        created_at_utc="2026-09-29T00:00:00Z",
        mode="rehearsal",
        expected_commit="a" * 40,
        repository_root=str(Path(__file__).resolve().parents[1]),
        expected_runtime_python="/canonical/python",
        wandb_project="rd1-test-project",
        wandb_sweep_id=wandb_sweep_id,
        wandb_entity="rd1-test-entity",
        output_root=str(output_root),
        package_root=str(output_root / "pkg"),
        screening_basin_ids_path=str(output_root / "screening.txt"),
        screening_basin_ids_sha256="b" * 64,
        fixed_support_contract_path=str(output_root / "support.json"),
        fixed_support_contract_version=_TEST_SUPPORT_CONTRACT_VERSION,
        fixed_support_contract_sha256=_TEST_SUPPORT_CONTRACT_SHA256,
        baseline_policy_path=str(output_root / "baseline.yaml"),
        policy_overlay_path=str(output_root / "overlay.yaml"),
        base_pilot_policy_path=str(output_root / "pilot.yaml"),
        proposal_order=1,
        execution_generation=1,
        stop_before_training=True,
        max_agents=1,
        campaign_id=CAMPAIGN_ID_V2,
        domain_version=DOMAIN_VERSION_V2,
        canonicalization_version=CONFIGURATION_CANONICALIZATION_VERSION_V2,
        objective_id=OBJECTIVE_ID_V2,
    )


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


def test_run_agent_with_retry_invokes_on_attempt_for_every_attempt():
    calls = []

    def audit_fn():
        return ["r1"]

    def attempt_fn():
        if not calls:
            return AgentAttemptResult(
                success=False, returncode=1, output_text=WANDB_SERVICE_STARTUP_FAILURE_MARKER, run_id=None
            )
        return AgentAttemptResult(success=True, returncode=0, output_text="ok", run_id="r2")

    def on_attempt(attempt_index, result):
        calls.append((attempt_index, result.success))

    result = run_agent_with_retry(
        attempt_fn=attempt_fn, audit_run_ids_fn=audit_fn, max_attempts=2, on_attempt=on_attempt
    )
    assert result.success is True
    assert calls == [(1, False), (2, True)]


def test_redact_secrets_redacts_all_known_credential_shapes():
    text = (
        "WANDB_API_KEY=deadbeefdeadbeefdeadbeefdeadbeef\n"
        "Authorization: Bearer abcdef0123456789abcdef0123456789abcdef01\n"
        "machine api.wandb.ai login someuser password deadbeefdeadbeefdeadbeefdeadbeef\n"
        "password: hunter2hunter2hunter2hunter2hunter2\n"
        "normal log line: epoch 3 loss=0.512\n"
    )
    redacted = redact_secrets(text)
    assert "deadbeef" not in redacted
    assert "hunter2" not in redacted
    assert "abcdef0123456789abcdef0123456789abcdef01" not in redacted
    assert redacted.count("***REDACTED***") >= 4
    assert "normal log line: epoch 3 loss=0.512" in redacted


def test_find_local_execution_provenance_unique_match(short_tmp_path):
    output_root = short_tmp_path
    (output_root / "trial-a").mkdir()
    (output_root / "trial-a" / "execution_provenance.json").write_text(
        json.dumps({"proposal_order": 20, "wandb_run_id": "run-a"}), encoding="utf-8"
    )
    found = find_local_execution_provenance(output_root, expected_order=20, expected_run_id="run-a")
    assert found == output_root / "trial-a" / "execution_provenance.json"


def test_find_local_execution_provenance_refuses_no_match(short_tmp_path):
    output_root = short_tmp_path
    (output_root / "trial-a").mkdir()
    (output_root / "trial-a" / "execution_provenance.json").write_text(
        json.dumps({"proposal_order": 19, "wandb_run_id": "run-a"}), encoding="utf-8"
    )
    with pytest.raises(RunAcceptanceError) as excinfo:
        find_local_execution_provenance(output_root, expected_order=20, expected_run_id="run-a")
    assert excinfo.value.reason == "local_execution_provenance_not_found"


def test_find_local_execution_provenance_refuses_ambiguous_match(short_tmp_path):
    output_root = short_tmp_path
    for name in ("trial-a", "trial-b"):
        (output_root / name).mkdir()
        (output_root / name / "execution_provenance.json").write_text(
            json.dumps({"proposal_order": 20, "wandb_run_id": "run-a"}), encoding="utf-8"
        )
    with pytest.raises(RunAcceptanceError) as excinfo:
        find_local_execution_provenance(output_root, expected_order=20, expected_run_id="run-a")
    assert excinfo.value.reason == "local_execution_provenance_ambiguous"


def test_validate_local_provenance_record_accepts_fully_valid_record(tmp_path):
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x", objective_score=0.777)
    provenance_path = tmp_path / "execution_provenance.json"
    provenance_path.write_text(json.dumps(record), encoding="utf-8")
    objective = validate_local_provenance_record(
        provenance_path, expected_order=20, expected_sweep_id="sweep-x", expected_run_id="run-x"
    )
    assert objective == pytest.approx(0.777)


def test_validate_local_provenance_record_refuses_non_valid_status(tmp_path):
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x", execution_status="INVALID")
    provenance_path = tmp_path / "execution_provenance.json"
    provenance_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RunAcceptanceError) as excinfo:
        validate_local_provenance_record(
            provenance_path, expected_order=20, expected_sweep_id="sweep-x", expected_run_id="run-x"
        )
    assert excinfo.value.reason == "local_provenance_not_valid"


def test_validate_local_provenance_record_refuses_non_finite_objective(tmp_path):
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x", objective_score=float("nan"))
    provenance_path = tmp_path / "execution_provenance.json"
    provenance_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RunAcceptanceError) as excinfo:
        validate_local_provenance_record(
            provenance_path, expected_order=20, expected_sweep_id="sweep-x", expected_run_id="run-x"
        )
    assert excinfo.value.reason == "local_provenance_objective_not_finite"


def test_validate_local_provenance_record_refuses_identity_mismatch(tmp_path):
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x")
    provenance_path = tmp_path / "execution_provenance.json"
    provenance_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RunAcceptanceError) as excinfo:
        validate_local_provenance_record(
            provenance_path, expected_order=20, expected_sweep_id="sweep-x", expected_run_id="a-different-run-id"
        )
    assert excinfo.value.reason == "local_provenance_identity_mismatch"


def test_validate_run_for_registry_acceptance_refuses_when_wandb_run_not_finished(monkeypatch, short_tmp_path):
    """Root-cause-#2 scenario: the agent launcher subprocess can exit 0
    while the W&B run it dispatched never reached a finished state -- the
    orchestrator must refuse on the independent W&B lookup alone, even
    though the local provenance record itself looks fully valid."""
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x")
    _write_provenance_file(output_root, record)

    def fake_validate_terminal_wandb_run(_manifest_path, _run_id):
        raise RunAcceptanceError("wandb_run_not_finished", "state=crashed")

    monkeypatch.setattr(rd1_job, "validate_terminal_wandb_run", fake_validate_terminal_wandb_run)

    with pytest.raises(RunAcceptanceError) as excinfo:
        validate_run_for_registry_acceptance(
            manifest_path=manifest_path, wandb_sweep_id="sweep-x", run_id="run-x", order=20
        )
    assert excinfo.value.reason == "wandb_run_not_finished"


def test_validate_run_for_registry_acceptance_refuses_when_wandb_run_not_flagged_valid(monkeypatch, short_tmp_path):
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x")
    _write_provenance_file(output_root, record)

    def fake_validate_terminal_wandb_run(_manifest_path, _run_id):
        raise RunAcceptanceError("wandb_run_not_flagged_valid", "flashnh/valid=False")

    monkeypatch.setattr(rd1_job, "validate_terminal_wandb_run", fake_validate_terminal_wandb_run)

    with pytest.raises(RunAcceptanceError) as excinfo:
        validate_run_for_registry_acceptance(
            manifest_path=manifest_path, wandb_sweep_id="sweep-x", run_id="run-x", order=20
        )
    assert excinfo.value.reason == "wandb_run_not_flagged_valid"


def test_validate_run_for_registry_acceptance_refuses_when_wandb_objective_not_finite(monkeypatch, short_tmp_path):
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x")
    _write_provenance_file(output_root, record)

    def fake_validate_terminal_wandb_run(_manifest_path, _run_id):
        raise RunAcceptanceError("wandb_objective_not_finite", "flashnh/common120_raw_space_nse_v001=nan")

    monkeypatch.setattr(rd1_job, "validate_terminal_wandb_run", fake_validate_terminal_wandb_run)

    with pytest.raises(RunAcceptanceError) as excinfo:
        validate_run_for_registry_acceptance(
            manifest_path=manifest_path, wandb_sweep_id="sweep-x", run_id="run-x", order=20
        )
    assert excinfo.value.reason == "wandb_objective_not_finite"


def test_validate_run_for_registry_acceptance_refuses_local_identity_mismatch(short_tmp_path):
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x")
    trial_dir_name = record["trial_id"]
    record["configuration_id"] = "sweep_v2_cfg_" + "0" * 20  # corrupted: disagrees with re-derivation
    _write_provenance_file(output_root, record, trial_dir_name=trial_dir_name)

    with pytest.raises(RunAcceptanceError) as excinfo:
        validate_run_for_registry_acceptance(
            manifest_path=manifest_path, wandb_sweep_id="sweep-x", run_id="run-x", order=20
        )
    assert excinfo.value.reason == "local_provenance_identity_mismatch"


def test_validate_run_for_registry_acceptance_accepts_fully_valid_matching_run(monkeypatch, short_tmp_path):
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=20, sweep_id="sweep-x", run_id="run-x", objective_score=0.9)
    _write_provenance_file(output_root, record)

    monkeypatch.setattr(rd1_job, "validate_terminal_wandb_run", lambda _manifest_path, _run_id: 0.9)

    acceptance = validate_run_for_registry_acceptance(
        manifest_path=manifest_path, wandb_sweep_id="sweep-x", run_id="run-x", order=20
    )
    assert acceptance["objective"] == pytest.approx(0.9)
    assert acceptance["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_cmd_append_registry_row_refuses_and_leaves_registry_untouched_on_exit0_but_crashed_run(
    monkeypatch, short_tmp_path
):
    """Scenario 1: the agent launcher subprocess exited 0, but the run it
    dispatched never reached a finished state -- ``_cmd_append_registry_row``
    must refuse and the on-disk registry must remain byte-identical (no
    append, no backup)."""
    monkeypatch.setattr(rd1_job, "RegistryLock", _NullLock)

    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=1, sweep_id="sweep-x", run_id="run-x")
    _write_provenance_file(output_root, record)

    def fake_validate_terminal_wandb_run(_manifest_path, _run_id):
        raise RunAcceptanceError("wandb_run_not_finished", "agent exited 0 but the dispatched run crashed")

    monkeypatch.setattr(rd1_job, "validate_terminal_wandb_run", fake_validate_terminal_wandb_run)

    registry_path = output_root / "registry.json"
    registry_before = json.dumps({"runs": []})
    registry_path.write_text(registry_before, encoding="utf-8")
    backup_dir = output_root / "backups"

    args = argparse.Namespace(
        order=1,
        run_id="run-x",
        recorded_at_utc="2026-09-29T00:00:00Z",
        registry_path=registry_path,
        lock_path=output_root / "registry.lock",
        backup_dir=backup_dir,
        manifest_path=manifest_path,
        wandb_sweep_id="sweep-x",
    )
    exit_code = _cmd_append_registry_row(args)
    assert exit_code == 1
    assert registry_path.read_text(encoding="utf-8") == registry_before
    assert not backup_dir.exists()


def test_cmd_append_registry_row_appends_complete_fields_on_fully_valid_run(monkeypatch, short_tmp_path):
    """Scenario 5: a fully-valid finished run is appended with every
    required field (order, run id, timestamp, objective, manifest
    checksum) populated."""
    monkeypatch.setattr(rd1_job, "RegistryLock", _NullLock)

    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=1, sweep_id="sweep-x", run_id="run-x", objective_score=0.5)
    _write_provenance_file(output_root, record)
    monkeypatch.setattr(rd1_job, "validate_terminal_wandb_run", lambda _manifest_path, _run_id: 0.5)

    registry_path = output_root / "registry.json"
    registry_path.write_text(json.dumps({"runs": []}), encoding="utf-8")
    backup_dir = output_root / "backups"

    args = argparse.Namespace(
        order=1,
        run_id="run-x",
        recorded_at_utc="2026-09-29T00:00:00Z",
        registry_path=registry_path,
        lock_path=output_root / "registry.lock",
        backup_dir=backup_dir,
        manifest_path=manifest_path,
        wandb_sweep_id="sweep-x",
    )
    exit_code = _cmd_append_registry_row(args)
    assert exit_code == 0

    registry = read_registry(registry_path)
    assert len(registry["runs"]) == 1
    row = registry["runs"][0]
    assert row["order"] == 1
    assert row["wandb_run_id"] == "run-x"
    assert row["recorded_at_utc"] == "2026-09-29T00:00:00Z"
    assert row["objective"] == pytest.approx(0.5)
    assert row["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_persist_attempt_evidence_persists_and_redacts_on_success_and_failure(tmp_path):
    """Scenarios 6 and 7: evidence is written for both a successful and a
    failed attempt, secrets are redacted in the persisted copy, and nothing
    is written outside the caller-supplied attempts directory."""
    attempts_dir = tmp_path / "agent_attempts"

    success_result = AgentAttemptResult(
        success=True,
        returncode=0,
        output_text="wandb: run created\nWANDB_API_KEY=deadbeefdeadbeefdeadbeefdeadbeef",
        run_id=None,
    )
    failure_result = AgentAttemptResult(
        success=False,
        returncode=1,
        output_text="ERROR password: hunter2hunter2hunter2hunter2",
        run_id=None,
    )

    _persist_attempt_evidence(attempts_dir, 1, success_result)
    _persist_attempt_evidence(attempts_dir, 2, failure_result)

    written = sorted(p.relative_to(attempts_dir) for p in attempts_dir.rglob("*") if p.is_file())
    assert written == [Path("attempt_01.json"), Path("attempt_02.json")]

    record_1 = json.loads((attempts_dir / "attempt_01.json").read_text(encoding="utf-8"))
    assert record_1["success"] is True
    assert record_1["returncode"] == 0
    assert "deadbeef" not in record_1["output_text_redacted"]
    assert "***REDACTED***" in record_1["output_text_redacted"]

    record_2 = json.loads((attempts_dir / "attempt_02.json").read_text(encoding="utf-8"))
    assert record_2["success"] is False
    assert record_2["returncode"] == 1
    assert "hunter2" not in record_2["output_text_redacted"]
    assert "***REDACTED***" in record_2["output_text_redacted"]


class _FakeWandbRun:
    def __init__(self, run_id, *, created_at="2026-01-01T00:00:00Z", state="finished", summary=None):
        self.id = run_id
        self.created_at = created_at
        self.state = state
        self.summary = summary or {}


def test_find_local_execution_provenance_by_order_unique_match(short_tmp_path):
    output_root = short_tmp_path
    (output_root / "trial-a").mkdir()
    (output_root / "trial-a" / "execution_provenance.json").write_text(
        json.dumps({"proposal_order": 21, "wandb_run_id": "run-a"}), encoding="utf-8"
    )
    found = find_local_execution_provenance_by_order(output_root, expected_order=21)
    assert found == output_root / "trial-a" / "execution_provenance.json"


def test_find_local_execution_provenance_by_order_refuses_no_match(short_tmp_path):
    output_root = short_tmp_path
    (output_root / "trial-a").mkdir()
    (output_root / "trial-a" / "execution_provenance.json").write_text(
        json.dumps({"proposal_order": 20, "wandb_run_id": "run-a"}), encoding="utf-8"
    )
    with pytest.raises(RunAcceptanceError) as excinfo:
        find_local_execution_provenance_by_order(output_root, expected_order=21)
    assert excinfo.value.reason == "local_execution_provenance_not_found"


def test_find_local_execution_provenance_by_order_refuses_ambiguous_match(short_tmp_path):
    output_root = short_tmp_path
    for name in ("trial-a", "trial-b"):
        (output_root / name).mkdir()
        (output_root / name / "execution_provenance.json").write_text(
            json.dumps({"proposal_order": 21, "wandb_run_id": f"run-{name}"}), encoding="utf-8"
        )
    with pytest.raises(RunAcceptanceError) as excinfo:
        find_local_execution_provenance_by_order(output_root, expected_order=21)
    assert excinfo.value.reason == "local_execution_provenance_ambiguous"


def test_resolve_run_id_from_local_provenance_selects_own_order_over_later_crashed_run(
    monkeypatch, short_tmp_path
):
    """The regression scenario this fix exists for: a later-created run
    from a DIFFERENT proposal order (e.g. a crashed retry of another
    order) exists in the same sweep. The removed sweep-wide "latest run"
    lookup would have picked that other run; this function must instead
    select the run id recorded in THIS order's own local provenance,
    regardless of what else was created later in the sweep."""
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")

    own_record = _build_provenance_record(order=21, sweep_id="sweep-x", run_id="run-own", objective_score=0.5)
    _write_provenance_file(output_root, own_record)
    # A different order's later-created run, still present in the same output_root/sweep.
    other_record = _build_provenance_record(order=22, sweep_id="sweep-x", run_id="run-other-later", objective_score=0.9)
    _write_provenance_file(output_root, other_record)

    seen_run_ids = []

    def fake_validate_terminal_wandb_run(_manifest_path, run_id, **_kwargs):
        seen_run_ids.append(run_id)
        assert run_id == "run-own"
        return 0.5

    monkeypatch.setattr(rd1_job, "validate_terminal_wandb_run", fake_validate_terminal_wandb_run)

    run_id = resolve_run_id_from_local_provenance(
        manifest_path=manifest_path, wandb_sweep_id="sweep-x", order=21
    )

    assert run_id == "run-own"
    assert seen_run_ids == ["run-own"]  # never even looked at "run-other-later"


def test_resolve_run_id_from_local_provenance_refuses_missing_provenance(short_tmp_path):
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    # No execution_provenance.json written for order=21 at all.

    with pytest.raises(RunAcceptanceError) as excinfo:
        resolve_run_id_from_local_provenance(manifest_path=manifest_path, wandb_sweep_id="sweep-x", order=21)
    assert excinfo.value.reason == "local_execution_provenance_not_found"


def test_resolve_run_id_from_local_provenance_refuses_ambiguous_provenance(short_tmp_path):
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")

    for run_id, trial_dir_name in (("run-a", "trial-a"), ("run-b", "trial-b")):
        record = _build_provenance_record(order=21, sweep_id="sweep-x", run_id=run_id)
        _write_provenance_file(output_root, record, trial_dir_name=trial_dir_name)

    with pytest.raises(RunAcceptanceError) as excinfo:
        resolve_run_id_from_local_provenance(manifest_path=manifest_path, wandb_sweep_id="sweep-x", order=21)
    assert excinfo.value.reason == "local_execution_provenance_ambiguous"


def test_resolve_run_id_from_local_provenance_refuses_when_wandb_run_not_finished(monkeypatch, short_tmp_path):
    """Even with unambiguous local provenance, a fresh W&B lookup that
    disagrees (not finished) must still refuse -- the strict terminal-
    valid-finite acceptance contract is unchanged by this fix."""
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=21, sweep_id="sweep-x", run_id="run-own")
    _write_provenance_file(output_root, record)

    def fake_validate_terminal_wandb_run(_manifest_path, _run_id, **_kwargs):
        raise RunAcceptanceError("wandb_run_not_finished", "state=crashed")

    monkeypatch.setattr(rd1_job, "validate_terminal_wandb_run", fake_validate_terminal_wandb_run)

    with pytest.raises(RunAcceptanceError) as excinfo:
        resolve_run_id_from_local_provenance(manifest_path=manifest_path, wandb_sweep_id="sweep-x", order=21)
    assert excinfo.value.reason == "wandb_run_not_finished"


def test_resolve_run_id_from_local_provenance_never_lists_sweep_runs(monkeypatch, short_tmp_path):
    """No sweep-wide fallback anywhere: this function must never call the
    sweep-listing API at all, only a single-run lookup for its own
    already-known run id."""
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")
    record = _build_provenance_record(order=21, sweep_id="sweep-x", run_id="run-own", objective_score=0.5)
    _write_provenance_file(output_root, record)

    good_run = _FakeWandbRun("run-own", state="finished", summary={"flashnh/valid": True, V2_METRIC_NAME: 0.5})

    class _FakeApi:
        def sweep(self, path):
            raise AssertionError("must never list the sweep -- no sweep-wide fallback")

        def run(self, path):
            assert path == "rd1-test-entity/rd1-test-project/run-own"
            return good_run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=_FakeApi))

    run_id = resolve_run_id_from_local_provenance(
        manifest_path=manifest_path, wandb_sweep_id="sweep-x", order=21
    )
    assert run_id == "run-own"


def test_validate_terminal_wandb_run_retries_transient_failure_then_succeeds(monkeypatch, short_tmp_path):
    """Registry-acceptance-gate sibling of the above: a transient W&B API
    error on the first call must not be fatal, and the full acceptance
    contract (state/``flashnh/valid``/finite-objective) is still enforced
    once the retried call succeeds."""
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")

    good_run = _FakeWandbRun(
        "run-x", state="finished", summary={"flashnh/valid": True, V2_METRIC_NAME: 0.42}
    )
    calls = {"n": 0}

    class _FakeApi:
        def run(self, path):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("An error occurred while verifying the API key.")
            assert path == "rd1-test-entity/rd1-test-project/run-x"
            return good_run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=_FakeApi))
    sleeps = []

    objective = rd1_job.validate_terminal_wandb_run(manifest_path, "run-x", sleep_fn=sleeps.append)

    assert objective == pytest.approx(0.42)
    assert calls["n"] == 2
    assert len(sleeps) == 1


def test_validate_terminal_wandb_run_raises_run_acceptance_error_after_exhausting_backoff(
    monkeypatch, short_tmp_path
):
    output_root = short_tmp_path
    manifest_path = output_root / "manifest.json"
    _write_test_manifest(manifest_path, output_root=output_root, wandb_sweep_id="sweep-x")

    class _FakeApi:
        def run(self, path):
            raise RuntimeError("An error occurred while verifying the API key.")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=_FakeApi))
    sleeps = []

    with pytest.raises(RunAcceptanceError) as excinfo:
        rd1_job.validate_terminal_wandb_run(manifest_path, "run-x", sleep_fn=sleeps.append)

    assert excinfo.value.reason == "wandb_run_lookup_failed"
    assert len(sleeps) == 2
