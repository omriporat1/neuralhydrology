"""Bounded same-order retry eligibility for the RD1 P20-P24 fixed chain.

Unlike the retired controller (which retried by submitting a brand-new
Slurm job from inside another job -- the exact pattern this replacement
retires), each fixed-chain job runs its own ``wandb agent`` attempt
in-process and, if that attempt fails, may retry in-process at most once,
under narrow conditions. There is no cross-job sacct polling here: the
retry decision is made by the same job that just observed the failure.

Eligibility requires ALL of:

1. the attempt budget is not exhausted (``attempt_index < max_attempts``);
2. the failed attempt's captured output contains the exact, previously
   observed local wandb-core service/socket startup-failure marker;
3. no W&B run id was resolved from that same attempt's own captured output;
4. a bounded, read-only W&B sweep-runs audit (before vs. after the attempt)
   independently confirms no new run appeared in the sweep -- defense in
   depth against a race where ``wandb.init()`` partially created a run
   before the socket error surfaced.

Any other condition is fail-closed: no retry, and the caller must treat the
attempt as a terminal failure exactly as if this policy did not exist.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, TypeVar

__all__ = [
    "WANDB_SERVICE_STARTUP_FAILURE_MARKER",
    "RetryAuditError",
    "is_socket_failure_retry_eligible",
    "fetch_with_backoff",
    "no_new_run_appeared",
]

WANDB_SERVICE_STARTUP_FAILURE_MARKER = "Failed to connect to service on socket"

_T = TypeVar("_T")


class RetryAuditError(Exception):
    """Raised when a bounded read-only W&B audit exhausts its attempt
    budget without a successful fetch. Fail-closed: the caller must treat
    this exactly as an ineligible retry, never as a silent "assume ok"."""


def fetch_with_backoff(
    fetch_fn: "Callable[[], _T]",
    *,
    max_attempts: int = 3,
    base_delay_sec: float = 2.0,
    sleep_fn: "Callable[[float], None]" = time.sleep,
) -> "_T":
    """Call ``fetch_fn`` (a read-only W&B audit call) up to ``max_attempts``
    times with exponential backoff (``base_delay_sec * 2**attempt``,
    0-indexed, no sleep after the final attempt), returning its result on
    the first success. Raises :class:`RetryAuditError` if every attempt
    raises. Never mutates state -- only ever used for read-only audits.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last_exc: "Exception | None" = None
    for attempt in range(max_attempts):
        try:
            return fetch_fn()
        except Exception as exc:  # noqa: BLE001 -- any fetch failure is retried, then surfaced
            last_exc = exc
            if attempt < max_attempts - 1:
                sleep_fn(base_delay_sec * (2 ** attempt))
    raise RetryAuditError(f"exhausted {max_attempts} read-only audit attempts: {last_exc}") from last_exc


def no_new_run_appeared(run_ids_before: "Iterable[str]", run_ids_after: "Iterable[str]") -> bool:
    """True only if ``run_ids_after`` introduces no id absent from
    ``run_ids_before`` (a pure set-difference check; never contacts W&B
    itself -- callers obtain both snapshots via :func:`fetch_with_backoff`
    around a read-only sweep-runs listing)."""
    return set(run_ids_after) - set(run_ids_before) == set()


def is_socket_failure_retry_eligible(
    *,
    attempt_index: int,
    max_attempts: int,
    attempt_output_text: str,
    run_id_resolved: "str | None",
    no_new_run_confirmed: bool,
) -> "tuple[bool, str]":
    """Return ``(eligible, reason)`` for one bounded same-order retry.

    ``attempt_index`` is 1-based (the attempt that just failed). ``reason``
    is always a specific machine-readable code, whether eligible or not, so
    callers can log/audit the exact basis for the decision.
    """
    if attempt_index >= max_attempts:
        return False, "retry_budget_exhausted"
    if WANDB_SERVICE_STARTUP_FAILURE_MARKER not in attempt_output_text:
        return False, "failure_output_missing_exact_socket_marker"
    if run_id_resolved:
        return False, "wandb_run_already_exists_retry_forbidden"
    if not no_new_run_confirmed:
        return False, "sweep_audit_could_not_confirm_no_new_run"
    return True, "all_conditions_met"
