"""Stage 1 lead-6 optimization pilot: restart-safe early-stopping
integration (task item 5).

Composes, unmodified: :mod:`src.baseline.early_stopping`'s pure state
machine (``load_early_stopping_policy`` / ``new_state`` / ``load_state`` /
``save_state`` / ``record_official_validation_event`` /
``best_checkpoint_epoch``) and
:func:`src.baseline.pilot_screening_eval.classify_screening_epoch_role`.
This module adds only:

1. **An "effective policy"** (:func:`build_effective_policy`) that layers
   this pilot's stricter epoch sub-cap (``min(base_max_epoch_budget,
   pilot_max_epoch_budget)`` = 36) on top of the unmodified, still-binding
   ``config/stage1_early_stopping_policy_v001.yaml`` -- that committed base
   policy file is never edited or loosened, only ever tightened for this
   pilot. Also asserts the base policy's ``min_epoch_before_stop`` /
   ``min_delta`` / ``patience_events`` exactly match this pilot's frozen
   values (eligible from epoch 6, 0.005, patience 3): if the base policy is
   later retuned by an unrelated Stage 1 decision, this pilot fails loudly
   rather than silently drifting onto different thresholds mid-run.
2. **One restart-safe entry point** (:func:`record_screening_event`) that:
   - is a no-op for a ``"diagnostic_only"`` epoch (epoch 3) -- never fed
     into the stopping state machine, matching task item 5's "never stop
     before epoch 6" / "epoch 3 diagnostic only" requirement;
   - feeds a ``"stopping_eligible"`` epoch's primary metric (median
     per-basin raw-space NSE, from
     :func:`src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`)
     into ``early_stopping.record_official_validation_event``;
   - persists the updated state to ``run_dir/pilot_early_stopping_state.json``
     after every stopping-eligible event (never after a diagnostic-only one,
     since nothing in the stopping decision changed);
   - rejects an off-cadence (``"not_a_screening_epoch"``) request.
3. **Resume helpers** (:func:`load_or_init_pilot_state`) that reload
   persisted state on restart, refusing to reload state written under a
   different effective policy. Out-of-order/contradictory epoch replay is
   already rejected by ``early_stopping.record_official_validation_event``
   itself -- not reimplemented here. This module does not change NH's own
   resume mechanics or its RNG-non-identical-replay caveat.
"""
from __future__ import annotations

from pathlib import Path

from .early_stopping import (
    StoppingError,
    best_checkpoint_epoch,
    load_early_stopping_policy,
    load_state,
    new_state,
    record_official_validation_event,
    save_state,
)
from .pilot_lead06_config import PilotPolicy

__all__ = [
    "PilotEarlyStoppingError",
    "STATE_FILENAME",
    "build_effective_policy",
    "load_or_init_pilot_state",
    "record_screening_event",
    "pilot_best_checkpoint_epoch",
]

STATE_FILENAME = "pilot_early_stopping_state.json"

_EXPECTED_METRIC_NAME = "median_per_basin_raw_space_nse"
_EXPECTED_MIN_DELTA = 0.005
_EXPECTED_PATIENCE_EVENTS = 3


class PilotEarlyStoppingError(Exception):
    """Raised for a pilot/base early-stopping policy mismatch, contradictory
    resumed state, or an attempt to feed a non-stopping-eligible event into
    the stopping state machine."""


def build_effective_policy(pilot_policy: PilotPolicy) -> dict:
    """Load the unmodified base early-stopping policy and layer this
    pilot's stricter epoch sub-cap on top. See module docstring."""
    base_policy = load_early_stopping_policy(pilot_policy.base_early_stopping_policy_path)

    if base_policy["metric_name"] != _EXPECTED_METRIC_NAME:
        raise PilotEarlyStoppingError(
            f"base early-stopping policy metric_name={base_policy['metric_name']!r}, "
            f"expected {_EXPECTED_METRIC_NAME!r}"
        )
    if not base_policy["higher_is_better"]:
        raise PilotEarlyStoppingError("base early-stopping policy must have higher_is_better=true for NSE")
    if base_policy["min_epoch_before_stop"] != pilot_policy.stopping_eligible_from_epoch:
        raise PilotEarlyStoppingError(
            f"base policy min_epoch_before_stop={base_policy['min_epoch_before_stop']} != "
            f"pilot stopping_eligible_from_epoch={pilot_policy.stopping_eligible_from_epoch}"
        )
    if base_policy["min_delta"] != _EXPECTED_MIN_DELTA:
        raise PilotEarlyStoppingError(
            f"base policy min_delta={base_policy['min_delta']} != frozen pilot value {_EXPECTED_MIN_DELTA}"
        )
    if base_policy["patience_events"] != _EXPECTED_PATIENCE_EVENTS:
        raise PilotEarlyStoppingError(
            f"base policy patience_events={base_policy['patience_events']} != "
            f"frozen pilot value {_EXPECTED_PATIENCE_EVENTS}"
        )

    effective = dict(base_policy)
    effective["max_epoch_budget"] = min(base_policy["max_epoch_budget"], pilot_policy.pilot_max_epoch_budget)
    effective["policy_name"] = f"{base_policy['policy_name']}__pilot_subcap_{effective['max_epoch_budget']}"
    return effective


def load_or_init_pilot_state(run_dir, effective_policy: dict) -> dict:
    """Reload the persisted early-stopping state for this run, or create a
    fresh one if none exists yet (first stopping-eligible screening event of
    a new run). Refuses to reload state persisted under a different
    effective policy (e.g. a stale run_dir reused with a different pilot
    sub-cap)."""
    state_path = Path(run_dir) / STATE_FILENAME
    state = load_state(state_path)
    if state is None:
        return new_state(effective_policy)
    if state["policy_name"] != effective_policy["policy_name"]:
        raise PilotEarlyStoppingError(
            f"persisted state at {state_path} was created under policy_name "
            f"{state['policy_name']!r}, but the effective policy for this resume is "
            f"{effective_policy['policy_name']!r} -- refusing to reload contradictory state"
        )
    return state


def record_screening_event(
    *, run_dir, epoch: int, epoch_role: str, primary_metric_median: float, effective_policy: dict
) -> dict:
    """Feed one screening checkpoint's result into the restart-safe
    early-stopping state machine, persisting the updated state on disk.

    ``epoch_role`` must be
    :func:`src.baseline.pilot_screening_eval.classify_screening_epoch_role`'s
    output for this same epoch. A ``"diagnostic_only"`` event never touches
    the stopping decision (returned state is the reloaded/fresh state,
    unchanged, and nothing new is written to disk). ``"not_a_screening_epoch"``
    is rejected.
    """
    if epoch_role == "not_a_screening_epoch":
        raise PilotEarlyStoppingError(f"epoch {epoch} is not a screening epoch; refusing to record")
    if epoch_role not in ("diagnostic_only", "stopping_eligible"):
        raise PilotEarlyStoppingError(f"unknown epoch_role: {epoch_role!r}")

    state = load_or_init_pilot_state(run_dir, effective_policy)
    if epoch_role == "diagnostic_only":
        return state

    try:
        new = record_official_validation_event(state, epoch, primary_metric_median, effective_policy)
    except StoppingError as exc:
        raise PilotEarlyStoppingError(str(exc)) from exc

    save_state(Path(run_dir) / STATE_FILENAME, new)
    return new


def pilot_best_checkpoint_epoch(state: dict) -> "int | None":
    return best_checkpoint_epoch(state)
