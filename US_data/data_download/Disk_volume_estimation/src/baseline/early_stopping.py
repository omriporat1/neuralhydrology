"""Early-stopping policy engine for Stage 1 NeuralHydrology training runs.

Implements section 2.3's binding decisions (docs/
stage1_validation_optimization_foundation.md, "Part E"): save a checkpoint
every epoch (an operational convention for the training harness, not
something this module enforces); never stop before a minimum epoch; treat
"official validation events" (development-validation raw-space metric
computations, expected every 2-3 epochs) as the unit patience is measured
in, not epochs themselves; require a minimum absolute improvement to reset
patience; stop after a fixed number of non-improving official validation
events, or at a hard epoch-budget ceiling, whichever comes first; always
track the best epoch independent of the final epoch.

This module is deliberately NH/torch-independent -- it is a pure state
machine over (epoch, metric_value) pairs recorded by a training harness. It
has no parameter, field, or code path through which temporal-test or
spatial-holdout data could enter: every public function's signature is
restricted to development-validation metric values and policy/state
dictionaries. `tests/test_early_stopping.py` inspects these signatures
directly to prove this structurally, mirroring the same discipline used for
Part C's event-selection "no prediction-error-based selection" proof.

Persisted state (see `new_state`/`load_state`/`save_state`) is a single
small JSON document making the whole history restart-safe: a training job
resumed after a crash reloads exactly the state it left off with and never
needs to recompute or guess prior official-validation results.
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path

__all__ = [
    "StoppingError",
    "load_early_stopping_policy",
    "new_state",
    "load_state",
    "save_state",
    "record_official_validation_event",
    "best_checkpoint_epoch",
]

_REQUIRED_POLICY_KEYS = [
    "policy_name",
    "metric_name",
    "higher_is_better",
    "min_epoch_before_stop",
    "min_delta",
    "patience_events",
    "max_epoch_budget",
]

_STATE_SCHEMA_VERSION = 1


class StoppingError(Exception):
    """Raised for invalid early-stopping policy/state usage."""


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_early_stopping_policy(path) -> dict:
    import yaml

    p = Path(path)
    if not p.is_file():
        raise StoppingError(f"early-stopping policy file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise StoppingError(f"early-stopping policy {p} did not parse to a mapping")
    missing = [k for k in _REQUIRED_POLICY_KEYS if k not in data]
    if missing:
        raise StoppingError(f"early-stopping policy {p} missing required key(s): {missing}")
    if data["min_epoch_before_stop"] < 1:
        raise StoppingError("min_epoch_before_stop must be >= 1")
    if data["min_delta"] < 0:
        raise StoppingError("min_delta must be >= 0")
    if data["patience_events"] < 1:
        raise StoppingError("patience_events must be >= 1")
    if data["max_epoch_budget"] < data["min_epoch_before_stop"]:
        raise StoppingError("max_epoch_budget must be >= min_epoch_before_stop")
    return data


# ---------------------------------------------------------------------------
# State lifecycle
# ---------------------------------------------------------------------------

def new_state(policy: dict) -> dict:
    """Build a fresh, empty persisted state for the given policy."""
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "policy_name": policy["policy_name"],
        "metric_name": policy["metric_name"],
        "higher_is_better": bool(policy["higher_is_better"]),
        "history": [],
        "best_epoch": None,
        "best_metric_value": None,
        "events_since_best_improvement": 0,
        "stopped": False,
        "stop_reason": None,
        "stop_epoch": None,
    }


def load_state(path) -> dict | None:
    """Load a persisted state, or return None if no state file exists yet."""
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    if state.get("schema_version") != _STATE_SCHEMA_VERSION:
        raise StoppingError(
            f"early-stopping state {p} has unsupported schema_version={state.get('schema_version')!r}"
        )
    return state


def save_state(path, state: dict) -> None:
    """Persist state atomically (write to a temp file, then rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, default=str)
        os.replace(tmp_name, p)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------

def _is_improvement(metric_value: float, best_metric_value: float, min_delta: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return (metric_value - best_metric_value) >= min_delta
    return (best_metric_value - metric_value) >= min_delta


def record_official_validation_event(state: dict, epoch: int, metric_value: float, policy: dict) -> dict:
    """Record one official development-validation metric value and return a
    new state reflecting the updated best-checkpoint tracking and stopping
    decision.

    Never accepts, reads, or references temporal-test or spatial-holdout
    data -- ``metric_value`` must already be a development-validation
    (screening-subset or full-population) raw-space metric computed by the
    caller.

    Restart-safe: replaying the exact same (epoch, metric_value) that was
    already the last recorded event is a no-op (returns an equivalent
    state). Any other out-of-order or inconsistent replay raises.
    """
    if not isinstance(epoch, int) or epoch < 1:
        raise StoppingError(f"epoch must be a positive int, got {epoch!r}")
    if not math.isfinite(metric_value):
        raise StoppingError(f"metric_value must be finite, got {metric_value!r}")

    history = state["history"]
    if history:
        last = history[-1]
        if epoch == last["epoch"]:
            if metric_value == last["metric_value"]:
                return copy.deepcopy(state)
            raise StoppingError(
                f"epoch {epoch} already recorded with a different metric_value "
                f"({last['metric_value']!r} != {metric_value!r}) -- inconsistent replay"
            )
        if epoch < last["epoch"]:
            raise StoppingError(
                f"epoch {epoch} is not after the last recorded epoch {last['epoch']} -- out of order"
            )

    if state["stopped"]:
        raise StoppingError(
            f"cannot record a new official validation event: training already stopped "
            f"at epoch {state['stop_epoch']} ({state['stop_reason']})"
        )

    new = copy.deepcopy(state)
    higher_is_better = policy["higher_is_better"]
    min_delta = policy["min_delta"]

    is_new_best = new["best_metric_value"] is None or _is_improvement(
        metric_value, new["best_metric_value"], min_delta, higher_is_better
    )
    if is_new_best:
        new["best_epoch"] = epoch
        new["best_metric_value"] = metric_value
        new["events_since_best_improvement"] = 0
    else:
        new["events_since_best_improvement"] += 1

    new["history"].append({"epoch": epoch, "metric_value": metric_value, "is_new_best": is_new_best})

    should_stop, reason = _evaluate_stopping(new, epoch, policy)
    if should_stop:
        new["stopped"] = True
        new["stop_reason"] = reason
        new["stop_epoch"] = epoch

    return new


def _evaluate_stopping(state: dict, epoch: int, policy: dict) -> tuple[bool, str | None]:
    if not policy.get("performance_early_stopping_enabled", True):
        if epoch >= policy["max_epoch_budget"]:
            return True, "max_epoch_budget_reached"
        return False, None
    if epoch < policy["min_epoch_before_stop"]:
        return False, None
    if epoch >= policy["max_epoch_budget"]:
        return True, "max_epoch_budget_reached"
    if state["events_since_best_improvement"] >= policy["patience_events"]:
        return True, "patience_exhausted"
    return False, None


def best_checkpoint_epoch(state: dict) -> int | None:
    """Return the epoch of the best-tracked checkpoint (None if no official
    validation event has been recorded yet). Always retained independent of
    the final/stopping epoch, per section 2.3."""
    return state["best_epoch"]
