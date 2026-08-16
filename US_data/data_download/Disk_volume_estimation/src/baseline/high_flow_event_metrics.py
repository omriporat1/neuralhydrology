"""High-flow conditional and event-window metrics for the Dynamic-Input-
Family-A event/high-flow audit.

Implements sections 1 and 3 of the pre-registered methodology at
``.scratch_local/moriah_evidence/dynamic_input_family_a_event_audit/
METHODOLOGY_preregistered.md``. Pure, deterministic functions operating on
already-extracted raw-space (m^3/s) arrays -- no NeuralHydrology inference,
no file I/O. Reuses :func:`src.baseline.nh_raw_space_evaluation.raw_space_metrics`
for the correlation/KGE bundle rather than re-implementing it; adds only the
conditional-subset framing (threshold masking, NSE suppression, normalized
RMSE, sample-size gating) and the event-window peak/timing/volume/shape
metrics that module does not provide.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .hydrograph_atlas_events import EventWindow
from .nh_raw_space_evaluation import raw_space_metrics

__all__ = [
    "HighFlowEventMetricsError",
    "basin_high_flow_threshold",
    "high_flow_conditional_metrics",
    "event_metrics",
]


class HighFlowEventMetricsError(ValueError):
    """Raised for a setup/contract problem (shape mismatch, no finite
    values, invalid quantile), never for an ordinary poor-skill outcome."""


def basin_high_flow_threshold(observed_m3s: np.ndarray, quantile: float) -> float:
    """Basin-specific high-flow threshold from OBSERVED discharge only.

    ``quantile`` in (0, 1), e.g. 0.90 or 0.95. Uses only finite values;
    raises if none are finite."""
    if not (0.0 < quantile < 1.0):
        raise HighFlowEventMetricsError(f"quantile must be in (0, 1), got {quantile}")
    obs = np.asarray(observed_m3s, dtype=np.float64)
    finite = obs[np.isfinite(obs)]
    if finite.size == 0:
        raise HighFlowEventMetricsError("no finite observed values to derive a threshold from")
    return float(np.quantile(finite, quantile))


def high_flow_conditional_metrics(
    obs_m3s: np.ndarray,
    sim_m3s: np.ndarray,
    *,
    threshold: float,
    min_n_for_correlation: int = 10,
) -> dict:
    """Metrics on the subset where admitted obs >= ``threshold``.

    Never computes NSE (a conditional high-flow-only subset makes NSE's
    variance-normalization misleading; see methodology doc section 1).
    Pearson r / KGE are populated only when ``n >= min_n_for_correlation``
    AND both obs and sim have nonzero variance on the subset; otherwise left
    NaN (not silently defaulted, not omitted from the dict -- the key is
    always present so downstream tables have a stable schema)."""
    obs = np.asarray(obs_m3s, dtype=np.float64)
    sim = np.asarray(sim_m3s, dtype=np.float64)
    if obs.shape != sim.shape:
        raise HighFlowEventMetricsError(f"obs shape {obs.shape} != sim shape {sim.shape}")
    if not np.isfinite(threshold):
        raise HighFlowEventMetricsError(f"threshold must be finite, got {threshold}")

    finite = np.isfinite(obs) & np.isfinite(sim)
    mask = finite & (obs >= threshold)
    obs_c = obs[mask]
    sim_c = sim[mask]
    n = int(obs_c.size)

    result = {
        "threshold": float(threshold),
        "n": n,
        "rmse": float("nan"),
        "mae": float("nan"),
        "bias": float("nan"),
        "nrmse": float("nan"),
        "pbias": float("nan"),
        "pearson_r": float("nan"),
        "kge": float("nan"),
    }
    if n == 0:
        return result

    error = sim_c - obs_c
    obs_mean = float(np.mean(obs_c))
    result["rmse"] = float(np.sqrt(np.mean(error ** 2)))
    result["mae"] = float(np.mean(np.abs(error)))
    result["bias"] = float(np.mean(error))
    if obs_mean != 0.0:
        result["nrmse"] = result["rmse"] / obs_mean

    obs_sum = float(np.sum(obs_c))
    if obs_sum != 0.0:
        result["pbias"] = float(100.0 * np.sum(error) / obs_sum)

    if n >= min_n_for_correlation and np.std(obs_c) > 0.0 and np.std(sim_c) > 0.0:
        full = raw_space_metrics(obs_c, sim_c)
        result["pearson_r"] = full["pearson_r"]
        result["kge"] = full["kge"]

    return result


def event_metrics(
    dates_window: Sequence,
    obs_m3s_window: np.ndarray,
    sim_m3s_window: np.ndarray,
    *,
    event: EventWindow,
) -> dict:
    """Peak-magnitude, peak-timing, volume, and shape metrics for one
    (basin, event) window against one family/epoch's predictions.

    ``dates_window``/``obs_m3s_window``/``sim_m3s_window`` must already be
    sliced to exactly ``[event.window_start, event.window_end]`` (inclusive)
    and equal length. ``event.peak_value``/``event.peak_time`` (fixed at
    observed-only selection time) are used as the observed peak -- never
    re-derived from this window's obs array, so the observed side of every
    metric is identical across every family/epoch by construction.

    Volume uses a rectangular (per-admitted-hourly-sample) sum, not
    trapezoidal integration -- see methodology doc section 3 for why.
    Event-window NSE/KGE are supplementary only (unstable on short windows;
    never used to drive the decision)."""
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates_window)))
    obs = np.asarray(obs_m3s_window, dtype=np.float64)
    sim = np.asarray(sim_m3s_window, dtype=np.float64)
    if not (len(dates) == obs.shape[0] == sim.shape[0]):
        raise HighFlowEventMetricsError(
            f"dates ({len(dates)}) / obs ({obs.shape[0]}) / sim ({sim.shape[0]}) length mismatch"
        )

    finite = np.isfinite(obs) & np.isfinite(sim)
    n_admitted = int(finite.sum())

    result = {
        "n_admitted": n_admitted,
        "n_total": int(len(dates)),
        "obs_peak": float(event.peak_value),
        "obs_peak_time": event.peak_time.isoformat(),
        "sim_peak": float("nan"),
        "sim_peak_time": None,
        "abs_peak_error": float("nan"),
        "signed_peak_bias": float("nan"),
        "relative_peak_error": float("nan"),
        "abs_timing_error_hours": float("nan"),
        "obs_volume_m3": float("nan"),
        "sim_volume_m3": float("nan"),
        "relative_volume_bias": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "nrmse_by_obs_peak": float("nan"),
        "nse_supplementary": float("nan"),
        "kge_supplementary": float("nan"),
    }
    if n_admitted == 0:
        return result

    obs_f = obs[finite]
    sim_f = sim[finite]
    dates_f = dates[finite]

    sim_peak_idx = int(np.argmax(sim_f))
    sim_peak = float(sim_f[sim_peak_idx])
    sim_peak_time = dates_f[sim_peak_idx]
    result["sim_peak"] = sim_peak
    result["sim_peak_time"] = sim_peak_time.isoformat()
    result["abs_peak_error"] = abs(sim_peak - event.peak_value)
    result["signed_peak_bias"] = sim_peak - event.peak_value
    if event.peak_value != 0.0:
        result["relative_peak_error"] = (sim_peak - event.peak_value) / event.peak_value
    result["abs_timing_error_hours"] = abs(
        (sim_peak_time - event.peak_time).total_seconds()
    ) / 3600.0

    seconds_per_sample = 3600.0
    obs_volume = float(np.sum(obs_f) * seconds_per_sample)
    sim_volume = float(np.sum(sim_f) * seconds_per_sample)
    result["obs_volume_m3"] = obs_volume
    result["sim_volume_m3"] = sim_volume
    if obs_volume != 0.0:
        result["relative_volume_bias"] = (sim_volume - obs_volume) / obs_volume

    error = sim_f - obs_f
    result["rmse"] = float(np.sqrt(np.mean(error ** 2)))
    result["mae"] = float(np.mean(np.abs(error)))
    if event.peak_value != 0.0:
        result["nrmse_by_obs_peak"] = result["rmse"] / event.peak_value

    if n_admitted >= 2 and np.std(obs_f) > 0.0:
        full = raw_space_metrics(obs_f, sim_f)
        result["nse_supplementary"] = full["nse"]
        result["kge_supplementary"] = full["kge"]

    return result
