#!/usr/bin/env python3
"""
Build human-review hydrograph cards for Flash-NH WY2024 basin screening.

Reads already-downloaded hourly streamflow parquet files and existing
WY2024 metrics/candidate-class tables, then produces event-centred review
cards for a bounded, diverse set of basins.

Usage:
    python scripts/build_hydrograph_review_cards.py
    python scripts/build_hydrograph_review_cards.py --max-basins 30 --no-show
    python scripts/build_hydrograph_review_cards.py \\
        --metrics-dir reports/flashnh_wy2024_streamflow_metrics_v002 \\
        --output-dir  reports/flashnh_hydrograph_review_cards_v001
"""

import argparse
import ast
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not available – plot generation will be skipped.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_METRICS_DIR = REPO_ROOT / "reports" / "flashnh_wy2024_streamflow_metrics_v002"
DEFAULT_OUTPUT_DIR  = REPO_ROOT / "reports" / "flashnh_hydrograph_review_cards_v001"
DEFAULT_MAX_BASINS  = 80
DEFAULT_SEED        = 42

# Budget fractions per review group (must sum <= 1.0; remainder goes to class reps)
GROUP_FRACTIONS = {
    "extreme_rbi":              0.10,
    "extreme_rise_per_km2":     0.07,
    "extreme_q95_ratio":        0.07,
    "manual_review_context":    0.08,
    "reference_flashy_core":    0.15,
    "reference_flashy_moderate": 0.10,
    "reference_flashy_possible": 0.12,
    "reference_low_flashiness":  0.05,
    "reference_exclude_hard_qc": 0.07,
    "reference_rbi_low":         0.07,
    "reference_zero_flow":       0.06,
}

QC_JUMP_THRESH  = 20.0   # max_abs_hourly_jump_over_Q50
QC_ZERO_THRESH  = 0.10   # zero_flow_fraction
QC_SPIKE_FLAG   = "CONTEXT_SUSPICIOUS_SPIKE_SEVERE"

PLOT_COLORS = {
    "series":   "#2563eb",
    "q50":      "#6b7280",
    "q95":      "#f59e0b",
    "q99":      "#ef4444",
    "peak":     "#dc2626",
    "rise":     "#16a34a",
    "fall":     "#7c3aed",
    "zero":     "#94a3b8",
}

HUMAN_REVIEW_ALLOWED = {
    "human_decision":      "KEEP | KEEP_LOW_CONFIDENCE | EXCLUDE | UNSURE",
    "hydrograph_behavior": (
        "smooth_event_response | flashy_event_response | ephemeral_pulse_response | "
        "step_like_response | noisy_low_flow | unclear"
    ),
    "artifact_type": (
        "none | single_point_spike | gap_edge_jump | sensor_noise | "
        "rating_shift_or_step | regulated_or_managed | ice_or_seasonal_artifact | "
        "ephemeral_valid | very_flashy_valid | other "
        "(multiple values allowed, separated by semicolons)"
    ),
    "confidence":      "high | medium | low",
    "reviewer_notes":  "free text",
}

MEDIUM_HALF_HOURS = 72   # +-72 h around event centre
TIGHT_HALF_HOURS  = 36   # +-36 h around event centre
CLOSE_HALF_HOURS  = 12   # +-12 h around event centre (key view for peak/rise/fall)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR,
                   help=f"Directory with WY2024 tables/ and hourly_streamflow/. "
                        f"Default: {DEFAULT_METRICS_DIR}")
    p.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Root output directory. Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--max-basins",  type=int,  default=DEFAULT_MAX_BASINS,
                   help=f"Maximum basins in review set. Default: {DEFAULT_MAX_BASINS}")
    p.add_argument("--no-show",     action="store_true",
                   help="Never display plots interactively (always save to disk).")
    p.add_argument("--seed",        type=int,  default=DEFAULT_SEED,
                   help=f"Random seed. Default: {DEFAULT_SEED}")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _parse_flag_list(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s in ("", "[]", "nan"):
        return []
    try:
        r = ast.literal_eval(s)
        return r if isinstance(r, list) else []
    except Exception:
        return []


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    tables_dir = metrics_dir / "tables"
    for p in [tables_dir / "wy2024_streamflow_metrics.csv",
              metrics_dir / "wy2024_streamflow_metrics.csv"]:
        if p.exists():
            print(f"  Loading metrics: {p}")
            df = pd.read_csv(p, dtype={"STAID": str})
            df["context_flags_list"] = df.get("context_flags",
                                               pd.Series(dtype=str)).apply(_parse_flag_list)
            df["hard_flags_list"]    = df.get("hard_flags",
                                               pd.Series(dtype=str)).apply(_parse_flag_list)
            df["is_hard_qc_pass"] = (
                df.get("candidate_class", pd.Series(dtype=str)) != "EXCLUDE_HARD_QC"
            )
            return df
    raise FileNotFoundError(
        f"wy2024_streamflow_metrics.csv not found in {metrics_dir}. Check --metrics-dir."
    )


def load_hourly(staid: str, hourly_dir: Path):
    for name in (staid.zfill(8), staid):
        p = hourly_dir / f"{name}.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if "time_utc" in df.columns:
                    df = df.set_index("time_utc")
                df.index = pd.to_datetime(df.index, utc=True)
                return df.sort_index()
            except Exception as exc:
                warnings.warn(f"Could not load {p}: {exc}")
                return None
    return None


def _resolve_dirs(args):
    """Return (metrics_dir, hourly_dir) resolving v001/v002 fallbacks."""
    md = args.metrics_dir
    hd = md / "hourly_streamflow"
    if not md.exists() or not (md / "tables").exists():
        alt = REPO_ROOT / "reports" / "flashnh_wy2024_streamflow_metrics_v002"
        if alt.exists():
            print(f"  Metrics dir not found at {md}; falling back to {alt}")
            md = alt
    hd = md / "hourly_streamflow"
    if not hd.exists():
        alt_hd = REPO_ROOT / "reports" / "flashnh_wy2024_streamflow_metrics_v002" / "hourly_streamflow"
        if alt_hd.exists():
            print(f"  hourly_streamflow not found under {md}; using {alt_hd}")
            hd = alt_hd
        else:
            print(f"  WARNING: hourly_streamflow not found at {hd}. Plots will be skipped.")
    return md, hd


# ---------------------------------------------------------------------------
# Review-set selection (budget-based, stratified)
# ---------------------------------------------------------------------------
def _budget(max_basins: int) -> dict:
    """Compute per-group slot counts from fractions, minimum 1 per group."""
    total_frac = sum(GROUP_FRACTIONS.values())
    budgets = {}
    for g, f in GROUP_FRACTIONS.items():
        budgets[g] = max(1, round(f / total_frac * max_basins))
    # Trim to max_basins
    total = sum(budgets.values())
    if total > max_basins:
        excess = total - max_basins
        for g in sorted(budgets, key=lambda x: -budgets[x]):
            cut = min(excess, budgets[g] - 1)
            budgets[g] -= cut
            excess -= cut
            if excess == 0:
                break
    return budgets


def _stratified_sample(pool: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """
    Sample n rows from pool, preferring diversity across area_bin and BFI_bin.
    Falls back to completeness-sorted selection if too few rows.
    """
    if len(pool) <= n:
        return pool
    # Try to cover distinct area_bin / BFI_bin combinations
    strat_col = None
    for col in ("area_bin", "BFI_bin"):
        if col in pool.columns and pool[col].nunique() > 1:
            strat_col = col
            break
    if strat_col:
        groups = pool.groupby(strat_col, group_keys=False)
        per_group = max(1, n // pool[strat_col].nunique())
        sampled = groups.apply(
            lambda g: g.sort_values(
                "hourly_completeness_pct" if "hourly_completeness_pct" in g.columns else g.columns[0],
                ascending=False,
            ).head(per_group)
        )
        sampled = sampled.reset_index(drop=True)
        if len(sampled) < n:
            remaining = pool[~pool["STAID"].isin(sampled["STAID"])]
            extra = remaining.sample(
                n=min(n - len(sampled), len(remaining)),
                random_state=int(rng.integers(0, 2**31)),
            )
            sampled = pd.concat([sampled, extra])
        return sampled.head(n)
    # Fallback: top by completeness then random
    top = pool.sort_values(
        "hourly_completeness_pct" if "hourly_completeness_pct" in pool.columns
        else pool.columns[0], ascending=False,
    ).head(n * 3)
    return top.sample(n=min(n, len(top)), random_state=int(rng.integers(0, 2**31)))


def select_review_set(df: pd.DataFrame, max_basins: int, seed: int) -> pd.DataFrame:
    """
    Build a deduplicated, diverse review set.

    Groups are filled in priority order up to their budgeted slot count.
    Every group gets at least 1 basin if any candidates exist.
    """
    rng  = np.random.default_rng(seed)
    bdg  = _budget(max_basins)
    seen = {}   # staid -> {"review_group": ..., "reason_selected": ..., "row": ...}

    def add_rows(rows: pd.DataFrame, group: str, reason_fn):
        for _, row in rows.iterrows():
            s = row["STAID"]
            if s not in seen:
                seen[s] = {"review_group": group,
                           "reason_selected": reason_fn(row),
                           "row": row}

    hqp = df[df["is_hard_qc_pass"]].copy()

    # ── Extreme: top RBI ────────────────────────────────────────────────────
    if "RBI" in df.columns:
        pool = df.sort_values("RBI", ascending=False).head(bdg["extreme_rbi"])
        add_rows(pool, "extreme_rbi",
                 lambda r: f"extreme_rbi RBI={r.get('RBI', np.nan):.4f}")

    # ── Extreme: top max hourly rise per km² ────────────────────────────────
    col = "max_hourly_rise_per_km2"
    if col in df.columns:
        pool = df.sort_values(col, ascending=False).head(bdg["extreme_rise_per_km2"])
        add_rows(pool, "extreme_rise_per_km2",
                 lambda r: f"extreme_rise_per_km2={r.get(col, np.nan):.4g}")

    # ── Extreme: top Q95/Q50 ratio ──────────────────────────────────────────
    col = "q95_q50_ratio"
    if col in df.columns:
        pool = df.sort_values(col, ascending=False).head(bdg["extreme_q95_ratio"])
        add_rows(pool, "extreme_q95_ratio",
                 lambda r: f"extreme_q95_ratio={r.get(col, np.nan):.2f}")

    # ── MANUAL_REVIEW_CONTEXT ───────────────────────────────────────────────
    mr = df[df.get("candidate_class", pd.Series(dtype=str)) == "MANUAL_REVIEW_CONTEXT"]
    if len(mr) > 0:
        sc = "pilot_score" if "pilot_score" in mr.columns else (
             "RBI" if "RBI" in mr.columns else None)
        pool = mr.sort_values(sc, ascending=False) if sc else mr
        pool = pool[~pool["STAID"].isin(seen)].head(bdg["manual_review_context"])
        add_rows(pool, "manual_review_context",
                 lambda r: "MANUAL_REVIEW_CONTEXT")

    # ── Reference: by candidate class ──────────────────────────────────────
    class_map = {
        "reference_flashy_core":    "FLASHY_CORE",
        "reference_flashy_moderate": "FLASHY_MODERATE",
        "reference_flashy_possible": "FLASHY_POSSIBLE",
        "reference_low_flashiness":  "LOW_FLASHINESS_CONTROL",
        "reference_exclude_hard_qc": "EXCLUDE_HARD_QC",
    }
    for group, cls in class_map.items():
        src = hqp if cls != "EXCLUDE_HARD_QC" else df[~df["is_hard_qc_pass"]]
        pool = src[
            src.get("candidate_class", pd.Series(dtype=str)) == cls
        ]
        pool = pool[~pool["STAID"].isin(seen)]
        if len(pool) == 0:
            continue
        sample = _stratified_sample(pool, bdg[group], rng)
        add_rows(sample, group,
                 lambda r, g=group: f"{g} RBI={r.get('RBI', np.nan):.4f}")

    # ── Reference: low-RBI basins (RBI < P25) ──────────────────────────────
    if "RBI" in hqp.columns:
        p25 = hqp["RBI"].quantile(0.25)
        pool = hqp[hqp["RBI"] < p25]
        pool = pool[~pool["STAID"].isin(seen)]
        sample = _stratified_sample(pool, bdg["reference_rbi_low"], rng)
        add_rows(sample, "reference_rbi_low",
                 lambda r: f"low-RBI ref RBI={r.get('RBI', np.nan):.4f}")

    # ── Reference: zero-flow / ephemeral ───────────────────────────────────
    if "zero_flow_fraction" in hqp.columns:
        pool = hqp[hqp["zero_flow_fraction"] >= QC_ZERO_THRESH]
        pool = pool[~pool["STAID"].isin(seen)]
        sample = _stratified_sample(pool, bdg["reference_zero_flow"], rng)
        add_rows(sample, "reference_zero_flow",
                 lambda r: f"zero_flow_frac={r.get('zero_flow_fraction', np.nan):.3f}")

    # ── Build DataFrame ─────────────────────────────────────────────────────
    records = []
    for staid, info in seen.items():
        row = info["row"]
        rec = {
            "STAID":           staid,
            "review_group":    info["review_group"],
            "reason_selected": info["reason_selected"],
        }
        for col in row.index:
            if col not in rec:
                rec[col] = row[col]
        records.append(rec)

    out = pd.DataFrame(records).head(max_basins).reset_index(drop=True)

    # Report diversity
    n_groups  = out["review_group"].nunique()
    n_classes = out["candidate_class"].nunique() if "candidate_class" in out.columns else 0
    diverse   = n_groups >= 3 and n_classes >= 2
    print(f"  Diversity check: {n_groups} review groups, {n_classes} candidate classes "
          f"-> {'DIVERSE' if diverse else 'COLLAPSED (consider larger input table)'}")
    return out


# ---------------------------------------------------------------------------
# QC labels
# ---------------------------------------------------------------------------
def assign_qc_labels(row: pd.Series) -> list:
    labels = []
    jump = row.get("max_abs_hourly_jump_over_Q50", np.nan)
    zero = row.get("zero_flow_fraction", np.nan)
    rbi  = row.get("RBI", np.nan)
    ctx  = row.get("context_flags_list", [])
    if not isinstance(ctx, list):
        ctx = _parse_flag_list(ctx)

    if pd.notna(jump) and jump >= QC_JUMP_THRESH:
        labels.append("extreme_single_hour_jump")
    if pd.notna(zero) and zero >= QC_ZERO_THRESH:
        labels.append("high_zero_flow_fraction")
    if QC_SPIKE_FLAG in ctx:
        labels.append("possible_measurement_artifact")
    if pd.notna(jump) and jump >= 5.0 and pd.notna(rbi) and rbi < 0.02:
        labels.append("possible_flatline_or_step")
    if not labels:
        labels.append("reference_typical")
    return labels


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------
def find_event_peaks(q: pd.Series, threshold: float, min_sep_hours: int = 72) -> list:
    vals  = q.values
    above = [(i, float(vals[i])) for i in range(len(vals))
             if not np.isnan(vals[i]) and vals[i] >= threshold]
    if not above:
        return []

    clusters, cur = [], [above[0]]
    for k in range(1, len(above)):
        if above[k][0] - above[k - 1][0] <= 1:
            cur.append(above[k])
        else:
            clusters.append(cur); cur = [above[k]]
    clusters.append(cur)

    raw = [max(c, key=lambda x: x[1])[0] for c in clusters]
    kept = [raw[0]]
    for pk in raw[1:]:
        if pk - kept[-1] >= min_sep_hours:
            kept.append(pk)
        elif vals[pk] > vals[kept[-1]]:
            kept[-1] = pk
    return kept


def find_rise_peaks(q: pd.Series, n: int = 3) -> list:
    diff = q.diff()
    diff = diff[diff > 0].dropna()
    if diff.empty:
        return []
    top_times = diff.nlargest(n).index
    return [q.index.get_loc(t) for t in top_times if t in q.index]


def _window_slice(q: pd.Series, center_idx: int, half_hours: int):
    s = max(0, center_idx - half_hours)
    e = min(len(q) - 1, center_idx + half_hours)
    return q.iloc[s:e + 1], center_idx - s   # (slice, centre_offset_in_slice)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _plot_segments(ax, q: pd.Series, **kwargs):
    """Draw a discharge series with gap-aware segment breaks. Returns has_gap.
    Only the first plotted segment receives the legend label; subsequent
    segments use '_nolegend_' so the legend shows each series exactly once."""
    seg_t, seg_v, has_gap = [], [], False
    label_consumed = False
    for t, v in zip(q.index, q.values):
        if np.isnan(v):
            if seg_t:
                kw = dict(kwargs)
                if label_consumed:
                    kw["label"] = "_nolegend_"
                ax.plot(seg_t, seg_v, **kw)
                seg_t, seg_v = [], []
                label_consumed = True
            has_gap = True
        else:
            seg_t.append(t); seg_v.append(v)
    if seg_t:
        kw = dict(kwargs)
        if label_consumed:
            kw["label"] = "_nolegend_"
        ax.plot(seg_t, seg_v, **kw)
    return has_gap


def _add_q_hlines(ax, row: pd.Series):
    for key, color, ls in [("Q50",  PLOT_COLORS["q50"], "--"),
                            ("Q95",  PLOT_COLORS["q95"], "-."),
                            ("Q99",  PLOT_COLORS["q99"], ":")]:
        v = row.get(key, np.nan)
        if pd.notna(v) and float(v) > 0:
            ax.axhline(float(v), color=color, ls=ls, lw=0.75,
                       label=f"{key}={float(v):.4g}")


def _annot_box(ax, x, y, text, color, xoffset=8, yoffset=4):
    ax.annotate(text, xy=(x, y),
                xytext=(xoffset, yoffset), textcoords="offset points",
                fontsize=5.5, color=color, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, alpha=0.85, lw=0.6))


def _metric_block(row: pd.Series, qc_labels: list) -> str:
    def fv(k, fmt=".4g"):
        v = row.get(k, np.nan)
        return "N/A" if pd.isna(v) else format(float(v), fmt)
    return "\n".join([
        f"STAID={row.get('STAID','?')}  {row.get('STATE','?')}  HUC02={row.get('HUC02','?')}",
        f"Area={fv('DRAIN_SQKM','.1f')} km2  BFI={fv('BFI_AVE','.1f')}  "
        f"Class={_class_label(row.get('candidate_class','?'))}",
        f"RBI={fv('RBI','.4f')}  Compl={fv('hourly_completeness_pct','.1f')}%  "
        f"ZeroFrac={fv('zero_flow_fraction','.3f')}",
        f"Q50={fv('Q50')}  Q95={fv('Q95')}  Q99={fv('Q99')}  "
        f"Q95/Q50={fv('q95_q50_ratio','.2f')}",
        f"MaxRise={fv('max_hourly_rise')} m3/s/hr  "
        f"MaxRise/km2={fv('max_hourly_rise_per_km2','.4g')}  "
        f"MaxFall={fv('max_hourly_fall')} m3/s/hr",
        f"QC: {', '.join(qc_labels)}",
    ])


def _fmt_time(t):
    try:
        return pd.Timestamp(t).strftime("%m-%d %H:%M")
    except Exception:
        return str(t)


# ---------------------------------------------------------------------------
# Class label helpers (review-oriented wording)
# ---------------------------------------------------------------------------
_CLASS_LABELS = {
    "FLASHY_CORE":            "flashy core [RBI>=0.10]",
    "FLASHY_MODERATE":        "flashy moderate [RBI 0.05-0.10]",
    "FLASHY_POSSIBLE":        "flashy possible [RBI<0.05]",
    "LOW_FLASHINESS_CONTROL": "low-flashiness control",
    "MANUAL_REVIEW_CONTEXT":  "manual-review context (flagged)",
    "EXCLUDE_HARD_QC":        "QC-flagged (verify exclusion)",
}

def _class_label(cls: str) -> str:
    return _CLASS_LABELS.get(str(cls), str(cls))


# ---------------------------------------------------------------------------
# Event metrics (window-specific)
# ---------------------------------------------------------------------------
def _compute_event_metrics(q_win: pd.Series, centre_off: int, row: pd.Series) -> dict:
    area = row.get("DRAIN_SQKM", np.nan)
    q95  = row.get("Q95", np.nan)

    peak_q, peak_time = np.nan, None
    if 0 <= centre_off < len(q_win):
        pq = q_win.iloc[centre_off]
        if pd.notna(pq):
            peak_q   = float(pq)
            peak_time = q_win.index[centre_off]

    dq           = q_win.diff()
    valid_rise   = dq[dq > 0].dropna()
    valid_fall   = dq[dq < 0].dropna()
    max_rise      = float(valid_rise.max()) if not valid_rise.empty else np.nan
    max_rise_time = valid_rise.idxmax()     if not valid_rise.empty else None
    max_fall      = float(valid_fall.min()) if not valid_fall.empty else np.nan
    max_fall_time = valid_fall.idxmin()     if not valid_fall.empty else None

    rise_to_peak_h = np.nan
    if max_rise_time is not None and peak_time is not None:
        try:
            rise_to_peak_h = (peak_time - max_rise_time).total_seconds() / 3600
        except Exception:
            pass

    hours_above_q95 = 0
    if pd.notna(q95) and float(q95) > 0:
        hours_above_q95 = int((q_win.dropna() >= float(q95)).sum())

    def _per_km2(v):
        if pd.notna(v) and pd.notna(area) and float(area) > 0:
            return float(v) / float(area)
        return np.nan

    return {
        "peak_q":           peak_q,
        "peak_time":        peak_time,
        "max_rise":         max_rise,
        "max_rise_time":    max_rise_time,
        "max_rise_per_km2": _per_km2(max_rise),
        "max_fall":         max_fall,
        "max_fall_time":    max_fall_time,
        "max_fall_per_km2": _per_km2(max_fall),
        "rise_to_peak_h":   rise_to_peak_h,
        "hours_above_q95":  hours_above_q95,
        "missing_hours":    int(q_win.isna().sum()),
        "dq":               dq,
    }


def _format_event_metrics_text(em: dict, row: pd.Series, qc_labels: list) -> str:
    def fv(v, fmt=".4g"):
        return "N/A" if (v is None or pd.isna(v)) else format(float(v), fmt)

    peak_q = em["peak_q"]
    q50 = row.get("Q50", np.nan); q95 = row.get("Q95", np.nan); q99 = row.get("Q99", np.nan)
    pk50 = peak_q / float(q50) if pd.notna(peak_q) and pd.notna(q50) and float(q50) > 0 else np.nan
    pk95 = peak_q / float(q95) if pd.notna(peak_q) and pd.notna(q95) and float(q95) > 0 else np.nan

    lines = [
        f"STAID: {row.get('STAID','?')}",
        f"Class: {_class_label(row.get('candidate_class','?'))}",
        f"  raw: {row.get('candidate_class','?')}",
        f"Group: {row.get('review_group','?')}",
        f"Area:  {fv(row.get('DRAIN_SQKM'), '.1f')} km2  BFI: {fv(row.get('BFI_AVE'), '.1f')}",
        f"RBI:   {fv(row.get('RBI'), '.4f')}",
        f"Compl: {fv(row.get('hourly_completeness_pct'), '.1f')}%  "
        f"Zero: {fv(row.get('zero_flow_fraction'), '.3f')}",
        f"--- event ---",
        f"Peak Q:    {fv(peak_q)} m3/s",
        f"Q50/Q95/Q99: {fv(q50)} / {fv(q95)} / {fv(q99)}",
        f"Peak/Q50:  {fv(pk50, '.2f')}  Peak/Q95: {fv(pk95, '.2f')}",
        f"MaxRise:   {fv(em['max_rise'])} m3/s/hr",
        f"MaxFall:   {fv(em['max_fall'])} m3/s/hr",
        f"Rise/km2:  {fv(em['max_rise_per_km2'], '.4g')}",
        f"Fall/km2:  {fv(em['max_fall_per_km2'], '.4g')}",
        f"Rise->Peak:{fv(em['rise_to_peak_h'], '.1f')} h",
        f"Hrs>=Q95:  {em['hours_above_q95']}",
        f"Miss hrs:  {em['missing_hours']}",
        f"--- QC ---",
    ] + qc_labels
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_full_year(staid: str, q: pd.Series, row: pd.Series,
                   qc_labels: list, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.5),
                              gridspec_kw={"height_ratios": [3, 1.2]},
                              constrained_layout=True)
    ax, ax_log = axes

    # Linear panel
    _plot_segments(ax, q, color=PLOT_COLORS["series"], lw=0.55, alpha=0.85, label="Q (m3/s)")
    _add_q_hlines(ax, row)
    ax.set_ylabel("Q (m3/s)", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=6.5, loc="upper right", ncol=5)
    ax.grid(True, ls=":", lw=0.35, alpha=0.5)
    ax.text(0.01, 0.97, _metric_block(row, qc_labels),
            transform=ax.transAxes, fontsize=5.5, va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9ca3af", alpha=0.85))

    # Log context strip — positive values only
    q_pos = q.copy(); q_pos[q_pos <= 0] = np.nan
    _plot_segments(ax_log, q_pos, color=PLOT_COLORS["series"], lw=0.45, alpha=0.7,
                   label="Q (m3/s)")
    _add_q_hlines(ax_log, row)
    try:
        ax_log.set_yscale("log")
    except Exception:
        pass
    ax_log.legend(fontsize=6.0, loc="upper right", ncol=4)
    ax_log.set_ylabel("Q log (m3/s)", fontsize=7)
    ax_log.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_log.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_log.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6.5)
    ax_log.grid(True, ls=":", lw=0.35, alpha=0.5)
    ax_log.set_xlabel("Date (UTC)", fontsize=8)

    ax.set_title(
        f"WY2024 Full Hydrograph [linear+log] — {staid}  "
        f"Class={row.get('candidate_class','?')}  RBI={row.get('RBI', np.nan):.4f}",
        fontsize=8,
    )
    out_path = out_dir / f"{staid}_full_year.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_event_panel(staid: str, q_win: pd.Series, centre_off: int,
                     event_type: str, event_num: int, row: pd.Series,
                     qc_labels: list, out_dir: Path,
                     window_label: str, half_hours: int) -> tuple:
    """Two-panel event plot: upper Q (linear) + lower dQ. Returns (Path, has_gap)."""
    em = _compute_event_metrics(q_win, centre_off, row)
    dq = em["dq"]
    has_gap = em["missing_hours"] > 0
    area = row.get("DRAIN_SQKM", np.nan)
    q50  = row.get("Q50", np.nan)

    fig, (ax_q, ax_dq) = plt.subplots(
        2, 1, figsize=(13, 6.0),
        gridspec_kw={"height_ratios": [3, 1.6]},
        sharex=True, constrained_layout=True,
    )

    # ── Upper panel: Q (linear) ─────────────────────────────────────────────
    _plot_segments(ax_q, q_win, color=PLOT_COLORS["series"], lw=1.0, alpha=0.9,
                   label="Q (m3/s)")
    _add_q_hlines(ax_q, row)

    # Event centre vline (no inline text box — detail is in the metrics text box)
    if 0 <= centre_off < len(q_win):
        ct = q_win.index[centre_off]
        cv = q_win.iloc[centre_off]
        if pd.notna(cv):
            ccolor = PLOT_COLORS["peak"] if "peak" in event_type else PLOT_COLORS["rise"]
            ax_q.axvline(ct, color=ccolor, lw=1.6, ls="-", zorder=6,
                         label=f"Centre {cv:.4g} m3/s")

    # Max rise vline
    if em["max_rise_time"] is not None and pd.notna(em["max_rise"]):
        mr_t, mr_v = em["max_rise_time"], em["max_rise"]
        ax_q.axvline(mr_t, color=PLOT_COLORS["rise"], lw=1.2, ls="--", zorder=5,
                     label=f"MaxRise {mr_v:.4g} m3/s/hr")

    # Max fall vline
    if em["max_fall_time"] is not None and pd.notna(em["max_fall"]):
        mf_t, mf_v = em["max_fall_time"], em["max_fall"]
        ax_q.axvline(mf_t, color=PLOT_COLORS["fall"], lw=1.2, ls=":", zorder=5,
                     label=f"MaxFall {mf_v:.4g} m3/s/hr")

    # Adaptive y-limits: base on local event window, reference lines don't expand axis
    q_valid = q_win.dropna()
    if len(q_valid) > 0:
        ax_q.set_ylim(bottom=0, top=float(q_valid.max()) * 1.15)

    # Gap banner
    if has_gap:
        ax_q.text(0.5, 0.97, "[MISSING DATA IN WINDOW]",
                  transform=ax_q.transAxes, ha="center", va="top",
                  fontsize=7, color="#ef4444",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#ef4444"))

    # Metrics text box (upper right, monospace)
    ax_q.text(0.99, 0.99, _format_event_metrics_text(em, row, qc_labels),
              transform=ax_q.transAxes, fontsize=5.0, va="top", ha="right",
              family="monospace",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9ca3af", alpha=0.88))

    ax_q.set_ylabel("Q (m3/s)  [LINEAR]", fontsize=7)
    ax_q.legend(fontsize=6.0, loc="upper left", ncol=3)
    ax_q.grid(True, ls=":", lw=0.35, alpha=0.5)
    ax_q.set_title(
        f"[{window_label.upper()} +/-{half_hours}h] {event_type.upper()} event {event_num}"
        f" - {staid}  RBI={row.get('RBI', np.nan):.4f}"
        f"  {_class_label(row.get('candidate_class','?'))}",
        fontsize=7.5,
    )

    # ── Lower panel: dQ ─────────────────────────────────────────────────────
    t_arr   = dq.index
    dq_vals = dq.values
    ax_dq.fill_between(t_arr, dq_vals, 0,
                        where=(dq_vals > 0),
                        color=PLOT_COLORS["rise"], alpha=0.5, label="dQ>0 (rise)")
    ax_dq.fill_between(t_arr, dq_vals, 0,
                        where=(dq_vals < 0),
                        color=PLOT_COLORS["fall"], alpha=0.5, label="dQ<0 (fall)")
    ax_dq.axhline(0, color="#374151", lw=0.6, ls="-")

    if em["max_rise_time"] is not None and pd.notna(em["max_rise"]):
        ax_dq.scatter([em["max_rise_time"]], [em["max_rise"]],
                      color=PLOT_COLORS["rise"], s=40, zorder=7, marker="^",
                      label=f"MaxRise {em['max_rise']:.4g}")
    if em["max_fall_time"] is not None and pd.notna(em["max_fall"]):
        ax_dq.scatter([em["max_fall_time"]], [em["max_fall"]],
                      color=PLOT_COLORS["fall"], s=40, zorder=7, marker="v",
                      label=f"MaxFall {em['max_fall']:.4g}")

    # Adaptive dQ y-limits: symmetric around zero
    dq_clean = dq.dropna()
    if not dq_clean.empty:
        dq_abs_max = max(float(dq_clean.abs().max()), 1e-10)
        ax_dq.set_ylim(-dq_abs_max * 1.25, dq_abs_max * 1.25)

    ax_dq.set_ylabel("dQ (m3/s/hr)", fontsize=7)
    ax_dq.set_xlabel(f"Date (UTC)  [+/-{half_hours} h window]", fontsize=7)
    ax_dq.legend(fontsize=6.0, loc="upper left", ncol=4)
    ax_dq.grid(True, ls=":", lw=0.35, alpha=0.5)

    # x-axis ticks: finer resolution for narrower windows
    if half_hours <= 12:
        ax_dq.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        ax_dq.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        plt.setp(ax_dq.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=6.5)
    elif half_hours <= 36:
        ax_dq.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        ax_dq.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax_dq.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=6.5)
    else:
        ax_dq.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax_dq.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        plt.setp(ax_dq.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=6.5)

    out_path = out_dir / f"{staid}_{event_type}_e{event_num:02d}_{window_label}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path, has_gap


def plot_zero_flow_panel(staid: str, q: pd.Series, row: pd.Series,
                         qc_labels: list, out_dir: Path):
    valid = q.dropna()
    if len(valid) == 0:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
    _plot_segments(ax1, q, color=PLOT_COLORS["series"], lw=0.5, alpha=0.8)
    ax1.axhline(0, color=PLOT_COLORS["zero"], lw=0.8, ls="--", label="Q=0")
    ax1.set_title(f"Zero/low-flow context — {staid}  zero_frac="
                  f"{row.get('zero_flow_fraction', np.nan):.3f}", fontsize=8)
    ax1.set_xlabel("Date (UTC)", fontsize=7); ax1.set_ylabel("Q (m3/s)", fontsize=7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6.5)
    ax1.legend(fontsize=6.5); ax1.grid(True, ls=":", lw=0.35, alpha=0.5)
    pos = valid[valid > 0]
    if len(pos) > 0:
        ax2.hist(np.log10(pos.values), bins=40, color=PLOT_COLORS["series"],
                 alpha=0.7, edgecolor="none")
        ax2.set_xlabel("log10(Q) (m3/s)", fontsize=7)
        ax2.set_ylabel("Hours", fontsize=7)
        ax2.set_title("log10(Q) distribution", fontsize=8)
        ax2.grid(True, ls=":", lw=0.35, alpha=0.5)
    fig.tight_layout()
    out_path = out_dir / f"{staid}_zero_flow_context.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-basin processing
# ---------------------------------------------------------------------------
def process_basin(staid: str, row: pd.Series, hourly_dir: Path,
                  plots_dir: Path) -> list:
    records = []
    qc_labels = assign_qc_labels(row)

    ts = load_hourly(staid, hourly_dir)
    if ts is None:
        warnings.warn(f"No parquet for {staid} – skipping plots.")
        return records
    if "discharge_m3s" not in ts.columns:
        warnings.warn(f"{staid}: no discharge_m3s column – skipping.")
        return records

    q = ts["discharge_m3s"]

    # Full-year overview (linear + log strip)
    if HAS_MPL:
        fy = plot_full_year(staid, q, row, qc_labels, plots_dir)
        records.append(_manifest_rec(staid, row, qc_labels, fy,
                                     "full_year", None, None, False))

    # Top 3 events: use both Q99 peak detection and top-rise detection
    # Deduplicate centres to avoid redundant windows
    q99 = row.get("Q99", np.nan)
    threshold = float(q99) if pd.notna(q99) and float(q99) > 0 else (
        float(q.quantile(0.99)) if len(q.dropna()) > 10 else float(q.max())
    )
    peak_idxs = find_event_peaks(q, threshold, min_sep_hours=72)
    peak_idxs = sorted(peak_idxs, key=lambda i: q.iloc[i], reverse=True)[:3]
    peak_idxs = sorted(peak_idxs)

    rise_idxs = find_rise_peaks(q, n=3)

    # Collect (centre_idx, event_type) pairs, deduplicated within 12 h
    events = [(idx, "peak") for idx in peak_idxs]
    top_rise_added = False
    for ridx in rise_idxs:
        if all(abs(ridx - cidx) > 12 for cidx, _ in events):
            events.append((ridx, "rise"))
            top_rise_added = True
    # Force-include top rise if no rise event was added and it's >= 3 h from any peak
    if not top_rise_added and rise_idxs:
        top_ridx = rise_idxs[0]
        if all(abs(top_ridx - cidx) > 3 for cidx, _ in events):
            events.append((top_ridx, "rise"))
    events = events[:6]   # hard cap

    for ev_num, (cidx, etype) in enumerate(events, start=1):
        # Close window +-12 h (key view for manual peak/rise/fall inspection)
        q_cls, off_cls = _window_slice(q, cidx, CLOSE_HALF_HOURS)
        if HAS_MPL:
            cp, has_gap_c = plot_event_panel(
                staid, q_cls, off_cls, etype, ev_num, row, qc_labels, plots_dir,
                "close", CLOSE_HALF_HOURS)
            records.append(_manifest_rec(staid, row, qc_labels, cp,
                                         f"{etype}_close", q.index[cidx], ev_num, has_gap_c))
        # Tight window +-36 h
        q_tgt, off_tgt = _window_slice(q, cidx, TIGHT_HALF_HOURS)
        if HAS_MPL:
            tp, has_gap_t = plot_event_panel(
                staid, q_tgt, off_tgt, etype, ev_num, row, qc_labels, plots_dir,
                "tight", TIGHT_HALF_HOURS)
            records.append(_manifest_rec(staid, row, qc_labels, tp,
                                         f"{etype}_tight", q.index[cidx], ev_num, has_gap_t))
        # Medium window +-72 h
        q_med, off_med = _window_slice(q, cidx, MEDIUM_HALF_HOURS)
        if HAS_MPL:
            mp, has_gap_m = plot_event_panel(
                staid, q_med, off_med, etype, ev_num, row, qc_labels, plots_dir,
                "medium", MEDIUM_HALF_HOURS)
            records.append(_manifest_rec(staid, row, qc_labels, mp,
                                         f"{etype}_medium", q.index[cidx], ev_num, has_gap_m))

    # Zero-flow context panel
    zf = row.get("zero_flow_fraction", 0.0)
    if pd.notna(zf) and float(zf) >= QC_ZERO_THRESH and HAS_MPL:
        zp = plot_zero_flow_panel(staid, q, row, qc_labels, plots_dir)
        if zp is not None:
            records.append(_manifest_rec(staid, row, qc_labels, zp,
                                         "zero_flow_context", None, None, False))
    return records


def _manifest_rec(staid, row, qc_labels, plot_path,
                  event_type, event_time, event_num, has_gap):
    def fv(k):
        v = row.get(k, np.nan)
        return None if pd.isna(v) else float(v)
    return {
        "STAID":           staid,
        "review_group":    row.get("review_group", ""),
        "candidate_class": row.get("candidate_class", ""),
        "reason_selected": row.get("reason_selected", ""),
        "plot_path":       str(plot_path),
        "event_type":      event_type,
        "event_time":      str(event_time) if event_time is not None else "",
        "event_num":       "" if event_num is None else event_num,
        "has_gap_in_window": has_gap,
        "qc_labels":       "|".join(qc_labels),
        "STATE":           row.get("STATE", ""),
        "HUC02":           row.get("HUC02", ""),
        "DRAIN_SQKM":      fv("DRAIN_SQKM"),
        "BFI_AVE":         fv("BFI_AVE"),
        "RBI":             fv("RBI"),
        "hourly_completeness_pct":      fv("hourly_completeness_pct"),
        "Q50":             fv("Q50"),
        "Q95":             fv("Q95"),
        "Q99":             fv("Q99"),
        "Q_max":           fv("Q_max"),
        "q95_q50_ratio":   fv("q95_q50_ratio"),
        "zero_flow_fraction":            fv("zero_flow_fraction"),
        "max_hourly_rise":               fv("max_hourly_rise"),
        "max_hourly_fall":               fv("max_hourly_fall"),
        "max_hourly_rise_per_km2":       fv("max_hourly_rise_per_km2"),
        "max_abs_hourly_jump_over_Q50":  fv("max_abs_hourly_jump_over_Q50"),
    }


# ---------------------------------------------------------------------------
# Human review template (one row per basin)
# ---------------------------------------------------------------------------
def write_human_review_template(review_df: pd.DataFrame,
                                 all_records: list,
                                 out_dir: Path) -> Path:
    # Gather per-basin plot paths; prefer close (+-12h) over tight (+-36h) for template
    basin_plots = {}
    for rec in all_records:
        s = rec["STAID"]
        et = rec["event_type"]
        p  = Path(rec["plot_path"]).name
        if s not in basin_plots:
            basin_plots[s] = {"full_year": "", "close_events": [],
                              "tight_events": [], "zero_flow_context": ""}
        if et == "full_year":
            basin_plots[s]["full_year"] = p
        elif et == "zero_flow_context":
            basin_plots[s]["zero_flow_context"] = p
        elif "close" in et:
            basin_plots[s]["close_events"].append(p)
        elif "tight" in et:
            basin_plots[s]["tight_events"].append(p)

    def fv(row, k):
        v = row.get(k, np.nan)
        return "" if pd.isna(v) else v

    rows = []
    for _, row in review_df.iterrows():
        s = row["STAID"]
        bp = basin_plots.get(s, {"full_year": "", "close_events": [],
                                   "tight_events": [], "zero_flow_context": ""})
        evs = (bp.get("close_events") or bp.get("tight_events") or [])[:3]
        while len(evs) < 3:
            evs.append("")
        qc_labels = assign_qc_labels(row)
        rows.append({
            "STAID":               s,
            "candidate_class":     fv(row, "candidate_class"),
            "review_group":        fv(row, "review_group"),
            "RBI":                 fv(row, "RBI"),
            "BFI_AVE":             fv(row, "BFI_AVE"),
            "DRAIN_SQKM":          fv(row, "DRAIN_SQKM"),
            "HUC02":               fv(row, "HUC02"),
            "STATE":               fv(row, "STATE"),
            "qc_labels":           "|".join(qc_labels),
            "context_flags":       fv(row, "context_flags"),
            "zero_flow_fraction":  fv(row, "zero_flow_fraction"),
            "hourly_completeness_pct": fv(row, "hourly_completeness_pct"),
            "q50":                 fv(row, "Q50"),
            "q95":                 fv(row, "Q95"),
            "q99":                 fv(row, "Q99"),
            "q95_q50_ratio":       fv(row, "q95_q50_ratio"),
            "max_hourly_rise":     fv(row, "max_hourly_rise"),
            "max_hourly_fall":     fv(row, "max_hourly_fall"),
            "max_hourly_rise_per_km2":      fv(row, "max_hourly_rise_per_km2"),
            "max_abs_hourly_jump_over_Q50": fv(row, "max_abs_hourly_jump_over_Q50"),
            "full_year_plot":           bp["full_year"],
            "event_plot_1":             evs[0],
            "event_plot_2":             evs[1],
            "event_plot_3":             evs[2],
            "zero_flow_context_plot":   bp["zero_flow_context"],
            # Blank reviewer columns — fill in during manual review
            "human_decision":      "",
            "hydrograph_behavior": "",
            "artifact_type":       "",
            "confidence":          "",
            "reviewer_notes":      "",
        })

    tmpl = pd.DataFrame(rows)
    p = out_dir / "tables" / "human_review_template.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmpl.to_csv(p, index=False)

    # Allowed-values README
    readme_lines = [
        "# human_review_template.csv — Allowed Values",
        "",
        "Fill in the five blank columns for each basin after reviewing the plots.",
        "",
    ]
    for field, allowed in HUMAN_REVIEW_ALLOWED.items():
        readme_lines.append(f"## {field}")
        readme_lines.append(f"  {allowed}")
        readme_lines.append("")
    readme_p = out_dir / "tables" / "human_review_template_README.md"
    readme_p.write_text("\n".join(readme_lines), encoding="utf-8")

    return p


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------
def write_manifest_csv(records: list, out_dir: Path) -> Path:
    df = pd.DataFrame(records)
    p  = out_dir / "tables" / "review_card_manifest.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def _rbi_bin(rbi):
    if pd.isna(rbi): return "unknown"
    r = float(rbi)
    if r < 0.01:  return "RBI<0.01"
    if r < 0.03:  return "RBI 0.01-0.03"
    if r < 0.10:  return "RBI 0.03-0.10"
    if r < 0.30:  return "RBI 0.10-0.30"
    return "RBI>=0.30"


def write_summary(review_df: pd.DataFrame, records: list, out_dir: Path) -> Path:
    summ_dir = out_dir / "summaries"
    summ_dir.mkdir(parents=True, exist_ok=True)

    def vc(col):
        if col in review_df.columns:
            return review_df[col].fillna("unknown").value_counts().to_dict()
        return {}

    review_df = review_df.copy()
    review_df["rbi_bin"] = review_df.get("RBI", pd.Series(dtype=float)).apply(_rbi_bin)

    n_groups  = review_df["review_group"].nunique()
    n_classes = review_df["candidate_class"].nunique() if "candidate_class" in review_df.columns else 0
    diverse   = n_groups >= 3 and n_classes >= 2

    summary = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "total_basins":       len(review_df),
        "total_plots":        len(records),
        "diverse":            diverse,
        "n_review_groups":    n_groups,
        "n_candidate_classes": n_classes,
        "by_review_group":    vc("review_group"),
        "by_candidate_class": vc("candidate_class"),
        "by_area_bin":        vc("area_bin"),
        "by_bfi_bin":         vc("BFI_bin"),
        "by_huc02":           vc("HUC02"),
        "by_rbi_bin":         review_df["rbi_bin"].value_counts().to_dict(),
        "by_plot_type":       pd.Series([r["event_type"] for r in records]).value_counts().to_dict(),
    }
    with open(summ_dir / "review_set_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = [
        "# Flash-NH Hydrograph Review Cards — Summary",
        f"\nGenerated: {summary['generated_at']}",
        f"\n**Diverse selection:** {'YES' if diverse else 'NO — input table may be too small'}",
        f"\n## Totals\n",
        f"- Basins: **{summary['total_basins']}**",
        f"- Plots: **{summary['total_plots']}**",
    ]
    for section, label in [
        ("by_review_group",    "By review group"),
        ("by_candidate_class", "By candidate class"),
        ("by_area_bin",        "By area bin"),
        ("by_bfi_bin",         "By BFI bin"),
        ("by_huc02",           "By HUC02 (top 10)"),
        ("by_rbi_bin",         "By RBI bin"),
        ("by_plot_type",       "By plot type"),
    ]:
        md.append(f"\n## {label}\n")
        items = sorted(summary[section].items(), key=lambda x: -x[1])
        if section == "by_huc02":
            items = items[:10]
        for k, n in items:
            md.append(f"- {k}: {n}")

    (summ_dir / "review_set_summary.md").write_text("\n".join(md), encoding="utf-8")
    return summ_dir / "review_set_summary.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)

    print("=" * 62)
    print("Flash-NH Hydrograph Review Card Builder")
    print("=" * 62)
    print(f"  Metrics dir : {args.metrics_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Max basins  : {args.max_basins}")
    print(f"  Seed        : {args.seed}")
    print(f"  matplotlib  : {'available' if HAS_MPL else 'NOT AVAILABLE'}")

    md, hd = _resolve_dirs(args)

    # Output dirs
    plots_dir = args.output_dir / "plots"
    for d in (plots_dir, args.output_dir / "tables", args.output_dir / "summaries"):
        d.mkdir(parents=True, exist_ok=True)

    # Load metrics
    print("\nLoading metrics table...")
    try:
        df = load_metrics(md)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}"); return
    print(f"  {len(df)} basins loaded.")

    # Select review set
    print(f"\nSelecting review set (max {args.max_basins} basins, seed {args.seed})...")
    review_df = select_review_set(df, args.max_basins, args.seed)
    print(f"  {len(review_df)} basins selected.")
    for g, n in review_df["review_group"].value_counts().items():
        print(f"    {g}: {n}")

    # Per-class counts
    if "candidate_class" in review_df.columns:
        print("  Candidate class counts:")
        for c, n in review_df["candidate_class"].value_counts().items():
            print(f"    {c}: {n}")

    # Area / BFI / HUC02
    for col in ("area_bin", "BFI_bin", "HUC02"):
        if col in review_df.columns:
            vals = review_df[col].value_counts()
            top = ", ".join(f"{k}:{v}" for k, v in list(vals.items())[:5])
            print(f"  {col}: {top}")

    # Generate plots
    all_records = []
    print(f"\nGenerating plots for {len(review_df)} basins...")
    for i, (_, rev_row) in enumerate(review_df.iterrows()):
        staid = rev_row["STAID"]
        print(f"  [{i+1:3d}/{len(review_df)}] {staid} "
              f"({rev_row.get('candidate_class','?')})", end="", flush=True)
        recs = process_basin(staid, rev_row, hd, plots_dir)
        all_records.extend(recs)
        print(f"  -> {len(recs)} plot(s)")

    # Write outputs
    print("\nWriting outputs...")
    manifest_p  = write_manifest_csv(all_records, args.output_dir)
    template_p  = write_human_review_template(review_df, all_records, args.output_dir)
    summary_p   = write_summary(review_df, all_records, args.output_dir)

    print("\n" + "=" * 62)
    print("Done.")
    print(f"  Output dir             : {args.output_dir}")
    print(f"  Total basins           : {len(review_df)}")
    print(f"  Full-year plots        : "
          f"{len([r for r in all_records if r['event_type']=='full_year'])}")
    print(f"  Event plots (total)    : "
          f"{len([r for r in all_records if r['event_type']!='full_year' and r['event_type']!='zero_flow_context'])}")
    print(f"  Zero-flow panels       : "
          f"{len([r for r in all_records if r['event_type']=='zero_flow_context'])}")
    print(f"  Manifest CSV           : {manifest_p}")
    print(f"  Human review template  : {template_p}")
    print(f"  Summary                : {summary_p}")
    print("=" * 62)


if __name__ == "__main__":
    main()
