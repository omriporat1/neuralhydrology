"""Matplotlib rendering for the Phase-B Sweep-v1 durable review layer.

Renders a Checkpoint Decision Board and twelve standalone figures from
authoritative Flash-NH tabular evidence (via
:mod:`src.baseline.sweep_v1_review_analysis`).  No W&B dependency, no
network access, no automated CONTINUE/UNCERTAIN/EXPAND decision -- every
figure is evidence for a human reviewer, never a verdict.

The public entry point is :func:`render_checkpoint_packet`, which writes a
directory of PNGs plus derived CSV/JSON and a README index.  Callers supply
already-sliced-to-checkpoint DataFrames; this module performs no checkpoint
selection of its own beyond what :mod:`sweep_v1_review_analysis` computes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baseline import sweep_v1_campaign as sweep
from src.baseline import sweep_v1_review_analysis as analysis

__all__ = ["SYNTHETIC_BANNER", "render_checkpoint_packet"]

SYNTHETIC_BANNER = "SYNTHETIC / DEMONSTRATION DATA — NOT SCIENTIFIC RESULTS"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "figure.dpi": 100, "savefig.dpi": 170, "axes.grid": True,
    "grid.alpha": 0.25, "font.family": "DejaVu Sans",
})

ARM_STYLE = {
    "bayesian": {"color": "#1f5fa8", "marker": "o", "label": "Bayesian"},
    "random_control": {"color": "#e08214", "marker": "^", "label": "Random control"},
}
TOP_QUARTILE_EDGE = "#111111"
FAIL_STYLE = {"color": "#b22222", "marker": "x", "label": "Failed attempt (excluded)"}
FAILURE_CATEGORY_COLORS = {"node_failure": "#b22222", "oom": "#8a1c9c", "timeout": "#c9761f",
                           "data_error": "#555555"}

_DEFAULT_REVIEW_NAMES = {"checkpoint_12": "Boundary Review 1", "checkpoint_24": "Boundary Review 2",
                          "final": "Final Closure"}


def _review_name_for(checkpoint_label: str, review_name: str | None) -> str:
    return review_name or _DEFAULT_REVIEW_NAMES.get(checkpoint_label, checkpoint_label)


# ------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------

def _banner(fig: plt.Figure, synthetic: bool) -> None:
    if synthetic:
        fig.text(0.5, 0.995, SYNTHETIC_BANNER, ha="center", va="top", fontsize=11,
                  color="#8a1c1c", fontweight="bold")


def _save(fig: plt.Figure, path: Path, also_pdf: bool = False) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    saved = [path]
    if also_pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        saved.append(pdf_path)
    plt.close(fig)
    return saved


def _arm_scatter(ax, df: pd.DataFrame, x_col: str, y_col: str, *, top_mask: pd.Series | None = None,
                  size: int = 46) -> None:
    for arm, style in ARM_STYLE.items():
        sub = df[df["search_arm"] == arm]
        if not len(sub):
            continue
        ax.scatter(sub[x_col], sub[y_col], s=size, c=style["color"], marker=style["marker"],
                   label=style["label"], alpha=0.85, edgecolors="none", zorder=3)
        if top_mask is not None:
            top_sub = sub[top_mask.reindex(sub.index, fill_value=False)]
            if len(top_sub):
                ax.scatter(top_sub[x_col], top_sub[y_col], s=size * 1.9, facecolors="none",
                           edgecolors=TOP_QUARTILE_EDGE, linewidths=1.4, marker=style["marker"],
                           label=f"{style['label']} (top quartile)", zorder=4)


def _mark_continuous_bounds(ax, lower: float, upper: float, geometry: str, orientation: str = "x",
                             boundary_fraction: float = analysis.BOUNDARY_FRACTION) -> None:
    set_lim = ax.set_xlim if orientation == "x" else ax.set_ylim
    axvline = ax.axvline if orientation == "x" else ax.axhline
    axvspan = ax.axvspan if orientation == "x" else ax.axhspan
    axvline(lower, color="black", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
    axvline(upper, color="black", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
    if geometry == "log":
        log_lower, log_upper = np.log10(lower), np.log10(upper)
        span = log_upper - log_lower
        lower_band = 10 ** (log_lower + boundary_fraction * span)
        upper_band = 10 ** (log_upper - boundary_fraction * span)
    else:
        span = upper - lower
        lower_band = lower + boundary_fraction * span
        upper_band = upper - boundary_fraction * span
    axvspan(lower, lower_band, color="#c94b4b", alpha=0.08, zorder=0)
    axvspan(upper_band, upper, color="#c94b4b", alpha=0.08, zorder=0)


def _table_ax(ax, dataframe: pd.DataFrame, col_widths: list[float] | None = None, fontsize: float = 8.6) -> None:
    ax.axis("off")
    table = ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc="center",
                     cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.6)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f4f6f8" if row % 2 == 0 else "white")


_BOUNDARY_MATRIX_LEGEND = (
    "Legend: top-Q near-frac (pooled/Bay/rand) = fraction of that arm's top-quartile-by-best_score trials landing "
    "within the 10% boundary zone.  shift→bound = signed median Bayesian proposal-position move toward this side "
    "over the search, in [0,1] boundary-normalized units (positive = moved toward this boundary).  spearman→bound = "
    "rank correlation of proposal order with position toward this side (positive = later proposals sit closer).  "
    "prev tier / evolution = this axis/side's tier at the previous checkpoint and the strengthening/stable/weakening "
    "read.  Natural bounds (dropout=0) are always shown as a preference, never as expansion pressure."
)


def _boundary_pressure_display_table_v2(boundary_df: pd.DataFrame, evolution_df: pd.DataFrame | None) -> pd.DataFrame:
    """Human-readable Panel E matrix (§9): every cell has defined units, a
    shared abbreviations legend is rendered alongside it, and (when a prior
    checkpoint is available) previous-tier + strengthening/stable/weakening
    columns make evolution visible without a second lookup (§10)."""
    display = boundary_df.copy()
    if evolution_df is not None and len(evolution_df):
        display = display.merge(evolution_df[["axis", "boundary_side", "tier_previous", "direction"]],
                                 on=["axis", "boundary_side"], how="left")
    else:
        display["tier_previous"] = None
        display["direction"] = "n/a (first checkpoint)"
    display["tier_previous"] = display["tier_previous"].fillna("—")
    display["direction"] = display["direction"].fillna("n/a (first checkpoint)")

    def _fmt(v):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"

    display["shift_display"] = display["position_shift_toward_boundary"].map(_fmt)
    display["spearman_display"] = display["spearman_toward_boundary"].map(_fmt)
    columns = ["axis", "boundary_side", "interpretation", "top_quartile_near_fraction",
               "bayesian_top_quartile_near_fraction", "random_top_quartile_near_fraction",
               "shift_display", "spearman_display", "tier_previous", "direction"]
    return display[columns].rename(columns={
        "boundary_side": "side", "interpretation": "interpretation",
        "top_quartile_near_fraction": "top-Q near-frac (pooled)",
        "bayesian_top_quartile_near_fraction": "top-Q near-frac (Bay)",
        "random_top_quartile_near_fraction": "top-Q near-frac (rand)",
        "shift_display": "shift→bound", "spearman_display": "spearman→bound",
        "tier_previous": "prev tier", "direction": "evolution"})


def _status_header_text(status: dict[str, Any]) -> str:
    overshoot = f" (+{status['bounded_overshoot']} bounded overshoot)" if status["bounded_overshoot"] else ""
    bay_gpu = status["cumulative_bayesian_gpu_hours"]
    rand_gpu = status["cumulative_random_gpu_hours"]
    bay_best = status["bayesian_incumbent_best"]
    rand_best = status["random_best_at_common_n"]
    bay_best_s = f"{bay_best:.3f}" if bay_best is not None else "n/a"
    rand_best_s = f"{rand_best:.3f}" if rand_best is not None else "n/a"
    return (
        f"WHERE ARE WE?   {status['review_name']}\n"
        f"  Valid Bayesian: {status['actual_valid_bayesian']} / target {status['target_valid_bayesian']}"
        f"{overshoot}    Valid random-control: {status['valid_random_control']}    "
        f"Failed/retry attempts: {status['failed_or_retry_attempts']}    Common-N: {status['common_n']}\n"
        f"  Cumulative GPU-hours — Bayesian: {bay_gpu:.1f}h | Random: {rand_gpu:.1f}h    "
        f"Incumbent best — Bayesian: {bay_best_s} | Random @ common-N: {rand_best_s}"
    )


def _draw_evolution_panel(ax, evolution_df: pd.DataFrame | None) -> None:
    """STRENGTHENING? mini-panel (§10): compact arrow notation per
    axis/side, or an explicit first-checkpoint notice when there is nothing
    to compare against yet."""
    ax.axis("off")
    if evolution_df is None or not len(evolution_df) or (evolution_df["direction"] == "n/a (first checkpoint)").all():
        ax.text(0.0, 0.85, "No prior checkpoint available — this is the first boundary review.\n"
                            "Evolution tracking begins at the next checkpoint.",
                fontsize=9.5, color="#555555", va="top")
        return
    arrow = {"strengthening": "↑ strengthening", "weakening": "↓ weakening", "stable": "→ stable"}
    lines = [f"{'axis':<20}{'side':<13}{'prev→now':<20}{'evolution'}"]
    for _, row in evolution_df.sort_values("axis").iterrows():
        prev = row["tier_previous"] or "—"
        transition = f"{prev} → {row['tier_current']}"
        lines.append(f"{row['axis']:<20}{str(row['boundary_side']):<13}{transition:<20}"
                     f"{arrow.get(row['direction'], row['direction'])}")
    ax.text(0.0, 0.95, "\n".join(lines), fontsize=8.2, family="monospace", va="top")


def _jitter_categorical_positions(values: pd.Series, categories: list, arm: pd.Series, rng_seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    base = values.map({c: i for i, c in enumerate(categories)}).to_numpy(dtype=float)
    arm_offset = np.where(arm.to_numpy() == "bayesian", -0.10, 0.10)
    jitter = rng.uniform(-0.06, 0.06, size=len(base))
    return base + arm_offset + jitter


# ------------------------------------------------------------------
# panel-drawing primitives (reused by board + standalone figures)
# ------------------------------------------------------------------

def _draw_search_progress(ax, trial_slice: pd.DataFrame, checkpoint_valid_bayesian_count: int,
                           checkpoint_label: str, review_name: str | None = None) -> None:
    valid = analysis.valid_trials(trial_slice)
    n = analysis.common_n(trial_slice)
    label_of = {"bayesian": "Bayesian", "random_control": "Random control (one frozen realization)"}
    curves: dict[str, pd.Series] = {}
    for arm, style in ARM_STYLE.items():
        sub = valid[valid["search_arm"] == arm].sort_values("valid_result_order")
        if not len(sub):
            continue
        cumulative = analysis.cumulative_best_by_order(sub["best_score"])
        curves[arm] = pd.Series(cumulative, index=sub["valid_result_order"].to_numpy())
        # each arm's line stops at its own last valid trial -- no extrapolation.
        ax.plot(sub["valid_result_order"], cumulative, color=style["color"], marker=style["marker"],
                markersize=5, linewidth=1.8, label=f"{label_of[arm]} (n={len(sub)})", zorder=3)
    if n > 0:
        ax.axvspan(0, n, color="#2c3e50", alpha=0.06, zorder=0, label="common-N region (both arms observed)")
        ax.axvline(n, color="#555555", linestyle=":", linewidth=1.2, zorder=1)
        ymin, ymax = ax.get_ylim()
        ax.text(n, ymax, f" common N={n}", fontsize=8.5, color="#555555", va="top")
        if "bayesian" in curves and "random_control" in curves:
            bayes_at_n = curves["bayesian"].reindex(curves["bayesian"].index[curves["bayesian"].index <= n]).iloc[-1]
            random_at_n = curves["random_control"].reindex(
                curves["random_control"].index[curves["random_control"].index <= n]).iloc[-1]
            ax.text(0.02, 0.02, f"at common-N: Bayesian best={bayes_at_n:.3f}  |  random best={random_at_n:.3f}",
                    transform=ax.transAxes, fontsize=8.3, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#999999", alpha=0.85))
    ax.set_xlim(left=0)
    ax.set_xlabel("valid trial index within arm")
    ax.set_ylabel("cumulative best objective (raw-space NSE)")
    title_prefix = f"{review_name} — " if review_name else ""
    ax.set_title(f"{title_prefix}Search progress ({checkpoint_valid_bayesian_count} valid Bayesian target)",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=7.8)


def _continuous_geometry(axis: str) -> str:
    spec = sweep.SEARCH_DOMAIN[axis]
    return "log" if spec.get("distribution") == "log_uniform" else "linear"


def _draw_continuous_pressure_panel(ax_perf, ax_drift, trial_slice: pd.DataFrame, axis: str, side: str) -> None:
    """Combined performance-vs-parameter + proposal-movement-vs-order view
    for the single most-pressured continuous axis (§8): one logical panel,
    two subpanels, sharing the legal bound / 10% zone / top-performer
    markers so a reviewer sees where performance concentrates AND whether
    the Bayesian search is actively moving toward that boundary."""
    domain = sweep.SEARCH_DOMAIN
    geometry = _continuous_geometry(axis)
    lower, upper = domain[axis]["lower"], domain[axis]["upper"]
    valid = analysis.valid_trials(trial_slice)
    bayesian = valid[valid["search_arm"] == "bayesian"].sort_values("proposal_order")
    top_mask, threshold = analysis.is_top_quartile(valid)

    _arm_scatter(ax_perf, valid, axis, "best_score", top_mask=top_mask)
    _mark_continuous_bounds(ax_perf, lower, upper, geometry, orientation="x")
    if geometry == "log":
        ax_perf.set_xscale("log")
    ax_perf.set_xlabel(f"{axis}" + (" (log scale)" if geometry == "log" else ""))
    ax_perf.set_ylabel("best_score (raw-space NSE)")
    ax_perf.set_title(f"Performance vs {axis}\n(top quartile ≥ {threshold:.3f}; red band = 10% near-{side} zone)",
                       fontsize=9.5)
    ax_perf.legend(loc="lower left" if side == "lower" else "lower right", fontsize=7.2)

    evidence = analysis.proposal_drift_evidence(bayesian, axis, side, geometry=geometry, lower=lower, upper=upper)
    ax_drift.scatter(bayesian["proposal_order"], bayesian[axis], c="#1f5fa8", s=42, zorder=3)
    if geometry == "log":
        ax_drift.set_yscale("log")
    _mark_continuous_bounds(ax_drift, lower, upper, geometry, orientation="y")
    ax_drift.set_xlabel("Bayesian proposal order")
    ax_drift.set_ylabel(axis)
    shift = evidence["position_shift_toward_boundary"]
    spearman = evidence["spearman_toward_boundary"]
    shift_str = f"{shift:+.2f}" if shift is not None else "n/a"
    spearman_str = f"{spearman:+.2f}" if spearman is not None else "n/a"
    ax_drift.set_title(f"Proposal movement vs order\n(shift→{side}={shift_str} of [0,1]; "
                        f"spearman→{side}={spearman_str}; drift flagged: {evidence['drift_toward_boundary']})",
                        fontsize=9)


def _draw_categorical_panel(ax, trial_slice: pd.DataFrame, axis: str) -> None:
    """Per-arm occupancy as a FRACTION of that arm's own valid n (§5): raw
    counts are misleading here because Bayesian n and random-control n
    differ substantially, so the primary bar height is normalized within
    each arm and the raw count is kept only as a small supplemental
    annotation on the bar."""
    occ = analysis.categorical_occupancy_table(trial_slice)
    occ = occ[occ["axis"] == axis]
    values = sweep.SEARCH_DOMAIN[axis]["values"]
    width = 0.35
    x = np.arange(len(values))
    for offset, arm in zip((-width / 2, width / 2), ("bayesian", "random_control")):
        sub = occ[occ["search_arm"] == arm].set_index("value").reindex(values)
        total_n = int(sub["total"].iloc[0]) if len(sub) and pd.notna(sub["total"].iloc[0]) else 0
        ax.bar(x + offset, sub["fraction"], width=width, color=ARM_STYLE[arm]["color"],
               label=f"{ARM_STYLE[arm]['label']} (n={total_n})", alpha=0.85)
        for xi, (frac, count, top_frac) in enumerate(zip(sub["fraction"], sub["count"], sub["top_quartile_fraction"])):
            if pd.notna(frac):
                ax.text(xi + offset, frac + 0.015, f"n={int(count)}", ha="center", fontsize=7, color="#333333")
            if pd.notna(top_frac):
                ax.text(xi + offset, (frac or 0) + 0.06, f"top-Q {top_frac:.0%}", ha="center", fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(axis)
    ax.set_ylabel("fraction of arm's valid trials")
    ax.set_title(f"{axis} occupancy — per-arm fraction\n(bar labels: raw n, top-quartile fraction landing here)",
                 fontsize=9)
    ax.legend(fontsize=8)


def _draw_learning_behavior_panels(ax_best_epoch, ax_best_minus_final, trial_slice: pd.DataFrame) -> None:
    valid = analysis.valid_trials(trial_slice)
    bayesian = valid[valid["search_arm"] == "bayesian"]
    epochs = list(range(1, 13))
    counts = bayesian["best_epoch"].value_counts().reindex(epochs, fill_value=0)
    colors = ["#c94b4b" if e >= 11 else "#1f5fa8" for e in epochs]
    ax_best_epoch.bar(epochs, counts.values, color=colors)
    ax_best_epoch.set_xlabel("best_epoch", fontsize=8.5)
    ax_best_epoch.set_ylabel("Bayesian trial count")
    n_late = int((bayesian["late_best"]).sum())
    ax_best_epoch.set_title(f"best_epoch distribution\n(late_best ≥ 11: {n_late}/{len(bayesian)}, red)",
                            fontsize=9.5)

    ax_best_minus_final.hist(valid["best_minus_final"], bins=14, color="#4c6b8a", edgecolor="white")
    ax_best_minus_final.set_xlabel("best_minus_final\n(best_score − final_epoch_score)", fontsize=8.5)
    ax_best_minus_final.set_ylabel("trial count")
    ax_best_minus_final.set_title("Post-peak instability", fontsize=9.5)


# ------------------------------------------------------------------
# decision board
# ------------------------------------------------------------------

def render_decision_board(output_dir: Path, *, trial_slice: pd.DataFrame, ops_slice: pd.DataFrame,
                           checkpoint_label: str, checkpoint_valid_bayesian_count: int, synthetic: bool,
                           previous_boundary_df: pd.DataFrame | None = None,
                           review_name: str | None = None) -> Path:
    """Decision Board v2 (§6): panels are arranged in the reading order a
    reviewer actually needs — WHERE ARE WE? (status header) → VALUE-ADD?
    (search progress vs random) → BOUNDARY PRESSURE? (most-pressured
    continuous axis + categorical occupancy) → STRENGTHENING? (checkpoint
    evolution) → the full boundary-evidence matrix, with budget/failure
    counts surfaced up front in the header rather than buried in Panel E.
    """
    review_name = _review_name_for(checkpoint_label, review_name)
    boundary_df = analysis.derive_boundary_pressure_table(trial_slice)
    evolution_df = analysis.boundary_pressure_evolution(boundary_df, previous_boundary_df)
    status = analysis.checkpoint_status_summary(
        trial_slice, ops_slice, checkpoint_valid_bayesian_count=checkpoint_valid_bayesian_count,
        review_name=review_name)
    pressured_axis, pressured_side = analysis.most_pressured_continuous_axis(boundary_df)

    fig = plt.figure(figsize=(21, 18))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.42, 1.0, 1.0, 1.35], hspace=0.60, wspace=0.28)

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    ax_header.text(0.0, 1.0, _status_header_text(status), fontsize=11, family="monospace", va="top",
                   linespacing=1.7)

    ax_a = fig.add_subplot(gs[1, 0])
    _draw_search_progress(ax_a, trial_slice, checkpoint_valid_bayesian_count, checkpoint_label, review_name)
    ax_a.set_title("VALUE-ADD?\n" + ax_a.get_title(), fontsize=10)

    ax_b1 = fig.add_subplot(gs[1, 1])
    ax_b2 = ax_b1.inset_axes([1.18, 0, 0.82, 1])
    _draw_continuous_pressure_panel(ax_b1, ax_b2, trial_slice, pressured_axis, pressured_side)
    ax_b1.annotate(f"BOUNDARY PRESSURE? — most-pressured continuous axis: {pressured_axis} ({pressured_side})",
                   xy=(0, 1.32), xycoords="axes fraction", fontsize=10, fontweight="bold")

    ax_c = fig.add_subplot(gs[2, 0])
    _draw_categorical_panel(ax_c, trial_slice, "hidden_size")
    ax_c.set_title("BOUNDARY PRESSURE (categorical)?\n" + ax_c.get_title(), fontsize=9.5)

    ax_d = fig.add_subplot(gs[2, 1])
    ax_d.set_title("STRENGTHENING? — checkpoint-to-checkpoint tier evolution", fontsize=11, loc="left")
    _draw_evolution_panel(ax_d, evolution_df if previous_boundary_df is not None else None)

    ax_e = fig.add_subplot(gs[3, :])
    _table_ax(ax_e, _boundary_pressure_display_table_v2(boundary_df, evolution_df),
              col_widths=[0.07, 0.075, 0.20, 0.11, 0.10, 0.10, 0.09, 0.09, 0.075, 0.10], fontsize=7.4)
    ax_e.text(0.0, -0.10, _BOUNDARY_MATRIX_LEGEND, transform=ax_e.transAxes, fontsize=7.4, color="#333333",
              wrap=True, va="top")
    ax_e.set_title("Panel E — Boundary-evidence matrix, all 5 axes × 2 sides "
                   "(evidence table, not a composite score; BUDGET/STABILITY WARNINGS in header above)",
                   fontsize=11, pad=14)

    fig.suptitle(f"Sweep-v1 Checkpoint Decision Board — {review_name}", fontsize=17, y=1.015, fontweight="bold")
    for ax, letter, dy in ((ax_a, "A", 1.05), (ax_b1, "B", 1.42), (ax_c, "C", 1.18), (ax_d, "D", 1.05),
                           (ax_e, "E", 1.14)):
        ax.annotate(letter, xy=(-0.10, dy), xycoords="axes fraction", fontsize=13, fontweight="bold")
    _banner(fig, synthetic)
    path = output_dir / "decision_board.png"
    _save(fig, path, also_pdf=True)
    return path


# ------------------------------------------------------------------
# 12 standalone figures
# ------------------------------------------------------------------

def fig1_search_progress(output_dir: Path, trial_slice: pd.DataFrame, checkpoint_label: str,
                          checkpoint_valid_bayesian_count: int, synthetic: bool) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.2))
    _draw_search_progress(ax, trial_slice, checkpoint_valid_bayesian_count, checkpoint_label)
    fig.suptitle("Figure 1 — Bayesian vs Random Search Progress", fontsize=13, y=1.02)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig01_search_progress.png", also_pdf=True)[0]


def fig2_compute_efficiency(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    valid = analysis.valid_trials(trial_slice)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for arm, style in ARM_STYLE.items():
        sub = valid[valid["search_arm"] == arm].sort_values("valid_result_order")
        if not len(sub):
            continue
        cumulative_gpu = analysis.cumulative_gpu_hours_by_order(sub["gpu_hours"])
        cumulative_best = analysis.cumulative_best_by_order(sub["best_score"])
        ax.plot(cumulative_gpu, cumulative_best, color=style["color"], marker=style["marker"],
                markersize=5, linewidth=1.8, label=f"{style['label']} (n={len(sub)})")
    ax.set_xlabel("cumulative GPU-hours (secondary axis — valid-trial-index is primary, see Fig. 1)")
    ax.set_ylabel("cumulative best objective (raw-space NSE)")
    fig.suptitle("Figure 2 — Bayesian vs Random Compute Efficiency", fontsize=13, y=1.02)
    ax.legend(loc="lower right")
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig02_compute_efficiency.png", also_pdf=True)[0]


def fig3_objective_distribution(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    valid = analysis.valid_trials(trial_slice)
    top_mask, threshold = analysis.is_top_quartile(valid)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for i, (arm, style) in enumerate(ARM_STYLE.items()):
        sub = valid[valid["search_arm"] == arm]
        if not len(sub):
            continue
        y = np.full(len(sub), i) + np.random.default_rng(3).uniform(-0.12, 0.12, len(sub))
        ax.scatter(sub["best_score"], y, s=50, c=style["color"], marker=style["marker"], alpha=0.85,
                   label=f"{style['label']} (n={len(sub)})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.1, label=f"top-quartile threshold ({threshold:.3f})")
    top_sorted = valid[top_mask].sort_values("best_score", ascending=False).head(5)
    # ranks with near-identical best_score in the same row (Bayesian/random)
    # would otherwise stack their "#N" labels on top of one another, so
    # nearby labels within a row are staggered to increasing heights.
    x_span = float(valid["best_score"].max() - valid["best_score"].min()) or 1.0
    cluster_gap = 0.02 * x_span
    last_x_by_row: dict[int, float] = {}
    level_by_row: dict[int, int] = {}
    for _, row in top_sorted.sort_values("best_score").iterrows():
        y_pos = 0 if row["search_arm"] == "bayesian" else 1
        x = float(row["best_score"])
        if y_pos in last_x_by_row and (x - last_x_by_row[y_pos]) < cluster_gap:
            level_by_row[y_pos] += 1
        else:
            level_by_row[y_pos] = 0
        last_x_by_row[y_pos] = x
        display_rank = int(top_sorted.index.get_loc(row.name)) + 1
        ax.annotate(f"#{display_rank}", (x, y_pos), xytext=(0, 14 + 14 * level_by_row[y_pos]),
                    textcoords="offset points", fontsize=8, ha="center")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Bayesian", "Random control"])
    ax.set_xlabel("best_score (raw-space NSE)")
    ax.set_title("no significance test implied — spread/overlap only", fontsize=9.5, color="#555555")
    fig.suptitle("Figure 3 — Candidate Objective Distribution", fontsize=13, y=1.03)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig03_objective_distribution.png", also_pdf=True)[0]


def fig4_hyperparameter_response(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    valid = analysis.valid_trials(trial_slice)
    top_mask, _ = analysis.is_top_quartile(valid)
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.4))
    domain = sweep.SEARCH_DOMAIN

    ax = axes[0]
    _arm_scatter(ax, valid, "learning_rate", "best_score", top_mask=top_mask)
    _mark_continuous_bounds(ax, domain["learning_rate"]["lower"], domain["learning_rate"]["upper"], "log")
    ax.set_xscale("log"); ax.set_xlabel("learning_rate (log)"); ax.set_ylabel("best_score")

    for ax, axis in zip(axes[1:3], ("embedding_dropout", "output_dropout")):
        _arm_scatter(ax, valid, axis, "best_score", top_mask=top_mask)
        _mark_continuous_bounds(ax, domain[axis]["lower"], domain[axis]["upper"], "linear")
        ax.set_xlabel(axis)

    for ax, axis in zip(axes[3:5], ("hidden_size", "batch_size")):
        categories = domain[axis]["values"]
        x = _jitter_categorical_positions(valid[axis], categories, valid["search_arm"])
        for arm, style in ARM_STYLE.items():
            mask = valid["search_arm"] == arm
            ax.scatter(x[mask.to_numpy()], valid.loc[mask, "best_score"], s=46, c=style["color"],
                      marker=style["marker"], alpha=0.85, label=style["label"])
            top_arm_mask = (top_mask & mask).to_numpy()
            if top_arm_mask.any():
                ax.scatter(x[top_arm_mask], valid.loc[top_arm_mask, "best_score"], s=90, facecolors="none",
                          edgecolors=TOP_QUARTILE_EDGE, linewidths=1.3, marker=style["marker"])
        ax.set_xticks(range(len(categories))); ax.set_xticklabels([str(c) for c in categories])
        ax.set_xlabel(axis)

    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Figure 4 — Five-Axis Hyperparameter Response (top-quartile candidates outlined)",
                fontsize=14, y=1.05)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig04_five_axis_response.png", also_pdf=True)[0]


def fig5_boundary_occupancy(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    valid = analysis.valid_trials(trial_slice)
    bayesian = valid[valid["search_arm"] == "bayesian"]
    top_mask, threshold = analysis.is_top_quartile(valid)
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), gridspec_kw={"wspace": 0.32, "hspace": 0.4})
    domain = sweep.SEARCH_DOMAIN

    ax = axes[0, 0]
    lr_lower, lr_upper = domain["learning_rate"]["lower"], domain["learning_rate"]["upper"]
    bins = np.logspace(np.log10(lr_lower), np.log10(lr_upper), 16)
    ax.hist(valid["learning_rate"], bins=bins, color="#9fb8d8", edgecolor="white", label="all valid")
    ax.hist(valid.loc[top_mask, "learning_rate"], bins=bins, color="#1f5fa8", edgecolor="white",
           label="top quartile", alpha=0.9)
    _mark_continuous_bounds(ax, lr_lower, lr_upper, "log")
    ax.set_xscale("log"); ax.set_xlabel("learning_rate (log)"); ax.set_ylabel("count")
    ax.set_title("LR distribution: all vs top-quartile (shaded = near-boundary 10%)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ordered = bayesian.sort_values("proposal_order")
    half = len(ordered) // 2
    colors = ["#7a9cc6"] * half + ["#c94b4b"] * (len(ordered) - half)
    ax.scatter(ordered["proposal_order"], ordered["learning_rate"], c=colors, s=44)
    ax.set_yscale("log")
    _mark_continuous_bounds(ax, lr_lower, lr_upper, "log", orientation="y")
    ax.set_xlabel("Bayesian proposal order"); ax.set_ylabel("learning_rate (log)")
    ax.set_title("Early (blue) vs late (red) proposals")

    ax = axes[0, 2]
    od_lower, od_upper = domain["output_dropout"]["lower"], domain["output_dropout"]["upper"]
    bins = np.linspace(od_lower, od_upper, 14)
    ax.hist(valid["output_dropout"], bins=bins, color="#9fd8ba", edgecolor="white", label="all valid")
    ax.hist(valid.loc[top_mask, "output_dropout"], bins=bins, color="#1f8a4c", edgecolor="white",
           label="top quartile", alpha=0.9)
    _mark_continuous_bounds(ax, od_lower, od_upper, "linear")
    ax.set_xlabel("output_dropout (natural lower bound)"); ax.set_ylabel("count")
    ax.set_title("output_dropout distribution (0 = natural, not expandable)")
    ax.legend(fontsize=8)

    _draw_categorical_panel(axes[1, 0], trial_slice, "hidden_size")
    _draw_categorical_panel(axes[1, 1], trial_slice, "batch_size")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.85, f"top-quartile threshold: {threshold:.3f}", fontsize=10)
    axes[1, 2].text(0.02, 0.65, f"n valid = {len(valid)} (Bayesian {len(bayesian)}, "
                                f"random {len(valid) - len(bayesian)})", fontsize=10)
    axes[1, 2].text(0.02, 0.45, "This figure directly supports the\nCONTINUE / UNCERTAIN / EXPAND review:\n"
                                "compare shaded near-boundary bands\nagainst top-quartile concentration.",
                    fontsize=9.5, color="#333333")

    fig.suptitle("Figure 5 — Boundary / Proposal Occupancy", fontsize=14, y=1.02)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig05_boundary_occupancy.png", also_pdf=True)[0]


def fig6_proposal_drift(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    valid = analysis.valid_trials(trial_slice)
    bayesian = valid[valid["search_arm"] == "bayesian"].sort_values("proposal_order")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    domain = sweep.SEARCH_DOMAIN

    ax = axes[0]
    ax.scatter(bayesian["proposal_order"], bayesian["learning_rate"], c="#1f5fa8", s=46, zorder=3)
    if len(bayesian) >= 4:
        window = max(3, len(bayesian) // 6)
        rolling = bayesian["learning_rate"].rolling(window, min_periods=1, center=True).median()
        ax.plot(bayesian["proposal_order"], rolling, color="#c94b4b", linewidth=1.6,
               linestyle="--", label=f"rolling median (window={window}, secondary)", zorder=2)
        ax.legend(fontsize=8)
    ax.set_yscale("log")
    _mark_continuous_bounds(ax, domain["learning_rate"]["lower"], domain["learning_rate"]["upper"], "log",
                            orientation="y")
    ax.set_xlabel("Bayesian proposal order"); ax.set_ylabel("learning_rate (log)")
    ax.set_title("learning_rate proposal drift")

    ax = axes[1]
    window = max(3, len(bayesian) // 6)
    is_h256 = (bayesian["hidden_size"] == 256).astype(float)
    rolling_fraction = is_h256.rolling(window, min_periods=1).mean()
    ax.plot(bayesian["proposal_order"], rolling_fraction, color="#1f8a4c", linewidth=2, marker="o", markersize=4)
    ax.axhline(1 / 3, color="black", linestyle=":", linewidth=1, label="uniform-prior rate (1/3)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Bayesian proposal order"); ax.set_ylabel(f"rolling fraction hidden_size=256 (window={window})")
    ax.set_title("hidden_size=256 proposal drift")
    ax.legend(fontsize=8)

    fig.suptitle("Figure 6 — Bayesian Proposal Drift", fontsize=14, y=1.03)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig06_proposal_drift.png", also_pdf=True)[0]


_ROLE_ARCHETYPE_REASON = {
    "strong_stable": "top-quartile best_score, best_minus_final near 0",
    "typical": "median best_score, noisy but stable post-peak",
    "late_best": "best_epoch ≥ 11 (still improving near budget end)",
    "unstable": "largest best_minus_final in this slice",
}


def _short_hyperparam_str(row: pd.Series) -> str:
    return (f"lr={row['learning_rate']:.1e} h={int(row['hidden_size'])} "
            f"ed={row['embedding_dropout']:.2f} od={row['output_dropout']:.2f} bs={int(row['batch_size'])}")


def fig7_representative_trajectories(output_dir: Path, trial_slice: pd.DataFrame, trajectory_slice: pd.DataFrame,
                                     synthetic: bool) -> Path:
    """§15: each representative's legend entry now carries a short config ID,
    search arm, abbreviated hyperparameters, and the archetype/selection
    reason -- not just a role name -- without overloading the legend (one
    compact multi-line entry per representative, legend moved below the
    axes and sized for four entries)."""
    picks = analysis.representative_trajectory_selection(trial_slice)
    valid = analysis.valid_trials(trial_slice).set_index("trial_id")
    role_style = {
        "strong_stable": {"color": "#1f8a4c", "marker": "o", "linestyle": "-", "label": "strong / stable"},
        "typical": {"color": "#1f5fa8", "marker": "s", "linestyle": "-", "label": "typical / noisy-stable"},
        "late_best": {"color": "#8a5fc9", "marker": "D", "linestyle": "-.", "label": "late-best (best_epoch≥11)"},
        "unstable": {"color": "#c94b4b", "marker": "X", "linestyle": ":", "label": "post-peak unstable"},
    }
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    for role, trial_id in picks.items():
        if trial_id is None or trial_id not in valid.index:
            continue
        traj = trajectory_slice[trajectory_slice["trial_id"] == trial_id].sort_values("epoch")
        if not len(traj):
            continue
        style = role_style[role]
        row = valid.loc[trial_id]
        arm_short = "Bay" if row["search_arm"] == "bayesian" else "rand"
        short_id = str(row["configuration_id"])[-8:]
        legend_label = (f"{style['label']}  [{short_id}, {arm_short}]\n"
                        f"  {_short_hyperparam_str(row)}\n"
                        f"  reason: {_ROLE_ARCHETYPE_REASON[role]}")
        ax.plot(traj["epoch"], traj["median_raw_space_nse"], color=style["color"], marker=style["marker"],
               linestyle=style["linestyle"], linewidth=1.9, markersize=6, label=legend_label, zorder=3)
        best_epoch, final_epoch = int(row["best_epoch"]), 12
        best_val = traj.set_index("epoch").loc[best_epoch, "median_raw_space_nse"]
        final_val = traj.set_index("epoch").loc[final_epoch, "median_raw_space_nse"]
        ax.scatter([best_epoch], [best_val], marker="*", s=220, color=style["color"], edgecolors="black",
                  linewidths=0.8, zorder=5)
        ax.scatter([final_epoch], [final_val], marker="P", s=110, color=style["color"], edgecolors="black",
                  linewidths=0.8, zorder=5)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("epoch (1–12)"); ax.set_ylabel("median raw-space NSE (screening)")
    ax.set_title("★ = best checkpoint, + = final (epoch 12) checkpoint", fontsize=9.5, color="#555555")
    fig.suptitle("Figure 7 — Representative Epoch Trajectories", fontsize=14, y=1.03)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.6, frameon=True)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig07_representative_trajectories.png", also_pdf=True)[0]


def fig8_best_epoch_late_gain(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    """§13: population is explicit (Bayesian primary, random shown
    separately since it's a single frozen realization); the x-axis is
    zero-anchored because ``late_gain_10_to_12`` is the difference of two
    cumulative maxima and is mathematically non-negative -- no invented
    threshold is applied, but a zoomed inset + rug resolves the small
    positive gains that a coarse histogram would otherwise flatten to a
    single left-edge bar."""
    fig = plt.figure(figsize=(15, 6.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.85], wspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_learning_behavior_panels(ax1, ax2, trial_slice)

    ax3 = fig.add_subplot(gs[0, 2])
    valid = analysis.valid_trials(trial_slice)
    bayesian = valid[valid["search_arm"] == "bayesian"]
    gains = bayesian["late_gain_10_to_12"].astype(float)
    assert (gains >= -1e-9).all(), "late_gain_10_to_12 must be non-negative (cumulative-max difference)"
    max_gain = float(gains.max()) if len(gains) else 0.0
    bins = np.linspace(0, max(max_gain, 1e-6), 16)
    ax3.hist(gains, bins=bins, color="#8a5fc9", edgecolor="white")
    for y in gains:
        ax3.plot([y, y], [0, -0.35], color="#333333", linewidth=0.8, alpha=0.6, clip_on=False)
    ax3.set_xlim(left=0)
    ax3.set_xlabel("late_gain_10_to_12\n(best_score_12 − best_score_10, ≥ 0 by construction)", fontsize=7.6)
    ax3.set_ylabel("Bayesian trial count")
    ax3.set_title(f"Late-gain distribution (Bayesian, n={len(bayesian)}; rug below axis;\n"
                 f"random control n={int((valid['search_arm'] == 'random_control').sum())}, not pooled here)",
                 fontsize=9)
    fig.suptitle("Figure 8 — Best-Epoch / Late-Gain Diagnostics", fontsize=14, y=1.03)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig08_best_epoch_late_gain.png", also_pdf=True)[0]


def fig9_best_vs_final(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    """§14: annotates the top few (not just one) instability cases with
    compact config IDs, spreading callouts around the point cloud so labels
    don't clip the axes or overlap each other; identity line and "no fixed
    threshold" framing are preserved."""
    valid = analysis.valid_trials(trial_slice)
    fig, ax = plt.subplots(figsize=(9.4, 8.2))
    _arm_scatter(ax, valid, "best_score", "final_epoch_score")
    lims = [min(valid["best_score"].min(), valid["final_epoch_score"].min()) - 0.015,
            max(valid["best_score"].max(), valid["final_epoch_score"].max()) + 0.015]
    ax.plot(lims, lims, color="black", linewidth=1, linestyle="--", label="identity (best = final)")
    ax.set_xlim(lims); ax.set_ylim(lims)

    n_annotate = min(4, len(valid))
    worst = valid.sort_values("best_minus_final", ascending=False).head(n_annotate)
    offsets = [(24, 18), (24, -30), (-70, 22), (-70, -34)]
    for rank, ((_, row), offset) in enumerate(zip(worst.iterrows(), offsets), start=1):
        short_id = str(row["configuration_id"])[-8:]
        ax.annotate(f"#{rank} {short_id}\n(best−final={row['best_minus_final']:.3f})",
                   (row["best_score"], row["final_epoch_score"]), xytext=offset,
                   textcoords="offset points", fontsize=7.8,
                   arrowprops=dict(arrowstyle="->", color="#333333", shrinkA=2, shrinkB=6),
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", alpha=0.9),
                   annotation_clip=False)
    ax.set_xlabel("best_score"); ax.set_ylabel("final_epoch_score (epoch 12)")
    ax.set_title(f"top {n_annotate} instability cases annotated — no fixed instability threshold is applied",
                fontsize=9.5, color="#555555")
    fig.suptitle("Figure 9 — Best-vs-Final / Instability Diagnostics", fontsize=13, y=1.03)
    ax.legend(loc="upper left", fontsize=8.5)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig09_best_vs_final.png", also_pdf=True)[0]


def fig10_parallel_coordinates(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    """§17: axis endpoints are labeled with actual legal-domain values (not
    just 0/1), categorical axes show their category labels, the
    log-geometry of learning_rate is called out explicitly, and the figure
    is widened with generous margins so no endpoint label clips."""
    valid = analysis.valid_trials(trial_slice)
    domain = sweep.SEARCH_DOMAIN
    axes_order = ["learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size"]

    def _norm(axis, value):
        spec = domain[axis]
        if spec["kind"] == "categorical":
            values = spec["values"]
            return values.index(value) / (len(values) - 1)
        lower, upper = spec["lower"], spec["upper"]
        if spec["distribution"] == "log_uniform":
            return (np.log10(value) - np.log10(lower)) / (np.log10(upper) - np.log10(lower))
        return (value - lower) / (upper - lower)

    def _endpoint_labels(axis) -> tuple[str, str]:
        spec = domain[axis]
        if spec["kind"] == "categorical":
            values = spec["values"]
            return str(values[0]), str(values[-1])
        if spec["distribution"] == "log_uniform":
            return f"{spec['lower']:.0e}\n(log scale)", f"{spec['upper']:.0e}\n(log scale)"
        return f"{spec['lower']:g}", f"{spec['upper']:g}"

    fig, ax = plt.subplots(figsize=(14, 7.2))
    cmap = plt.get_cmap("viridis")
    vmin, vmax = valid["best_score"].min(), valid["best_score"].max()
    order = valid.sort_values("best_score")
    for _, row in order.iterrows():
        ys = [_norm(axis, row[axis]) for axis in axes_order]
        color = cmap((row["best_score"] - vmin) / (vmax - vmin + 1e-12))
        linewidth = 2.4 if row["best_score"] >= order["best_score"].quantile(0.75) else 0.8
        alpha = 0.9 if linewidth > 1 else 0.35
        ax.plot(range(len(axes_order)), ys, color=color, linewidth=linewidth, alpha=alpha)
    for i, axis in enumerate(axes_order):
        ax.axvline(i, color="#cccccc", linewidth=0.8, zorder=0)
        low_label, high_label = _endpoint_labels(axis)
        ax.text(i, -0.06, low_label, ha="center", va="top", fontsize=7.5, transform=ax.get_xaxis_transform())
        ax.text(i, 1.05, high_label, ha="center", va="bottom", fontsize=7.5, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(axes_order)))
    ax.set_xticklabels(axes_order, rotation=0, fontsize=9.5)
    ax.set_xlim(-0.35, len(axes_order) - 0.65)
    ax.set_yticks([])
    ax.set_ylim(-0.12, 1.12)
    ax.set_ylabel("normalized within legal domain (see endpoint labels above/below each axis)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax, label="best_score", pad=0.02)
    ax.set_title("Exploratory / non-causal visualization only — not a sensitivity analysis; "
                "learning_rate axis is log-spaced", fontsize=9.5, color="#555555")
    fig.suptitle("Figure 10 — Multivariate Hyperparameter Structure", fontsize=14, y=1.05)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig10_parallel_coordinates.png", also_pdf=True)[0]


def fig11_pairwise_interactions(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    valid = analysis.valid_trials(trial_slice)
    top_mask, _ = analysis.is_top_quartile(valid)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    domain = sweep.SEARCH_DOMAIN

    ax = axes[0]
    hidden_values = domain["hidden_size"]["values"]
    cmap = plt.get_cmap("plasma")
    for value, marker in zip(hidden_values, ("o", "s", "^")):
        sub = valid[valid["hidden_size"] == value]
        sc = ax.scatter(sub["learning_rate"], sub["best_score"], c=sub["best_score"], cmap=cmap,
                       marker=marker, s=60, label=f"hidden_size={value}", edgecolors="black", linewidths=0.4)
    ax.set_xscale("log")
    _mark_continuous_bounds(ax, domain["learning_rate"]["lower"], domain["learning_rate"]["upper"], "log")
    ax.set_xlabel("learning_rate (log)"); ax.set_ylabel("best_score")
    ax.set_title("learning_rate × hidden_size")
    ax.legend(fontsize=8)

    ax = axes[1]
    sc = ax.scatter(valid["embedding_dropout"], valid["output_dropout"], c=valid["best_score"], cmap=cmap,
                    s=70, edgecolors="black", linewidths=0.4)
    top = valid[top_mask]
    ax.scatter(top["embedding_dropout"], top["output_dropout"], facecolors="none", edgecolors=TOP_QUARTILE_EDGE,
              s=140, linewidths=1.4, label="top quartile")
    ax.set_xlabel("embedding_dropout"); ax.set_ylabel("output_dropout")
    ax.set_title("embedding_dropout × output_dropout")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=axes.tolist(), label="best_score", pad=0.02, shrink=0.85)

    fig.suptitle("Figure 11 — Selected Pairwise / Interaction Views (curated, not an all-pairs matrix)",
                fontsize=13, y=1.03)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig11_pairwise_interactions.png", also_pdf=True)[0]


def fig12_operations(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    """§16: failed/incomplete attempts NEVER receive a fabricated scientific
    score. The runtime-vs-objective scatter (previously placing failures at
    an arbitrary y-position below the valid trials) now plots VALID trials
    only; failed attempts get their own strip panel keyed on
    runtime/cost/failure_category, with a compact table alongside."""
    valid = analysis.valid_trials(trial_slice)
    failed = trial_slice[trial_slice["workflow_status"] != analysis.VALID_STATUS].copy()
    fig = plt.figure(figsize=(19, 10.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.30)
    domain = sweep.SEARCH_DOMAIN

    ax = fig.add_subplot(gs[0, 0])
    for value in domain["hidden_size"]["values"]:
        sub = valid[valid["hidden_size"] == value]
        x = np.full(len(sub), value) + np.random.default_rng(11).uniform(-6, 6, len(sub))
        ax.scatter(x, sub["gpu_hours"], s=42, c=[ARM_STYLE[a]["color"] for a in sub["search_arm"]], alpha=0.8)
    ax.set_xticks(domain["hidden_size"]["values"])
    ax.set_xlabel("hidden_size"); ax.set_ylabel("gpu_hours"); ax.set_title("Cost vs hidden_size (valid trials)")

    ax = fig.add_subplot(gs[0, 1])
    for value in domain["batch_size"]["values"]:
        sub = valid[valid["batch_size"] == value]
        x = np.full(len(sub), value) + np.random.default_rng(12).uniform(-6, 6, len(sub))
        ax.scatter(x, sub["gpu_hours"], s=42, c=[ARM_STYLE[a]["color"] for a in sub["search_arm"]], alpha=0.8)
    ax.set_xticks(domain["batch_size"]["values"])
    ax.set_xlabel("batch_size"); ax.set_ylabel("gpu_hours"); ax.set_title("Cost vs batch_size (valid trials)")

    ax = fig.add_subplot(gs[0, 2])
    _arm_scatter(ax, valid, "runtime_seconds", "best_score")
    ax.set_xlabel("runtime_seconds"); ax.set_ylabel("best_score")
    ax.set_title(f"Runtime vs objective — VALID trials only (n={len(valid)})")
    ax.legend(fontsize=8)

    ax_strip = fig.add_subplot(gs[1, 0:2])
    categories = sorted(failed["failure_category"].dropna().unique().tolist()) or ["unknown"]
    cat_y = {c: i for i, c in enumerate(categories)}
    if len(failed):
        rng = np.random.default_rng(13)
        for cat in categories:
            sub = failed[failed["failure_category"] == cat]
            if not len(sub):
                continue
            y = np.full(len(sub), cat_y[cat]) + rng.uniform(-0.15, 0.15, len(sub))
            color = FAILURE_CATEGORY_COLORS.get(cat, FAIL_STYLE["color"])
            ax_strip.scatter(sub["runtime_seconds"], y, s=60, c=color, marker=FAIL_STYLE["marker"])
        ax_strip.set_yticks(range(len(categories)))
        ax_strip.set_yticklabels(categories)
        ax_strip.set_xlabel("runtime_seconds (at time of failure/retry)")
        ax_strip.set_title(f"Failed/incomplete attempts by category — NO score plotted (n={len(failed)})",
                           fontsize=10)
    else:
        ax_strip.axis("off")
        ax_strip.text(0.02, 0.5, "No failed/incomplete attempts in this checkpoint slice.", fontsize=10,
                      color="#555555")

    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis("off")
    if len(failed):
        table_df = failed[["configuration_id", "search_arm", "failure_category", "runtime_seconds", "gpu_hours"]].copy()
        table_df["configuration_id"] = table_df["configuration_id"].astype(str).str.slice(-8)
        table_df = table_df.rename(columns={"configuration_id": "config", "search_arm": "arm",
                                            "failure_category": "category"})
        _table_ax(ax_table, table_df.reset_index(drop=True), fontsize=7.4)
        ax_table.set_title("Failed attempts — runtime/cost, no fabricated score", fontsize=9.5)
    else:
        ax_table.text(0.02, 0.5, "—", fontsize=10)

    fig.suptitle("Figure 12 — Runtime / Operations (operational evidence, not a scientific ranking)",
                fontsize=13, y=1.02)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig12_operations.png", also_pdf=True)[0]


_SENSITIVITY_AXES = (("learning_rate", "log"), ("embedding_dropout", "linear"), ("output_dropout", "linear"))


def _boundary_band_sensitivity_all_axes(trial_slice: pd.DataFrame) -> pd.DataFrame:
    valid = analysis.valid_trials(trial_slice)
    top_mask, _ = analysis.is_top_quartile(valid)
    top = valid[top_mask]
    domain = sweep.SEARCH_DOMAIN
    frames = []
    for axis, geometry in _SENSITIVITY_AXES:
        lower, upper = domain[axis]["lower"], domain[axis]["upper"]
        sens = analysis.boundary_band_sensitivity(top[axis], lower, upper, geometry)
        sens.insert(0, "axis", axis)
        frames.append(sens)
    return pd.concat(frames, ignore_index=True)


def fig13_boundary_evolution(output_dir: Path, boundary_df: pd.DataFrame,
                              previous_boundary_df: pd.DataFrame | None, synthetic: bool) -> Path:
    """Standalone checkpoint-to-checkpoint evolution figure (§10): the same
    tier-evolution evidence shown in the decision board's STRENGTHENING?
    panel, plus a bar comparison of top-quartile near-boundary fraction now
    vs at the previous checkpoint, for reviewers who want this evidence on
    its own page rather than folded into the board."""
    evolution_df = analysis.boundary_pressure_evolution(boundary_df, previous_boundary_df)
    fig = plt.figure(figsize=(14.5, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.35)

    ax_text = fig.add_subplot(gs[0, 0])
    _draw_evolution_panel(ax_text, evolution_df if previous_boundary_df is not None else None)
    ax_text.set_title("Tier evolution (previous → current)", fontsize=11, loc="left")

    ax_bar = fig.add_subplot(gs[0, 1])
    if previous_boundary_df is not None and len(evolution_df):
        labels = [f"{r['axis']} ({r['boundary_side']})" for _, r in evolution_df.iterrows()]
        x = np.arange(len(evolution_df))
        width = 0.35
        prev_vals = evolution_df["top_quartile_near_fraction_previous"].astype(float)
        cur_vals = evolution_df["top_quartile_near_fraction_current"].astype(float)
        ax_bar.bar(x - width / 2, prev_vals, width=width, color="#9fb8d8", label="previous checkpoint")
        ax_bar.bar(x + width / 2, cur_vals, width=width, color="#1f5fa8", label="current checkpoint")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, fontsize=7.3, rotation=38, ha="right", rotation_mode="anchor")
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_ylabel("top-quartile near-boundary fraction")
        ax_bar.set_title("Occupancy fraction: previous vs current checkpoint", fontsize=10.5)
        ax_bar.legend(fontsize=8)
    else:
        ax_bar.axis("off")
        ax_bar.text(0.05, 0.5, "No prior checkpoint available for comparison\n"
                                "(this is the first boundary review).", fontsize=10, color="#555555")
    fig.suptitle("Figure 13 — Checkpoint-to-Checkpoint Boundary-Pressure Evolution", fontsize=14, y=1.03)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig13_boundary_evolution.png", also_pdf=True)[0]


def fig14_boundary_band_sensitivity(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    """Supplemental boundary-band sensitivity diagnostic (§12): for each
    continuous axis, shows how the top-quartile near-boundary occupancy
    fraction would read under narrower/wider bands (5% / 20%) than the
    canonical 10% rule (dotted line). This is a robustness check only -- it
    does not alter :data:`analysis.BOUNDARY_FRACTION` or any tier."""
    sensitivity_df = _boundary_band_sensitivity_all_axes(trial_slice)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, (axis, _geometry) in zip(axes, _SENSITIVITY_AXES):
        sens = sensitivity_df[sensitivity_df["axis"] == axis].reset_index(drop=True)
        x = np.arange(len(sens))
        width = 0.35
        ax.bar(x - width / 2, sens["near_lower_fraction"], width=width, color="#c94b4b", label="near-lower fraction")
        ax.bar(x + width / 2, sens["near_upper_fraction"], width=width, color="#1f5fa8", label="near-upper fraction")
        for xi, canonical in zip(x, sens["canonical"]):
            if canonical:
                ax.axvline(xi, color="black", linestyle=":", linewidth=1.3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(f * 100)}%" for f in sens["band_fraction"]])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("boundary-band width (dotted = canonical 10% rule)")
        ax.set_ylabel("fraction of top-quartile trials")
        n = int(sens["n"].iloc[0]) if len(sens) else 0
        ax.set_title(f"{axis} (top-quartile n={n})", fontsize=10.5)
    axes[0].legend(fontsize=7.8)
    fig.suptitle("Figure 14 — Boundary-Band Sensitivity (5% / 10% / 20%) — supplemental robustness "
                "diagnostic; canonical 10% rule unchanged", fontsize=12.5, y=1.04)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig14_boundary_band_sensitivity.png", also_pdf=True)[0]


def fig15_top_configurations(output_dir: Path, trial_slice: pd.DataFrame, synthetic: bool) -> Path:
    """Standalone Top Configurations Matrix (§18): top Bayesian + top
    random-control candidates side by side across all 5 search axes plus
    derived diagnostics, so a reviewer can see whether several independent
    strong candidates agree on a hyperparameter story or the apparent
    conclusion rests on one isolated winner."""
    table = analysis.top_configurations_table(trial_slice)
    display = table.copy()
    display["best_score"] = display["best_score"].map(lambda v: f"{v:.4f}")
    display["learning_rate"] = display["learning_rate"].map(lambda v: f"{v:.2e}")
    display["embedding_dropout"] = display["embedding_dropout"].map(lambda v: f"{v:.3f}")
    display["output_dropout"] = display["output_dropout"].map(lambda v: f"{v:.3f}")
    display["late_gain_10_to_12"] = display["late_gain_10_to_12"].map(lambda v: f"{v:.4f}")
    display["best_minus_final"] = display["best_minus_final"].map(lambda v: f"{v:.4f}")
    display["late_best"] = display["late_best"].map(lambda v: "yes" if bool(v) else "no")
    n_bayesian = int((table["search_arm"] == "bayesian").sum())
    n_random = int((table["search_arm"] == "random_control").sum())
    display["search_arm"] = display["search_arm"].map({"bayesian": "Bay", "random_control": "rand"})
    display = display.rename(columns={
        "configuration_short_id": "config", "search_arm": "arm", "best_score": "best",
        "learning_rate": "lr", "hidden_size": "hidden", "embedding_dropout": "emb_do",
        "output_dropout": "out_do", "batch_size": "batch", "proposal_order": "order",
        "best_epoch": "best_ep", "late_gain_10_to_12": "late_gain", "best_minus_final": "best-final",
        "late_best": "late?"})
    display = display.drop(columns=["configuration_id"], errors="ignore")
    fig, ax = plt.subplots(figsize=(16.5, 0.9 + 0.42 * len(display)))
    _table_ax(ax, display, fontsize=8.2)
    fig.suptitle(f"Figure 15 — Top Configurations Matrix (top {n_bayesian} Bayesian + top {n_random} "
                f"random-control by best_score, all 5 axes + derived diagnostics)", fontsize=12.5, y=1.08)
    _banner(fig, synthetic)
    return _save(fig, output_dir / "fig15_top_configurations.png", also_pdf=True)[0]


# ------------------------------------------------------------------
# orchestration
# ------------------------------------------------------------------

_FIGURE_ORDER = [
    ("fig01_search_progress.png", "Bayesian vs random cumulative best objective by valid trial index (common-N marked)"),
    ("fig02_compute_efficiency.png", "Same comparison against cumulative GPU-hours (secondary axis)"),
    ("fig03_objective_distribution.png", "Spread/overlap of best_score by arm; top-quartile threshold marked"),
    ("fig04_five_axis_response.png", "best_score vs each of the 5 search axes; top-quartile candidates outlined"),
    ("fig05_boundary_occupancy.png", "LR/output_dropout distributions, early-vs-late proposals, categorical occupancy"),
    ("fig06_proposal_drift.png", "Bayesian proposal drift over time for learning_rate and hidden_size=256 fraction"),
    ("fig07_representative_trajectories.png", "Strong-stable / typical / late-best / unstable epoch trajectories"),
    ("fig08_best_epoch_late_gain.png", "best_epoch distribution and late_gain_10_to_12 distribution"),
    ("fig09_best_vs_final.png", "best_score vs final_epoch_score with identity line; instability visible, no threshold"),
    ("fig10_parallel_coordinates.png", "5-axis parallel-coordinates view colored by best_score (exploratory, non-causal)"),
    ("fig11_pairwise_interactions.png", "Curated pairwise views: LR×hidden_size and embedding×output dropout"),
    ("fig12_operations.png", "Runtime/GPU-hours vs hidden_size/batch_size; failures/retries shown operationally "
     "(valid trials only -- failed attempts get their own strip/table, never a fabricated score)"),
    ("fig13_boundary_evolution.png", "Checkpoint-to-checkpoint tier evolution and near-boundary occupancy, "
     "previous vs current (n/a marker at the first checkpoint)"),
    ("fig14_boundary_band_sensitivity.png", "Supplemental 5%/10%/20% boundary-band robustness check "
     "(canonical 10% rule unchanged)"),
    ("fig15_top_configurations.png", "Top Bayesian + top random-control configurations, all 5 axes + derived diagnostics"),
]


def render_checkpoint_packet(output_dir: Path, *, trial_df: pd.DataFrame, trajectory_df: pd.DataFrame,
                              proposal_df: pd.DataFrame, operations_df: pd.DataFrame, checkpoint_label: str,
                              checkpoint_valid_bayesian_count: int, random_control_count: int = 12,
                              synthetic: bool = True, previous_boundary_df: pd.DataFrame | None = None,
                              review_name: str | None = None) -> dict[str, Any]:
    """Render one checkpoint's full review packet (decision board + 15 figures
    + derived data + README) into ``output_dir``.  ``trial_df`` /
    ``trajectory_df`` are the FULL campaign tables; this function performs
    the checkpoint slice itself via :mod:`sweep_v1_review_analysis`.

    ``previous_boundary_df`` is the immediately-preceding checkpoint's
    :func:`analysis.derive_boundary_pressure_table` output (``None`` at the
    first checkpoint) and drives the §10 checkpoint-evolution evidence on
    both the decision board and the standalone evolution figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_name = _review_name_for(checkpoint_label, review_name)

    trial_slice = analysis.checkpoint_slice(trial_df, checkpoint_valid_bayesian_count, random_control_count)
    ops_slice = analysis.checkpoint_operations_slice(trial_df, checkpoint_valid_bayesian_count, random_control_count)
    trial_ids = set(trial_slice["trial_id"])
    trajectory_slice = trajectory_df[trajectory_df["trial_id"].isin(trial_ids)]
    boundary_df = analysis.derive_boundary_pressure_table(trial_slice)
    occupancy_df = analysis.categorical_occupancy_table(trial_slice)
    evolution_df = analysis.boundary_pressure_evolution(boundary_df, previous_boundary_df)
    top_configs_df = analysis.top_configurations_table(trial_slice)
    sensitivity_df = _boundary_band_sensitivity_all_axes(trial_slice)

    generated: dict[str, str] = {}
    board_path = render_decision_board(output_dir, trial_slice=trial_slice, ops_slice=ops_slice,
                                       checkpoint_label=checkpoint_label,
                                       checkpoint_valid_bayesian_count=checkpoint_valid_bayesian_count,
                                       synthetic=synthetic, previous_boundary_df=previous_boundary_df,
                                       review_name=review_name)
    generated["decision_board"] = str(board_path)

    generated["fig01_search_progress"] = str(fig1_search_progress(
        output_dir, trial_slice, checkpoint_label, checkpoint_valid_bayesian_count, synthetic))
    generated["fig02_compute_efficiency"] = str(fig2_compute_efficiency(output_dir, trial_slice, synthetic))
    generated["fig03_objective_distribution"] = str(fig3_objective_distribution(output_dir, trial_slice, synthetic))
    generated["fig04_five_axis_response"] = str(fig4_hyperparameter_response(output_dir, trial_slice, synthetic))
    generated["fig05_boundary_occupancy"] = str(fig5_boundary_occupancy(output_dir, trial_slice, synthetic))
    generated["fig06_proposal_drift"] = str(fig6_proposal_drift(output_dir, trial_slice, synthetic))
    generated["fig07_representative_trajectories"] = str(fig7_representative_trajectories(
        output_dir, trial_slice, trajectory_slice, synthetic))
    generated["fig08_best_epoch_late_gain"] = str(fig8_best_epoch_late_gain(output_dir, trial_slice, synthetic))
    generated["fig09_best_vs_final"] = str(fig9_best_vs_final(output_dir, trial_slice, synthetic))
    generated["fig10_parallel_coordinates"] = str(fig10_parallel_coordinates(output_dir, trial_slice, synthetic))
    generated["fig11_pairwise_interactions"] = str(fig11_pairwise_interactions(output_dir, trial_slice, synthetic))
    generated["fig12_operations"] = str(fig12_operations(output_dir, ops_slice, synthetic))
    generated["fig13_boundary_evolution"] = str(fig13_boundary_evolution(
        output_dir, boundary_df, previous_boundary_df, synthetic))
    generated["fig14_boundary_band_sensitivity"] = str(fig14_boundary_band_sensitivity(
        output_dir, trial_slice, synthetic))
    generated["fig15_top_configurations"] = str(fig15_top_configurations(output_dir, trial_slice, synthetic))

    trial_slice.to_csv(output_dir / "derived_trial_slice.csv", index=False)
    ops_slice.to_csv(output_dir / "derived_operations_slice.csv", index=False)
    boundary_df.to_csv(output_dir / "derived_boundary_pressure_table.csv", index=False)
    occupancy_df.to_csv(output_dir / "derived_categorical_occupancy.csv", index=False)
    evolution_df.to_csv(output_dir / "derived_boundary_pressure_evolution.csv", index=False)
    top_configs_df.to_csv(output_dir / "derived_top_configurations.csv", index=False)
    sensitivity_df.to_csv(output_dir / "derived_boundary_band_sensitivity.csv", index=False)
    with open(output_dir / "derived_boundary_pressure_table.json", "w", encoding="utf-8") as fh:
        json.dump(boundary_df.to_dict(orient="records"), fh, indent=2)

    common_n_value = analysis.common_n(trial_slice)
    valid = analysis.valid_trials(trial_slice)
    n_bayesian = int((valid["search_arm"] == "bayesian").sum())
    n_random = int((valid["search_arm"] == "random_control").sum())
    n_failed = int((ops_slice["workflow_status"] != analysis.VALID_STATUS).sum())

    readme_lines = [f"# Sweep-v1 review packet — {review_name} ({checkpoint_label})", ""]
    if synthetic:
        readme_lines += [f"**{SYNTHETIC_BANNER}**", "",
                         "This packet was generated from a deterministic synthetic fixture for visualization-system",
                         "development only. No real Sweep-v1 trial, proposal, or W&B evidence is represented.", ""]
    readme_lines += [
        f"Checkpoint target: {checkpoint_valid_bayesian_count} valid Bayesian results "
        f"(actual: {n_bayesian}, allowing bounded overshoot).",
        f"Valid random-control results: {n_random}. Common-N (both arms): {common_n_value}.",
        f"Failed/incomplete attempts in this slice (excluded from scientific curves): {n_failed}.",
        f"Previous checkpoint available for evolution tracking: {'yes' if previous_boundary_df is not None else 'no (first checkpoint)'}.",
        "", "## Decision board (start here)", "",
        "- `decision_board.png` (+ .pdf) — reading hierarchy: WHERE ARE WE? -> VALUE-ADD? -> BOUNDARY PRESSURE? "
        "-> STRENGTHENING? -> Panel E boundary-evidence matrix (budget/failure counts are in the header)",
        "", "## Figures", "",
    ]
    for filename, purpose in _FIGURE_ORDER:
        readme_lines.append(f"- `{filename}` — {purpose}")
    readme_lines += ["", "## Derived data",
                     "- `derived_trial_slice.csv` — exact trial rows visible at this checkpoint",
                     "- `derived_operations_slice.csv` — trial rows plus failed/incomplete attempts in this window",
                     "- `derived_boundary_pressure_table.csv`/`.json` — per-axis, per-side boundary-pressure evidence "
                     "(pooled + Bayesian-only + random-control-only occupancy, drift effect size, interpretation)",
                     "- `derived_boundary_pressure_evolution.csv` — previous-vs-current tier per axis/side",
                     "- `derived_boundary_band_sensitivity.csv` — 5%/10%/20% band robustness check (supplemental)",
                     "- `derived_top_configurations.csv` — top Bayesian + top random-control configurations",
                     "- `derived_categorical_occupancy.csv` — hidden_size/batch_size occupancy counts",
                     "", "This packet requires no W&B access to interpret."]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    generated["readme"] = str(output_dir / "README.md")
    generated["derived_trial_slice_csv"] = str(output_dir / "derived_trial_slice.csv")
    generated["derived_operations_slice_csv"] = str(output_dir / "derived_operations_slice.csv")
    generated["derived_boundary_pressure_csv"] = str(output_dir / "derived_boundary_pressure_table.csv")
    generated["derived_boundary_pressure_json"] = str(output_dir / "derived_boundary_pressure_table.json")
    generated["derived_boundary_pressure_evolution_csv"] = str(output_dir / "derived_boundary_pressure_evolution.csv")
    generated["derived_boundary_band_sensitivity_csv"] = str(output_dir / "derived_boundary_band_sensitivity.csv")
    generated["derived_top_configurations_csv"] = str(output_dir / "derived_top_configurations.csv")
    generated["derived_categorical_occupancy_csv"] = str(output_dir / "derived_categorical_occupancy.csv")
    return generated
