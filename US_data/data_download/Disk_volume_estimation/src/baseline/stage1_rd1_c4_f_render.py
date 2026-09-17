"""RD1-C4-F descriptive scientific-review packet renderer.

Reads the already-produced, hash-verified RD1-C4-E formal-results evidence
bundle (job 46183269, commit 11e79fa) plus the frozen 24-configuration
six-axis roster, and renders a compact set of tables and figures into an
output directory (normally under ``.scratch_local/``, which stays untracked
per repository policy).

This is a synthesis/visualization tool only:
  * no new scientific computation beyond descriptive aggregation
    (medians/IQR/ECDFs/counts) of columns already present in the qualified
    evidence bundle;
  * no classifier, winner, promotion, or tolerance decision;
  * the 9,600 trial x basin cells are never pooled into a falsely
    independent comparison -- ECDFs/points are always per-configuration
    (24 configurations = 24 search-level units) or per-basin (400 basins),
    never both axes collapsed together.

Usage:
    python -m src.baseline.stage1_rd1_c4_f_render \
        --evidence-dir .scratch_local/rd1_c4_e_rerun_46183269_evidence \
        --roster .scratch_local/rd1_supervisor_meeting_c2/tables/output_A_configuration_table_24row.csv \
        --out-dir .scratch_local/rd1_c4_f_review_packet_v001
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from src.baseline.stage1_rd1_c4_f_synthesis import (
    RAW_SIX_AXES,
    basin_arm_medians,
    build_configuration_table,
    select_percentile_basins,
    sign_counts,
    validate_canonical_basin_ids,
    validate_roster_against_trial_ids,
)

ARMS = ("bayesian", "random_control")
ARM_COLORS = {"bayesian": "#1f77b4", "random_control": "#ff7f0e"}
PROPOSAL_ORDERS = list(range(1, 13))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ecdf(ax, values, color, label=None, linewidth=1.2, alpha=1.0):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label)


def load_bundle(evidence_dir: Path) -> dict:
    # basin_id is a fixed-width 8-character USGS identifier (e.g. "01464000") and
    # must never be parsed as numeric -- pandas' default dtype inference would
    # silently strip leading zeros. dtype=str preserves the canonical string as
    # written by the RD1-C4-E producer.
    per_basin = pd.read_csv(evidence_dir / "per_basin_metrics.csv", dtype={"basin_id": str})
    q98 = pd.read_csv(evidence_dir / "q98_diagnostics.csv", dtype={"basin_id": str})
    prov = pd.read_csv(evidence_dir / "provenance_audit.csv", dtype={"basin_id": str})
    bds = pd.read_csv(evidence_dir / "basin_distribution_summary.csv")
    cq98 = pd.read_csv(evidence_dir / "canonical_q98_facts.csv", dtype={"basin_id": str})
    review_identity = json.loads((evidence_dir / "review_identity.json").read_text())
    for name, df in (("per_basin", per_basin), ("q98", q98), ("prov", prov), ("cq98", cq98)):
        validate_canonical_basin_ids(df["basin_id"], context=name)
    return dict(
        per_basin=per_basin, q98=q98, prov=prov, bds=bds, cq98=cq98,
        review_identity=review_identity,
    )


def _run_by_run_progress_figure(per_basin, config_table, metric, trajectory_col, trajectory_label,
                                  show_objective_as_context, title, out_path):
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.25, 1.6], hspace=0.55, wspace=0.28)
    cmap = matplotlib.colormaps.get_cmap("viridis").resampled(12)
    norm = matplotlib.colors.Normalize(vmin=1, vmax=12)

    for col, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, col])
        arm_pb = per_basin[per_basin.search_arm == arm]
        for order in PROPOSAL_ORDERS:
            trial = arm_pb[arm_pb.proposal_order == order]
            if trial.empty:
                continue
            _ecdf(ax, trial[metric].to_numpy(), cmap(norm(order)))
        ax.set_xlim(-1, 1)
        ax.set_title(f"{arm}\nper-configuration {metric.upper()} ECDF (400 basins x 12 configs)", fontsize=10)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.3)

    cax = fig.add_subplot(gs[1, :])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=PROPOSAL_ORDERS)
    cbar.set_label("proposal order")

    ax = fig.add_subplot(gs[2, :])
    for arm in ARMS:
        arm_tbl = config_table[config_table.search_arm == arm].sort_values("proposal_order")
        ax.plot(arm_tbl.proposal_order, arm_tbl[trajectory_col], "o-", color=ARM_COLORS[arm],
                 label=f"{arm} {trajectory_label}")
        objective_label = f"{arm} best-so-far (official objective)" if not show_objective_as_context \
            else f"{arm} best-so-far (official objective, context)"
        ax.plot(arm_tbl.proposal_order, arm_tbl.cumulative_best, "--", color=ARM_COLORS[arm], alpha=0.6,
                 label=objective_label)
    ax.set_xlabel("proposal order")
    ax.set_ylabel("value")
    ax.set_title(f"Per-proposal-order {trajectory_label} and best-so-far incumbent objective", fontsize=10)
    ax.set_xticks(PROPOSAL_ORDERS)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(title, y=0.99)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig1_nse_progress(per_basin: pd.DataFrame, config_table: pd.DataFrame, out_path: Path) -> None:
    _run_by_run_progress_figure(
        per_basin, config_table, metric="nse", trajectory_col="median_nse",
        trajectory_label="median NSE", show_objective_as_context=False,
        title="RD1-C4-F Fig.1 -- run-by-run NSE progress (job 46183269, 24 configs x 400 basins)",
        out_path=out_path,
    )


def fig2_kge_progress(per_basin: pd.DataFrame, config_table: pd.DataFrame, out_path: Path) -> None:
    _run_by_run_progress_figure(
        per_basin, config_table, metric="kge", trajectory_col="median_kge",
        trajectory_label="median KGE", show_objective_as_context=True,
        title="RD1-C4-F Fig.2 -- run-by-run KGE progress (job 46183269, 24 configs x 400 basins)",
        out_path=out_path,
    )


def fig3_config_level_arm_comparison(config_table: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    metrics = [
        ("official_objective", "official objective"),
        ("median_nse", "median NSE (per config)"),
        ("median_kge", "median KGE (per config)"),
        ("median_q98_normalized_rmse", "median Q98-norm. RMSE"),
        ("median_relative_volume_bias", "median rel. volume bias"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    summary_rows = []
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, metrics):
        for i, arm in enumerate(ARMS):
            vals = config_table.loc[config_table.search_arm == arm, col].to_numpy()
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=ARM_COLORS[arm], s=28, alpha=0.85)
            median = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            ax.plot([i - 0.15, i + 0.15], [median, median], color=ARM_COLORS[arm], linewidth=2.2)
            ax.plot([i, i], [q25, q75], color=ARM_COLORS[arm], linewidth=1, alpha=0.6)
            summary_rows.append(
                {"metric": col, "search_arm": arm, "n": len(vals), "median": median,
                 "q25": q25, "q75": q75, "min": float(vals.min()), "max": float(vals.max())}
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(ARMS, rotation=15)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("RD1-C4-F Fig.3 -- configuration-level arm comparison (12 points/arm; descriptive only, no p-values)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def fig4_basin_heterogeneity(per_basin: pd.DataFrame, out_path: Path) -> dict:
    nse_medians = basin_arm_medians(per_basin, "nse")
    kge_medians = basin_arm_medians(per_basin, "kge")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    focus_bound = {"NSE": 2.0, "KGE": 2.0}
    for ax, medians, label in zip(axes, (nse_medians, kge_medians), ("NSE", "KGE")):
        diff = medians["diff_bayesian_minus_random"]
        bound = focus_bound[label]
        in_range = diff[diff.abs() <= bound]
        n_out = int((diff.abs() > bound).sum())
        ax.hist(in_range, bins=40, color="#4c72b0", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        counts = sign_counts(diff)
        ax.set_title(
            f"Basin-level median {label} difference\n(Bayesian - random_control, across each arm's 12 configs)\n"
            f"n_positive={counts['n_positive']}, n_negative={counts['n_negative']}, n_zero={counts['n_zero']}\n"
            f"focus range [-{bound},{bound}]; n_out_of_range={n_out}/{len(diff)} "
            f"(full range [{diff.min():.3g}, {diff.max():.3g}])",
            fontsize=8.5,
        )
        ax.set_xlabel(f"median-{label} difference, focus range (descriptive, across-basin, not independent replication)", fontsize=8)
        ax.set_ylabel("basin count")
        ax.grid(alpha=0.3)
    fig.suptitle("RD1-C4-F Fig.4 -- basin-level heterogeneity (400 basins; descriptive across-basin comparison)")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    def _tail_stats(diff, bound):
        return {
            "sign_counts": sign_counts(diff),
            "focus_bound": bound,
            "n_out_of_focus_range": int((diff.abs() > bound).sum()),
            "min": float(diff.min()),
            "max": float(diff.max()),
        }

    return {
        "nse": _tail_stats(nse_medians["diff_bayesian_minus_random"], focus_bound["NSE"]),
        "kge": _tail_stats(kge_medians["diff_bayesian_minus_random"], focus_bound["KGE"]),
    }, nse_medians, kge_medians


def fig5_q98_tail_safe(q98: pd.DataFrame, config_table: pd.DataFrame, out_path: Path, focus_max: float = 2.0) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    rng = np.random.default_rng(1)
    for i, arm in enumerate(ARMS):
        vals = config_table.loc[config_table.search_arm == arm, "median_q98_normalized_rmse"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=ARM_COLORS[arm], s=28)
    ax.set_xticks([0, 1]); ax.set_xticklabels(ARMS)
    ax.set_title("Configuration-level median Q98-norm. RMSE (12 pts/arm)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    vals = q98["q98_normalized_rmse"].dropna()
    in_range = vals[vals <= focus_max]
    n_out = int((vals > focus_max).sum())
    ax.hist(in_range, bins=60, color="#55a868", alpha=0.85)
    ax.set_title(
        f"Per-cell Q98-norm. RMSE, focus range [0, {focus_max}]\n"
        f"n_out_of_range(> {focus_max}) = {n_out}/{len(vals)} "
        f"(max={vals.max():.4g})",
        fontsize=9,
    )
    ax.set_xlabel("q98_normalized_rmse (focus range only; full-range tail reported in table, not clipped from data)", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i, arm in enumerate(ARMS):
        vals = config_table.loc[config_table.search_arm == arm, "median_relative_volume_bias"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=ARM_COLORS[arm], s=28)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(ARMS)
    ax.set_title("Configuration-level median relative volume bias\n(positive = overestimate, negative = underestimate)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    for i, arm in enumerate(ARMS):
        vals = config_table.loc[config_table.search_arm == arm, "median_observed_peak_time_magnitude_error"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=ARM_COLORS[arm], s=28)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(ARMS)
    ax.set_title("Configuration-level median observed peak-time/magnitude error\n(sign per qualified metric definition)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("RD1-C4-F Fig.5 -- Q98 diagnostics, readable under long tails (no silent clipping of underlying data)")
    fig.tight_layout(rect=(0, 0, 0.99, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    tail_rows = []
    for metric in ("q98_normalized_rmse", "relative_volume_bias", "observed_peak_time_magnitude_error"):
        v = q98[metric].dropna()
        tail_rows.append({
            "metric": metric, "n": len(v), "min": float(v.min()), "max": float(v.max()),
            "p99": float(np.percentile(v, 99)), "p999": float(np.percentile(v, 99.9)),
        })
    return pd.DataFrame(tail_rows)


def fig6_config_space_mapping(config_table: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, axis_col in zip(axes.flat, RAW_SIX_AXES):
        for arm in ARMS:
            sub = config_table[config_table.search_arm == arm]
            x = pd.to_numeric(sub[axis_col], errors="coerce")
            ax.scatter(x, sub["official_objective"], color=ARM_COLORS[arm], label=arm, s=32, alpha=0.85)
        axis_name = axis_col.replace("raw_", "")
        ax.set_xlabel(axis_name)
        if axis_name == "learning_rate":
            ax.set_xscale("log")
        ax.set_ylabel("official objective")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "RD1-C4-F Fig.6 -- frozen six-axis configuration-space mapping vs official objective\n"
        "(24 explored configurations; descriptive only, no causal/effect inference, no space expansion)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig7_provenance_appendix(prov: pd.DataFrame, review_identity: dict, out_path: Path) -> pd.DataFrame:
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    for arm in ARMS:
        vals = prov.loc[prov.search_arm == arm, "max_abs_diff_m3s"]
        vals = vals[vals > 0]
        if len(vals):
            _ecdf(ax, np.log10(vals.to_numpy()), ARM_COLORS[arm], label=arm)
    ax.axvline(np.log10(1e-6), color="gray", linestyle="--", linewidth=1, label="atol=1e-6 m^3/s (report-only)")
    ax.set_xlabel("log10(max abs diff, m^3/s) [package-canonical vs admitted, per cell]")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-cell provenance max-abs-diff\n(PROVISIONAL, report-only -- NOT a pass/fail threshold)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    summary = prov.groupby("search_arm").agg(
        n_cells=("basin_id", "count"),
        n_cells_exceeding=("envelope_exceeded", "sum"),
        n_elements_compared=("n_compared", "sum"),
        n_elements_exceeding=("n_exceeding", "sum"),
        max_abs_diff_m3s=("max_abs_diff_m3s", "max"),
    )
    ps = review_identity["provenance_audit_summary"]
    lines = [
        "Aggregate provenance-audit summary (REPORT-ONLY provenance metadata)",
        f"total: n_cells={ps['n_cells']}, n_cells_with_exceedance={ps['n_cells_with_exceedance']}, "
        f"n_elements_compared={ps['n_elements_compared']}, n_elements_exceeding={ps['n_elements_exceeding']}",
        f"worst_cell: basin={ps['worst_cell']['basin_id']}, "
        f"max_abs_diff_m3s={ps['worst_cell']['max_abs_diff_m3s']:.6g}",
        "",
        "Does NOT alter official NSE/KGE or canonical Q98 paths.",
        "No new tolerance or exclusion is proposed here.",
        "",
    ]
    for arm, row in summary.iterrows():
        lines.append(f"{arm}:")
        lines.append(f"  n_cells={int(row.n_cells)}  n_cells_exceeding={int(row.n_cells_exceeding)}")
        lines.append(f"  n_elements_compared={int(row.n_elements_compared)}  n_elements_exceeding={int(row.n_elements_exceeding)}")
        lines.append(f"  max_abs_diff_m3s(extremum)={row.max_abs_diff_m3s:.6g}")
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=9.5, family="monospace", va="top", ha="left", transform=ax.transAxes)

    fig.suptitle("RD1-C4-F Fig.7 -- provenance appendix (job 46183269, commit 11e79fa)")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.evidence_dir)
    roster = pd.read_csv(args.roster)
    validate_roster_against_trial_ids(roster, bundle["review_identity"]["trial_ids"])

    config_table = build_configuration_table(bundle["per_basin"], bundle["q98"], roster)
    # bring in the roster's own cumulative_best / is_new_incumbent (already-qualified
    # RD1-C2 analysis-core output) as the best-so-far incumbent trajectory.
    roster_extra = roster.set_index("proposal_id")[["cumulative_best", "is_new_incumbent"]]
    config_table = config_table.join(roster_extra, on="trial_id")
    config_table.to_csv(args.out_dir / "tables" / "configuration_level_table.csv", index=False)

    fig1_nse_progress(bundle["per_basin"], config_table, args.out_dir / "figures" / "fig1_nse_run_by_run_progress.png")
    fig2_kge_progress(bundle["per_basin"], config_table, args.out_dir / "figures" / "fig2_kge_run_by_run_progress.png")

    arm_summary = fig3_config_level_arm_comparison(config_table, args.out_dir / "figures" / "fig3_config_level_arm_comparison.png")
    arm_summary.to_csv(args.out_dir / "tables" / "arm_level_metric_summary.csv", index=False)

    het_summary, nse_medians, kge_medians = fig4_basin_heterogeneity(bundle["per_basin"], args.out_dir / "figures" / "fig4_basin_level_heterogeneity.png")
    validate_canonical_basin_ids(nse_medians.index, context="basin_level_nse_medians_and_diff")
    validate_canonical_basin_ids(kge_medians.index, context="basin_level_kge_medians_and_diff")
    nse_medians.to_csv(args.out_dir / "tables" / "basin_level_nse_medians_and_diff.csv")
    kge_medians.to_csv(args.out_dir / "tables" / "basin_level_kge_medians_and_diff.csv")
    (args.out_dir / "tables" / "basin_heterogeneity_sign_counts.json").write_text(json.dumps(het_summary, indent=2))

    tail_table = fig5_q98_tail_safe(bundle["q98"], config_table, args.out_dir / "figures" / "fig5_q98_diagnostics_tail_safe.png")
    tail_table.to_csv(args.out_dir / "tables" / "q98_tail_diagnostics.csv", index=False)

    fig6_config_space_mapping(config_table, args.out_dir / "figures" / "fig6_frozen_configuration_space_mapping.png")

    prov_summary = fig7_provenance_appendix(bundle["prov"], bundle["review_identity"], args.out_dir / "figures" / "fig7_provenance_appendix.png")
    prov_summary.to_csv(args.out_dir / "tables" / "provenance_appendix_summary.csv", index=False)

    # selection manifest for the hydrograph supplement (basin IDs only here;
    # series extraction is a separate step against frozen stored predictions)
    diff_selection = select_percentile_basins(nse_medians["diff_bayesian_minus_random"], percentiles=(10, 50, 90))
    validate_canonical_basin_ids(diff_selection.values(), context="hydrograph_basin_selection_manifest")
    selection_manifest = {
        "rule": "closest basin to empirical 10th/50th/90th percentile of "
                "(median-NSE-bayesian - median-NSE-random_control) across 400 basins; "
                "ties broken by lexicographically smallest basin_id",
        "source_table": "basin_level_nse_medians_and_diff.csv",
        "selected_basins": {str(p): b for p, b in diff_selection.items()},
    }
    (args.out_dir / "tables" / "hydrograph_basin_selection_manifest.json").write_text(json.dumps(selection_manifest, indent=2))

    # receipts
    receipts = {}
    for f in sorted((args.out_dir / "tables").glob("*")) + sorted((args.out_dir / "figures").glob("*")):
        receipts[str(f.relative_to(args.out_dir))] = _sha256(f)
    receipts["_input_evidence_dir"] = str(args.evidence_dir)
    receipts["_input_roster"] = str(args.roster)
    receipts["_input_roster_sha256"] = _sha256(args.roster)
    (args.out_dir / "receipts.json").write_text(json.dumps(receipts, indent=2))

    print("selection manifest:", selection_manifest["selected_basins"])
    print("wrote outputs to", args.out_dir)


if __name__ == "__main__":
    main()
