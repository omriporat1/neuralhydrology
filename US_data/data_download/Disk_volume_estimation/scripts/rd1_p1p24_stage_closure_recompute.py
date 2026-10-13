"""Independent recomputation of the decisive RD1 P1-P24/R1-R12 dashboard-v2 claims.

Closure audit for the RD1 P1-P24 / R1-R12 development-screening evidence stage
(see docs/rd1_p1p24_r1r12_evidence_stage_closure.md). Reads the hash-verified
standardized long table of the retained dashboard-v2 bundle, the upstream
per-basin evidence it was built from, and the Session A v3 recovery outputs.
It deliberately does not import the dashboard's own build scripts, so every
check here is a second, independent implementation of the same definitions.

Read-only with respect to all evidence; writes one JSON result file. Exits 1 if
any consistency check fails. Development-screening scope only (Common-120
support, 400 basins, Seed A, mf12x50000); no sealed scope is read.

Usage:
    python scripts/rd1_p1p24_stage_closure_recompute.py [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch_local"
DASH = SCRATCH / "rd1_candidate_dashboard_v2_20261008T093605Z"
RECOVERY = SCRATCH / "rd1_p17p19_recovery_eval_20261008T071622Z"
DEFAULT_OUT = SCRATCH / "rd1_p1p24_stage_closure_audit_20261013" / "closure_recompute_results.json"

LABELS = [f"P{i}" for i in range(1, 25)] + [f"R{i}" for i in range(1, 13)]
CONTINUATION = [f"P{i}" for i in range(13, 25)]
FLOOD = ["q98_normalized_rmse", "abs_relative_volume_bias", "abs_observed_peak_time_magnitude_error"]
MAIN = ["nse", "kge"] + FLOOD
HIGHER_IS_BETTER = {"nse": True, "kge": True, "q98_normalized_rmse": False,
                    "abs_relative_volume_bias": False, "abs_observed_peak_time_magnitude_error": False}
SIX = ["P18", "P9", "P15", "P19", "P22", "P23"]
RECOVERED = {"P17": 4, "P18": 5, "P19": 7}
TIE_ATOL = 1e-12
CSV_ROUNDTRIP_ATOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pareto_front(med: pd.DataFrame, labels: list[str], flood_metric: str) -> list[str]:
    """Non-dominated set on (median NSE higher-better, median flood metric lower-better)."""
    pts = {lab: (med.loc[lab, "nse"], med.loc[lab, flood_metric]) for lab in labels}
    front = []
    for i, (xi, yi) in pts.items():
        dominated = any(xj >= xi and yj <= yi and (xj != xi or yj != yi)
                        for j, (xj, yj) in pts.items() if j != i)
        if not dominated:
            front.append(i)
    return sorted(front, key=LABELS.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    res: dict = {"checks": {}}
    failures: list[str] = []

    def check(name: str, cond: bool, detail=None) -> None:
        res["checks"][name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            failures.append(name)

    long_path = DASH / "tables/standardized_results_long.csv"
    res["inputs"] = {"standardized_results_long_sha256": sha256(long_path),
                     "dashboard_manifest_sha256": sha256(DASH / "09_manifest.json"),
                     "recovery_v3_manifest_sha256": sha256(RECOVERY / "11_final_handoff_manifest_v3.json")}
    df = pd.read_csv(long_path, dtype={"basin_id": str})

    def piv(metric: str) -> pd.DataFrame:
        return df.pivot(index="basin_id", columns="label", values=metric)[LABELS]

    # --- A. population identity -------------------------------------------
    counts = df.groupby("label").size()
    check("A1_rows_14400", len(df) == 14400, len(df))
    check("A2_label_set_P1_24_R1_12", sorted(counts.index) == sorted(LABELS))
    check("A3_400_basins_per_label", bool((counts == 400).all()))
    check("A4_identical_basin_set", df.groupby("label")["basin_id"].apply(frozenset).nunique() == 1)
    check("A5_no_duplicate_label_basin", not df.duplicated(["label", "basin_id"]).any())
    check("A6_all_main_metrics_finite", bool(np.isfinite(df[MAIN].to_numpy()).all()))
    obj_gap = (df.groupby("label")["nse"].median() - df.groupby("label")["official_objective"].first()).abs()
    check("A7_official_objective_equals_median_nse", float(obj_gap.max()) < 1e-12, float(obj_gap.max()))
    res["objective_by_label"] = df.groupby("label")["official_objective"].first().reindex(LABELS).to_dict()
    res["best_epoch_by_label"] = df.groupby("label")["best_epoch"].first().reindex(LABELS).astype(int).to_dict()
    res["source_tag_by_label"] = df.groupby("label")["source_tag"].first().reindex(LABELS).to_dict()

    # --- B. per-row upstream source hashes still match the retained files --
    upstream = []
    groups = df.groupby(["source_evidence_dir", "source_per_basin_metrics_sha256",
                         "source_q98_diagnostics_sha256"]).size().reset_index(name="rows")
    for _, row in groups.iterrows():
        d = ROOT / row["source_evidence_dir"]
        if (d / "per_basin_metrics.csv").exists():
            hashes = {"per_basin_metrics.csv": sha256(d / "per_basin_metrics.csv"),
                      "q98_diagnostics.csv": sha256(d / "q98_diagnostics.csv")}
            ok = (hashes["per_basin_metrics.csv"] == row["source_per_basin_metrics_sha256"]
                  and hashes["q98_diagnostics.csv"] == row["source_q98_diagnostics_sha256"])
        else:
            hashes = {f.name: sha256(f) for f in sorted(d.glob("P*_recovered_evaluation.json"))}
            ok = row["source_per_basin_metrics_sha256"] in hashes.values()
        upstream.append({"dir": row["source_evidence_dir"], "rows": int(row["rows"]), "ok": ok, "sha256": hashes})
    res["upstream_sources"] = upstream
    check("B1_upstream_source_hashes_match", all(u["ok"] for u in upstream))

    # --- C. P17-P19 rows vs Session A v3 authoritative recovery ------------
    v3 = json.loads((RECOVERY / "11_final_handoff_manifest_v3.json").read_text())
    for lab, epoch in RECOVERED.items():
        rec = json.loads((RECOVERY / "09_recovered_evaluation_outputs_v2_q98fix"
                          / f"{lab}_recovered_evaluation.json").read_text())
        jb = pd.DataFrame(rec["per_basin"]).astype({"basin_id": str}).set_index("basin_id")
        sub = df[df.label == lab].set_index("basin_id")
        jb = jb.loc[sub.index]
        maxdiff = {m: float((sub[m] - jb[m]).abs().max()) for m in
                   ["nse", "kge", "q98_normalized_rmse", "relative_volume_bias",
                    "observed_peak_time_magnitude_error"]}
        trial = v3["trials"][lab]
        q98 = trial["q98_high_flow_diagnostics_400_basins"]
        res.setdefault("recovered_trials", {})[lab] = {
            "best_epoch": trial["best_epoch"], "n_basins": trial["n_basins_evaluated"],
            "n_q98_evaluated": q98["n_q98_basins_evaluated"], "n_q98_failed": q98["n_q98_basins_failed"],
            "objective_recorded": trial["objective_agreement"]["recorded"],
            "objective_exact_match": trial["objective_agreement"]["exact_match"],
            "per_basin_max_abs_diff_long_table_vs_json": maxdiff}
        check(f"C_{lab}_best_epoch_{epoch}", int(sub["best_epoch"].iloc[0]) == epoch == trial["best_epoch"])
        check(f"C_{lab}_400_basins_400_q98_0_failed", trial["n_basins_evaluated"] == 400
              and q98["n_q98_basins_evaluated"] == 400 and q98["n_q98_basins_failed"] == 0)
        check(f"C_{lab}_objective_exact_match", trial["objective_agreement"]["exact_match"] is True)
        check(f"C_{lab}_per_basin_values_match_json", max(maxdiff.values()) < CSV_ROUNDTRIP_ATOL, maxdiff)
        check(f"C_{lab}_q98_median_matches_v3",
              abs(float(sub["q98_normalized_rmse"].median()) - q98["q98_normalized_rmse_median"]) < 1e-12)

    # --- D. medians ------------------------------------------------------------
    med = df.groupby("label")[MAIN + ["relative_volume_bias", "observed_peak_time_magnitude_error"]].median()
    res["medians"] = {lab: {k: float(v) for k, v in med.loc[lab].items()} for lab in LABELS}

    # --- E. win credit (per-basin best; ties share credit) -------------------
    win = {}
    for m in MAIN:
        arr = piv(m).to_numpy()
        best = arr.max(1, keepdims=True) if HIGHER_IS_BETTER[m] else arr.min(1, keepdims=True)
        is_best = np.isclose(arr, best, rtol=0, atol=TIE_ATOL)
        win[m] = dict(zip(LABELS, ((is_best / is_best.sum(1, keepdims=True)).sum(0) / 4.0).tolist()))
    stored = json.loads((DASH / "tables/win_credit_matrix.json").read_text())
    gap = max(abs(win[m][lab] - stored[m]["win_pct"][lab]) for m in MAIN for lab in LABELS)
    check("E1_win_credit_matches_dashboard_table", gap < 1e-9, gap)
    res["win_pct"] = win
    res["win_pct_lowest_of_six"] = {m: min(SIX, key=lambda lab: win[m][lab]) for m in MAIN}
    res["win_pct_rank_P18_of_36"] = {m: sorted(LABELS, key=lambda lab: -win[m][lab]).index("P18") + 1 for m in MAIN}
    res["win_pct_P19_minus_P9"] = {m: win[m]["P19"] - win[m]["P9"] for m in MAIN}

    # --- F. post-hoc exclusion sensitivities (primary population stays 400) -
    pn = piv("nse")
    all_neg = set(pn.index[(pn < 0).all(axis=1)])
    any_neg = set(pn.index[(pn < 0).any(axis=1)])
    subsets = {"primary_400": set(pn.index),
               "excl_all_candidates_negative_nse": set(pn.index) - all_neg,
               "excl_any_candidate_negative_nse": set(pn.index) - any_neg}
    sens = {}
    for name, basins in subsets.items():
        sub_med = df[df.basin_id.isin(basins)].groupby("label")[MAIN].median()
        sens[name] = {"n_basins": len(basins)}
        for m in MAIN:
            order = sub_med[m].sort_values(ascending=not HIGHER_IS_BETTER[m]).index.tolist()
            sens[name][m] = {"top5": order[:5], "rank_P18": order.index("P18") + 1,
                             "rank_P19": order.index("P19") + 1}
    res["sensitivity"] = sens
    stored_sens = json.loads((DASH / "tables/exclusion_sensitivity_analysis.json").read_text())
    for m in MAIN:
        c = stored_sens["top5_rank_comparison_per_metric"][m]
        check(f"F1_{m}_top5_matches_dashboard_table",
              c["top5_primary"] == sens["primary_400"][m]["top5"]
              and c["top5_sensitivity_excl29"] == sens["excl_all_candidates_negative_nse"][m]["top5"]
              and c["top5_sensitivity_excl129"] == sens["excl_any_candidate_negative_nse"][m]["top5"])

    # --- G. same-basin joint improvement vs P9 --------------------------------
    d_nse = pn.sub(pn["P9"], axis=0)
    joint = {}
    for fm in FLOOD:
        pf = piv(fm)
        imp = pf.rsub(pf["P9"], axis=0)  # positive = lower flood error than P9
        joint[fm] = {lab: int(((d_nse[lab] > 0) & (imp[lab] > 0)).sum()) for lab in LABELS if lab != "P9"}
    res["joint_improvement_vs_P9"] = joint
    res["joint_improvement_max"] = {fm: max(joint[fm].items(), key=lambda kv: kv[1]) for fm in FLOOD}
    stored_joint = json.loads((DASH / "tables/joint_improvement_quadrants_vs_P9.json").read_text())
    check("G1_joint_counts_match_dashboard_table",
          all(stored_joint[fm][lab]["both_improve"] == joint[fm][lab] for fm in FLOOD for lab in joint[fm]))

    # --- H. two-metric Pareto fronts on medians -------------------------------
    pareto = {}
    for fm in FLOOD:
        before = pareto_front(med, LABELS[:12], fm)
        after24 = pareto_front(med, LABELS[:24], fm)
        after36 = pareto_front(med, LABELS, fm)
        pareto[fm] = {"before_P1_12": before, "after_P1_24": after24, "after_all_36": after36,
                      "added_vs_before": [x for x in after24 if x not in before],
                      "displaced_vs_before": [x for x in before if x not in after24]}
        check(f"H1_{fm}_R_labels_do_not_change_front", after24 == after36)
    res["pareto"] = pareto
    nse_max = med["nse"].idxmax()
    res["nse_max_label"] = nse_max
    res["pareto_added_union_excluding_nse_max"] = sorted(
        {x for fm in FLOOD for x in pareto[fm]["added_vs_before"] if x != nse_max}, key=LABELS.index)

    # --- I. shared-failure categories, 33- vs 36-candidate population ---------
    pp = piv("observed_peak_time_magnitude_error")
    under_all = set(pp.index[(pp < 0).all(axis=1)])
    l33 = [lab for lab in LABELS if lab not in RECOVERED]
    all_neg33 = set(pn.index[(pn[l33] < 0).all(axis=1)])
    any_neg33 = set(pn.index[(pn[l33] < 0).any(axis=1)])
    res["shared_failure"] = {
        "n36": {"all_negative_nse": len(all_neg), "any_negative_nse": len(any_neg),
                "all_underpredict_at_observed_peak_time": len(under_all)},
        "n33": {"all_negative_nse": len(all_neg33), "any_negative_nse": len(any_neg33),
                "all_underpredict_at_observed_peak_time": len(set(pp.index[(pp[l33] < 0).all(axis=1)]))},
        "left_all_negative_set": {b: [lab for lab in LABELS if pn.loc[b, lab] >= 0]
                                  for b in sorted(all_neg33 - all_neg)},
        "joined_any_negative_set": {b: [lab for lab in RECOVERED if pn.loc[b, lab] < 0]
                                    for b in sorted(any_neg - any_neg33)},
        "nse_std_median_within_all_negative": float(pn.loc[sorted(all_neg)].std(axis=1).median()),
        "nse_std_median_elsewhere": float(pn.drop(index=sorted(all_neg)).std(axis=1).median()),
        "n_basins_with_1_or_2_candidates_nse_ge_0": int((pn.ge(0).sum(axis=1).between(1, 2)).sum()),
        "underpredict_overlap": {"with_all_negative": len(under_all & all_neg),
                                 "with_any_negative": len(under_all & any_neg)},
    }
    stored_sf = json.loads((DASH / "tables/shared_failure_categories.json").read_text())["reproduced_counts"]
    check("I1_shared_failure_counts_match_dashboard_table",
          (stored_sf["n_basins_negative_nse_every_candidate"],
           stored_sf["n_basins_negative_nse_at_least_one_candidate"],
           stored_sf["n_basins_every_candidate_underpredicts_at_observed_peak_time"])
          == (len(all_neg), len(any_neg), len(under_all)))

    # --- J. basin-area association (fig10 caption) -----------------------------
    area = df[df.label == "P1"].set_index("basin_id")["area_km2"].loc[pn.index]
    res["pearson_r_log_area_vs_median_nse"] = float(np.corrcoef(np.log(area), pn.median(axis=1))[0, 1])

    res["failures"] = failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {args.out}")
    print(f"checks: {len(res['checks']) - len(failures)}/{len(res['checks'])} pass")
    for name in failures:
        print(f"FAIL {name}: {res['checks'][name]['detail']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
