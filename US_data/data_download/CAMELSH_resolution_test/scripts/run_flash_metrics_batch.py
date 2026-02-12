# scripts/run_flash_metrics_batch.py
from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd

from src.data_usgs import fetch_usgs_iv_discharge
from src.aggregation import to_hourly_candidates
from src.events import detect_pot_events
from src.flash_metrics import event_peak_comparison


def summarize_events(df: pd.DataFrame) -> dict[str, float]:
    """Summarize per-event comparison table into a few robust stats."""
    if df.empty:
        return {
            "recall": np.nan,
            "med_rel_err": np.nan,
            "p90_rel_err": np.nan,
            "p10_rel_err": np.nan,
            "med_dt_min": np.nan,
            "p90_abs_dt_min": np.nan,
        }

    detected = df["detected"].astype(bool)
    recall = float(detected.mean())

    rel = df.loc[detected, "rel_peak_error"].dropna()
    dt = df.loc[detected, "dt_minutes"].dropna()

    out = {"recall": recall}

    if len(rel):
        out.update({
            "med_rel_err": float(rel.median()),
            "p90_rel_err": float(rel.quantile(0.90)),
            "p10_rel_err": float(rel.quantile(0.10)),
        })
    else:
        out.update({"med_rel_err": np.nan, "p90_rel_err": np.nan, "p10_rel_err": np.nan})

    if len(dt):
        out.update({
            "med_dt_min": float(dt.median()),
            "p90_abs_dt_min": float(dt.abs().quantile(0.90)),
        })
    else:
        out.update({"med_dt_min": np.nan, "p90_abs_dt_min": np.nan})

    return out


def main():
    year = 2016
    n_success_target = 20
    max_tries = 200
    seed = 123

    pot_quantile = 0.95
    min_separation = "12h"

    # hourly definitions
    nearest_tol = "10min"
    search_window = "1h"  # window to find hourly peak around hi-res peak time

    camelsh_dir = Path("data/raw/camelsh")
    files = sorted(camelsh_dir.glob("*_hourly.nc"))
    if not files:
        raise SystemExit(f"No *_hourly.nc files found in {camelsh_dir}")

    gauge_ids = [f.name.replace("_hourly.nc", "") for f in files]
    random.seed(seed)
    random.shuffle(gauge_ids)

    out_csv = Path(f"data/processed/flash_metrics_{year}_q{int(pot_quantile*100)}_n{n_success_target}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    start = f"{year-1}-12-31"
    end = f"{year+1}-01-02"

    rows = []
    n_success = 0
    n_fail = 0

    for i, gauge in enumerate(gauge_ids[:max_tries], 1):
        print(f"\n[{i}/{min(max_tries, len(gauge_ids))}] Gauge {gauge} (success={n_success}, fail={n_fail})")

        try:
            # Fetch IV
            q_iv = fetch_usgs_iv_discharge(gauge, start, end)

            # High-res series restricted to target year for event detection
            q_hi = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
            if len(q_hi) < 2000:
                raise RuntimeError("Too little IV data in target year after dropping NaNs")

            # Hourly candidates
            cands = to_hourly_candidates(q_iv, nearest_tolerance=nearest_tol)
            q_hourly_nearest = cands[f"nearest_{nearest_tol}"].loc[f"{year}-01-01":f"{year}-12-31 23:00:00"]
            q_hourly_mean = cands["mean"].loc[f"{year}-01-01":f"{year}-12-31 23:00:00"]

            # Detect POT events on high-res
            events = detect_pot_events(q_hi, quantile=pot_quantile, min_separation=min_separation)
            n_events = len(events)
            if n_events < 5:
                raise RuntimeError(f"Too few events detected (n={n_events})")

            # Compare event peaks
            comp_nearest = event_peak_comparison(q_hi, q_hourly_nearest, events, search_window=search_window)
            comp_mean = event_peak_comparison(q_hi, q_hourly_mean, events, search_window=search_window)

            s_near = summarize_events(comp_nearest)
            s_mean = summarize_events(comp_mean)

            # Diagnostics: IV cadence
            diffs = q_iv.index.to_series().diff().dropna()
            dt_mode_min = float(diffs.mode().iloc[0].total_seconds() / 60.0) if len(diffs) and len(diffs.mode()) else np.nan

            rows.append({
                "gauge": str(gauge),
                "year": year,
                "pot_quantile": pot_quantile,
                "min_separation": min_separation,
                "nearest_tol": nearest_tol,
                "search_window": search_window,
                "n_events": n_events,
                "iv_dt_mode_min": dt_mode_min,

                # nearest hourly
                "recall_nearest": s_near["recall"],
                "med_rel_err_nearest": s_near["med_rel_err"],
                "p10_rel_err_nearest": s_near["p10_rel_err"],
                "p90_rel_err_nearest": s_near["p90_rel_err"],
                "med_dt_min_nearest": s_near["med_dt_min"],
                "p90_abs_dt_min_nearest": s_near["p90_abs_dt_min"],

                # mean hourly
                "recall_mean": s_mean["recall"],
                "med_rel_err_mean": s_mean["med_rel_err"],
                "p10_rel_err_mean": s_mean["p10_rel_err"],
                "p90_rel_err_mean": s_mean["p90_rel_err"],
                "med_dt_min_mean": s_mean["med_dt_min"],
                "p90_abs_dt_min_mean": s_mean["p90_abs_dt_min"],

                "status": "ok",
                "error": None,
            })

            n_success += 1
            print(f"OK: events={n_events} recall_nearest={s_near['recall']:.3f} med_rel_err_nearest={s_near['med_rel_err']:.4f}")

            if n_success >= n_success_target:
                break

        except Exception as e:
            n_fail += 1
            rows.append({
                "gauge": str(gauge),
                "year": year,
                "pot_quantile": pot_quantile,
                "min_separation": min_separation,
                "nearest_tol": nearest_tol,
                "search_window": search_window,
                "n_events": None,
                "iv_dt_mode_min": None,
                "recall_nearest": None,
                "med_rel_err_nearest": None,
                "p10_rel_err_nearest": None,
                "p90_rel_err_nearest": None,
                "med_dt_min_nearest": None,
                "p90_abs_dt_min_nearest": None,
                "recall_mean": None,
                "med_rel_err_mean": None,
                "p10_rel_err_mean": None,
                "p90_rel_err_mean": None,
                "med_dt_min_mean": None,
                "p90_abs_dt_min_mean": None,
                "status": "fail",
                "error": repr(e),
            })
            print("FAIL:", e)

    df = pd.DataFrame(rows)
    df["gauge"] = df["gauge"].astype(str)
    df["gauge_excel_safe"] = "'" + df["gauge"]
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    ok = df[df["status"] == "ok"].copy()
    if len(ok):
        print("\n=== Summary across successful gauges ===")
        for col in ["recall_nearest", "med_rel_err_nearest", "p90_rel_err_nearest", "p90_abs_dt_min_nearest",
                    "recall_mean", "med_rel_err_mean", "p90_rel_err_mean", "p90_abs_dt_min_mean"]:
            s = ok[col].astype(float)
            print(f"{col}: median={s.median():.4f}  p10={s.quantile(0.10):.4f}  p90={s.quantile(0.90):.4f}")


if __name__ == "__main__":
    main()
