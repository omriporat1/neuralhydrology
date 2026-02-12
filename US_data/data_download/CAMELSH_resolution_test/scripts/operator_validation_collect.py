# scripts/operator_validation_collect.py
from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd

from src.data_camelsh import load_camelsh_hourly
from src.data_usgs import fetch_usgs_iv_discharge
from src.aggregation import to_hourly_candidates


def rmse(a: pd.Series, b: pd.Series) -> float:
    d = (a - b).dropna()
    return float(np.sqrt(np.mean(d.values ** 2))) if len(d) else np.nan


def mae(a: pd.Series, b: pd.Series) -> float:
    d = (a - b).dropna()
    return float(np.mean(np.abs(d.values))) if len(d) else np.nan


def modal_dt_minutes(idx: pd.DatetimeIndex) -> float | None:
    if len(idx) < 3:
        return None
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return None
    mode = diffs.mode()
    if mode.empty:
        return None
    return float(mode.iloc[0].total_seconds() / 60.0)


def main():
    year = 2016
    n_success_target = 30     # <-- increase later (e.g., 100)
    max_tries = 200           # prevents infinite loops
    seed = 42

    camelsh_dir = Path("data/raw/camelsh")
    out_csv = Path(f"data/processed/operator_map_{year}_n{n_success_target}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(camelsh_dir.glob("*_hourly.nc"))
    if not files:
        raise SystemExit(f"No *_hourly.nc files found in {camelsh_dir}")

    gauge_ids = [f.name.replace("_hourly.nc", "") for f in files]  # keep as strings
    random.seed(seed)
    random.shuffle(gauge_ids)

    start = f"{year-1}-12-31"
    end = f"{year+1}-01-02"

    rows = []
    n_success = 0
    n_fail = 0

    for i, gauge in enumerate(gauge_ids[:max_tries], 1):
        print(f"\n[{i}/{min(max_tries, len(gauge_ids))}] Gauge {gauge} (success={n_success}, fail={n_fail})")

        try:
            q_cam = load_camelsh_hourly(gauge, year)
            q_iv = fetch_usgs_iv_discharge(gauge, start, end)

            # IV cadence diagnostic
            dt_mode_min = modal_dt_minutes(q_iv.index)

            candidates = to_hourly_candidates(q_iv)

            scores = []
            for op, q_h in candidates.items():
                q_h_aligned = q_h.reindex(q_cam.index)
                scores.append({
                    "operator": op,
                    "rmse": rmse(q_h_aligned, q_cam),
                    "mae": mae(q_h_aligned, q_cam),
                    "n_overlap": int((~q_cam.isna() & ~q_h_aligned.isna()).sum()),
                    "n_cam_nonan": int(q_cam.notna().sum()),
                    "n_hourly_nonan": int(q_h_aligned.notna().sum()),
                })

            sdf = pd.DataFrame(scores).sort_values(["rmse", "mae"])
            best = sdf.iloc[0]
            second = sdf.iloc[1] if len(sdf) > 1 else None

            rows.append({
                "gauge": str(gauge),
                "year": year,
                "best_operator": best["operator"],
                "rmse_best": float(best["rmse"]),
                "mae_best": float(best["mae"]),
                "n_overlap": int(best["n_overlap"]),
                "rmse_second": None if second is None else float(second["rmse"]),
                "second_operator": None if second is None else second["operator"],
                "rmse_ratio_second_over_best": None if second is None or best["rmse"] == 0 else float(second["rmse"] / best["rmse"]),
                "iv_modal_dt_min": dt_mode_min,
                "status": "ok",
                "error": None,
            })

            n_success += 1
            print("Best:", best["operator"], "RMSE:", best["rmse"], "MAE:", best["mae"], "dt_mode_min:", dt_mode_min)

            if n_success >= n_success_target:
                break

        except Exception as e:
            n_fail += 1
            rows.append({
                "gauge": str(gauge),
                "year": year,
                "best_operator": None,
                "rmse_best": None,
                "mae_best": None,
                "n_overlap": None,
                "rmse_second": None,
                "second_operator": None,
                "rmse_ratio_second_over_best": None,
                "iv_modal_dt_min": None,
                "status": "fail",
                "error": repr(e),
            })
            print("FAIL:", e)

    df = pd.DataFrame(rows)
    df["gauge"] = df["gauge"].astype(str)
    df["gauge_excel_safe"] = "'" + df["gauge"]  # helps Excel keep leading zeros

    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    ok = df[df["status"] == "ok"].copy()
    print("\n=== Operator counts (successful only) ===")
    print(ok["best_operator"].value_counts(dropna=False).to_string())

    if len(ok) > 0:
        print("\n=== Operator share (%) ===")
        print((100 * ok["best_operator"].value_counts(normalize=True)).round(1).to_string())


if __name__ == "__main__":
    main()
