# scripts/operator_validation_sample.py
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


def main():
    year = 2016
    n_gauges = 10
    seed = 42

    camelsh_dir = Path("data/raw/camelsh")
    out_csv = Path(f"data/processed/operator_validation_{year}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Collect gauge IDs from filenames like "{gauge}_hourly.nc"
    files = sorted(camelsh_dir.glob("*_hourly.nc"))
    if not files:
        raise SystemExit(f"No *_hourly.nc files found in {camelsh_dir}. Check your unzip location.")

    gauge_ids = [f.name.replace("_hourly.nc", "") for f in files]

    random.seed(seed)
    sample = random.sample(gauge_ids, k=min(n_gauges, len(gauge_ids)))
    print(f"Sampling {len(sample)} gauges (seed={seed}):", sample)

    rows = []
    for i, gauge in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] Gauge {gauge}")

        try:
            q_cam = load_camelsh_hourly(gauge, year)
            # Fetch a whole year (+ 1 day buffer)
            start = f"{year-1}-12-31"
            end = f"{year+1}-01-02"
            q_iv = fetch_usgs_iv_discharge(gauge, start, end)
            candidates = to_hourly_candidates(q_iv)

            # Evaluate each candidate
            scores = []
            for op, q_h in candidates.items():
                q_h_aligned = q_h.reindex(q_cam.index)
                scores.append({
                    "operator": op,
                    "rmse": rmse(q_h_aligned, q_cam),
                    "mae": mae(q_h_aligned, q_cam),
                    "n_overlap": int((~q_cam.isna() & ~q_h_aligned.isna()).sum()),
                })

            sdf = pd.DataFrame(scores).sort_values(["rmse", "mae"])
            if gauge == "01671025":
                print("\nDEBUG 01671025 score table:")
                print(sdf.to_string(index=False))

            best = sdf.iloc[0]
            second = sdf.iloc[1] if len(sdf) > 1 else None

            rows.append({
                "gauge": gauge,
                "year": year,
                "best_operator": best["operator"],
                "rmse_best": float(best["rmse"]),
                "mae_best": float(best["mae"]),
                "n_overlap": int(best["n_overlap"]),
                "second_operator": None if second is None else second["operator"],
                "rmse_second": None if second is None else float(second["rmse"]),
                "rmse_ratio_second_over_best": None if second is None else float(second["rmse"] / best["rmse"]) if best["rmse"] and best["rmse"] > 0 else None,
            })

            print("Best:", best["operator"], "RMSE:", best["rmse"], "MAE:", best["mae"])
            if second is not None:
                print("Second:", second["operator"], "RMSE:", second["rmse"])

        except Exception as e:
            rows.append({
                "gauge": gauge,
                "year": year,
                "best_operator": None,
                "rmse_best": None,
                "mae_best": None,
                "n_overlap": None,
                "second_operator": None,
                "rmse_second": None,
                "rmse_ratio_second_over_best": None,
                "error": repr(e),
            })
            print("ERROR:", e)

    df = pd.DataFrame(rows)
    df["gauge"] = df["gauge"].astype(str)
    df["gauge_excel_safe"] = "'" + df["gauge"]


    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
