# scripts/run_operator_test.py
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


def bias(a: pd.Series, b: pd.Series) -> float:
    # mean(a-b)
    d = (a - b).dropna()
    return float(np.mean(d.values)) if len(d) else np.nan


def main():
    gauge = "0136230002"
    year = 2016

    q_cam = load_camelsh_hourly(gauge, year)

    # Fetch USGS IV for the entire year (+ a small buffer helps with edges)
    q_iv = fetch_usgs_iv_discharge(gauge, f"{year}-01-01", f"{year+1}-01-02")

    candidates = to_hourly_candidates(q_iv)

    rows = []
    for name, q_h in candidates.items():
        # Align exactly to CAMELSH hourly timestamps
        q_h_aligned = q_h.reindex(q_cam.index)

        rows.append({
            "operator": name,
            "n_overlap": int((~q_cam.isna() & ~q_h_aligned.isna()).sum()),
            "mae": mae(q_h_aligned, q_cam),
            "rmse": rmse(q_h_aligned, q_cam),
            "bias": bias(q_h_aligned, q_cam),
            "peak_cam": float(q_cam.max(skipna=True)),
            "peak_usgs": float(q_h_aligned.max(skipna=True)),
            "peak_ratio_usgs_over_cam": float(q_h_aligned.max(skipna=True) / q_cam.max(skipna=True)),
        })

    df = pd.DataFrame(rows).sort_values(["rmse", "mae"])
    print(df.to_string(index=False))

    true_peak = float(q_iv.loc[q_cam.index.min():q_cam.index.max()].max())
    sampled_peak = float(candidates["first"].max())
    mean_peak = float(candidates["mean"].max())

    print("True 15-min peak:", true_peak)
    print("Hourly sampled peak:", sampled_peak)
    print("Hourly mean peak:", mean_peak)
    print("Sampled / True:", sampled_peak / true_peak)
    print("Mean / True:", mean_peak / true_peak)


if __name__ == "__main__":
    main()
