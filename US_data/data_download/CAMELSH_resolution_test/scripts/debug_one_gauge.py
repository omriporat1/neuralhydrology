import pandas as pd

from src.data_camelsh import load_camelsh_hourly
from src.data_usgs import fetch_usgs_iv_discharge
from src.aggregation import to_hourly_candidates

def main():
    gauge = "01671025"
    year = 2016

    q_cam = load_camelsh_hourly(gauge, year)
    q_iv = fetch_usgs_iv_discharge(gauge, f"{year}-01-01", f"{year+1}-01-02")
    cand = to_hourly_candidates(q_iv)

    # Align candidates
    aligned = {k: v.reindex(q_cam.index) for k, v in cand.items()}

    # Print overlap + missingness
    print("CAMELSH:", len(q_cam), "missing:", q_cam.isna().sum())
    for k, s in aligned.items():
        print(k, "missing:", s.isna().sum(), "overlap:", (~s.isna() & ~q_cam.isna()).sum())

    # Show a few timestamps where first is NaN but cam isn't (or vice versa)
    bad = q_cam.notna() & aligned["first"].isna()
    print("\nHours where CAMELSH has data but USGS(first) doesn't:", bad.sum())
    if bad.sum() > 0:
        print(q_cam[bad].head(10))

    # Quick compare peak timing
    print("\nPeak CAMELSH:", q_cam.idxmax(), float(q_cam.max()))
    print("Peak USGS first:", aligned["first"].idxmax(), float(aligned["first"].max()))

if __name__ == "__main__":
    main()
