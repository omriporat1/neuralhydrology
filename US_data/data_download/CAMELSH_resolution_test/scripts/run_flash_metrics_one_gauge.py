# scripts/run_flash_metrics_one_gauge.py
import pandas as pd

from src.data_usgs import fetch_usgs_iv_discharge
from src.aggregation import to_hourly_candidates
from src.events import detect_pot_events
from src.flash_metrics import event_peak_comparison


def main():
    gauge = "0136230002"
    year = 2016

    # fetch with buffer for clean edges
    q_iv = fetch_usgs_iv_discharge(gauge, f"{year-1}-12-31", f"{year+1}-01-02")

    # keep only the year for event detection
    q_hi = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"]

    # build hourly variants
    cands = to_hourly_candidates(q_iv, nearest_tolerance="10min")

    q_hourly_nearest = cands["nearest_10min"].loc[f"{year}-01-01":f"{year}-12-31 23:00:00"]
    q_hourly_mean = cands["mean"].loc[f"{year}-01-01":f"{year}-12-31 23:00:00"]

    # detect POT events on high-res
    events = detect_pot_events(q_hi, quantile=0.95, min_separation="12h")
    print("Detected events:", len(events), "threshold:", events["threshold"].iloc[0] if len(events) else None)

    # compare peaks
    comp_nearest = event_peak_comparison(q_hi, q_hourly_nearest, events, search_window="1h")
    comp_mean = event_peak_comparison(q_hi, q_hourly_mean, events, search_window="1h")

    def summarize(df, name):
        det = df["detected"].mean() if len(df) else float("nan")
        med_err = df["rel_peak_error"].median() if len(df) else float("nan")
        med_dt = df["dt_minutes"].median() if len(df) else float("nan")
        print(f"\n== {name} ==")
        print("events:", len(df), "recall:", det)
        print("median rel peak error:", med_err)
        print("median dt (min):", med_dt)

    summarize(comp_nearest, "Hourly nearest (10min)")
    summarize(comp_mean, "Hourly mean")

    # show worst peak underestimation (nearest)
    if len(comp_nearest):
        worst = comp_nearest.sort_values("rel_peak_error").head(5)
        print("\nWorst 5 (nearest):")
        print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
