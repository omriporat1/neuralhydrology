import matplotlib.pyplot as plt

from src.data_camelsh import load_camelsh_hourly
from src.data_usgs import fetch_usgs_iv_discharge
from src.aggregation import to_hourly_candidates

def main():
    gauge = "0136230002"
    year = 2016

    q_cam = load_camelsh_hourly(gauge, year)
    q_iv = fetch_usgs_iv_discharge(gauge, f"{year}-01-01", f"{year+1}-01-02")
    cand = to_hourly_candidates(q_iv)

    # Pick a window around the annual max (in CAMELSH)
    t_peak = q_cam.idxmax()
    t0 = t_peak -  pd.Timedelta("2D")
    t1 = t_peak +  pd.Timedelta("2D")

    q_iv_win = q_iv.loc[t0:t1]
    q_first = cand["first"].reindex(q_cam.index).loc[t0:t1]
    q_mean  = cand["mean"].reindex(q_cam.index).loc[t0:t1]
    q_cam_win = q_cam.loc[t0:t1]

    plt.figure()
    plt.plot(q_iv_win.index, q_iv_win.values, label="USGS IV (sub-hourly)")
    plt.plot(q_first.index, q_first.values, label="USGS hourly (first)")
    plt.plot(q_mean.index, q_mean.values, label="USGS hourly (mean)")
    plt.plot(q_cam_win.index, q_cam_win.values, label="CAMELSH hourly", linestyle=":")

    plt.legend()
    plt.title(f"{gauge} window around CAMELSH annual peak ({year})")
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    main()
