from __future__ import annotations

from pathlib import Path
import argparse
import random
from datetime import datetime
import textwrap
import numpy as np
import pandas as pd

from src.data_usgs import fetch_usgs_iv_discharge
from src.aggregation import to_hourly_candidates
from src.events import detect_pot_events

from src.flashdiag import (
    zfill8,
    max_slope_native, max_slope_1h_from_hi, max_slope_1h_from_hourly,
    make_event_table, basin_summary_from_events,
    panel_event_scatter, plot_event_hydrograph,
    write_run_artifacts,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--n_success", type=int, default=40)
    ap.add_argument("--max_tries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--nearest_tol", type=str, default="10min")
    ap.add_argument("--min_sep", type=str, default="12h")
    ap.add_argument("--search_window", type=str, default="1h")
    ap.add_argument("--quantiles", type=str, default="0.95,0.98,0.99,0.995")
    ap.add_argument("--hydro_quantile", type=float, default=0.99)

    ap.add_argument("--attr_csv", type=str, default=r"C:\PhD\Python\neuralhydrology\US_data\iv_scan_results.csv")
    ap.add_argument("--camelsh_dir", type=str, default="data/raw/camelsh")

    args = ap.parse_args()

    year = args.year
    quantiles = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]

    # pick hydro quantile early
    q_hydro = args.hydro_quantile
    if q_hydro not in quantiles:
        q_hydro = sorted(quantiles, key=lambda x: abs(x - q_hydro))[0]

    start = f"{year-1}-12-31"
    end = f"{year+1}-01-02"

    # ---- run folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    qtag = "-".join([str(q) for q in quantiles])
    run_name = (
        f"flashdiag_{year}"
        f"_n{args.n_success}"
        f"_seed{args.seed}"
        f"_tol{args.nearest_tol}"
        f"_sep{args.min_sep}"
        f"_win{args.search_window}"
        f"_q{qtag}"
        f"_{ts}"
    )
    run_dir = Path("reports/runs") / run_name
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # attributes (area)
    attr = pd.read_csv(args.attr_csv)
    attr["gauge"] = attr["site_id"].astype(str).map(zfill8)
    attr = attr[["gauge", "drainage_km2"]].drop_duplicates("gauge")
    area_map = dict(zip(attr["gauge"], attr["drainage_km2"]))

    # gauge list (from CAMELSH filenames)
    files = sorted(Path(args.camelsh_dir).glob("*_hourly.nc"))
    gauges = [zfill8(f.name.replace("_hourly.nc", "")) for f in files]
    gauges = [g for g in gauges if g in area_map]

    random.seed(args.seed)
    random.shuffle(gauges)

    # ---- stage 1: collect basins + slopes + cache series
    basin_rows = []
    cache = {}  # gauge -> (q_iv_year, q_near_year, q_mean_year)
    success = []

    for i, g in enumerate(gauges[: args.max_tries], 1):
        if len(success) >= args.n_success:
            break
        print(f"[{i}] {g} (success={len(success)})")

        try:
            q_iv = fetch_usgs_iv_discharge(g, start, end)
            q_iv_year = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
            if len(q_iv_year) < 2000:
                raise RuntimeError("Too little IV data in target year")

            cands = to_hourly_candidates(q_iv, nearest_tolerance=args.nearest_tol)
            q_near_year = cands[f"nearest_{args.nearest_tol}"].loc[f"{year}-01-01":f"{year}-12-31 23:59:59"]
            q_mean_year = cands["mean"].loc[f"{year}-01-01":f"{year}-12-31 23:59:59"]

            area = float(area_map[g])

            basin_rows.append({
                "gauge": g,
                "year": year,
                "drainage_km2": area,
                "slope_hi_native_max": max_slope_native(q_iv_year),
                "slope_hr_native_near_max": max_slope_native(q_near_year),
                "slope_hr_native_mean_max": max_slope_native(q_mean_year),
                "slope_hi_1h_max": max_slope_1h_from_hi(q_iv_year),
                "slope_hr_1h_near_max": max_slope_1h_from_hourly(q_near_year),
                "slope_hr_1h_mean_max": max_slope_1h_from_hourly(q_mean_year),
            })

            cache[g] = (q_iv_year, q_near_year, q_mean_year)
            success.append(g)

        except Exception as e:
            print("  FAIL:", e)

    if not basin_rows:
        raise SystemExit("No successful basins.")

    df_basin = pd.DataFrame(basin_rows)
    df_basin.to_csv(run_dir / f"basin_slopes_{year}_n{len(success)}.csv", index=False)

    # ---- stage 2: event tables
    all_events = []
    for q in quantiles:
        print(f"\nComputing event tables for quantile {q} ...")
        for g in success:
            q_iv_year, q_near_year, q_mean_year = cache[g]
            ev = make_event_table(
                gauge=g,
                year=year,
                area_km2=float(area_map[g]),
                q_iv_year=q_iv_year,
                q_hr_near_year=q_near_year,
                q_hr_mean_year=q_mean_year,
                detect_pot_events_fn=detect_pot_events,
                quantile=q,
                min_sep=args.min_sep,
                search_window=args.search_window,
            )
            if len(ev):
                all_events.append(ev)

    if not all_events:
        raise SystemExit("No events detected across basins/quantiles.")

    df_events = pd.concat(all_events, ignore_index=True)
    df_events.to_csv(run_dir / f"events_{year}_n{len(success)}.csv", index=False)

    df_bsum = basin_summary_from_events(df_events)
    df_bsum.to_csv(run_dir / f"basin_summary_{year}_n{len(success)}.csv", index=False)

    # ---- panel plots (event level) with N + RMSE + Spearman
    dfs_by_q = {q: df_events[df_events["quantile"] == q].copy() for q in sorted(df_events["quantile"].unique())}

    panel_event_scatter(
        dfs_by_q,
        xcol="q_peak_hi",
        ycol="q_peak_hr_near",
        ccol="drainage_km2",
        title=f"Event peaks: 15-min vs hourly nearest (N varies by Q). Basins={len(success)}",
        xlabel="Q_peak 15-min (m3/s)",
        ylabel="Q_peak hourly nearest (m3/s)",
        outpath=fig_dir / f"panel_event_peak_scatter_near_{year}.png",
        logxy=True,
    )

    panel_event_scatter(
        dfs_by_q,
        xcol="q_peak_hi",
        ycol="q_peak_hr_mean",
        ccol="drainage_km2",
        title=f"Event peaks: 15-min vs hourly mean (centered). Basins={len(success)}",
        xlabel="Q_peak 15-min (m3/s)",
        ylabel="Q_peak hourly mean (m3/s)",
        outpath=fig_dir / f"panel_event_peak_scatter_mean_{year}.png",
        logxy=True,
    )

    # ---- worst-event hydrographs (separate PNGs) for chosen quantile
    q_vis = q_hydro
    dvis = df_events[df_events["quantile"] == q_vis].copy()
    event_fig_dir = fig_dir / "event_hydrographs"
    event_fig_dir.mkdir(parents=True, exist_ok=True)

    def safe_tag(ts: pd.Timestamp) -> str:
        return pd.Timestamp(ts).strftime("%Y%m%dT%H%M%S")

    # Rank: worst underestimation & worst timing (near + mean)
    cases = []
    dn = dvis[dvis["detected_near"]].copy()
    dm = dvis[dvis["detected_mean"]].copy()
    dn["abs_dt"] = dn["dt_minutes_near"].abs()
    dm["abs_dt"] = dm["dt_minutes_mean"].abs()

    for _, r in dn.sort_values("rel_peak_error_near").head(5).iterrows():
        cases.append(("worst_peakerr", "near", r))
    for _, r in dm.sort_values("rel_peak_error_mean").head(5).iterrows():
        cases.append(("worst_peakerr", "mean", r))
    for _, r in dn.sort_values("abs_dt", ascending=False).head(5).iterrows():
        cases.append(("worst_timing", "near", r))
    for _, r in dm.sort_values("abs_dt", ascending=False).head(5).iterrows():
        cases.append(("worst_timing", "mean", r))

    # Extreme not preserved
    for _, r in dn[dn["extreme_preserved_near"] == False].head(5).iterrows():
        cases.append(("not_preserved", "near", r))
    for _, r in dm[dm["extreme_preserved_mean"] == False].head(5).iterrows():
        cases.append(("not_preserved", "mean", r))

    # Dedup
    seen = set()
    cases2 = []
    for case_type, method, r in cases:
        key = (case_type, method, r["gauge"], r["t_peak_hi"])
        if key not in seen:
            cases2.append((case_type, method, r))
            seen.add(key)

    print(f"\nPlotting {len(cases2)} worst-event hydrographs at Q={q_vis} ...")
    for case_type, method, r in cases2:
        g = r["gauge"]
        t_hi = pd.Timestamp(r["t_peak_hi"])
        q_iv_year, q_near_year, q_mean_year = cache[g]

        out = event_fig_dir / f"{case_type}_{method}_{g}_{safe_tag(t_hi)}_Q{q_vis}.png"
        plot_event_hydrograph(
            gauge=g,
            year=year,
            quantile=q_vis,
            method=method,
            t_peak_hi=t_hi,
            q_iv_year=q_iv_year,
            q_hr_near_year=q_near_year,
            q_hr_mean_year=q_mean_year,
            outpath=out,
            hours_before=12,
            hours_after=24,
        )

    # ---- run README + params
    readme = textwrap.dedent(f"""
    # Flash-resolution diagnostics run

    **Run:** {run_name}

    ## Goal
    Compare USGS IV hi-res discharge with two hourly representations:
    - hourly nearest-to-hour within {args.nearest_tol}
    - hourly mean (centered, timestamp t represents ~[t-30min, t+30min))

    POT events detected on hi-res at quantiles: {quantiles}
    min separation: {args.min_sep}
    matching window: ±{args.search_window}

    ## Outputs
    - basin_slopes_*.csv: basin-level flashiness metrics (quantile-independent)
    - events_*.csv: event-level table (main truth)
    - basin_summary_*.csv: basin-level summaries derived from events
    - figures/panel_event_peak_scatter_*.png: scatter panels annotated with N, RMSE, Spearman rho, bias
    - figures/event_hydrographs/*.png: separate worst-event hydrographs for Q={q_vis}
    """).strip()

    params = vars(args).copy()
    params.update({"run_name": run_name, "start": start, "end": end, "quantiles": quantiles, "q_hydro_used": q_vis})
    write_run_artifacts(run_dir, params, readme)

    print(f"\nDone. Run saved under: {run_dir.resolve()}")
    print(f"Figures: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
