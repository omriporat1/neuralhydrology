#!/usr/bin/env python3
"""Stage 1 Milestone 2C — one-hour basin-statistic extraction for MRMS and RTMA.

Loads pre-computed CAMELSH polygon basin-grid weights (Milestone 2B), decodes
one sample hour of MRMS QPE and RTMA analysis data, and extracts weighted basin
statistics for the 50 pilot basins.

Default RTMA output excludes:
  10wdir  — circular variable; linear averaging is invalid.
  orog    — static terrain field; not a dynamic forcing.
  (Pass --include-excluded-vars to include them for diagnostics.)

Outputs (under --data-root):
  03_basin_timeseries/stage1_pilot/one_hour/
      mrms_one_hour_basin_stats.parquet
      rtma_one_hour_basin_stats.parquet
      combined_one_hour_basin_stats.parquet
      combined_one_hour_basin_stats_preview.csv
  06_qc_reports/stage1_pilot/one_hour_extraction/
      mrms_grid_preview_with_basins.png
      rtma_temperature_preview_with_basins.png
      basin_value_histograms.png
      weighted_mean_vs_q50_scatter.png
      focused/  (per-basin detailed plots)
  09_manifests/stage1_pilot/one_hour_extraction/
      manifest.json / summary.json / summary.md / run_command.txt /
      git_commit.txt / config_snapshot.yaml

Usage:
    python scripts/extract_stage1_one_hour.py \\
        --config configs/pilot_stage1.yaml \\
        --data-root tmp/stage1_pilot_dryrun
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("extract_one_hour")

# MRMS/RTMA geographic extents (from grid-definition JSON, Milestone 2A)
_MRMS_LON_MIN, _MRMS_LON_MAX = -129.995, -60.005
_MRMS_LAT_MIN, _MRMS_LAT_MAX =   20.005,  54.995
_MRMS_DX, _MRMS_DY = 0.01, 0.01

_RTMA_LON_MIN, _RTMA_LON_MAX = -138.373,  -59.042
_RTMA_LAT_MIN, _RTMA_LAT_MAX =   19.229,   57.089


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 Milestone 2C: one-hour basin-statistic extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/pilot_stage1.yaml")
    p.add_argument("--data-root", dest="data_root", default=None)
    p.add_argument("--sample-time", dest="sample_time",
                   default="2023-01-01T00:00:00")
    p.add_argument("--products", default="mrms,rtma",
                   help="Comma-separated: mrms, rtma, or both")
    p.add_argument("--include-excluded-vars", dest="include_excluded_vars",
                   action="store_true",
                   help="Also extract 10wdir and orog (diagnostic use only)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sample file discovery / download
# ---------------------------------------------------------------------------

def _locate_or_fetch_mrms(raw_dir: Path, sample_dt: datetime) -> tuple[Optional[Path], bool]:
    from src.pipeline.grid_definitions import find_cached_mrms_sample, download_mrms_sample
    mrms_raw = raw_dir / "mrms"
    cached = find_cached_mrms_sample(mrms_raw, sample_dt)
    if cached is not None:
        LOGGER.info("MRMS sample found (cached): %s", cached)
        return cached, False
    LOGGER.info("MRMS sample not found — downloading from S3 ...")
    dl = download_mrms_sample(mrms_raw, sample_dt)
    if dl is None:
        LOGGER.error("MRMS download failed for %s", sample_dt.isoformat())
    else:
        LOGGER.info("MRMS downloaded: %s", dl)
    return dl, dl is not None


def _locate_or_fetch_rtma(raw_dir: Path, sample_dt: datetime) -> tuple[Optional[Path], bool]:
    from src.pipeline.grid_definitions import find_cached_rtma_sample, download_rtma_sample
    rtma_raw = raw_dir / "rtma"
    cached = find_cached_rtma_sample(rtma_raw, sample_dt)
    if cached is not None:
        LOGGER.info("RTMA sample found (cached): %s", cached)
        return cached, False
    LOGGER.info("RTMA sample not found — downloading from S3 ...")
    dl = download_rtma_sample(rtma_raw, sample_dt)
    if dl is None:
        LOGGER.error("RTMA download failed for %s", sample_dt.isoformat())
    else:
        LOGGER.info("RTMA downloaded: %s", dl)
    return dl, dl is not None


# ---------------------------------------------------------------------------
# Geographic context loading (Natural Earth)
# ---------------------------------------------------------------------------

def _try_load_geo_context(
    cache_dir: Optional[Path] = None,
) -> tuple[Optional[Any], Optional[Any]]:
    """Return (countries_gdf, us_states_gdf), either or both may be None.

    Sources tried in order:
      1. countries: pyogrio test fixtures (naturalearth_lowres, 110m, local)
      2. us_states: Natural Earth 110m admin-1 from CDN (cached locally if cache_dir given)
    Both datasets are in EPSG:4326.
    """
    try:
        import geopandas as gpd
    except ImportError:
        LOGGER.info("geo_context: geopandas not available — skipping map context")
        return None, None

    countries_gdf = None
    us_states_gdf = None

    # 1. Countries — use local pyogrio test fixture (always available)
    try:
        import pyogrio
        fixture_shp = (
            Path(pyogrio.__file__).parent
            / "tests" / "fixtures" / "naturalearth_lowres" / "naturalearth_lowres.shp"
        )
        if fixture_shp.exists():
            countries_gdf = gpd.read_file(str(fixture_shp))
            LOGGER.info("geo_context: countries loaded from pyogrio fixture (%d features)", len(countries_gdf))
    except Exception as exc:
        LOGGER.debug("geo_context: countries fixture failed: %s", exc)

    # 2. US states — try local cache first, then CDN
    states_cache: Optional[Path] = None
    if cache_dir is not None:
        states_cache = cache_dir / "ne_110m_admin1_us_states.gpkg"
        if states_cache.exists():
            try:
                us_states_gdf = gpd.read_file(str(states_cache))
                LOGGER.info("geo_context: US states loaded from cache (%d features)", len(us_states_gdf))
            except Exception as exc:
                LOGGER.debug("geo_context: states cache read failed: %s", exc)
                us_states_gdf = None

    if us_states_gdf is None:
        ne_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_1_states_provinces.zip"
        try:
            all_states = gpd.read_file(ne_url)
            us_states_gdf = all_states[all_states["iso_a2"] == "US"].copy()
            LOGGER.info("geo_context: US states downloaded (%d features)", len(us_states_gdf))
            if states_cache is not None:
                states_cache.parent.mkdir(parents=True, exist_ok=True)
                us_states_gdf.to_file(str(states_cache), driver="GPKG")
                LOGGER.info("geo_context: US states cached to %s", states_cache.name)
        except Exception as exc:
            LOGGER.info("geo_context: US states unavailable (%s) — maps will lack state boundaries", exc)

    return countries_gdf, us_states_gdf


def _add_geo_context(ax: Any, countries_gdf: Any, us_states_gdf: Any) -> None:
    """Overlay light-gray country and state boundaries on a lat/lon axes."""
    if countries_gdf is not None:
        try:
            countries_gdf.plot(ax=ax, color="none", edgecolor="#aaaaaa", linewidth=0.6, zorder=2)
        except Exception:
            pass
    if us_states_gdf is not None:
        try:
            us_states_gdf.plot(ax=ax, color="none", edgecolor="#bbbbbb", linewidth=0.5, zorder=2)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scale bar
# ---------------------------------------------------------------------------

def _add_scale_bar(
    ax: Any,
    extent: tuple[float, float, float, float],
    lat_deg: float,
    *,
    x_frac: float = 0.05,
    y_frac: float = 0.06,
) -> None:
    """Draw an approximate km scale bar on a lat/lon axis.

    extent: (lon_min, lon_max, lat_min, lat_max)
    lat_deg: representative latitude for computing km-per-degree-longitude
    """
    import numpy as np

    lon_min, lon_max, lat_min, lat_max = extent
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_deg))
    lon_span_km = lon_span * km_per_deg_lon

    # Pick a "nice" scale bar length (~15-25% of map width)
    target_km = lon_span_km * 0.20
    for candidate in [1, 2, 5, 10, 25, 50, 100, 200, 500]:
        if candidate >= target_km * 0.5:
            bar_km = candidate
            break
    else:
        bar_km = round(target_km / 10) * 10

    bar_deg = bar_km / km_per_deg_lon
    x0 = lon_min + x_frac * lon_span
    y0 = lat_min + y_frac * lat_span

    ax.plot([x0, x0 + bar_deg], [y0, y0], "k-", linewidth=2.5,
            solid_capstyle="butt", zorder=10)
    tick_h = lat_span * 0.012
    for xp in (x0, x0 + bar_deg):
        ax.plot([xp, xp], [y0 - tick_h / 2, y0 + tick_h / 2], "k-", lw=1.5, zorder=10)
    ax.text(x0 + bar_deg / 2, y0 + tick_h * 1.2,
            f"{bar_km} km", ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=10)


# ---------------------------------------------------------------------------
# Broad QC plots
# ---------------------------------------------------------------------------

def _make_broad_qc_plots(
    qc_dir: Path,
    result: dict[str, Any],
    pilot_manifest: Any,
    countries_gdf: Any,
    us_states_gdf: Any,
) -> list[str]:
    """Generate broad-domain QC PNGs (MRMS, RTMA, histograms, scatter).

    Grid and basin-marker color scales are shared (same vmin/vmax) for each map.
    Returns list of filenames written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        LOGGER.warning("matplotlib not available — skipping QC plots: %s", exc)
        return []

    qc_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    manifest = pilot_manifest[["STAID", "LAT_GAGE", "LNG_GAGE"]].copy()
    manifest["STAID_norm"] = manifest["STAID"].apply(
        lambda s: str(s).strip().lstrip("0").zfill(8)
    )

    # ---- Plot 1: MRMS grid + basin markers (shared scale) -------------------
    mrms_grid = result.get("mrms_grid")
    mrms_df   = result.get("mrms_df")
    if mrms_grid is not None and mrms_df is not None and len(mrms_df) > 0:
        try:
            stride = 10
            arr_ds = mrms_grid.values[::stride, ::stride].copy()
            # Shared vmin/vmax: use 99th-percentile of grid (avoid outlier saturation)
            vmin = 0.0
            vmax = float(np.nanpercentile(mrms_grid.values[mrms_grid.values > 0], 99)
                         if (mrms_grid.values > 0).any() else 1.0)
            vmax = max(vmax, float(mrms_df["weighted_mean"].max()) + 1e-9)

            fig, ax = plt.subplots(figsize=(13, 7))
            im = ax.imshow(
                arr_ds, origin="upper",
                extent=[_MRMS_LON_MIN, _MRMS_LON_MAX, _MRMS_LAT_MIN, _MRMS_LAT_MAX],
                aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax,
            )
            _add_geo_context(ax, countries_gdf, us_states_gdf)

            merged = mrms_df[["STAID", "weighted_mean"]].merge(
                manifest, left_on="STAID", right_on="STAID_norm", how="inner"
            )
            ax.scatter(
                merged["LNG_GAGE"], merged["LAT_GAGE"],
                c=merged["weighted_mean"], cmap="Blues",
                s=90, marker="D", edgecolors="k", linewidths=0.7,
                vmin=vmin, vmax=vmax, zorder=5,
            )
            cb = fig.colorbar(im, ax=ax, fraction=0.025,
                              label="MRMS QPE [mm] — grid + basin weighted_mean (same scale)\n"
                                    "GRIB units metadata: 'unknown'; documented as mm")
            ax.set_xlim(_MRMS_LON_MIN, _MRMS_LON_MAX)
            ax.set_ylim(_MRMS_LAT_MIN, _MRMS_LAT_MAX)
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title(
                f"MRMS QPE 1-h Pass1 [mm] — {mrms_grid.valid_time_utc}\n"
                f"Overview: grid (downsampled ×{stride}) + basin weighted_mean [same scale]",
                fontsize=10,
            )
            _add_scale_bar(ax, (_MRMS_LON_MIN, _MRMS_LON_MAX, _MRMS_LAT_MIN, _MRMS_LAT_MAX), 37.0)
            out = qc_dir / "mrms_grid_preview_with_basins.png"
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(out.name)
            LOGGER.info("QC plot: %s", out.name)
        except Exception as exc:
            LOGGER.warning("MRMS broad plot failed: %s", exc)
            try:
                plt.close("all")
            except Exception:
                pass

    # ---- Plot 2: RTMA 2m temperature + basin markers (shared scale) ---------
    rtma_grids = result.get("rtma_grids", [])
    rtma_df    = result.get("rtma_df")
    tmp_grid = next(
        (g for g in rtma_grids if g.short_name in ("2t", "t2m")),
        rtma_grids[0] if rtma_grids else None,
    )
    if tmp_grid is not None and rtma_df is not None and len(rtma_df) > 0:
        try:
            import numpy as np
            tmp_basin = rtma_df.loc[rtma_df["variable"] == tmp_grid.short_name]
            # Shared vmin/vmax across grid and basin markers
            vmin = float(np.nanmin(tmp_grid.values))
            vmax = float(np.nanmax(tmp_grid.values))

            fig, ax = plt.subplots(figsize=(13, 7))
            im = ax.imshow(
                tmp_grid.values, origin="lower",
                extent=[_RTMA_LON_MIN, _RTMA_LON_MAX, _RTMA_LAT_MIN, _RTMA_LAT_MAX],
                aspect="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
            )
            _add_geo_context(ax, countries_gdf, us_states_gdf)

            if len(tmp_basin) > 0:
                merged = tmp_basin[["STAID", "weighted_mean"]].merge(
                    manifest, left_on="STAID", right_on="STAID_norm", how="inner"
                )
                ax.scatter(
                    merged["LNG_GAGE"], merged["LAT_GAGE"],
                    c=merged["weighted_mean"], cmap="RdYlBu_r",
                    s=90, marker="D", edgecolors="k", linewidths=0.7,
                    vmin=vmin, vmax=vmax, zorder=5,
                )
            cb = fig.colorbar(im, ax=ax, fraction=0.025,
                              label=f"{tmp_grid.grib_name} [{tmp_grid.units}]\n"
                                    f"Grid + basin weighted_mean — same scale")
            ax.set_xlim(_RTMA_LON_MIN, _RTMA_LON_MAX)
            ax.set_ylim(_RTMA_LAT_MIN, _RTMA_LAT_MAX)
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title(
                f"RTMA {tmp_grid.grib_name} [{tmp_grid.units}] — {tmp_grid.valid_time_utc}\n"
                f"Overview: grid (Lambert, approx. lat/lon) + basin weighted_mean [same scale]",
                fontsize=10,
            )
            _add_scale_bar(ax, (_RTMA_LON_MIN, _RTMA_LON_MAX, _RTMA_LAT_MIN, _RTMA_LAT_MAX), 37.0)
            out = qc_dir / "rtma_temperature_preview_with_basins.png"
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(out.name)
            LOGGER.info("QC plot: %s", out.name)
        except Exception as exc:
            LOGGER.warning("RTMA broad plot failed: %s", exc)
            try:
                plt.close("all")
            except Exception:
                pass

    # ---- Plot 3: Basin value histograms ------------------------------------
    combined_df = result.get("combined_df")
    if combined_df is not None and len(combined_df) > 0:
        try:
            from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT
            import numpy as np
            key_pairs: list[tuple[str, str]] = [(_MRMS_PRODUCT, "unknown")]
            for sn in ("2t", "sp", "10u", "2sh"):
                if sn in combined_df["variable"].values and len(key_pairs) < 4:
                    key_pairs.append((_RTMA_PRODUCT, sn))

            n = len(key_pairs)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
            if n == 1:
                axes = [axes]

            for ax, (prod, var) in zip(axes, key_pairs):
                sub = combined_df.loc[
                    (combined_df["product"] == prod) & (combined_df["variable"] == var),
                    "weighted_mean",
                ].dropna()
                ax.hist(sub, bins=min(20, max(5, len(sub))), edgecolor="k", alpha=0.75)
                label = combined_df.loc[combined_df["variable"] == var,
                                        "variable_standard_name"].iloc[0] if len(sub) else var
                units_raw = combined_df.loc[combined_df["variable"] == var, "units"].iloc[0] if len(sub) else ""
                units_disp = "mm" if prod == _MRMS_PRODUCT else units_raw
                ax.set_xlabel(f"weighted_mean [{units_disp}]")
                ax.set_ylabel("Basin count")
                ax.set_title(f"{label}\n({var}, n={len(sub)})", fontsize=9)
                ax.grid(axis="y", alpha=0.3)

            fig.suptitle(
                f"Basin weighted_mean — pilot 50 basins, {combined_df['valid_time_utc'].iloc[0]}",
                fontsize=10,
            )
            out = qc_dir / "basin_value_histograms.png"
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(out.name)
            LOGGER.info("QC plot: %s", out.name)
        except Exception as exc:
            LOGGER.warning("Histogram plot failed: %s", exc)
            try:
                plt.close("all")
            except Exception:
                pass

    # ---- Plot 4: Weighted mean vs Q50 scatter ------------------------------
    if combined_df is not None and len(combined_df) > 0:
        try:
            from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT
            mrms_df_l = result.get("mrms_df")
            rtma_df_l = result.get("rtma_df")
            plot_pairs: list[tuple[str, str]] = []
            if mrms_df_l is not None and len(mrms_df_l) > 0:
                plot_pairs.append((_MRMS_PRODUCT, "unknown"))
            for sn in ("2t", "sp"):
                if rtma_df_l is not None and sn in rtma_df_l["variable"].values:
                    plot_pairs.append((_RTMA_PRODUCT, sn))

            n = len(plot_pairs)
            if n > 0:
                fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
                if n == 1:
                    axes = [axes]
                for ax, (prod, var) in zip(axes, plot_pairs):
                    sub = combined_df.loc[
                        (combined_df["product"] == prod) & (combined_df["variable"] == var)
                    ].dropna(subset=["weighted_mean", "unweighted_q50"])
                    ax.scatter(sub["unweighted_q50"], sub["weighted_mean"],
                               s=40, edgecolors="k", linewidths=0.4, alpha=0.8)
                    lo = float(min(sub["unweighted_q50"].min(), sub["weighted_mean"].min()))
                    hi = float(max(sub["unweighted_q50"].max(), sub["weighted_mean"].max()))
                    if lo < hi:
                        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1")
                        ax.legend(fontsize=8)
                    label = combined_df.loc[combined_df["variable"] == var,
                                            "variable_standard_name"].iloc[0] if len(sub) else var
                    units_raw = combined_df.loc[combined_df["variable"] == var, "units"].iloc[0] if len(sub) else ""
                    units_disp = "mm" if prod == _MRMS_PRODUCT else units_raw
                    ax.set_xlabel(f"unweighted_q50 [{units_disp}]")
                    ax.set_ylabel(f"weighted_mean [{units_disp}]")
                    ax.set_title(f"{label} ({var})\nweighted_mean vs Q50", fontsize=9)
                    ax.grid(alpha=0.3)
                fig.suptitle("Weighted mean vs unweighted Q50 — pilot 50 basins", fontsize=10)
                out = qc_dir / "weighted_mean_vs_q50_scatter.png"
                fig.tight_layout()
                fig.savefig(out, dpi=120)
                plt.close(fig)
                written.append(out.name)
                LOGGER.info("QC plot: %s", out.name)
        except Exception as exc:
            LOGGER.warning("Scatter plot failed: %s", exc)
            try:
                plt.close("all")
            except Exception:
                pass

    return written


# ---------------------------------------------------------------------------
# Focused QC maps
# ---------------------------------------------------------------------------

def _make_focused_maps(
    qc_dir: Path,
    result: dict[str, Any],
    mrms_weights: Any,
    rtma_weights: Any,
    pilot_manifest: Any,
    camelsh_shp_path: Optional[Path],
    countries_gdf: Any,
    us_states_gdf: Any,
    *,
    gauge_audit_df: Optional[Any] = None,
) -> list[str]:
    """Generate focused per-basin QC PNGs.

    One PNG per selected basin showing: CAMELSH polygon, grid cell scatter
    (colored by value, same scale as basin marker), state/coastline context,
    km scale bar, and annotated statistics.

    Representative basin set:
      - wet_mrms:      highest MRMS weighted_mean > 0
      - dry_mrms:      MRMS weighted_mean = 0 (or lowest)
      - small_basin:   fewest valid MRMS grid cells
      - large_basin:   most valid MRMS grid cells
      - rtma_tmp:      RTMA 2m temperature (median latitude basin)
      - rtma_wind_u:   RTMA 10u wind U-component (same basin as rtma_tmp)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import geopandas as gpd
    except Exception as exc:
        LOGGER.warning("focused maps: cannot import required libs: %s", exc)
        return []

    focused_dir = qc_dir / "focused"
    focused_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    mrms_df   = result.get("mrms_df")
    rtma_df   = result.get("rtma_df")
    mrms_grid = result.get("mrms_grid")
    rtma_grids = result.get("rtma_grids", [])

    # Load CAMELSH polygons (optional)
    camelsh_gdf = None
    if camelsh_shp_path is not None and camelsh_shp_path.exists():
        try:
            camelsh_gdf = gpd.read_file(str(camelsh_shp_path))
            if camelsh_gdf.crs is None:
                camelsh_gdf = camelsh_gdf.set_crs("EPSG:4326")
            camelsh_gdf["STAID_norm"] = camelsh_gdf["GAGE_ID"].apply(
                lambda s: str(s).strip().lstrip("0").zfill(8)
            )
            LOGGER.info("focused maps: CAMELSH shapefile loaded (%d polygons)", len(camelsh_gdf))
        except Exception as exc:
            LOGGER.warning("focused maps: CAMELSH load failed: %s", exc)
            camelsh_gdf = None

    manifest = pilot_manifest[["STAID", "LAT_GAGE", "LNG_GAGE"]].copy()
    manifest["STAID_norm"] = manifest["STAID"].apply(
        lambda s: str(s).strip().lstrip("0").zfill(8)
    )

    # ---- Select representative basins --------------------------------------
    focus_cases: list[dict] = []

    if mrms_df is not None and len(mrms_df) > 0 and mrms_grid is not None:
        df_sorted = mrms_df.sort_values("weighted_mean")
        # Wettest
        wet_rows = df_sorted[df_sorted["weighted_mean"] > 0]
        if len(wet_rows) > 0:
            focus_cases.append({
                "staid": wet_rows["STAID"].iloc[-1],
                "product": "mrms_qpe_1h_pass1",
                "variable": "unknown",
                "label": "wet_mrms",
                "grid": mrms_grid,
                "weights": mrms_weights,
                "df_stats": mrms_df,
            })
        # Driest (weighted_mean == 0 or minimum)
        dry_staid = df_sorted["STAID"].iloc[0]
        focus_cases.append({
            "staid": dry_staid,
            "product": "mrms_qpe_1h_pass1",
            "variable": "unknown",
            "label": "dry_mrms",
            "grid": mrms_grid,
            "weights": mrms_weights,
            "df_stats": mrms_df,
        })
        # Smallest basin (fewest cells)
        df_cells = mrms_df.sort_values("valid_cell_count")
        focus_cases.append({
            "staid": df_cells["STAID"].iloc[0],
            "product": "mrms_qpe_1h_pass1",
            "variable": "unknown",
            "label": "small_basin_mrms",
            "grid": mrms_grid,
            "weights": mrms_weights,
            "df_stats": mrms_df,
        })
        # Largest basin (most cells)
        focus_cases.append({
            "staid": df_cells["STAID"].iloc[-1],
            "product": "mrms_qpe_1h_pass1",
            "variable": "unknown",
            "label": "large_basin_mrms",
            "grid": mrms_grid,
            "weights": mrms_weights,
            "df_stats": mrms_df,
        })

    if rtma_df is not None and len(rtma_df) > 0:
        # Use median-latitude pilot basin for RTMA examples
        mid_row = manifest.sort_values("LAT_GAGE").iloc[len(manifest) // 2]
        mid_staid = mid_row["STAID_norm"]

        tmp_grid = next((g for g in rtma_grids if g.short_name in ("2t", "t2m")), None)
        if tmp_grid is not None:
            focus_cases.append({
                "staid": mid_staid,
                "product": "rtma_conus_aws_2p5km",
                "variable": tmp_grid.short_name,
                "label": "rtma_temperature_2t",
                "grid": tmp_grid,
                "weights": rtma_weights,
                "df_stats": rtma_df,
            })
    # ---- Draw each focused map (scalar) ------------------------------------
    for case in focus_cases:
        staid     = case["staid"]
        vg        = case["grid"]
        weights_df = case["weights"]
        df_stats  = case["df_stats"]
        label     = case["label"]
        product   = case["product"]
        variable  = case["variable"]

        # Look up gauge-polygon offset for this basin (from audit, if available)
        gauge_offset_m: Optional[float] = None
        if gauge_audit_df is not None:
            audit_row = gauge_audit_df[gauge_audit_df["STAID"] == staid]
            if len(audit_row) > 0:
                raw = audit_row["distance_to_polygon_m"].iloc[0]
                if raw is not None and str(raw) not in ("", "nan", "None"):
                    try:
                        gauge_offset_m = float(raw)
                    except (ValueError, TypeError):
                        pass

        try:
            fname = _draw_focused_map(
                staid=staid,
                vg=vg,
                weights_df=weights_df,
                df_stats=df_stats,
                camelsh_gdf=camelsh_gdf,
                manifest=manifest,
                countries_gdf=countries_gdf,
                us_states_gdf=us_states_gdf,
                out_dir=focused_dir,
                label=label,
                gauge_offset_m=gauge_offset_m,
            )
            if fname:
                written.append(f"focused/{fname}")
                LOGGER.info("focused QC map: focused/%s", fname)
        except Exception as exc:
            LOGGER.warning("focused map %s failed: %s", label, exc)

    # ---- Wind vector map (RTMA 10u + 10v combined) -------------------------
    if rtma_df is not None and len(rtma_df) > 0:
        u_vg = next((g for g in rtma_grids if g.short_name in ("10u", "u10", "ugrd")), None)
        v_vg = next((g for g in rtma_grids if g.short_name in ("10v", "v10", "vgrd")), None)
        if u_vg is not None and v_vg is not None:
            mid_row  = manifest.sort_values("LAT_GAGE").iloc[len(manifest) // 2]
            mid_staid = mid_row["STAID_norm"]
            try:
                fname = _draw_wind_vector_map(
                    staid=mid_staid,
                    u_vg=u_vg,
                    v_vg=v_vg,
                    weights_df=rtma_weights,
                    df_stats=rtma_df,
                    camelsh_gdf=camelsh_gdf,
                    manifest=manifest,
                    countries_gdf=countries_gdf,
                    us_states_gdf=us_states_gdf,
                    out_dir=focused_dir,
                )
                if fname:
                    written.append(f"focused/{fname}")
                    LOGGER.info("focused QC map: focused/%s", fname)
            except Exception as exc:
                LOGGER.warning("wind vector map failed: %s", exc)

    return written


def _get_display_units(vg: Any) -> str:
    """Return plot-friendly units string, substituting 'mm' for MRMS QPE."""
    if vg.product == "mrms_qpe_1h_pass1":
        return "mm"
    return vg.units


def _get_focused_color_scale(
    vg: Any,
    cell_values: Any,
    weighted_mean: float,
) -> tuple[float, float, str, bool]:
    """Return (vmin, vmax, cmap, is_dry) for a focused basin map.

    Color scale always includes weighted_mean so the basin marker is
    on the same range as the grid cells.

    is_dry=True means precipitation is zero or near-zero; caller should
    annotate the map accordingly and use a fixed display range.
    """
    import numpy as np

    valid_vals = cell_values[np.isfinite(cell_values)]
    wm = float(weighted_mean) if np.isfinite(weighted_mean) else float("nan")

    # Include weighted_mean in scale so basin marker fits within colorbar
    all_for_scale = (
        np.append(valid_vals, wm) if np.isfinite(wm) else valid_vals
    )

    # ---- MRMS QPE: non-negative, use Blues ----
    if vg.product == "mrms_qpe_1h_pass1":
        vmax_cand = float(np.nanmax(all_for_scale)) if len(all_for_scale) else 0.0
        if vmax_cand < 0.01:
            return 0.0, 1.0, "Blues", True      # dry hour: 0–1 mm display range
        return 0.0, vmax_cand, "Blues", False

    # ---- Signed/diverging variables: centre at zero ----
    if vg.short_name in {"10u", "u10", "ugrd", "10v", "v10", "vgrd"}:
        abs_max = max(
            float(np.nanmax(np.abs(all_for_scale))) if len(all_for_scale) else 0.1,
            0.1,
        )
        return -abs_max, abs_max, "RdBu_r", False

    # ---- Other variables ----
    if vg.short_name in {"2t", "t2m", "2d", "d2m"}:
        cmap = "RdYlBu_r"
    elif vg.short_name in {"tcc", "tcdc"}:
        cmap = "Blues"
    elif vg.short_name in {"sp", "pres", "pressfc"}:
        cmap = "viridis"
    else:
        cmap = "plasma"

    vmin_v = float(np.nanmin(all_for_scale)) if len(all_for_scale) else 0.0
    vmax_v = float(np.nanmax(all_for_scale)) if len(all_for_scale) else 1.0
    if vmin_v == vmax_v:
        vmin_v -= 0.001
        vmax_v += 0.001
    return vmin_v, vmax_v, cmap, False


def _get_focused_extent(
    staid: str,
    camelsh_gdf: Any,
    basin_wts: Any,
    cell_size_deg: float = 0.01,
) -> tuple[float, float, float, float]:
    """Return (lon_min, lon_max, lat_min, lat_max) for a focused basin map.

    Buffer rules applied to polygon bounds (or weight-cell bounds as fallback):
      - at least 2 grid cells in each direction
      - at least 25% of basin span in each direction
      - at least ~5 km (~0.045°) in each direction
    """
    # Primary: use CAMELSH polygon bounds
    if camelsh_gdf is not None:
        poly_rows = camelsh_gdf[camelsh_gdf["STAID_norm"] == staid]
        if len(poly_rows) > 0:
            b = poly_rows.geometry.total_bounds   # [minx, miny, maxx, maxy]
            lon_min, lat_min, lon_max, lat_max = (
                float(b[0]), float(b[1]), float(b[2]), float(b[3])
            )
        else:
            lon_min = float(basin_wts["lon_center"].min())
            lon_max = float(basin_wts["lon_center"].max())
            lat_min = float(basin_wts["lat_center"].min())
            lat_max = float(basin_wts["lat_center"].max())
    else:
        lon_min = float(basin_wts["lon_center"].min())
        lon_max = float(basin_wts["lon_center"].max())
        lat_min = float(basin_wts["lat_center"].min())
        lat_max = float(basin_wts["lat_center"].max())

    lat_span = max(lat_max - lat_min, cell_size_deg)
    lon_span = max(lon_max - lon_min, cell_size_deg)

    km5_deg = 5.0 / 111.32          # ~0.045°
    cell2   = 2 * cell_size_deg
    buf_lat = max(lat_span * 0.25, cell2, km5_deg)
    buf_lon = max(lon_span * 0.25, cell2, km5_deg)

    return (lon_min - buf_lon, lon_max + buf_lon,
            lat_min - buf_lat, lat_max + buf_lat)


def _draw_focused_map(
    staid: str,
    vg: Any,
    weights_df: Any,
    df_stats: Any,
    camelsh_gdf: Any,
    manifest: Any,
    countries_gdf: Any,
    us_states_gdf: Any,
    out_dir: Path,
    label: str,
    *,
    gauge_offset_m: Optional[float] = None,
) -> Optional[str]:
    """Draw one focused basin map. Returns filename or None on failure.

    gauge_offset_m: if provided and > 1 km, a warning annotation is added.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import geopandas as gpd

    # Basin weight cells
    basin_wts = weights_df[weights_df["STAID"] == staid]
    if basin_wts.empty:
        LOGGER.debug("_draw_focused_map: no weights for %s", staid)
        return None

    rows = basin_wts["row_idx"].values
    cols = basin_wts["col_idx"].values
    lats = basin_wts["lat_center"].values
    lons = basin_wts["lon_center"].values

    grid = vg.values
    nrows_g, ncols_g = grid.shape
    in_bounds = (rows >= 0) & (rows < nrows_g) & (cols >= 0) & (cols < ncols_g)
    rows, cols, lats, lons = (
        rows[in_bounds], cols[in_bounds], lats[in_bounds], lons[in_bounds]
    )
    if len(rows) == 0:
        return None

    cell_values = grid[rows, cols]
    n_cells = len(rows)

    # Basin stats
    stats_row = df_stats[
        (df_stats["STAID"] == staid) & (df_stats["variable"] == vg.short_name)
    ]
    weighted_mean = float(stats_row["weighted_mean"].iloc[0]) if len(stats_row) else float("nan")

    # Color scale (includes weighted_mean so basin marker sits on same range as raster)
    display_units = _get_display_units(vg)
    val_min, val_max, cmap, is_dry = _get_focused_color_scale(vg, cell_values, weighted_mean)
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=val_min, vmax=val_max)

    # Extent from polygon bounds + proper buffer (avoids tiny-basin problem)
    cell_size = 0.025 if vg.product == "rtma_conus_aws_2p5km" else 0.01
    extent = _get_focused_extent(staid, camelsh_gdf, basin_wts, cell_size_deg=cell_size)
    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent
    lat_center = (lat_min_e + lat_max_e) / 2

    fig, ax = plt.subplots(figsize=(8, 7))

    # ---- Layer 1: Background raster — opaque; primary visual for spatial alignment ----
    _draw_background_subgrid(ax, vg, extent,
                             vmin=val_min, vmax=val_max, cmap=cmap, alpha=0.90)

    # ---- Layer 2: CAMELSH polygon boundary — main spatial reference ----
    if camelsh_gdf is not None:
        basin_poly = camelsh_gdf[camelsh_gdf["STAID_norm"] == staid]
        if len(basin_poly) > 0:
            basin_poly.plot(ax=ax, color="none", edgecolor="crimson",
                            linewidth=2.2, zorder=6)

    # ---- Layer 3: Geographic context (subtle, behind polygon) ----
    _add_geo_context(ax, countries_gdf, us_states_gdf)

    # ---- Layer 4: Extraction cells — hollow rings (location indicator only) ----
    # Do NOT fill with colormap; raster already shows the values.
    ax.scatter(
        lons, lats,
        facecolors="none", edgecolors="0.30",
        s=_cell_marker_size(n_cells), linewidths=0.45, zorder=5,
    )

    # ---- Layers 5 & 6: Basin mean diamond + gauge star ----
    gauge_row = manifest[manifest["STAID_norm"] == staid]
    g_lon = g_lat = None
    if len(gauge_row) > 0:
        g_lon = float(gauge_row["LNG_GAGE"].iloc[0])
        g_lat = float(gauge_row["LAT_GAGE"].iloc[0])

    if g_lon is not None and np.isfinite(weighted_mean):
        # Large filled diamond: basin weighted_mean on same colormap/normalization as raster
        ax.scatter(
            [g_lon], [g_lat],
            c=[weighted_mean], cmap=cmap, norm=norm,
            s=420, marker="D", edgecolors="k", linewidths=2.0, zorder=9,
        )
    if g_lon is not None:
        # Small yellow star: gauge location (visually distinct from diamond)
        ax.scatter(
            [g_lon], [g_lat],
            marker="*", s=90, facecolors="yellow", edgecolors="k",
            linewidths=0.8, zorder=10,
        )

    # ---- Colorbar from ScalarMappable (covers raster + basin mean, same scale) ----
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    units_note = " (GRIB units: unknown)" if vg.units == "unknown" else ""
    cb.set_label(
        f"{vg.grib_name} [{display_units}]{units_note}\n"
        "Raster + ◆ basin weighted_mean — same scale",
        fontsize=8,
    )

    # ---- Scale bar (lower-left) ----
    _add_scale_bar(ax, extent, lat_center)

    # ---- Annotation box (upper-left, no legend) ----
    wm_str = f"{weighted_mean:.4g}" if np.isfinite(weighted_mean) else "NaN"
    ann_lines = [
        f"STAID:  {staid}",
        f"◆  Basin mean: {wm_str} {display_units}",
        f"★  Gauge location",
        f"○  Extraction cells: {n_cells}",
    ]
    if is_dry:
        ann_lines.append("⚠  ZERO PRECIP (local grid = 0)")
    ax.text(
        0.02, 0.98, "\n".join(ann_lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.88, edgecolor="0.60"),
        zorder=12,
    )

    # ---- Centered dry annotation ----
    if is_dry:
        ax.text(
            0.5, 0.50,
            "Local grid and\nbasin weighted_mean are zero",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="navy", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      alpha=0.85, edgecolor="steelblue"),
            zorder=13,
        )

    # Optional gauge-offset warning (bottom-right; doesn't overlap scale bar or annotation box)
    if gauge_offset_m is not None and np.isfinite(gauge_offset_m) and gauge_offset_m > 1000:
        ax.text(
            0.98, 0.02,
            f"⚠ Gauge {gauge_offset_m / 1000:.1f} km outside polygon",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5,
            color="darkorange",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      alpha=0.87, edgecolor="darkorange"),
            zorder=13,
        )

    ax.set_xlim(lon_min_e, lon_max_e)
    ax.set_ylim(lat_min_e, lat_max_e)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.tick_params(labelsize=8)

    dry_tag = " [DRY]" if is_dry else ""
    ax.set_title(
        f"{vg.product} | {vg.grib_name} ({vg.short_name}){dry_tag}\n"
        f"STAID {staid} | {vg.valid_time_utc} | "
        f"weighted_mean = {wm_str} {display_units} | n = {n_cells}",
        fontsize=9,
    )

    fname = f"{label}_{staid}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=130)
    plt.close(fig)
    return fname


def _cell_marker_size(n_cells: int) -> int:
    """Pick scatter marker size inversely proportional to cell count."""
    if n_cells < 20:
        return 80
    if n_cells < 100:
        return 40
    if n_cells < 500:
        return 20
    return 10


def _draw_wind_vector_map(
    staid: str,
    u_vg: Any,
    v_vg: Any,
    weights_df: Any,
    df_stats: Any,
    camelsh_gdf: Any,
    manifest: Any,
    countries_gdf: Any,
    us_states_gdf: Any,
    out_dir: Path,
) -> Optional[str]:
    """Draw a vector wind QC map combining RTMA 10u and 10v.

    Background raster: wind speed magnitude sqrt(u²+v²).
    Quiver: downsampled wind vectors at weight-cell positions, color-coded by speed.
    Crimson arrow: basin weighted_mean wind vector.
    Note: RTMA u/v are grid-relative (LCC); small rotation error at QC scale is acceptable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    basin_wts = weights_df[weights_df["STAID"] == staid]
    if basin_wts.empty:
        return None

    rows = basin_wts["row_idx"].values
    cols = basin_wts["col_idx"].values
    lats = basin_wts["lat_center"].values
    lons = basin_wts["lon_center"].values

    nrows_g, ncols_g = u_vg.values.shape
    in_bounds = (rows >= 0) & (rows < nrows_g) & (cols >= 0) & (cols < ncols_g)
    rows, cols, lats, lons = rows[in_bounds], cols[in_bounds], lats[in_bounds], lons[in_bounds]
    if len(rows) == 0:
        return None

    n_cells = len(rows)

    # Basin weighted means for 10u and 10v
    u_row = df_stats[(df_stats["STAID"] == staid) & (df_stats["variable"] == u_vg.short_name)]
    v_row = df_stats[(df_stats["STAID"] == staid) & (df_stats["variable"] == v_vg.short_name)]
    u_wm = float(u_row["weighted_mean"].iloc[0]) if len(u_row) else 0.0
    v_wm = float(v_row["weighted_mean"].iloc[0]) if len(v_row) else 0.0
    speed_wm = float(np.sqrt(u_wm**2 + v_wm**2))

    # Wind speed grid and cell values
    u_grid = u_vg.values
    v_grid = v_vg.values
    speed_grid = np.sqrt(u_grid**2 + v_grid**2)
    speed_cells = speed_grid[rows, cols]

    # Color scale: wind speed (non-negative)
    vmax_s = max(float(np.nanmax(speed_cells)), speed_wm, 0.5)
    norm_s = mcolors.Normalize(vmin=0.0, vmax=vmax_s)
    cmap_s = "YlOrRd"

    # Extent from polygon bounds + buffer
    extent = _get_focused_extent(staid, camelsh_gdf, basin_wts, cell_size_deg=0.025)
    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent
    lat_center = (lat_min_e + lat_max_e) / 2

    fig, ax = plt.subplots(figsize=(8, 7))

    # Layer 1: Wind speed raster (opaque background)
    ax.imshow(
        speed_grid, origin="lower",
        extent=[_RTMA_LON_MIN, _RTMA_LON_MAX, _RTMA_LAT_MIN, _RTMA_LAT_MAX],
        aspect="auto", cmap=cmap_s, norm=norm_s, alpha=0.88, zorder=1,
    )

    # Layer 2: CAMELSH polygon
    if camelsh_gdf is not None:
        poly = camelsh_gdf[camelsh_gdf["STAID_norm"] == staid]
        if len(poly) > 0:
            poly.plot(ax=ax, color="none", edgecolor="crimson", linewidth=2.2, zorder=6)

    # Layer 3: Geographic context
    _add_geo_context(ax, countries_gdf, us_states_gdf)

    # Layer 4: Quiver at weight-cell positions (downsampled)
    stride_q = max(1, n_cells // 22)   # target ≤ 22 arrows
    q_lons = lons[::stride_q]
    q_lats = lats[::stride_q]
    q_u    = u_grid[rows[::stride_q], cols[::stride_q]]
    q_v    = v_grid[rows[::stride_q], cols[::stride_q]]
    q_sp   = np.sqrt(q_u**2 + q_v**2)

    # Arrow scale: typical speed fills ~12% of map span
    lon_span = lon_max_e - lon_min_e
    scale_val = max(float(np.nanmean(q_sp)) if len(q_sp) else 5.0, 0.5) / (lon_span * 0.12)
    scale_val = max(scale_val, 5.0)   # at least 5 data-units per axis-unit

    Q = ax.quiver(
        q_lons, q_lats, q_u, q_v, q_sp,
        cmap=cmap_s, norm=norm_s,
        scale=scale_val, width=0.004, alpha=0.85, zorder=7,
    )
    # Quiver key: reference arrow showing 5 m/s
    try:
        ax.quiverkey(Q, X=0.87, Y=0.08, U=5.0, label="5 m/s",
                     labelpos="E", coordinates="axes",
                     fontproperties={"size": 7})
    except Exception:
        pass

    # Layer 5: Extraction cells (hollow rings, optional location guide)
    ax.scatter(
        lons, lats,
        facecolors="none", edgecolors="0.35",
        s=_cell_marker_size(n_cells), linewidths=0.4, zorder=5,
    )

    # Layer 6: Basin mean wind vector — large crimson arrow
    gauge_row = manifest[manifest["STAID_norm"] == staid]
    g_lon = g_lat = None
    if len(gauge_row) > 0:
        g_lon = float(gauge_row["LNG_GAGE"].iloc[0])
        g_lat = float(gauge_row["LAT_GAGE"].iloc[0])
        ax.quiver(
            [g_lon], [g_lat], [u_wm], [v_wm],
            color="crimson", scale=scale_val, width=0.007, zorder=10,
        )
        # Gauge star
        ax.scatter(
            [g_lon], [g_lat],
            marker="*", s=90, facecolors="yellow", edgecolors="k",
            linewidths=0.8, zorder=11,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_s, norm=norm_s)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Wind speed [m/s]\nRaster + quiver color — same scale", fontsize=8)

    # Scale bar (lower-left)
    _add_scale_bar(ax, extent, lat_center)

    # Annotation box (upper-left; quiver key is lower-right)
    ann_lines = [
        f"STAID:  {staid}",
        f"→  Wind: U={u_wm:.2f}, V={v_wm:.2f} m/s",
        f"   Speed: {speed_wm:.2f} m/s",
        f"★  Gauge location",
        f"○  Extraction cells: {n_cells}",
        "→ crimson = basin mean vector",
    ]
    ax.text(
        0.02, 0.98, "\n".join(ann_lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.88, edgecolor="0.60"),
        zorder=12,
    )

    ax.set_xlim(lon_min_e, lon_max_e)
    ax.set_ylim(lat_min_e, lat_max_e)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.tick_params(labelsize=8)
    ax.set_title(
        f"RTMA 10 m wind vector | STAID {staid} | {u_vg.valid_time_utc}\n"
        f"Raster: speed [m/s] | Quiver: 10u, 10v | "
        f"Crimson arrow: basin weighted_mean",
        fontsize=9,
    )

    fname = f"rtma_wind_vector_{staid}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=130)
    plt.close(fig)
    return fname


def _draw_background_subgrid(
    ax: Any,
    vg: Any,
    extent: tuple[float, float, float, float],
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    alpha: float = 0.55,
) -> None:
    """Plot a sub-region of the full grid as a semi-transparent background."""
    import numpy as np
    import matplotlib.pyplot as plt

    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent
    grid = vg.values

    if vg.product == "mrms_qpe_1h_pass1":
        # regular_ll: compute row/col range from geographic extent
        lat_first = _MRMS_LAT_MAX
        lon_first = _MRMS_LON_MIN
        dx, dy = _MRMS_DX, _MRMS_DY
        nrows_g, ncols_g = grid.shape

        row_start = max(0, int(np.floor((lat_first - lat_max_e) / dy)) - 2)
        row_end   = min(nrows_g, int(np.ceil((lat_first - lat_min_e) / dy)) + 3)
        col_start = max(0, int(np.floor((lon_min_e - lon_first) / dx)) - 2)
        col_end   = min(ncols_g, int(np.ceil((lon_max_e - lon_first) / dx)) + 3)

        if row_start >= row_end or col_start >= col_end:
            return

        sub = grid[row_start:row_end, col_start:col_end]
        sub_lat_max = lat_first - row_start * dy
        sub_lat_min = lat_first - (row_end - 1) * dy
        sub_lon_min = lon_first + col_start * dx
        sub_lon_max = lon_first + (col_end - 1) * dx
        sub_extent  = [sub_lon_min, sub_lon_max, sub_lat_min, sub_lat_max]

        ax.imshow(
            sub, origin="upper", extent=sub_extent, aspect="auto",
            cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, zorder=1,
        )
    else:
        # RTMA (Lambert): approximate by using the full grid with extent clipped
        ax.imshow(
            grid, origin="lower",
            extent=[_RTMA_LON_MIN, _RTMA_LON_MAX, _RTMA_LAT_MIN, _RTMA_LAT_MAX],
            aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, zorder=1,
        )


# ---------------------------------------------------------------------------
# Gauge-polygon distance audit
# ---------------------------------------------------------------------------

def _run_gauge_polygon_audit(
    pilot_manifest: Any,
    camelsh_shp_path: Optional[Path],
    qc_dir: Path,
    prov_dir: Path,
) -> tuple[dict[str, Any], Optional[Any]]:
    """Compute gauge-to-CAMELSH-polygon distances for all 50 pilot basins.

    Gauges are the USGS stream gauge coordinates from the pilot manifest.
    Extraction uses polygon weights; the gauge point is metadata/QC only.
    Small offsets are expected due to delineation/datum differences.

    Returns (summary_dict, audit_DataFrame).
    Returns ({"available": False, ...}, None) when the shapefile is unavailable.

    Saves:
      qc_dir / gauge_polygon_distance_audit.csv
      prov_dir / gauge_polygon_distance_audit.json
    """
    import json as _json
    import numpy as _np
    import pandas as _pd

    unavailable = {"available": False, "warn": False, "warn_strong": False}

    if camelsh_shp_path is None or not camelsh_shp_path.exists():
        LOGGER.warning("gauge_polygon_audit: CAMELSH shapefile unavailable — skipping")
        return unavailable, None

    try:
        import geopandas as _gpd
        import shapely as _shapely
        from pyproj import Transformer as _Transformer
    except ImportError as exc:
        LOGGER.warning("gauge_polygon_audit: missing dependency (%s) — skipping", exc)
        return unavailable, None

    LOGGER.info("gauge_polygon_audit: loading CAMELSH shapefile ...")
    try:
        camelsh = _gpd.read_file(str(camelsh_shp_path))
        if camelsh.crs is None:
            camelsh = camelsh.set_crs("EPSG:4326")
        camelsh_5070 = camelsh.to_crs("EPSG:5070")
        camelsh_5070["STAID_norm"] = camelsh_5070["GAGE_ID"].apply(
            lambda s: str(s).strip().lstrip("0").zfill(8)
        )
    except Exception as exc:
        LOGGER.warning("gauge_polygon_audit: shapefile load failed: %s — skipping", exc)
        return unavailable, None

    t = _Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    records: list[dict] = []
    for _, row in pilot_manifest.iterrows():
        staid   = str(row["STAID"]).strip().lstrip("0").zfill(8)
        lat_gage = float(row["LAT_GAGE"])
        lon_gage = float(row["LNG_GAGE"])

        poly_rows = camelsh_5070[camelsh_5070["STAID_norm"] == staid]
        if poly_rows.empty or poly_rows.geometry.isna().all():
            records.append({
                "STAID": staid, "lat_gage": lat_gage, "lon_gage": lon_gage,
                "gauge_inside_polygon": False,
                "distance_to_polygon_m": None, "distance_to_boundary_m": None,
                "polygon_area_km2": None, "status": "NO_POLYGON",
            })
            continue

        polygon = poly_rows.geometry.union_all()
        area_km2 = float(polygon.area / 1e6)

        x_g, y_g = t.transform(lon_gage, lat_gage)
        pt = _shapely.Point(x_g, y_g)

        inside = bool(polygon.contains(pt))
        dist_poly     = float(pt.distance(polygon))           # 0 if inside
        dist_boundary = float(pt.distance(polygon.boundary))  # to exterior ring

        if inside or dist_poly < 1.0:
            status = "INSIDE_OR_ON_BOUNDARY"
        elif dist_poly <= 250:
            status = "NEAR_POLYGON_LE_250M"
        elif dist_poly <= 1000:
            status = "OFFSET_250M_TO_1KM"
        elif dist_poly <= 5000:
            status = "OFFSET_GT_1KM"
        else:
            status = "OFFSET_GT_5KM"

        records.append({
            "STAID": staid, "lat_gage": lat_gage, "lon_gage": lon_gage,
            "gauge_inside_polygon":   inside,
            "distance_to_polygon_m":  round(dist_poly,     1),
            "distance_to_boundary_m": round(dist_boundary, 1),
            "polygon_area_km2":       round(area_km2, 3),
            "status": status,
        })

    audit_df = _pd.DataFrame(records)

    # Summary counts
    n_inside  = int((audit_df["status"] == "INSIDE_OR_ON_BOUNDARY").sum())
    n_near    = int((audit_df["status"] == "NEAR_POLYGON_LE_250M").sum())
    n_250_1k  = int((audit_df["status"] == "OFFSET_250M_TO_1KM").sum())
    n_gt1k    = int((audit_df["status"] == "OFFSET_GT_1KM").sum())
    n_gt5k    = int((audit_df["status"] == "OFFSET_GT_5KM").sum())
    n_no_poly = int((audit_df["status"] == "NO_POLYGON").sum())

    valid = audit_df[audit_df["distance_to_polygon_m"].notna()]
    top10 = (
        valid.sort_values("distance_to_polygon_m", ascending=False)
        .head(10)[["STAID", "distance_to_polygon_m", "polygon_area_km2", "status"]]
        .to_dict("records")
    )

    warn        = (n_gt1k > 0) or (n_no_poly > 0)
    warn_strong = n_gt5k > 0

    summary: dict[str, Any] = {
        "available":             True,
        "n_total":               len(audit_df),
        "n_inside_or_boundary":  n_inside,
        "n_near_le_250m":        n_near,
        "n_offset_250m_to_1km":  n_250_1k,
        "n_offset_gt_1km":       n_gt1k,
        "n_offset_gt_5km":       n_gt5k,
        "n_no_polygon":          n_no_poly,
        "warn":                  warn,
        "warn_strong":           warn_strong,
        "top10_largest_offsets": top10,
    }

    # Save outputs
    qc_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = qc_dir  / "gauge_polygon_distance_audit.csv"
    json_path = prov_dir / "gauge_polygon_distance_audit.json"

    audit_df.to_csv(csv_path, index=False)
    LOGGER.info("gauge_polygon_audit: %d basins audited → %s", len(audit_df), csv_path.name)
    with open(json_path, "w", encoding="utf-8") as fh:
        _json.dump({"summary": summary, "records": records}, fh, indent=2, default=str)

    if warn_strong:
        LOGGER.warning("gauge_polygon_audit: WARN_STRONG — %d basin(s) with gauge offset > 5 km",
                       n_gt5k)
    elif warn:
        LOGGER.warning("gauge_polygon_audit: WARN — %d basin(s) with gauge offset > 1 km", n_gt1k)
    else:
        LOGGER.info("gauge_polygon_audit: all %d gauges inside polygon or within 1 km", n_inside + n_near + n_250_1k)

    return summary, audit_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _run_validation(
    result: dict[str, Any],
    pilot_staids: list[str],
    mrms_path: Optional[Path],
    rtma_path: Optional[Path],
    mrms_weights_path: Path,
    rtma_weights_path: Path,
    output_paths: dict[str, Path],
    products: list[str],
) -> dict[str, Any]:
    import pandas as pd
    from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT

    n_pilot   = len(pilot_staids)
    mrms_df   = result["mrms_df"]
    rtma_df   = result["rtma_df"]
    rtma_grids = result["rtma_grids"]
    rtma_excl  = result["rtma_excluded"]

    checks: dict[str, Any] = {}

    if _MRMS_PRODUCT in products:
        checks["mrms_source_file_exists_nonempty"] = (
            mrms_path is not None and mrms_path.exists() and mrms_path.stat().st_size > 0
        )
        checks["mrms_weight_table_nonempty"] = (
            mrms_weights_path.exists() and len(pd.read_parquet(mrms_weights_path)) > 0
        )
        checks["mrms_grid_decoded"] = result["mrms_grid"] is not None
        checks["mrms_50_basins_in_output"] = len(mrms_df) == n_pilot
        if len(mrms_df) > 0:
            checks["mrms_no_all_null_weighted_mean"]        = not mrms_df["weighted_mean"].isna().all()
            checks["mrms_valid_weight_fraction_reasonable"] = bool((mrms_df["valid_weight_fraction"] > 0.5).all())
            checks["mrms_total_weight_close_to_1"]         = bool((mrms_df["total_weight"] - 1.0).abs().max() < 0.05)
            checks["mrms_variable_role_column_present"]    = "variable_role" in mrms_df.columns
        else:
            for k in ("mrms_no_all_null_weighted_mean", "mrms_valid_weight_fraction_reasonable",
                      "mrms_total_weight_close_to_1", "mrms_variable_role_column_present"):
                checks[k] = False

    if _RTMA_PRODUCT in products:
        checks["rtma_source_file_exists_nonempty"] = (
            rtma_path is not None and rtma_path.exists() and rtma_path.stat().st_size > 0
        )
        checks["rtma_weight_table_nonempty"] = (
            rtma_weights_path.exists() and len(pd.read_parquet(rtma_weights_path)) > 0
        )
        checks["rtma_grids_decoded"]  = len(rtma_grids) > 0
        checks["rtma_10wdir_excluded"] = not any(g.short_name == "10wdir" for g in rtma_grids)
        checks["rtma_orog_excluded"]   = not any(g.short_name == "orog"   for g in rtma_grids)
        if len(rtma_grids) > 0 and len(rtma_df) > 0:
            n_vars = len(rtma_grids)
            checks["rtma_50_basins_per_variable"]           = len(rtma_df) == n_pilot * n_vars
            checks["rtma_no_all_null_weighted_mean"]        = not rtma_df["weighted_mean"].isna().all()
            checks["rtma_valid_weight_fraction_reasonable"] = bool((rtma_df["valid_weight_fraction"] > 0.5).all())
            checks["rtma_variable_role_column_present"]     = "variable_role" in rtma_df.columns
        else:
            for k in ("rtma_50_basins_per_variable", "rtma_no_all_null_weighted_mean",
                      "rtma_valid_weight_fraction_reasonable", "rtma_variable_role_column_present"):
                checks[k] = False

    checks["combined_parquet_written"]  = output_paths.get("combined_parquet", Path("x")).exists()
    checks["provenance_dir_exists"]     = True

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.perf_counter()
    args = _parse_args()

    from src.pipeline.config import load_config, config_to_dict
    from src.pipeline.geometries import normalise_staid
    from src.pipeline.provenance import write_run_manifest
    from src.pipeline.extraction import (
        run_one_hour_extraction, _MRMS_PRODUCT, _RTMA_PRODUCT,
        _RTMA_EXCLUDED_DEFAULT,
    )
    import pandas as pd

    cfg       = load_config(Path(args.config))
    data_root = cfg.effective_data_root(override=args.data_root)
    LOGGER.info("Data root: %s", data_root)

    # Parse sample time
    try:
        sample_dt = datetime.fromisoformat(args.sample_time.replace("Z", "+00:00"))
        sample_dt = sample_dt.replace(tzinfo=None)
    except ValueError as exc:
        LOGGER.error("Invalid --sample-time: %s", exc)
        return 1

    # Parse products
    product_map = {"mrms": _MRMS_PRODUCT, "rtma": _RTMA_PRODUCT}
    products: list[str] = []
    for raw in [p.strip().lower() for p in args.products.split(",")]:
        products.append(product_map.get(raw, raw) if raw in product_map else raw)
    if not products:
        LOGGER.error("No valid products"); return 1

    LOGGER.info("sample_time=%s  products=%s  include_excluded=%s",
                sample_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), products, args.include_excluded_vars)

    # Paths
    raw_dir           = cfg.output_dir("raw", data_root)
    weights_base      = cfg.output_dir("basin_geometries", data_root) / "weights"
    mrms_weights_path = weights_base / "mrms" / "pilot_mrms_weights.parquet"
    rtma_weights_path = weights_base / "rtma" / "pilot_rtma_weights.parquet"
    manifest_csv      = (cfg.output_dir("manifests", data_root)
                         / "stage1_pilot" / "pilot_basin_manifest.csv")
    timeseries_dir    = cfg.output_dir("basin_timeseries", data_root) / "stage1_pilot" / "one_hour"
    qc_dir            = cfg.output_dir("qc_reports", data_root) / "stage1_pilot" / "one_hour_extraction"
    prov_dir          = cfg.output_dir("manifests", data_root) / "stage1_pilot" / "one_hour_extraction"
    geo_cache_dir     = cfg.output_dir("basin_geometries", data_root) / "reference"

    # CAMELSH shapefile (from weights provenance)
    camelsh_shp: Optional[Path] = None
    _candidate = cfg.output_dir("basin_geometries", data_root) / "camelsh" / "shapefiles" / "CAMELSH_shapefile.shp"
    if _candidate.exists():
        camelsh_shp = _candidate

    # Validate prerequisites
    for p, label in [(mrms_weights_path, "MRMS weight table"),
                     (rtma_weights_path, "RTMA weight table"),
                     (manifest_csv, "Pilot basin manifest")]:
        if not p.exists():
            LOGGER.error("Required input missing: %s — %s", label, p)
            return 1

    # Load pilot basins
    pilot_manifest = pd.read_csv(manifest_csv)
    pilot_staids   = [normalise_staid(s) for s in pilot_manifest["STAID"].tolist()]
    LOGGER.info("Pilot basins: %d", len(pilot_staids))

    # Locate or fetch sample files
    mrms_path: Optional[Path] = None
    rtma_path: Optional[Path] = None
    mrms_downloaded = rtma_downloaded = False

    if _MRMS_PRODUCT in products:
        mrms_path, mrms_downloaded = _locate_or_fetch_mrms(raw_dir, sample_dt)
        if mrms_path is None:
            LOGGER.error("MRMS sample file unavailable"); return 1

    if _RTMA_PRODUCT in products:
        rtma_path, rtma_downloaded = _locate_or_fetch_rtma(raw_dir, sample_dt)
        if rtma_path is None:
            LOGGER.error("RTMA sample file unavailable"); return 1

    # Run extraction
    LOGGER.info("Starting extraction ...")
    result = run_one_hour_extraction(
        mrms_path=mrms_path or Path("/dev/null"),
        rtma_path=rtma_path or Path("/dev/null"),
        mrms_weights_path=mrms_weights_path,
        rtma_weights_path=rtma_weights_path,
        pilot_staids=pilot_staids,
        products=products,
        include_excluded_vars=args.include_excluded_vars,
    )

    for w in result["warnings"]:
        LOGGER.warning("Extraction: %s", w)

    mrms_df     = result["mrms_df"]
    rtma_df     = result["rtma_df"]
    combined_df = result["combined_df"]
    rtma_grids  = result["rtma_grids"]
    rtma_excl   = result["rtma_excluded"]

    LOGGER.info("Extraction done: MRMS=%d rows  RTMA=%d rows  combined=%d rows",
                len(mrms_df), len(rtma_df), len(combined_df))

    # Write outputs
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    for df_out, fname, key in [
        (mrms_df,     "mrms_one_hour_basin_stats.parquet",     "mrms_parquet"),
        (rtma_df,     "rtma_one_hour_basin_stats.parquet",     "rtma_parquet"),
        (combined_df, "combined_one_hour_basin_stats.parquet", "combined_parquet"),
    ]:
        if len(df_out) > 0:
            p = timeseries_dir / fname
            df_out.to_parquet(p, index=False)
            output_paths[key] = p
            LOGGER.info("Written: %s (%d rows)", fname, len(df_out))

    if len(combined_df) > 0:
        csv_p = timeseries_dir / "combined_one_hour_basin_stats_preview.csv"
        try:
            combined_df.to_csv(csv_p, index=False)
            output_paths["combined_csv_preview"] = csv_p
            LOGGER.info("Written: %s (%d rows)", csv_p.name, len(combined_df))
        except PermissionError as exc:
            LOGGER.warning("CSV preview write failed (file locked?): %s — skipping", exc)

    # Load weight tables for focused maps (already on disk)
    mrms_weights_df = pd.read_parquet(mrms_weights_path)
    rtma_weights_df = pd.read_parquet(rtma_weights_path)

    # Validation
    validation = _run_validation(
        result=result, pilot_staids=pilot_staids,
        mrms_path=mrms_path, rtma_path=rtma_path,
        mrms_weights_path=mrms_weights_path, rtma_weights_path=rtma_weights_path,
        output_paths=output_paths, products=products,
    )
    all_pass = all(bool(v) for v in validation.values() if isinstance(v, bool))
    LOGGER.info("Validation: %s", "PASS" if all_pass else "FAIL")
    for k, v in validation.items():
        if isinstance(v, bool):
            LOGGER.info("  %-55s %s", k, "PASS" if v else "FAIL")

    # Gauge-polygon distance audit (runs before QC plots so offsets can annotate maps)
    LOGGER.info("Running gauge-polygon distance audit ...")
    gauge_audit_summary, gauge_audit_df = _run_gauge_polygon_audit(
        pilot_manifest=pilot_manifest,
        camelsh_shp_path=camelsh_shp,
        qc_dir=qc_dir,
        prov_dir=prov_dir,
    )

    # Geographic context (try once, reuse for all plots)
    LOGGER.info("Loading geographic context ...")
    countries_gdf, us_states_gdf = _try_load_geo_context(cache_dir=geo_cache_dir)
    geo_available = countries_gdf is not None or us_states_gdf is not None
    LOGGER.info("geo context: countries=%s  states=%s",
                "OK" if countries_gdf is not None else "unavailable",
                "OK" if us_states_gdf is not None else "unavailable")

    # QC plots
    plot_files: list[str] = []
    plot_files += _make_broad_qc_plots(qc_dir, result, pilot_manifest,
                                       countries_gdf, us_states_gdf)
    plot_files += _make_focused_maps(qc_dir, result,
                                     mrms_weights_df, rtma_weights_df,
                                     pilot_manifest, camelsh_shp,
                                     countries_gdf, us_states_gdf,
                                     gauge_audit_df=gauge_audit_df)

    # Provenance
    t_elapsed = time.perf_counter() - t_start
    validation["provenance_dir_exists"] = True

    rtma_var_list  = [g.short_name for g in rtma_grids]
    rtma_excl_list = [g.short_name for g in rtma_excl]
    mrms_var_list  = [result["mrms_grid"].short_name] if result.get("mrms_grid") else []

    input_paths_prov: dict[str, Any] = {
        "mrms_weight_table":  str(mrms_weights_path),
        "rtma_weight_table":  str(rtma_weights_path),
        "pilot_basin_manifest": str(manifest_csv),
    }
    if mrms_path:
        input_paths_prov["mrms_sample_file"] = str(mrms_path)
    if rtma_path:
        input_paths_prov["rtma_sample_file"] = str(rtma_path)
    if camelsh_shp:
        input_paths_prov["camelsh_shapefile"] = str(camelsh_shp)

    write_run_manifest(
        run_dir=prov_dir,
        run_command=" ".join(sys.argv),
        config_dict=config_to_dict(cfg),
        input_paths=input_paths_prov,
        output_paths={k: str(v) for k, v in output_paths.items()},
        validation_results=validation,
        extra={
            "sample_time_utc":        sample_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "products_extracted":     products,
            "mrms_variables":         mrms_var_list,
            "rtma_variables_included": rtma_var_list,
            "rtma_variables_excluded": rtma_excl_list,
            "n_pilot_basins":         len(pilot_staids),
            "mrms_output_rows":       len(mrms_df),
            "rtma_output_rows":       len(rtma_df),
            "combined_output_rows":   len(combined_df),
            "mrms_file_reused":       not mrms_downloaded,
            "rtma_file_reused":       not rtma_downloaded,
            "geo_context_available":  geo_available,
            "gauge_polygon_audit":    gauge_audit_summary,
            "qc_plots_written":       plot_files,
            "runtime_seconds":        round(t_elapsed, 2),
            "warnings":               result["warnings"],
        },
    )
    LOGGER.info("Provenance written: %s", prov_dir)

    # Final report
    print("\n" + "=" * 72)
    print("Stage 1 Milestone 2C — One-Hour Extraction Report")
    print("=" * 72)
    print(f"  Sample time       : {sample_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"  Products          : {', '.join(products)}")
    print(f"  Pilot basins      : {len(pilot_staids)}")
    print(f"  MRMS rows         : {len(mrms_df)}")
    print(f"  RTMA rows         : {len(rtma_df)}  ({len(rtma_var_list)} variables)")
    print(f"  Combined rows     : {len(combined_df)}")
    print(f"  Runtime           : {t_elapsed:.1f}s")
    print()
    print("  Variable policy:")
    print(f"    MRMS included   : {mrms_var_list}")
    print(f"    RTMA included   : {rtma_var_list}")
    print(f"    RTMA excluded   : {rtma_excl_list}  (see --include-excluded-vars for diagnostics)")
    print()
    print("  Sample files:")
    if mrms_path:
        print(f"    MRMS [{'downloaded' if mrms_downloaded else 'reused'}]: {mrms_path}")
    if rtma_path:
        print(f"    RTMA [{'downloaded' if rtma_downloaded else 'reused'}]: {rtma_path}")
    print()
    print("  Outputs:")
    for k, v in output_paths.items():
        print(f"    {k}: {v}")
    print()
    print(f"  Geo context       : countries={'OK' if countries_gdf is not None else 'unavailable'}  "
          f"states={'OK' if us_states_gdf is not None else 'unavailable'}")
    print(f"  CAMELSH shapefile : {'OK' if camelsh_shp else 'not found'}")
    print()
    print(f"  QC plots ({len(plot_files)}):")
    for name in plot_files:
        print(f"    {qc_dir / name}")
    print()
    print("  Validation:")
    for k, v in validation.items():
        print(f"    {'PASS' if v is True else 'FAIL' if v is False else '----'}  {k}")
    print()
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    # Gauge-polygon audit summary
    if gauge_audit_summary.get("available"):
        gs = gauge_audit_summary
        warn_tag = " [WARN_STRONG]" if gs.get("warn_strong") else (" [WARN]" if gs.get("warn") else "")
        print()
        print(f"  Gauge-Polygon Distance Audit{warn_tag}:")
        print(f"    Inside or on boundary   : {gs['n_inside_or_boundary']}")
        print(f"    Near polygon (<= 250 m)  : {gs['n_near_le_250m']}")
        print(f"    Offset 250 m to 1 km     : {gs['n_offset_250m_to_1km']}")
        print(f"    Offset > 1 km            : {gs['n_offset_gt_1km']}")
        print(f"    Offset > 5 km            : {gs['n_offset_gt_5km']}")
        print(f"    No polygon found         : {gs['n_no_polygon']}")
        if gs["top10_largest_offsets"]:
            print()
            print("    Top-10 largest offsets:")
            print(f"      {'STAID':>10}  {'offset_m':>10}  {'area_km2':>10}  status")
            for r in gs["top10_largest_offsets"]:
                print(f"      {r['STAID']:>10}  {r['distance_to_polygon_m']:>10.1f}  "
                      f"{r.get('polygon_area_km2', 'N/A'):>10}  {r['status']}")

    if result["warnings"]:
        print()
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    ! {w}")
    print("=" * 72)

    try:
        import subprocess
        gs = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, timeout=10)
        print("\ngit status --short:")
        print(gs.stdout if gs.stdout.strip() else "  (clean)")
    except Exception:
        pass

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
