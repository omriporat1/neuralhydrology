#!/usr/bin/env python3
"""Compute MRMS and RTMA basin-grid overlap weights for Stage 1 pilot basins.

This is Milestone 2B of the Stage 1 pilot data pipeline.
It reads the pilot basin manifest, loads or approximates basin geometries, then
computes fractional area overlap weights for each pilot basin against the MRMS
and RTMA grids.

GEOMETRY NOTE
-------------
Real CAMELSH basin polygons are required. The script auto-discovers them at:
  {data_root}/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp
or the path set in camelsh.basin_polygons in configs/pilot_stage1.yaml.

If polygons are absent, the script exits with a clear error and instructions.
A circular-buffer smoke-test fallback is available with --allow-fallback-circles
but produces weights that are NOT valid for training or scientific extraction.

PREREQUISITE
------------
Grid-definition JSON files must exist before this script runs.

Pilot (default, auto-discovered):
  {data_root}/09_manifests/stage1_pilot/grid_definitions/mrms_grid_definition.json
  {data_root}/09_manifests/stage1_pilot/grid_definitions/rtma_grid_definition.json

Full-period v001 (flat layout, auto-discovered if pilot path absent):
  {data_root}/grid_definitions/mrms_grid_definition.json
  {data_root}/grid_definitions/rtma_grid_definition.json

Explicit override (recommended for v001 to avoid ambiguity):
  --grid-def-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions

OUTPUTS (under configured data root)
--------------------------------------
  02_basin_geometries/weights/mrms/pilot_mrms_weights.parquet
  02_basin_geometries/weights/rtma/pilot_rtma_weights.parquet
  09_manifests/stage1_pilot/weights/
      weight_validation_mrms.json
      weight_validation_rtma.json
      weight_summary.json
      weight_summary.md
      manifest.json / summary.json / run_command.txt / git_commit.txt
  06_qc_reports/stage1_pilot/weights/
      overview_basin_map.png
      mrms_sample_basins.png
      rtma_sample_basins.png

Usage:
    python scripts/build_stage1_basin_weights.py --config configs/pilot_stage1.yaml
    python scripts/build_stage1_basin_weights.py --data-root /my/data/root
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("build_weights")

from src.pipeline.config import load_config, config_to_dict, PipelineConfig
from src.pipeline.provenance import write_run_manifest, git_commit_hash
from src.pipeline.geometries import (
    load_pilot_geometries, normalise_staid, resolve_polygon_path,
    CAMELSH_CURRENT_DOI, CAMELSH_POLYGON_SOURCE_DOI, CAMELSH_PREFERRED_FILE,
)
from src.pipeline.weights import (
    mrms_spec_from_json,
    rtma_spec_from_json,
    compute_mrms_weights,
    compute_rtma_weights,
    validate_weight_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "pilot_stage1.yaml"))
    p.add_argument("--data-root", default=None, help="Override data root path")
    p.add_argument(
        "--basin-list",
        default=None,
        metavar="CSV",
        help=(
            "Override the auto-discovered pilot basin manifest with this CSV file. "
            "The CSV must have a STAID column (8-char zero-padded gauge IDs). "
            "Optional columns: DRAIN_SQKM, LAT_GAGE, LNG_GAGE, pilot_role. "
            "Use this to build weights for the full v001 basin list (2,752 basins) "
            "rather than the 50-basin pilot manifest."
        ),
    )
    p.add_argument(
        "--out-tag",
        default="pilot",
        metavar="TAG",
        help=(
            "Output filename prefix for the weight Parquet files "
            "(default: 'pilot' → pilot_mrms_weights.parquet). "
            "Use e.g. 'v001_2752' to produce v001_2752_mrms_weights.parquet."
        ),
    )
    p.add_argument(
        "--grid-def-dir",
        default=None,
        metavar="DIR",
        dest="grid_def_dir",
        help=(
            "Directory containing mrms_grid_definition.json and rtma_grid_definition.json. "
            "If not given, the script searches in order: "
            "(1) {data_root}/grid_definitions/  [v001 full-period flat layout], "
            "(2) {data_root}/09_manifests/stage1_pilot/grid_definitions/  [pilot layout]. "
            "Use this flag to avoid ambiguity, e.g.: "
            "--grid-def-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions"
        ),
    )
    p.add_argument(
        "--skip-qc-plots",
        action="store_true",
        dest="skip_qc_plots",
        help=(
            "Skip optional QC plots (overview map and sample basin weight plots). "
            "Recommended for h2o operational runs where matplotlib may be slow or "
            "display columns (LNG_GAGE, DRAIN_SQKM) are absent from the basin list. "
            "Weight table validation and Parquet output are unaffected. "
            "Plot failures are always advisory — they never cause a nonzero exit — "
            "but this flag avoids the overhead entirely."
        ),
    )
    p.add_argument(
        "--allow-fallback-circles",
        action="store_true",
        help=(
            "SMOKE TEST ONLY: allow circular-buffer geometry fallback when "
            "CAMELSH shapefiles are unavailable. Weights produced with this flag "
            "are NOT valid for training or scientific extraction."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# QC plots
# ---------------------------------------------------------------------------

def _select_sample_basins(basin_gdf, weights_mrms, weights_rtma, n=6):
    """Return up to n representative STAIDs for QC detail plots.

    Selects basins spanning the drainage-area distribution. Works whether or
    not DRAIN_SQKM / pilot_role columns are present (v001 CAMELSH schema has
    neither; only LAYER, MAP_NAME, AREA, PERIMETER, GAGE_ID, geometry).
    """
    import numpy as np

    staids_with_weights = set(weights_mrms["STAID"].unique()) & set(weights_rtma["STAID"].unique())
    gdf = basin_gdf[basin_gdf["STAID"].isin(staids_with_weights)].copy()
    if gdf.empty:
        return basin_gdf["STAID"].tolist()[:n]

    # Determine sort column for size-stratified sampling.
    # Priority: DRAIN_SQKM → AREA (shapefile native) → geometry area → STAID sort.
    if "DRAIN_SQKM" in gdf.columns:
        sort_col = "DRAIN_SQKM"
    elif "AREA" in gdf.columns:
        sort_col = "AREA"
    else:
        try:
            gdf["_geom_area_km2"] = gdf.geometry.to_crs(5070).area / 1e6
            sort_col = "_geom_area_km2"
        except Exception:
            sort_col = None

    if sort_col is not None:
        gdf = gdf.sort_values(sort_col).reset_index(drop=True)
    else:
        gdf = gdf.sort_values("STAID").reset_index(drop=True)

    n_basins = len(gdf)
    indices = []
    # smallest, ~25th pct, median, ~75th pct, largest
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        i = min(int(frac * (n_basins - 1)), n_basins - 1)
        if i not in indices:
            indices.append(i)

    selected = set(gdf.iloc[indices]["STAID"].tolist())

    # Add a HOLDOUT_QC and an EXCLUDE_QC if pilot_role column is present
    if "pilot_role" in gdf.columns:
        for role in ("HOLDOUT_QC", "EXCLUDE_QC"):
            cands = gdf[gdf["pilot_role"] == role]
            if not cands.empty and len(selected) < n:
                selected.add(cands.iloc[0]["STAID"])

    return list(selected)[:n]


def _plot_overview_map(basin_gdf, mrms_spec, rtma_spec, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    try:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Grid extents
        ax.add_patch(plt.Rectangle(
            (mrms_spec.lon_first, mrms_spec.lat_last),
            mrms_spec.lon_last - mrms_spec.lon_first,
            mrms_spec.lat_first - mrms_spec.lat_last,
            fill=False, edgecolor="steelblue", linestyle="--", linewidth=1,
            label=f"MRMS grid ({mrms_spec.ncols}×{mrms_spec.nrows})",
        ))
        # RTMA bbox (hardcoded; does not require gauge columns)
        rtma_bbox_min_lon, rtma_bbox_max_lon = -138.4, -59.0
        rtma_bbox_min_lat, rtma_bbox_max_lat = 19.2, 57.1
        ax.add_patch(plt.Rectangle(
            (rtma_bbox_min_lon, rtma_bbox_min_lat),
            rtma_bbox_max_lon - rtma_bbox_min_lon,
            rtma_bbox_max_lat - rtma_bbox_min_lat,
            fill=False, edgecolor="darkorange", linestyle="-.", linewidth=1,
            label="RTMA grid (approx bbox)",
        ))
        # CONUS outline
        ax.add_patch(plt.Rectangle((-126, 24), 60, 26,
            fill=False, edgecolor="black", linestyle=":", linewidth=0.7, label="CONUS bbox"))

        # Basin centroids — derive lon/lat from gauge columns or geometry centroids
        has_lng_lat = "LNG_GAGE" in basin_gdf.columns and "LAT_GAGE" in basin_gdf.columns
        try:
            plot_gdf = basin_gdf.copy()
            if has_lng_lat:
                plot_gdf["_lon"] = basin_gdf["LNG_GAGE"].astype(float)
                plot_gdf["_lat"] = basin_gdf["LAT_GAGE"].astype(float)
            else:
                crs4326 = basin_gdf.to_crs(4326)
                plot_gdf["_lon"] = crs4326.geometry.centroid.x
                plot_gdf["_lat"] = crs4326.geometry.centroid.y
            # Derive area column for sizing
            if "DRAIN_SQKM" in basin_gdf.columns:
                plot_gdf["_area"] = basin_gdf["DRAIN_SQKM"].astype(float)
            elif "AREA" in basin_gdf.columns:
                plot_gdf["_area"] = basin_gdf["AREA"].astype(float)
            else:
                plot_gdf["_area"] = basin_gdf.geometry.to_crs(5070).area / 1e6
            sizes_all = np.clip(np.log10(plot_gdf["_area"] + 1) * 20, 10, 80)
            # Colour by pilot_role if present, else single colour
            if "pilot_role" in plot_gdf.columns:
                role_colors = {
                    "TRAIN": "#2196F3", "HOLDOUT_QC": "#FF9800", "EXCLUDE_QC": "#F44336",
                }
                for role, color in role_colors.items():
                    sub = plot_gdf[plot_gdf["pilot_role"] == role]
                    if sub.empty:
                        continue
                    sizes = np.clip(np.log10(sub["_area"] + 1) * 20, 10, 80)
                    ax.scatter(sub["_lon"], sub["_lat"], c=color, s=sizes,
                               alpha=0.85, zorder=5, label=f"{role} (n={len(sub)})")
            else:
                ax.scatter(plot_gdf["_lon"], plot_gdf["_lat"], c="#2196F3",
                           s=sizes_all, alpha=0.85, zorder=5,
                           label=f"all basins (n={len(plot_gdf)})")
        except Exception as exc:
            LOGGER.info("Could not plot basin scatter (missing columns): %s", exc)

        ax.set_xlim(-135, -55)
        ax.set_ylim(20, 55)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"Flash-NH Stage 1 pilot basins (n={len(basin_gdf)}) — overview map\n"
                     "marker size ∝ log(drainage area)")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return True
    except Exception as exc:
        LOGGER.warning("overview map failed: %s", exc)
        return False


def _plot_sample_basins(
    basin_gdf,
    weights_df,
    sample_staids: list[str],
    product: str,
    out_path: Path,
    geom_method: str = "unknown",
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    n = len(sample_staids)
    if n == 0:
        return False
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    use_real_polygons = geom_method == "camelsh_shapefile"

    try:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        axes_flat = np.array(axes).ravel() if n > 1 else [axes]

        for ax_idx, staid in enumerate(sample_staids):
            ax = axes_flat[ax_idx]
            basin_row = basin_gdf[basin_gdf["STAID"] == staid]
            if basin_row.empty:
                ax.set_visible(False)
                continue

            br = basin_row.iloc[0]
            # Derive centroid; fall back to geometry if gauge columns absent
            if "LAT_GAGE" in br.index and "LNG_GAGE" in br.index:
                lat_c = float(br["LAT_GAGE"])
                lon_c = float(br["LNG_GAGE"])
            else:
                try:
                    centroid = basin_row.to_crs(4326).geometry.centroid.iloc[0]
                    lon_c, lat_c = centroid.x, centroid.y
                except Exception:
                    ax.set_visible(False)
                    continue
            area = float(br["DRAIN_SQKM"]) if "DRAIN_SQKM" in br.index else (
                float(br["AREA"]) if "AREA" in br.index else 0.0
            )
            role = br.get("pilot_role", "?")

            # Get weights for this basin
            bw = weights_df[weights_df["STAID"] == staid]
            n_cells = len(bw)
            w_sum = float(bw["normalized_weight"].sum()) if n_cells > 0 else 0.0

            if n_cells > 0:
                weights_norm = bw["normalized_weight"].values
                w_max = weights_norm.max()
                sc = ax.scatter(
                    bw["lon_center"].values, bw["lat_center"].values,
                    c=weights_norm, cmap="viridis", s=15, alpha=0.85,
                    zorder=4, vmin=0, vmax=w_max,
                )
                fig.colorbar(sc, ax=ax, label="norm. weight", fraction=0.04)

            # Plot gauge point
            ax.plot(lon_c, lat_c, marker="*", color="red", markersize=10, zorder=6)

            if use_real_polygons:
                # Project the real CAMELSH polygon from EPSG:5070 → EPSG:4326 for plotting
                try:
                    basin_4326 = basin_row.to_crs(4326)
                    geom_4326 = basin_4326.iloc[0].geometry
                    geoms = list(geom_4326.geoms) if hasattr(geom_4326, "geoms") else [geom_4326]
                    for part in geoms:
                        if not hasattr(part, "exterior"):
                            continue
                        xp, yp = part.exterior.xy
                        ax.plot(xp, yp, color="black", linewidth=1.2, zorder=5)
                        for interior in part.interiors:
                            xi, yi = interior.xy
                            ax.plot(xi, yi, color="gray", linewidth=0.7, linestyle="--", zorder=5)
                    # Axis limits from actual polygon bounds + 15% padding
                    minx, miny, maxx, maxy = geom_4326.bounds
                    pad_lon = max((maxx - minx) * 0.15, 0.02)
                    pad_lat = max((maxy - miny) * 0.15, 0.02)
                    ax.set_xlim(minx - pad_lon, maxx + pad_lon)
                    ax.set_ylim(miny - pad_lat, maxy + pad_lat)
                except Exception as exc:
                    LOGGER.debug("Could not plot polygon for %s: %s — falling back to gauge view", staid, exc)
                    ax.set_xlim(lon_c - 0.5, lon_c + 0.5)
                    ax.set_ylim(lat_c - 0.5, lat_c + 0.5)
            else:
                # Circular-buffer fallback: draw the approximate ellipse used for geometry
                import matplotlib.patches as mpatches
                r_km = float(np.sqrt(max(area, 0.01) * 1e6 / np.pi)) / 1000.0
                r_deg_lat = r_km / 111.0
                r_deg_lon = r_km / (111.0 * abs(float(np.cos(np.radians(lat_c)))) + 1e-9)
                ell = mpatches.Ellipse(
                    (lon_c, lat_c), 2 * r_deg_lon, 2 * r_deg_lat,
                    fill=False, edgecolor="black", linestyle="-", linewidth=1.2,
                )
                ax.add_patch(ell)
                pad_lon = max(r_deg_lon * 1.5, 0.05)
                pad_lat = max(r_deg_lat * 1.5, 0.05)
                ax.set_xlim(lon_c - pad_lon, lon_c + pad_lon)
                ax.set_ylim(lat_c - pad_lat, lat_c + pad_lat)

            title = (f"STAID {staid}\n{role} | {area:.0f} km² | {n_cells} cells | sum={w_sum:.3f}")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("lon", fontsize=7)
            ax.set_ylabel("lat", fontsize=7)
            ax.tick_params(labelsize=6)

        # Hide unused axes
        for ax_idx in range(n, len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        geom_label = (
            "CAMELSH polygon geometry"
            if use_real_polygons
            else "TEST-ONLY circular-buffer geometry"
        )
        fig.suptitle(
            f"Sample basin weights — {product}\n"
            f"({geom_label}; points = cell centroids, colour = weight)",
            fontsize=10,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return True
    except Exception as exc:
        LOGGER.warning("%s sample basin plot failed: %s", product, exc)
        return False


# ---------------------------------------------------------------------------
# Validation summary writer
# ---------------------------------------------------------------------------

def _build_summary_md(
    mrms_val: dict, rtma_val: dict,
    geom_method: str, missing_geom: list,
    n_total: int, n_with_geom: int,
    polygon_source: Optional[str] = None,
) -> str:
    def _check_row(k, v) -> str:
        if isinstance(v, bool):
            return f"- **{k}**: {'PASS' if v else 'FAIL'}"
        return f"- **{k}**: {v}"

    test_only = geom_method == "circular_buffer_test_only"
    overall_ok = (
        all(mrms_val["checks"].values())
        and all(rtma_val["checks"].values())
        and n_with_geom == n_total
    )

    lines = [
        "# Stage 1 Basin Weight Summary",
        "",
    ]
    if test_only:
        lines += [
            "> **WARNING — TEST-ONLY WEIGHTS**",
            "> Geometry method is `circular_buffer_test_only`.",
            "> These weights MUST NOT be used for training data, scientific extraction,",
            "> or any downstream result. Re-run with real CAMELSH polygons.",
            "",
        ]
    lines += [
        f"**Overall:** {'PASS' if overall_ok else 'FAIL'}",
        f"**Geometry method:** {geom_method}",
        f"**Basins with geometry:** {n_with_geom}/{n_total}",
    ]
    if polygon_source:
        lines.append(f"**Polygon source:** `{polygon_source}`")
    lines += [
        f"**CAMELSH reference DOI:** {CAMELSH_CURRENT_DOI}",
        f"**Polygon source DOI:** {CAMELSH_POLYGON_SOURCE_DOI}",
    ]
    if missing_geom:
        lines.append(f"**Missing geometry:** {missing_geom}")

    for product, val in [("MRMS", mrms_val), ("RTMA", rtma_val)]:
        lines += ["", f"## {product}", ""]
        lines += [
            f"| Metric | Value |",
            f"|---|---|",
            f"| Basins with weights | {val['n_with_weights']}/{val['n_pilot_staids']} |",
            f"| Total weight records | {val['total_weight_records']} |",
            f"| Cells/basin (min/med/max) | {val['cells_per_basin_min']} / {val['cells_per_basin_median']} / {val['cells_per_basin_max']} |",
            f"| Weight sum (min/med/max) | {val.get('weight_sum_min','?')} / {val.get('weight_sum_median','?')} / {val.get('weight_sum_max','?')} |",
            f"| Negative weights | {val['n_negative_weights']} |",
            f"| Basins with bad sum | {val['n_basins_bad_sum']} |",
        ]
        if val["suspicious_basins"]:
            lines.append(f"| Suspicious basins | {len(val['suspicious_basins'])} |")
        lines += ["", "**Checks:**", ""]
        for k, v in val["checks"].items():
            lines.append(_check_row(k, v))

    lines += [
        "", "## Next step",
        "",
        "Implement one-hour basin-statistic extraction using these weight tables.",
        "See `docs/stage1_basin_weights.md` for the extraction plan.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        cfg = load_config(config_path)
        LOGGER.info("Config: %s", config_path)
    else:
        cfg = PipelineConfig()
        LOGGER.warning("Config not found at %s; using defaults.", config_path)

    data_root = cfg.effective_data_root(override=args.data_root)
    LOGGER.info("Data root: %s", data_root)
    LOGGER.info("Git:       %s", git_commit_hash() or "unknown")

    # ---- Locate required inputs ----
    if args.basin_list:
        pilot_manifest = Path(args.basin_list)
        if not pilot_manifest.exists():
            LOGGER.error("--basin-list file not found: %s", pilot_manifest)
            sys.exit(1)
        LOGGER.info("Basin list override: %s", pilot_manifest)
    else:
        pilot_manifest = data_root / "09_manifests" / "stage1_pilot" / "pilot_basin_manifest.csv"
        if not pilot_manifest.exists():
            LOGGER.error("Pilot manifest not found: %s  (run run_stage1_pilot_dry_run.py first)", pilot_manifest)
            sys.exit(1)

    # Resolve grid definition directory.
    # Priority: --grid-def-dir arg > flat v001 layout > pilot legacy layout.
    _pilot_grid_dir    = data_root / "09_manifests" / "stage1_pilot" / "grid_definitions"
    _fullperiod_grid_dir = data_root / "grid_definitions"

    if args.grid_def_dir:
        grid_def_dir = Path(args.grid_def_dir)
        LOGGER.info("Grid def dir (explicit): %s", grid_def_dir)
    elif _fullperiod_grid_dir.is_dir() and (_fullperiod_grid_dir / "mrms_grid_definition.json").exists():
        grid_def_dir = _fullperiod_grid_dir
        LOGGER.info("Grid def dir (v001 flat layout): %s", grid_def_dir)
    elif _pilot_grid_dir.is_dir() and (_pilot_grid_dir / "mrms_grid_definition.json").exists():
        grid_def_dir = _pilot_grid_dir
        LOGGER.info("Grid def dir (pilot legacy layout): %s", grid_def_dir)
    else:
        LOGGER.error(
            "Grid definition JSONs not found. Searched:\n"
            "  (1) --grid-def-dir: %s\n"
            "  (2) v001 flat:      %s\n"
            "  (3) pilot legacy:   %s\n"
            "Transfer grid definitions first (prepare_stage1_forcing_inputs_h2o.ps1)\n"
            "or pass --grid-def-dir <path> explicitly.",
            args.grid_def_dir or "(not provided)",
            _fullperiod_grid_dir,
            _pilot_grid_dir,
        )
        sys.exit(1)

    mrms_json = grid_def_dir / "mrms_grid_definition.json"
    rtma_json = grid_def_dir / "rtma_grid_definition.json"
    for p in [mrms_json, rtma_json]:
        if not p.exists():
            LOGGER.error(
                "Grid definition not found: %s\n"
                "  Grid def dir resolved to: %s\n"
                "  Pass --grid-def-dir to override.",
                p, grid_def_dir,
            )
            sys.exit(1)

    # ---- Output paths ----
    weights_manifest_dir = data_root / "09_manifests" / "stage1_pilot" / "weights"
    qc_dir = data_root / "06_qc_reports" / "stage1_pilot" / "weights"
    mrms_weights_dir = data_root / "02_basin_geometries" / "weights" / "mrms"
    rtma_weights_dir = data_root / "02_basin_geometries" / "weights" / "rtma"
    for d in [weights_manifest_dir, qc_dir, mrms_weights_dir, rtma_weights_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load grid specs ----
    LOGGER.info("Loading grid specs ...")
    mrms_spec = mrms_spec_from_json(mrms_json)
    rtma_spec = rtma_spec_from_json(rtma_json)
    LOGGER.info("MRMS: %dx%d grid, lat_first=%.3f, lon_first=%.3f, dy=%.4f",
                mrms_spec.nrows, mrms_spec.ncols, mrms_spec.lat_first, mrms_spec.lon_first, mrms_spec.dy)
    LOGGER.info("RTMA: %dx%d grid, dx=%.1fm, dy=%.1fm; computing origin ...",
                rtma_spec.nj, rtma_spec.ni, rtma_spec.dx_m, rtma_spec.dy_m)
    _ = rtma_spec.x0  # trigger origin computation and log it
    LOGGER.info("RTMA origin: x0=%.1fm, y0=%.1fm", rtma_spec.x0, rtma_spec.y0)

    # ---- Load basin geometries ----
    # Priority: explicit config path → auto-discovery under data_root → fallback (if allowed)
    polygon_path = cfg.camelsh.basin_polygons or None
    id_col = cfg.camelsh.basin_polygon_id_column or None

    # Resolve the actual polygon path up-front so we can record it in provenance,
    # regardless of whether it came from explicit config or auto-discovery.
    resolved_polygon_path = resolve_polygon_path(
        Path(polygon_path) if polygon_path else None,
        data_root,
    )
    LOGGER.info(
        "Loading pilot basin geometries (resolved_path=%s, id_col=%s, allow_fallback=%s) ...",
        resolved_polygon_path, id_col, args.allow_fallback_circles,
    )
    t0 = time.perf_counter()
    try:
        basin_gdf, geom_method, missing_geom = load_pilot_geometries(
            pilot_manifest,
            resolved_polygon_path,
            allow_fallback=args.allow_fallback_circles,
            id_col=id_col,
            data_root=data_root,
        )
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)
    LOGGER.info(
        "Geometries: method=%s, n=%d, missing=%d (%.1fs)",
        geom_method, len(basin_gdf), len(missing_geom),
        time.perf_counter() - t0,
    )
    test_only = geom_method == "circular_buffer_test_only"
    if test_only:
        LOGGER.warning(
            "============================================================\n"
            "  TEST-ONLY MODE: circular-buffer geometry fallback in use.\n"
            "  Weights produced in this run MUST NOT be used for:\n"
            "    - Training data generation\n"
            "    - Scientific extraction or evaluation\n"
            "  Re-run without --allow-fallback-circles once CAMELSH\n"
            "  polygons are available.\n"
            "============================================================"
        )

    import pandas as pd
    pilot_df = pd.read_csv(pilot_manifest, dtype={"STAID": str})
    from src.pipeline.geometries import normalise_staid as _ns
    pilot_staids = [_ns(s) for s in pilot_df["STAID"].tolist()]

    if basin_gdf.empty:
        LOGGER.error("No basin geometries loaded; cannot compute weights.")
        sys.exit(1)

    # ---- Compute MRMS weights ----
    LOGGER.info("Computing MRMS weights ...")
    t0 = time.perf_counter()
    mrms_weights = compute_mrms_weights(basin_gdf, mrms_spec, geom_method=geom_method)
    mrms_time = time.perf_counter() - t0
    LOGGER.info("MRMS weights: %d records in %.1fs", len(mrms_weights), mrms_time)

    # ---- Compute RTMA weights ----
    LOGGER.info("Computing RTMA weights ...")
    t0 = time.perf_counter()
    rtma_weights = compute_rtma_weights(basin_gdf, rtma_spec, geom_method=geom_method)
    rtma_time = time.perf_counter() - t0
    LOGGER.info("RTMA weights: %d records in %.1fs", len(rtma_weights), rtma_time)

    # ---- Write Parquet tables ----
    out_tag = args.out_tag or "pilot"
    mrms_parquet = mrms_weights_dir / f"{out_tag}_mrms_weights.parquet"
    rtma_parquet = rtma_weights_dir / f"{out_tag}_rtma_weights.parquet"
    mrms_weights.to_parquet(mrms_parquet, index=False)
    rtma_weights.to_parquet(rtma_parquet, index=False)
    LOGGER.info("Written: %s", mrms_parquet.name)
    LOGGER.info("Written: %s", rtma_parquet.name)

    # ---- Validate ----
    LOGGER.info("Validating weight tables ...")
    mrms_val = validate_weight_table(mrms_weights, pilot_staids, "mrms_qpe_1h_pass1")
    rtma_val = validate_weight_table(rtma_weights, pilot_staids, "rtma_conus_aws_2p5km")

    with open(weights_manifest_dir / "weight_validation_mrms.json", "w", encoding="utf-8") as fh:
        json.dump(mrms_val, fh, indent=2, default=str)
    with open(weights_manifest_dir / "weight_validation_rtma.json", "w", encoding="utf-8") as fh:
        json.dump(rtma_val, fh, indent=2, default=str)

    # ---- QC plots (advisory — failures never cause nonzero exit) ----
    overview_ok = False
    mrms_plot_ok = False
    rtma_plot_ok = False
    if not args.skip_qc_plots:
        LOGGER.info("Generating QC plots ...")
        overview_ok = _plot_overview_map(basin_gdf, mrms_spec, rtma_spec, qc_dir / "overview_basin_map.png")
        sample_staids = _select_sample_basins(basin_gdf, mrms_weights, rtma_weights, n=6)
        mrms_plot_ok = _plot_sample_basins(basin_gdf, mrms_weights, sample_staids, "mrms_qpe_1h_pass1",
                                            qc_dir / "mrms_sample_basins.png", geom_method=geom_method)
        rtma_plot_ok = _plot_sample_basins(basin_gdf, rtma_weights, sample_staids, "rtma_conus_aws_2p5km",
                                            qc_dir / "rtma_sample_basins.png", geom_method=geom_method)
    else:
        LOGGER.info("QC plots skipped (--skip-qc-plots)")

    # ---- Write summary ----
    polygon_source_str = str(resolved_polygon_path) if resolved_polygon_path else None
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "geometry_method": geom_method,
        "camelsh_polygon_path": polygon_source_str,
        "n_pilot_basins": len(pilot_staids),
        "n_basins_with_geometry": len(basin_gdf),
        "missing_geometry": missing_geom,
        "mrms_validation": mrms_val,
        "rtma_validation": rtma_val,
        "runtime_mrms_s": round(mrms_time, 2),
        "runtime_rtma_s": round(rtma_time, 2),
    }
    with open(weights_manifest_dir / "weight_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    (weights_manifest_dir / "weight_summary.md").write_text(
        _build_summary_md(
            mrms_val, rtma_val, geom_method, missing_geom,
            len(pilot_staids), len(basin_gdf),
            polygon_source=polygon_source_str,
        ),
        encoding="utf-8",
    )

    # ---- Combine all validation checks ----
    # Fatal checks: weight table correctness and Parquet output.
    # Exit code and "Overall" reflect only these.
    all_checks: dict[str, Any] = {
        "mrms_table_nonempty":               mrms_val["checks"]["table_nonempty"],
        "mrms_no_negative_weights":          mrms_val["checks"]["no_negative_weights"],
        "mrms_all_pilots_have_weights":      mrms_val["checks"]["all_pilots_have_weights"],
        "mrms_weight_sums_within_tolerance": mrms_val["checks"]["weight_sums_within_tolerance"],
        "mrms_parquet_written":              mrms_parquet.exists(),
        "rtma_table_nonempty":               rtma_val["checks"]["table_nonempty"],
        "rtma_no_negative_weights":          rtma_val["checks"]["no_negative_weights"],
        "rtma_all_pilots_have_weights":      rtma_val["checks"]["all_pilots_have_weights"],
        "rtma_weight_sums_within_tolerance": rtma_val["checks"]["weight_sums_within_tolerance"],
        "rtma_parquet_written":              rtma_parquet.exists(),
        "all_basins_have_geometry":          len(missing_geom) == 0,
    }
    # Advisory checks: QC plots are nice-to-have; their absence does not block
    # extraction. Reported in the final summary but excluded from sys.exit(1).
    advisory_checks: dict[str, Any] = {
        "overview_map_written":              overview_ok,
        "mrms_sample_plot_written":          mrms_plot_ok,
        "rtma_sample_plot_written":          rtma_plot_ok,
    }

    run_cmd = f"python scripts/build_stage1_basin_weights.py --config {args.config}"
    if args.data_root:
        run_cmd += f" --data-root {args.data_root}"
    if args.basin_list:
        run_cmd += f" --basin-list {args.basin_list}"
    if args.out_tag and args.out_tag != "pilot":
        run_cmd += f" --out-tag {args.out_tag}"
    if args.grid_def_dir:
        run_cmd += f" --grid-def-dir {args.grid_def_dir}"
    if args.skip_qc_plots:
        run_cmd += " --skip-qc-plots"

    write_run_manifest(
        weights_manifest_dir,
        run_command=run_cmd,
        config_dict=config_to_dict(cfg),
        input_paths={
            "pilot_manifest": str(pilot_manifest),
            "mrms_grid_def": str(mrms_json),
            "rtma_grid_def": str(rtma_json),
            "camelsh_polygons": polygon_source_str,
        },
        output_paths={
            "mrms_weights_parquet": str(mrms_parquet),
            "rtma_weights_parquet": str(rtma_parquet),
            "weight_summary_json": str(weights_manifest_dir / "weight_summary.json"),
            "overview_map": str(qc_dir / "overview_basin_map.png"),
            "mrms_sample_plot": str(qc_dir / "mrms_sample_basins.png"),
            "rtma_sample_plot": str(qc_dir / "rtma_sample_basins.png"),
        },
        validation_results={**all_checks, **advisory_checks},
        extra={
            "geometry_method": geom_method,
            "test_only": test_only,
            "camelsh_polygon_path": polygon_source_str,
            "n_basins_with_geometry": len(basin_gdf),
            "missing_geometry": missing_geom,
            "camelsh_reference_doi": CAMELSH_CURRENT_DOI,
            "camelsh_polygon_source_doi": CAMELSH_POLYGON_SOURCE_DOI,
        },
    )

    # ---- Final report ----
    print(f"\n{'='*60}")
    print("VALIDATION (fatal checks)")
    print(f"{'='*60}")
    for k, v in all_checks.items():
        tag = "PASS" if v else "FAIL"
        print(f"  {tag}  {k}")

    print(f"\n{'='*60}")
    print("ADVISORY (QC plots — do not affect exit code)")
    print(f"{'='*60}")
    for k, v in advisory_checks.items():
        tag = "OK  " if v else "SKIP"
        print(f"  {tag}  {k}")

    overall = "PASS" if all(all_checks.values()) else "FAIL"
    print(f"\nOverall: {overall}  (exit 0 if PASS; plots are advisory)")
    print(f"\nKey outputs:")
    print(f"  MRMS weights:  {mrms_parquet}")
    print(f"  RTMA weights:  {rtma_parquet}")
    print(f"  QC plots:      {qc_dir}")
    print(f"  Manifest:      {weights_manifest_dir / 'manifest.json'}")
    print()
    print("Geometry method:", geom_method)
    if test_only:
        print()
        print("  !! TEST-ONLY WEIGHTS — NOT VALID FOR PRODUCTION !!")
        print("  Circular-buffer geometry was used (--allow-fallback-circles).")
        print("  Obtain CAMELSH_shapefile.shp from DOI", CAMELSH_POLYGON_SOURCE_DOI)
        print("  Place at: {data_root}/02_basin_geometries/camelsh/CAMELSH_shapefile.shp")
        print("  Then rerun WITHOUT --allow-fallback-circles for production weights.")
    else:
        print("  Polygon source DOI:", CAMELSH_POLYGON_SOURCE_DOI)
        print("  CAMELSH reference:  ", CAMELSH_CURRENT_DOI)
    print()
    print("Next step: implement one-hour basin-statistic extraction using these weights.")

    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
