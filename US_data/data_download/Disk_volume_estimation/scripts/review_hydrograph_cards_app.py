#!/usr/bin/env python3
"""
Manual hydrograph review UI for Flash-NH WY2024 basin screening.

Usage:
    streamlit run scripts/review_hydrograph_cards_app.py
    streamlit run scripts/review_hydrograph_cards_app.py -- --review-dir reports/my_folder
    streamlit run scripts/review_hydrograph_cards_app.py -- --labels-file manual_review_labels_pass2.csv
    streamlit run scripts/review_hydrograph_cards_app.py -- --review-dir reports/my_folder --labels-file pass2.csv

Reads:
    <review_dir>/tables/human_review_template.csv
    <review_dir>/tables/review_card_manifest.csv
    <review_dir>/plots/*.png

Writes:
    <labels_csv>   (default: <review_dir>/manual_review_labels.csv)
    Use --labels-file to write to a separate file for each review pass.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_review_dir() -> Path:
    """Parse --review-dir from sys.argv (passed after -- in streamlit run)."""
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--review-dir" and i + 1 < len(args):
            return Path(args[i + 1])
    return REPO_ROOT / "reports" / "flashnh_hydrograph_review_cards_v003_diverse"


def _get_labels_csv(review_dir: Path) -> Path:
    """Resolve the active labels file from --labels-file CLI arg.

    Rules:
    - Omitted          → <review_dir>/manual_review_labels.csv  (default)
    - Filename only    → <review_dir>/<filename>
    - Path with dirs / absolute path → used as-is (resolved to absolute)
    """
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--labels-file" and i + 1 < len(args):
            raw = args[i + 1]
            p = Path(raw)
            # Filename only (no directory component) → place under review_dir
            if p.parent == Path(".") and not p.is_absolute():
                return review_dir / p
            # Path with directories or absolute → use as-is
            return p.resolve()
    return review_dir / "manual_review_labels.csv"


REVIEW_DIR   = _get_review_dir()
TEMPLATE_CSV = REVIEW_DIR / "tables" / "human_review_template.csv"
MANIFEST_CSV = REVIEW_DIR / "tables" / "review_card_manifest.csv"
PLOTS_DIR    = REVIEW_DIR / "plots"
LABELS_CSV   = _get_labels_csv(REVIEW_DIR)
METADATA_CSV = (
    REPO_ROOT
    / "reports"
    / "flashnh_usgs_site_metadata_v001"
    / "tables"
    / "wy2024_metrics_with_site_metadata.csv"
)

# Slim subset of metadata columns for site context display
_META_COLS = [
    "STAID", "LAT_GAGE", "LNG_GAGE", "latitude", "longitude",
    "monitoring_location_name", "site_type_code", "site_type_name",
    "metadata_policy_bucket",
]

# ---------------------------------------------------------------------------
# Review form vocabulary
# ---------------------------------------------------------------------------
DECISION_OPTS  = ["KEEP", "KEEP_LOW_CONFIDENCE", "EXCLUDE", "UNSURE"]
BEHAVIOR_OPTS  = [
    "smooth_event_response", "flashy_event_response", "ephemeral_pulse_response",
    "step_like_response", "noisy_low_flow", "mostly_zero_or_constant",
    "unnatural_or_artifact_like", "unclear",
]
ARTIFACT_OPTS  = [
    "none", "single_point_spike", "gap_edge_jump", "sensor_noise",
    "rating_shift_or_step", "regulated_or_managed", "ice_or_seasonal_artifact",
    "ephemeral_valid", "very_flashy_valid", "mostly_zero_or_constant",
    "all_or_most_zero", "physically_implausible_low_flow", "other",
]
CONFIDENCE_OPTS = ["high", "medium", "low"]

LABEL_COLS = ["STAID", "human_decision", "hydrograph_behavior",
              "artifact_type", "confidence", "reviewer_notes"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_template(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"STAID": str})
    for col in ["human_decision", "hydrograph_behavior", "artifact_type",
                "confidence", "reviewer_notes"]:
        if col not in df.columns:
            df[col] = ""
    return df


@st.cache_data
def load_manifest(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"STAID": str})


@st.cache_data
def load_site_metadata(path: str) -> pd.DataFrame:
    """Load a slim site-metadata table with lat/lon and site-context columns."""
    try:
        all_cols = pd.read_csv(path, dtype={"STAID": str}, nrows=0).columns.tolist()
        use_cols = [c for c in _META_COLS if c in all_cols]
        return pd.read_csv(path, dtype={"STAID": str}, usecols=use_cols)
    except Exception:
        return pd.DataFrame(columns=_META_COLS)


def load_labels() -> pd.DataFrame:
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV, dtype={"STAID": str})
    return pd.DataFrame(columns=LABEL_COLS)


def save_label(staid: str, decision: str, behavior: str,
               artifact: list, confidence: str, notes: str) -> None:
    """Upsert one basin's label into LABELS_CSV, preserving all other rows."""
    existing = load_labels()
    row = {
        "STAID":               staid,
        "human_decision":      decision,
        "hydrograph_behavior": behavior,
        "artifact_type":       ";".join(artifact) if artifact else "",
        "confidence":          confidence,
        "reviewer_notes":      notes,
    }
    updated = existing[existing["STAID"] != staid].copy()
    updated = pd.concat([updated, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(LABELS_CSV, index=False)
    st.session_state.labels = updated


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _safe(val, fmt: str = "") -> str:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        if fmt:
            return format(float(val), fmt)
        return str(val)
    except Exception:
        return str(val) if val else "N/A"


def _filt_opts(df: pd.DataFrame, col: str) -> list:
    return sorted(df[col].dropna().astype(str).unique().tolist())


def _area_bin(a) -> str:
    try:
        a = float(a)
        if a < 10:   return "<10 km2"
        if a < 100:  return "10-100 km2"
        if a < 1000: return "100-1000 km2"
        return ">=1000 km2"
    except Exception:
        return "unknown"


def _bfi_bin(b) -> str:
    try:
        b = float(b)
        if b <= 20: return "<=20"
        if b <= 30: return "20-30"
        if b <= 40: return "30-40"
        if b <= 50: return "40-50"
        return ">50"
    except Exception:
        return "unknown"


def _show_image(fname, caption: str) -> None:
    """Display a plot by filename; show a warning if missing."""
    if not fname or fname == "nan" or (isinstance(fname, float) and pd.isna(fname)):
        st.info(f"No {caption} plot available.")
        return
    p = PLOTS_DIR / str(fname)
    if not p.exists():
        p2 = REPO_ROOT / str(fname)
        if p2.exists():
            p = p2
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.warning(f"Plot file not found: {fname}")


def _get_label(labels: pd.DataFrame, staid: str, col: str, default="") -> str:
    rows = labels[labels["STAID"] == staid]
    if len(rows) > 0:
        v = rows.iloc[0].get(col, default)
        return str(v) if pd.notna(v) else default
    return default


def verify_saved_label(staid: str, decision: str) -> bool:
    """Re-read LABELS_CSV from disk and confirm the row exists with matching decision."""
    if not LABELS_CSV.exists():
        return False
    fresh = pd.read_csv(LABELS_CSV, dtype={"STAID": str})
    rows = fresh[fresh["STAID"] == staid]
    if len(rows) == 0:
        return False
    return str(rows.iloc[0]["human_decision"]) == decision


def _choose_next_staid(current_staid: str, staid_list: list, current_idx: int) -> str:
    """Return the next STAID, or stay on the current one if already at the end."""
    if current_idx < len(staid_list) - 1:
        return staid_list[current_idx + 1]
    return current_staid


def _find_latlon(meta_row: pd.Series) -> tuple:
    """Return (lat, lon) floats from the first non-null pair found in meta_row."""
    for lat_col, lon_col in [("LAT_GAGE", "LNG_GAGE"), ("latitude", "longitude")]:
        if lat_col in meta_row.index and lon_col in meta_row.index:
            try:
                lat_f = float(meta_row[lat_col])
                lon_f = float(meta_row[lon_col])
                if pd.notna(lat_f) and pd.notna(lon_f):
                    return lat_f, lon_f
            except (TypeError, ValueError):
                pass
    return None, None


# ---------------------------------------------------------------------------
# USGS site context block (link + map)
# ---------------------------------------------------------------------------
def _render_site_context(staid: str, meta_row) -> None:
    """Render the USGS link + map expander for the current basin."""
    usgs_url = f"https://waterdata.usgs.gov/monitoring-location/USGS-{staid}/"

    with st.expander("USGS site and map context", expanded=False):
        st.link_button("Open USGS monitoring-location page", usgs_url)

        if meta_row is None:
            st.warning("Site metadata not available for this basin.")
            return

        lat, lon = _find_latlon(meta_row)

        # Site info summary
        info_lines = []
        site_name = meta_row.get("monitoring_location_name", "")
        if site_name and pd.notna(site_name):
            info_lines.append(f"**Site:** {site_name}")
        for label, key in [
            ("Site type", "site_type_name"),
            ("Site type code", "site_type_code"),
            ("Metadata policy", "metadata_policy_bucket"),
        ]:
            val = meta_row.get(key, "")
            if val and pd.notna(val):
                info_lines.append(f"**{label}:** {val}")
        info_lines.append(
            f"**Lat/Lon:** {lat:.5f}, {lon:.5f}" if lat is not None else "**Lat/Lon:** N/A"
        )
        st.markdown("  \n".join(info_lines))

        if lat is None:
            st.info("Map unavailable: lat/lon not found for this basin.")
            return

        # Try folium (satellite basemap); fall back to st.map (OpenStreetMap)
        _rendered = False
        try:
            import folium
            from streamlit_folium import st_folium
            m = folium.Map(location=[lat, lon], zoom_start=12)
            folium.TileLayer(
                tiles=(
                    "https://server.arcgisonline.com/ArcGIS/rest/services"
                    "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ),
                attr="Esri WorldImagery",
                name="Satellite",
            ).add_to(m)
            folium.Marker(
                [lat, lon],
                tooltip=f"USGS {staid}",
                popup=site_name if (site_name and pd.notna(site_name)) else staid,
            ).add_to(m)
            st.caption("Map: gauge location only — satellite (Esri/folium). No basin boundary file in project.")
            st_folium(m, height=370, use_container_width=True)
            _rendered = True
        except ImportError:
            pass

        if not _rendered:
            point_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
            st.caption(
                "Map: gauge location only — OpenStreetMap (st.map). "
                "For satellite imagery: `python -m pip install streamlit-folium folium`"
            )
            try:
                st.map(point_df, zoom=12, use_container_width=True)
            except TypeError:
                st.map(point_df)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Flash-NH Review",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Load data ─────────────────────────────────────────────────────────
    if not TEMPLATE_CSV.exists():
        st.error(f"Template not found: {TEMPLATE_CSV}")
        st.stop()

    template  = load_template(str(TEMPLATE_CSV))
    manifest  = load_manifest(str(MANIFEST_CSV)) if MANIFEST_CSV.exists() else pd.DataFrame()
    site_meta = load_site_metadata(str(METADATA_CSV)) if METADATA_CSV.exists() else pd.DataFrame()

    if "labels" not in st.session_state:
        st.session_state.labels = load_labels()
    labels: pd.DataFrame = st.session_state.labels

    # Merge labels into template for filter / progress tracking
    df = template.copy()
    df = df.drop(columns=[c for c in LABEL_COLS[1:] if c in df.columns], errors="ignore")
    if len(labels) > 0:
        df = df.merge(labels[LABEL_COLS], on="STAID", how="left")
    else:
        for c in LABEL_COLS[1:]:
            df[c] = pd.NA

    df["_area_bin"] = df["DRAIN_SQKM"].apply(_area_bin)
    df["_bfi_bin"]  = df["BFI_AVE"].apply(_bfi_bin)

    n_total    = len(df)
    n_reviewed = int(df["human_decision"].apply(
        lambda x: pd.notna(x) and str(x).strip() != "").sum())
    n_unrev    = n_total - n_reviewed

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.title("Flash-NH Review")
    st.sidebar.caption(f"Review dir: {REVIEW_DIR.name}")
    st.sidebar.caption(f"Labels file: {LABELS_CSV.name}")

    # Archival-safety warnings
    _lname_lower = LABELS_CSV.name.lower()
    if any(kw in _lname_lower for kw in ("locked", "archive", "pass1_locked")):
        st.sidebar.warning(
            f"'{LABELS_CSV.name}' appears to be an archival labels file. "
            "It should not normally be edited."
        )
    st.sidebar.info(
        "For archival safety, copy completed review labels to a locked filename "
        "before starting a new review pass."
    )

    st.sidebar.markdown(
        f"**{n_reviewed} / {n_total}** reviewed &nbsp;|&nbsp; **{n_unrev}** remaining"
    )

    if n_reviewed > 0:
        dec_counts = df["human_decision"].value_counts()
        for d, c in dec_counts.items():
            st.sidebar.markdown(f"- {d}: {c}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    filt = df.copy()

    unrev_only = st.sidebar.checkbox("Unreviewed only", value=False)
    if unrev_only:
        filt = filt[filt["human_decision"].isna()]

    cls_sel = st.sidebar.multiselect("Candidate class", _filt_opts(df, "candidate_class"))
    if cls_sel:
        filt = filt[filt["candidate_class"].isin(cls_sel)]

    grp_sel = st.sidebar.multiselect("Review group", _filt_opts(df, "review_group"))
    if grp_sel:
        filt = filt[filt["review_group"].isin(grp_sel)]

    huc_sel = st.sidebar.multiselect("HUC02", _filt_opts(df, "HUC02"))
    if huc_sel:
        filt = filt[filt["HUC02"].astype(str).isin(huc_sel)]

    if "STATE" in df.columns:
        state_sel = st.sidebar.multiselect("State", _filt_opts(df, "STATE"))
        if state_sel:
            filt = filt[filt["STATE"].isin(state_sel)]

    abin_opts = sorted(df["_area_bin"].unique().tolist())
    abin_sel = st.sidebar.multiselect("Area bin", abin_opts)
    if abin_sel:
        filt = filt[filt["_area_bin"].isin(abin_sel)]

    bbin_opts = sorted(df["_bfi_bin"].unique().tolist())
    bbin_sel = st.sidebar.multiselect("BFI bin", bbin_opts)
    if bbin_sel:
        filt = filt[filt["_bfi_bin"].isin(bbin_sel)]

    dec_filter_opts = _filt_opts(df, "human_decision")
    if dec_filter_opts:
        dec_f = st.sidebar.multiselect("Decision label", dec_filter_opts)
        if dec_f:
            filt = filt[filt["human_decision"].isin(dec_f)]

    staid_search = st.sidebar.text_input("Search STAID")
    if staid_search.strip():
        filt = filt[filt["STAID"].str.contains(staid_search.strip(), case=False, na=False)]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(filt)} basins match filters")

    if len(filt) == 0:
        st.warning("No basins match the current filters.")
        st.stop()

    staid_list = filt["STAID"].tolist()

    # ── Basin navigation (top) ────────────────────────────────────────────
    if "current_staid" not in st.session_state or st.session_state.current_staid not in staid_list:
        prev = st.session_state.get("current_staid")
        fallback = staid_list[0]
        if prev is not None and prev not in staid_list:
            tmpl_order = template["STAID"].tolist()
            if prev in tmpl_order:
                for s in tmpl_order[tmpl_order.index(prev) + 1:]:
                    if s in staid_list:
                        fallback = s
                        break
        st.session_state.current_staid = fallback

    current_idx = staid_list.index(st.session_state.current_staid)

    nav_prev, nav_sel, nav_next = st.columns([1, 10, 1])
    with nav_prev:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("< Prev", key="top_prev", width="stretch"):
            if current_idx > 0:
                st.session_state.current_staid = staid_list[current_idx - 1]
                st.rerun()
    with nav_next:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next >", key="top_next", width="stretch"):
            if current_idx < len(staid_list) - 1:
                st.session_state.current_staid = staid_list[current_idx + 1]
                st.rerun()
    with nav_sel:
        st.selectbox(
            f"Basin  ({current_idx + 1} / {len(staid_list)})",
            staid_list,
            index=current_idx,
            key="current_staid",
        )

    current_staid = st.session_state.current_staid
    row = filt[filt["STAID"] == current_staid].iloc[0]

    # Look up the site-metadata row for map / USGS context
    meta_row = None
    if not site_meta.empty and "STAID" in site_meta.columns:
        meta_matches = site_meta[site_meta["STAID"] == current_staid]
        if len(meta_matches) > 0:
            meta_row = meta_matches.iloc[0]

    # ── Basin header ──────────────────────────────────────────────────────
    reviewed_badge = ""
    lbl_now = _get_label(labels, current_staid, "human_decision")
    if lbl_now:
        badge_color = {"KEEP": "green", "KEEP_LOW_CONFIDENCE": "orange",
                       "EXCLUDE": "red", "UNSURE": "gray"}.get(lbl_now, "gray")
        reviewed_badge = f" &nbsp; :{badge_color}[{lbl_now}]"

    st.markdown(f"## {current_staid}{reviewed_badge}", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Candidate class",  _safe(row.get("candidate_class")))
        st.metric("Review group",     _safe(row.get("review_group")))
        st.metric("State",            _safe(row.get("STATE")))
        st.metric("HUC02",            _safe(row.get("HUC02")))
    with c2:
        st.metric("Area (km2)",       _safe(row.get("DRAIN_SQKM"), ".1f"))
        st.metric("BFI",              _safe(row.get("BFI_AVE"), ".1f"))
        st.metric("RBI",              _safe(row.get("RBI"), ".4f"))
        st.metric("Completeness %",   _safe(row.get("hourly_completeness_pct"), ".1f"))
        st.metric("Zero fraction",    _safe(row.get("zero_flow_fraction"), ".3f"))
    with c3:
        q50 = row.get("q50"); q95 = row.get("q95"); q99 = row.get("q99")
        st.metric("Q50",              _safe(q50, ".4g"))
        st.metric("Q95",              _safe(q95, ".4g"))
        st.metric("Q99",              _safe(q99, ".4g"))
        st.metric("Q95/Q50",          _safe(row.get("q95_q50_ratio"), ".2f"))
        try:
            q99_q50 = float(q99) / float(q50) if pd.notna(q99) and pd.notna(q50) and float(q50) > 0 else None
        except Exception:
            q99_q50 = None
        st.metric("Q99/Q50",          _safe(q99_q50, ".2f"))
        st.metric("MaxRise/km2",      _safe(row.get("max_hourly_rise_per_km2"), ".4g"))
        st.metric("Jump/Q50",         _safe(row.get("max_abs_hourly_jump_over_Q50"), ".2f"))

    qc = row.get("qc_labels", "")
    if qc and not (isinstance(qc, float) and pd.isna(qc)):
        st.info(f"QC labels: {qc}")

    ctx = row.get("context_flags", "")
    if ctx and not (isinstance(ctx, float) and pd.isna(ctx)):
        with st.expander("Context flags"):
            st.code(str(ctx))

    # ── USGS site context and map ─────────────────────────────────────────
    _render_site_context(current_staid, meta_row)

    # ── Plot viewer ───────────────────────────────────────────────────────
    st.markdown("---")
    fy  = str(row.get("full_year_plot", ""))
    ep1 = str(row.get("event_plot_1", ""))
    ep2 = str(row.get("event_plot_2", ""))
    ep3 = str(row.get("event_plot_3", ""))
    zfp = str(row.get("zero_flow_context_plot", ""))

    tabs = st.tabs(["Full year", "Event 1", "Event 2", "Event 3", "All plots"])

    with tabs[0]:
        _show_image(fy, "Full year hydrograph")

    with tabs[1]:
        _show_image(ep1, "Event 1  [close +-12h]")

    with tabs[2]:
        _show_image(ep2, "Event 2  [close +-12h]")

    with tabs[3]:
        _show_image(ep3, "Event 3  [close +-12h]")

    with tabs[4]:
        if not manifest.empty:
            basin_rows = manifest[manifest["STAID"] == current_staid].copy()
            if len(basin_rows) > 0:
                sort_key = {"full_year": "0", "peak_close": "1", "peak_tight": "2",
                            "peak_medium": "3", "rise_close": "4", "rise_tight": "5",
                            "rise_medium": "6", "zero_flow_context": "7"}
                basin_rows["_sort"] = basin_rows["event_type"].map(
                    lambda x: sort_key.get(x, "9"))
                basin_rows = basin_rows.sort_values(["_sort", "event_num"])
                for _, pr in basin_rows.iterrows():
                    raw_path = str(pr["plot_path"])
                    p = Path(raw_path)
                    if not p.is_absolute():
                        if not p.exists():
                            p = REPO_ROOT / p
                    if p.exists():
                        caption = f"{pr['event_type']}  ev#{pr.get('event_num', '')}"
                        st.image(str(p), caption=caption, width="stretch")
                    else:
                        st.warning(f"Not found: {Path(raw_path).name}")
            else:
                st.info("No manifest entries for this basin.")
        else:
            st.info("Manifest not loaded.")

        if zfp and zfp != "nan":
            st.markdown("**Zero-flow context**")
            _show_image(zfp, "Zero-flow context")

    # ── Review form ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Manual review")

    lbl_decision = _get_label(labels, current_staid, "human_decision")
    lbl_behavior = _get_label(labels, current_staid, "hydrograph_behavior")
    lbl_artifact = _get_label(labels, current_staid, "artifact_type")
    lbl_conf     = _get_label(labels, current_staid, "confidence")
    lbl_notes    = _get_label(labels, current_staid, "reviewer_notes")

    # Seed session-state widget values from saved labels the first time this STAID appears.
    # After the first load, Streamlit tracks the keys; we do not overwrite user edits.
    _init_key = f"_init_{current_staid}"
    if _init_key not in st.session_state:
        st.session_state[f"human_decision_{current_staid}"] = (
            lbl_decision if lbl_decision in DECISION_OPTS else DECISION_OPTS[0])
        st.session_state[f"hydrograph_behavior_{current_staid}"] = (
            lbl_behavior if lbl_behavior in BEHAVIOR_OPTS else BEHAVIOR_OPTS[0])
        st.session_state[f"artifact_type_{current_staid}"] = (
            [a.strip() for a in lbl_artifact.split(";") if a.strip() in ARTIFACT_OPTS]
            if lbl_artifact else [])
        st.session_state[f"confidence_{current_staid}"] = (
            lbl_conf if lbl_conf in CONFIDENCE_OPTS else CONFIDENCE_OPTS[0])
        st.session_state[f"reviewer_notes_{current_staid}"] = lbl_notes
        st.session_state[_init_key] = True

    decision   = st.radio(
        "human_decision", DECISION_OPTS,
        key=f"human_decision_{current_staid}", horizontal=True)
    behavior   = st.radio(
        "hydrograph_behavior", BEHAVIOR_OPTS,
        key=f"hydrograph_behavior_{current_staid}", horizontal=True)
    artifact   = st.multiselect(
        "artifact_type  (select all that apply)", ARTIFACT_OPTS,
        key=f"artifact_type_{current_staid}")
    confidence = st.radio(
        "confidence", CONFIDENCE_OPTS,
        key=f"confidence_{current_staid}", horizontal=True)
    notes      = st.text_area(
        "reviewer_notes", key=f"reviewer_notes_{current_staid}", height=80)

    # ── Save buttons (after form, before bottom nav) ──────────────────────
    col_save, col_next = st.columns(2)
    with col_save:
        if st.button("Save", key="save_top", type="primary", width="stretch"):
            save_label(current_staid, decision, behavior, artifact, confidence, notes)
            if verify_saved_label(current_staid, decision):
                st.success(
                    f"Saved label for {current_staid}; verified in {LABELS_CSV.name}")
            else:
                st.error(f"Save failed for {current_staid} — row not found after write")
    with col_next:
        if st.button("Save and Next", key="save_next_top", width="stretch"):
            save_label(current_staid, decision, behavior, artifact, confidence, notes)
            if verify_saved_label(current_staid, decision):
                next_staid = _choose_next_staid(current_staid, staid_list, current_idx)
                st.session_state.current_staid = next_staid
                st.rerun()
            else:
                st.error(f"Save failed for {current_staid} — not advancing")

    # ── Bottom navigation ─────────────────────────────────────────────────
    st.markdown("---")
    bn_prev, bn_save, bn_save_next, bn_next = st.columns([1, 2, 2, 1])
    with bn_prev:
        if st.button("< Prev", key="bot_prev", width="stretch"):
            if current_idx > 0:
                st.session_state.current_staid = staid_list[current_idx - 1]
                st.rerun()
    with bn_save:
        if st.button("Save", key="save_bot", type="primary", width="stretch"):
            save_label(current_staid, decision, behavior, artifact, confidence, notes)
            if verify_saved_label(current_staid, decision):
                st.success(f"Saved {current_staid}")
            else:
                st.error(f"Save failed for {current_staid}")
    with bn_save_next:
        if st.button("Save and Next", key="save_next_bot", width="stretch"):
            save_label(current_staid, decision, behavior, artifact, confidence, notes)
            if verify_saved_label(current_staid, decision):
                next_staid = _choose_next_staid(current_staid, staid_list, current_idx)
                st.session_state.current_staid = next_staid
                st.rerun()
            else:
                st.error(f"Save failed for {current_staid} — not advancing")
    with bn_next:
        if st.button("Next >", key="bot_next", width="stretch"):
            if current_idx < len(staid_list) - 1:
                st.session_state.current_staid = staid_list[current_idx + 1]
                st.rerun()


if __name__ == "__main__":
    main()
