#!/usr/bin/env python3
"""
Manual hydrograph review UI for Flash-NH WY2024 basin screening.

Usage:
    streamlit run scripts/review_hydrograph_cards_app.py
    streamlit run scripts/review_hydrograph_cards_app.py -- --review-dir reports/my_folder
    streamlit run scripts/review_hydrograph_cards_app.py -- --review-dir=reports/my_folder
    streamlit run scripts/review_hydrograph_cards_app.py -- --labels-file manual_review_labels_pass2.csv
    streamlit run scripts/review_hydrograph_cards_app.py -- --review-dir=reports/my_folder --labels-file=pass2.csv

    # Smoke test (no Streamlit server needed):
    python scripts/review_hydrograph_cards_app.py --self-test-config

Env fallbacks (lower priority than CLI):
    FLASHNH_REVIEW_DIR    equivalent to --review-dir
    FLASHNH_LABELS_FILE   equivalent to --labels-file

Reads:
    <review_dir>/tables/human_review_template.csv
    <review_dir>/tables/review_card_manifest.csv
    <review_dir>/plots/*.png

Writes:
    <labels_csv>   (default: <review_dir>/manual_review_labels.csv)
    Use --labels-file to write to a separate file for each review pass.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_REVIEW_SUBDIR = "flashnh_hydrograph_review_cards_v004_main_training_candidate"
_DEFAULT_LABELS_NAME   = "manual_review_labels.csv"

# ---------------------------------------------------------------------------
# Authoritative config resolver
# ---------------------------------------------------------------------------

def resolve_app_config(argv=None, environ=None) -> dict:
    """Resolve --review-dir and --labels-file from argv and environment.

    Handles both --flag VALUE and --flag=VALUE forms.
    Precedence: CLI args > env vars > defaults.

    Returns a dict with keys:
        review_dir       Path (absolute, resolved)
        labels_file      Path (absolute, resolved)
        review_dir_src   "CLI" | "ENV" | "DEFAULT"
        labels_file_src  "CLI" | "ENV" | "DEFAULT"
    """
    if argv is None:
        argv = sys.argv[1:]
    if environ is None:
        environ = os.environ

    def _parse_flag(args, flag):
        """Return the value for --flag VALUE or --flag=VALUE, or None if absent."""
        prefix = flag + "="
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(prefix):
                return a[len(prefix):]
        return None

    # ── review_dir ──────────────────────────────────────────────────────────
    raw_dir = _parse_flag(argv, "--review-dir")
    if raw_dir is not None:
        p = Path(raw_dir)
        review_dir = (REPO_ROOT / p if not p.is_absolute() else p).resolve()
        review_dir_src = "CLI"
    elif "FLASHNH_REVIEW_DIR" in environ:
        p = Path(environ["FLASHNH_REVIEW_DIR"])
        review_dir = (REPO_ROOT / p if not p.is_absolute() else p).resolve()
        review_dir_src = "ENV"
    else:
        review_dir = (REPO_ROOT / "reports" / _DEFAULT_REVIEW_SUBDIR).resolve()
        review_dir_src = "DEFAULT"

    # ── labels_file ─────────────────────────────────────────────────────────
    raw_lf = _parse_flag(argv, "--labels-file")
    if raw_lf is not None:
        p = Path(raw_lf)
        # Filename only (no directory component) → place under review_dir
        if p.parent == Path(".") and not p.is_absolute():
            labels_file = (review_dir / p).resolve()
        else:
            labels_file = p.resolve()
        labels_file_src = "CLI"
    elif "FLASHNH_LABELS_FILE" in environ:
        p = Path(environ["FLASHNH_LABELS_FILE"])
        if p.parent == Path(".") and not p.is_absolute():
            labels_file = (review_dir / p).resolve()
        else:
            labels_file = p.resolve()
        labels_file_src = "ENV"
    else:
        labels_file = (review_dir / _DEFAULT_LABELS_NAME).resolve()
        labels_file_src = "DEFAULT"

    return {
        "review_dir":      review_dir,
        "labels_file":     labels_file,
        "review_dir_src":  review_dir_src,
        "labels_file_src": labels_file_src,
    }


# ---------------------------------------------------------------------------
# Module-level config — resolved once at script load time
# ---------------------------------------------------------------------------
_CFG = resolve_app_config()

REVIEW_DIR   = _CFG["review_dir"]
LABELS_CSV   = _CFG["labels_file"]
TEMPLATE_CSV = REVIEW_DIR / "tables" / "human_review_template.csv"
MANIFEST_CSV = REVIEW_DIR / "tables" / "review_card_manifest.csv"
PLOTS_DIR    = REVIEW_DIR / "plots"
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
# Each cached function takes the active path as an explicit string argument so
# that cache keys change automatically when REVIEW_DIR / LABELS_CSV changes.
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
# Startup guard
# ---------------------------------------------------------------------------
def _check_config_guard() -> None:
    """Stop the app if a CLI flag was present but the resolver did not pick it up.

    This catches edge cases where sys.argv layout is unexpected (e.g. Streamlit
    version differences) so the user gets an explicit error rather than silent
    fallback to a wrong review directory.
    """
    raw_argv = sys.argv[1:]

    has_review_dir_flag = any(
        a == "--review-dir" or a.startswith("--review-dir=") for a in raw_argv
    )
    has_labels_file_flag = any(
        a == "--labels-file" or a.startswith("--labels-file=") for a in raw_argv
    )

    if has_review_dir_flag and _CFG["review_dir_src"] != "CLI":
        st.error(
            "**CONFIG GUARD**: `--review-dir` was detected in `sys.argv` but "
            f"`resolve_app_config()` fell through to `{_CFG['review_dir_src']}`. "
            f"Active review dir: `{REVIEW_DIR}`  \n"
            f"Full `sys.argv`: `{raw_argv}`  \n\n"
            "This is a bug — please file an issue. As a workaround, set the "
            "`FLASHNH_REVIEW_DIR` environment variable instead of the CLI flag."
        )
        st.stop()

    if has_labels_file_flag and _CFG["labels_file_src"] != "CLI":
        st.error(
            "**CONFIG GUARD**: `--labels-file` was detected in `sys.argv` but "
            f"`resolve_app_config()` fell through to `{_CFG['labels_file_src']}`. "
            f"Active labels file: `{LABELS_CSV}`  \n"
            f"Full `sys.argv`: `{raw_argv}`  \n\n"
            "As a workaround, set the `FLASHNH_LABELS_FILE` environment variable."
        )
        st.stop()


# ---------------------------------------------------------------------------
# Self-test (run without Streamlit: python scripts/review_hydrograph_cards_app.py --self-test-config)
# ---------------------------------------------------------------------------
def _run_self_test() -> None:
    """Smoke-test resolve_app_config() with CLI, ENV, and DEFAULT scenarios."""
    v005_dir    = "reports/flashnh_hydrograph_review_cards_v005_second_pass_rules"
    v005_labels = "manual_review_labels_pass2.csv"
    pass_count  = 0
    fail_count  = 0

    def _check(label, cfg, exp_dir_substr, exp_labels_name, exp_dir_src, exp_lf_src):
        nonlocal pass_count, fail_count
        rd     = str(cfg["review_dir"])
        lf     = str(cfg["labels_file"])
        rd_src = cfg["review_dir_src"]
        lf_src = cfg["labels_file_src"]
        ok = True

        if exp_dir_substr not in rd:
            print(f"  FAIL  review_dir: expected to contain '{exp_dir_substr}', got '{rd}'")
            ok = False
        if not lf.endswith(exp_labels_name) and not lf.endswith(exp_labels_name.replace("/", "\\")):
            print(f"  FAIL  labels_file: expected to end with '{exp_labels_name}', got '{lf}'")
            ok = False
        if rd_src != exp_dir_src:
            print(f"  FAIL  review_dir_src: expected '{exp_dir_src}', got '{rd_src}'")
            ok = False
        if lf_src != exp_lf_src:
            print(f"  FAIL  labels_file_src: expected '{exp_lf_src}', got '{lf_src}'")
            ok = False

        status = "PASS" if ok else "FAIL"
        print(f"  {status}")
        print(f"       review_dir  [{rd_src}]: {rd}")
        print(f"       labels_file [{lf_src}]: {lf}")
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    print("=" * 70)
    print("resolve_app_config() self-test")
    print("=" * 70)

    # Test A — CLI with --flag=VALUE form (the form Streamlit passes)
    print(f"\nTest A — CLI (--flag=VALUE form):")
    cfg_a = resolve_app_config(
        argv=[f"--review-dir={v005_dir}", f"--labels-file={v005_labels}"],
        environ={},
    )
    _check("A", cfg_a, "v005", v005_labels, "CLI", "CLI")

    # Test A2 — CLI with --flag VALUE (space-separated) form
    print(f"\nTest A2 — CLI (--flag VALUE space form):")
    cfg_a2 = resolve_app_config(
        argv=["--review-dir", v005_dir, "--labels-file", v005_labels],
        environ={},
    )
    _check("A2", cfg_a2, "v005", v005_labels, "CLI", "CLI")

    # Test B — ENV fallback
    print(f"\nTest B — ENV (FLASHNH_REVIEW_DIR + FLASHNH_LABELS_FILE):")
    cfg_b = resolve_app_config(
        argv=[],
        environ={
            "FLASHNH_REVIEW_DIR":  v005_dir,
            "FLASHNH_LABELS_FILE": v005_labels,
        },
    )
    _check("B", cfg_b, "v005", v005_labels, "ENV", "ENV")

    # Test C — DEFAULT (no CLI, no env)
    print(f"\nTest C — DEFAULT (no CLI, no env):")
    cfg_c = resolve_app_config(argv=[], environ={})
    _check("C", cfg_c, "v004", "manual_review_labels.csv", "DEFAULT", "DEFAULT")

    # Test D — CLI overrides ENV for review_dir; labels_file falls to DEFAULT
    print(f"\nTest D — CLI review_dir overrides ENV, labels_file stays DEFAULT:")
    cfg_d = resolve_app_config(
        argv=[f"--review-dir={v005_dir}"],
        environ={"FLASHNH_REVIEW_DIR": "reports/flashnh_hydrograph_review_cards_v003_diverse"},
    )
    _check("D", cfg_d, "v005", "manual_review_labels.csv", "CLI", "DEFAULT")

    print("\n" + "=" * 70)
    total = pass_count + fail_count
    print(f"Results: {pass_count}/{total} passed, {fail_count}/{total} failed")
    print("=" * 70)

    if fail_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Flash-NH Review",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Guard: catch resolver failures before doing anything else
    _check_config_guard()

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

    # ── Config diagnostics (always visible) ───────────────────────────────
    st.sidebar.markdown("**Active configuration**")
    st.sidebar.caption(f"Review dir [{_CFG['review_dir_src']}]:  \n`{REVIEW_DIR}`")
    st.sidebar.caption(f"Labels file [{_CFG['labels_file_src']}]:  \n`{LABELS_CSV}`")
    st.sidebar.caption(
        f"Template:  \n`{TEMPLATE_CSV}`  \n({len(template)} rows)"
    )
    with st.sidebar.expander("sys.argv"):
        st.sidebar.code("\n".join(sys.argv) if sys.argv else "(empty)")

    st.sidebar.markdown("---")

    # ── Archival-safety warnings ───────────────────────────────────────────
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

    # ── Basin navigation ──────────────────────────────────────────────────
    # nav_current_staid is the authoritative current-basin key. It is never
    # used as a widget key, so it can always be written to safely.
    if "nav_current_staid" not in st.session_state or st.session_state.nav_current_staid not in staid_list:
        prev = st.session_state.get("nav_current_staid")
        fallback = staid_list[0]
        if prev is not None and prev not in staid_list:
            tmpl_order = template["STAID"].tolist()
            if prev in tmpl_order:
                for s in tmpl_order[tmpl_order.index(prev) + 1:]:
                    if s in staid_list:
                        fallback = s
                        break
        st.session_state.nav_current_staid = fallback

    current_idx = staid_list.index(st.session_state.nav_current_staid)

    # Callbacks defined before any widget renders so on_click wiring is safe.
    def _go_prev():
        idx = staid_list.index(st.session_state.nav_current_staid)
        if idx > 0:
            st.session_state.nav_current_staid = staid_list[idx - 1]

    def _go_next():
        idx = staid_list.index(st.session_state.nav_current_staid)
        if idx < len(staid_list) - 1:
            st.session_state.nav_current_staid = staid_list[idx + 1]

    def _on_staid_select():
        st.session_state.nav_current_staid = st.session_state.staid_selector

    nav_prev, nav_sel, nav_next = st.columns([1, 10, 1])
    with nav_prev:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("< Prev", key="top_prev", on_click=_go_prev, width="stretch")
    with nav_next:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Next >", key="top_next", on_click=_go_next, width="stretch")
    with nav_sel:
        st.selectbox(
            f"Basin  ({current_idx + 1} / {len(staid_list)})",
            staid_list,
            index=current_idx,
            key="staid_selector",
            on_change=_on_staid_select,
        )

    current_staid = st.session_state.nav_current_staid
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
                st.session_state.nav_current_staid = next_staid
                st.rerun()
            else:
                st.error(f"Save failed for {current_staid} — not advancing")

    # ── Bottom navigation ─────────────────────────────────────────────────
    st.markdown("---")
    bn_prev, bn_save, bn_save_next, bn_next = st.columns([1, 2, 2, 1])
    with bn_prev:
        st.button("< Prev", key="bot_prev", on_click=_go_prev, width="stretch")
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
                st.session_state.nav_current_staid = next_staid
                st.rerun()
            else:
                st.error(f"Save failed for {current_staid} — not advancing")
    with bn_next:
        st.button("Next >", key="bot_next", on_click=_go_next, width="stretch")


if __name__ == "__main__":
    if "--self-test-config" in sys.argv:
        _run_self_test()
        sys.exit(0)
    main()
