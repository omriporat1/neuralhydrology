#!/usr/bin/env python3
"""Build the Stage 1 derived static-attribute matrix (Milestone 2K-G-F-B).

Merges the rich GAGES-II + HydroATLAS + NLDAS-2 source mirror into a single
conservative "v001-core" static-attribute matrix for the Stage 1 basin
universe. See docs/stage1_static_attribute_matrix_plan.md for the full
inventory, audit, and merge/audit policy this script implements.

Source mirror (NOT tracked in git; h2o-resident):
  /data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/
  29 files: 26 attributes_gageii_*.csv + attributes_hydroATLAS.csv +
  attributes_nldas2_climate.csv + Var description_gageii.xlsx (not parsed by
  this script) + source_attributes_v001_checksums.sha256.

Output (NOT tracked in git; write under h2o
/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/
or a local --out-dir for dry runs, e.g. repo tmp/):
  stage1_static_attributes_v001.parquet         - the matrix (all roles, see column manifest)
  stage1_static_attributes_v001_column_manifest.json - per-column role/source/policy
  stage1_static_attributes_v001_provenance.json - build metadata, checksums, exclusions

Column roles (see column_manifest.json "role" field):
  model_input          - conservative v001-core numeric feature (float)
  split_support         - STATE/HUC02, retained for split construction/diagnostics,
                           EXCLUDED from model_input by design (not a model feature)
  diagnostic_latlon      - LAT_GAGE/LNG_GAGE/LAT_CENT/LONG_CENT (direct gauge and
                           basin-centroid coordinates), held out of model_input by
                           default; reserved for a dedicated future ablation
  diagnostic_record_network_qa - gauge-record history / gauge-network membership /
                           boundary-processing QA metadata (FLOWYRS_*,
                           FLOW_PCT_EST_VALUES, BASIN_BOUNDARY_CONFIDENCE, ACTIVE09,
                           HBN36, HCDN_2009, OLD_HCDN, NSIP_SENTINEL, PCT_DIFF_NWIS,
                           NWIS_DRAIN_SQKM); describes the observational record and
                           its provenance, not the basin itself -- excluded from
                           model_input (2026-07-20 static-attribute semantic
                           correction; see docs/decision_log.md)
  deferred_ambiguous      - fields whose semantics are not yet fully resolved
                           against the exact source catalog (currently lka_pc_use);
                           excluded from model_input pending resolution
  categorical_deferred    - genuine multi-class categorical codes (including
                           numeric-coded classes such as HydroATLAS "*_cl_smj"
                           and GAGES-II "*_DOM"/"*_SITE" region codes); kept as
                           raw values, NOT one-hot-encoded this round, NOT part
                           of model_input
  flag                  - derived per-basin QA/coverage flags added by this
                           builder (e.g. hydroatlas_coverage_flag)
  id                    - gauge_id index (not a data column)

Sentinel decoding (2026-07-20 static-attribute semantic correction, see
docs/decision_log.md): a narrowly scoped, explicit per-column map
(_SENTINEL_VALUES_BY_COLUMN) converts exact documented missing-value
sentinels (e.g. -999, -9999, -99) to NaN before role classification and
missingness calculation. This is NOT a blanket negative-value replacement --
only the listed columns are touched, and unrelated columns containing the
same literal number are left unchanged. This is what allows the 8
infrastructure-distance RAW_* columns to be excluded from model_input
through the existing high-missingness mechanism (not by hand-classification
by name) once their -999 sentinels are correctly counted as missing.

Filtering philosophy (conservative by default; see decision_log.md
2026-07-06 follow-up and stage1_static_attribute_matrix_plan.md secs 8-9):
  any column that is administrative, free-text, a disguised ID/class-code,
  near-constant, or high-missingness within the Stage 1 basin set is
  EXCLUDED from model_input, not kept on the chance it helps. Any non-numeric
  source column that isn't explicitly classified below causes a hard failure
  (no silent inclusion, no silent drop) so schema drift in the source files
  cannot slip an unreviewed field into the matrix.

HydroATLAS 5-basin gap (mandatory gate, see docs/stage1_static_attribute_matrix_plan.md
sec 4/9): the 5 non-standard 15-char coordinate-based STAIDs are absent from
HydroATLAS's raw STAID column under every representation tried (exact match,
zero-padded, leading-zero-stripped). This is a genuine data gap, not a
formatting mismatch. Policy applied here is option (b) from the gate:
intentionally retain those basins with HydroATLAS-sourced columns set to NaN,
tagged by an explicit `hydroatlas_coverage_flag` column (0 = missing from
HydroATLAS, 1 = present) written into the matrix -- no silent partial merge.
If the observed gap at build time does not exactly equal the expected/audited
5-STAID set, the build FAILS LOUD (option (c)) rather than proceeding, because
that would mean the source data changed in a way this policy hasn't reviewed.

Usage:
  python scripts/build_stage1_static_attribute_matrix.py \\
    --source-dir /data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/ \\
    --manifest   config/stage1_initial_training_basin_manifest.csv \\
    --out-dir    /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/ \\
    --matrix-name stage1_static_attributes_v001

Local dry run against the repo-local source fixture (validation only; the
canonical build must still be run on h2o against the h2o source mirror):
  python scripts/build_stage1_static_attribute_matrix.py \\
    --source-dir "C:/PhD/Python/neuralhydrology/US_data/attributes" \\
    --manifest   config/stage1_initial_training_basin_manifest.csv \\
    --out-dir    tmp/stage1_static_attribute_matrix_v001_dryrun \\
    --no-require-checksums --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Per-file column assembly below inserts columns one at a time into small
# (<=~120 col) per-source frames; pandas' fragmentation warning doesn't
# indicate a correctness issue at this scale and would otherwise be noisy
# across 28 source files.
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SCRIPT_NAME = Path(__file__).name
CREATED_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column classification policy (derived from Milestone 2K-G-F/2K-G-F-B audit
# of the 780 non-ID columns across the 28 source CSVs; see
# docs/stage1_static_attribute_matrix_plan.md secs 5/8/9)
# ---------------------------------------------------------------------------

MATRIX_NAME_DEFAULT = "stage1_static_attributes_v001"

# (source_file, column) pairs to drop as exact duplicates of a column kept
# from a different (canonical) source file.
_DUPLICATE_DROP: dict[str, set[str]] = {
    "attributes_gageii_Bound_QA.csv": {"DRAIN_SQKM"},  # canonical copy kept from BasinID.csv
}

# Administrative / free-text columns: never physically meaningful as model input.
_ADMIN_DROP_TEXT: set[str] = {
    "STANAME", "COUNTYNAME_SITE", "WR_REPORT_REMARKS", "ADR_CITATION",
    "SCREENING_COMMENTS", "NAWQA_SUID",
}

# Numeric-looking columns that are actually administrative IDs/codes, not
# physical quantities (drainage-area duplicate handling excluded; see above).
_ADMIN_DROP_NUMERIC_ID: set[str] = {"FIPS_SITE", "REACHCODE", "BOUND_SOURCE"}

# Sparse binary membership flags: "present" (non-blank) means member,
# "missing" means non-member. Retained as diagnostic_record_network_qa (see
# below), NOT model_input -- gauge-network membership is not a basin/
# hydro-environmental attribute. This set now only controls *encoding style*
# (0/1 presence) within that diagnostic role, not the role itself.
_BINARY_FLAGS: set[str] = {"HCDN_2009", "HBN36", "OLD_HCDN", "NSIP_SENTINEL", "ACTIVE09"}

# Genuine categorical fields (including numeric-coded classes) deferred out of
# v001-core per the conservative-by-default policy. Retained as raw values in
# the "categorical_deferred" column group for a future documented ablation.
_CATEGORICAL_DEFER_EXPLICIT: set[str] = {
    "CLASS", "AGGECOREGION", "HUC10_CHECK",
    "GEOL_REEDBUSH_DOM", "GEOL_REEDBUSH_SITE",
    "GEOL_HUNT_DOM_CODE", "GEOL_HUNT_DOM_DESC", "GEOL_HUNT_SITE_CODE",
    "USDA_LRR_SITE",
    "ECO2_BAS_DOM", "ECO3_BAS_DOM", "HLR_BAS_DOM_100M", "NUTR_BAS_DOM", "PNV_BAS_DOM",
    "ECO3_SITE", "HLR100M_SITE", "HUC8_SITE", "NUTR_ECO_SITE",
}
# HydroATLAS "single most common class in catchment" / admin-division-ID
# columns use a `_cl_smj` / `_id_smj` suffix convention.
_CATEGORICAL_DEFER_SUFFIXES: tuple[str, ...] = ("_cl_smj", "_id_smj")

# Geography columns kept for split construction/diagnostics, excluded from
# model_input by explicit decision (docs/decision_log.md, 2026-07-06 follow-up).
_GEO_SPLIT_SUPPORT: set[str] = {"STATE", "HUC02"}

# Coordinates held out of model_input by default; reserved for a dedicated
# future ablation on spatial generalization. Gauge and basin-centroid
# coordinates are both direct-location fields (2026-07-20 static-attribute
# semantic correction; see docs/decision_log.md).
_DIAGNOSTIC_LATLON: set[str] = {"LAT_GAGE", "LNG_GAGE", "LAT_CENT", "LONG_CENT"}

# Gauge-record history / gauge-network membership / boundary-processing QA
# metadata: these describe the *observational record and its provenance*,
# not the basin's physical/hydro-environmental setting, so they are held out
# of model_input as diagnostic metadata (2026-07-20 static-attribute semantic
# correction; see docs/decision_log.md). NWIS_DRAIN_SQKM and PCT_DIFF_NWIS
# still have their -9999 sentinels decoded (see _SENTINEL_VALUES_BY_COLUMN)
# for provenance/validation purposes even though they land here, not in
# model_input.
_DIAGNOSTIC_RECORD_NETWORK_QA: set[str] = {
    "FLOWYRS_1900_2009", "FLOWYRS_1950_2009", "FLOWYRS_1990_2009",
    "FLOW_PCT_EST_VALUES", "BASIN_BOUNDARY_CONFIDENCE",
    "ACTIVE09", "HBN36", "HCDN_2009", "OLD_HCDN", "NSIP_SENTINEL",
    "PCT_DIFF_NWIS", "NWIS_DRAIN_SQKM",
}

# Deferred/ambiguous fields: semantics not yet fully resolved against the
# exact HydroATLAS catalog; excluded from the first Stage 1 baseline
# model_input set pending that resolution (2026-07-20 static-attribute
# semantic correction; see docs/decision_log.md).
_DEFERRED_AMBIGUOUS: set[str] = {"lka_pc_use"}

# Explicit per-column sentinel-value maps: exact numeric values decoded to
# NaN before missingness calculation and role classification. Narrowly
# scoped -- only the listed column is affected, no blanket negative-value
# replacement (2026-07-20 static-attribute semantic correction; see
# docs/decision_log.md and the 496-column semantic audit it codifies).
_SENTINEL_VALUES_BY_COLUMN: dict[str, frozenset[float]] = {
    "RAW_DIS_NEAREST_DAM": frozenset({-999.0}),
    "RAW_AVG_DIS_ALLDAMS": frozenset({-999.0}),
    "RAW_DIS_NEAREST_MAJ_DAM": frozenset({-999.0}),
    "RAW_AVG_DIS_ALL_MAJ_DAMS": frozenset({-999.0}),
    "RAW_DIS_NEAREST_CANAL": frozenset({-999.0}),
    "RAW_AVG_DIS_ALLCANALS": frozenset({-999.0}),
    "RAW_DIS_NEAREST_MAJ_NPDES": frozenset({-999.0}),
    "RAW_AVG_DIS_ALL_MAJ_NPDES": frozenset({-999.0}),
    "NWIS_DRAIN_SQKM": frozenset({-9999.0}),
    "PCT_DIFF_NWIS": frozenset({-9999.0}),
    "PERHOR": frozenset({-9999.0}),
    "STRAHLER_MAX": frozenset({-99.0}),
}
_SENTINEL_DECODE_ALGORITHM_ID = "stage1_static_sentinel_decode_v1"

# Per-year raw series: drop outright because equivalent official summary
# columns already exist natively in the same file (FLOWYRS_*, FLOW_PCT_EST_VALUES).
_PER_YEAR_DROP: dict[str, str] = {
    "attributes_gageii_FlowRec.csv": r"^wy\d{4}$",
}
# Per-year raw series with no existing native summary: reduce to mean/std
# across the Stage1 basin's own row (interannual variability descriptors).
_PER_YEAR_REDUCE: dict[str, tuple[str, str, str]] = {
    # file -> (regex, output_prefix, unit_suffix)
    "attributes_gageii_Climate_Ppt_Annual.csv": (r"^PPT\d{4}_AVG$", "climate_ppt_annual", "mm"),
    "attributes_gageii_Climate_Tmp_Annual.csv": (r"^TMP\d{4}_AVG$", "climate_tmp_annual", "c"),
}

# Safety-net regex used by the auditor (and re-checked here defensively) to
# catch any ID/code-like column name that should never land in model_input.
ID_LIKE_PATTERN = re.compile(
    r"(CODE|FIPS|REACHCODE|SUID|_SOURCE$|_ID$|_id_smj$|_cl_smj$|COMID|SITENO|GAGEID|^HUC\d+$)",
    re.IGNORECASE,
)

_NEAR_CONSTANT_MAX_NUNIQUE = 1     # nunique(dropna=True) <= this -> excluded
_HIGH_MISSING_THRESHOLD = 0.20     # missing_frac > this -> excluded

# The exact HydroATLAS 5-basin gap identified and audited in 2K-G-F/2K-G-F-B.
# If the observed gap differs, the build fails loud (mandatory gate).
_EXPECTED_HYDROATLAS_GAP_STAIDS: frozenset[str] = frozenset({
    "393109104464500", "394839104570300", "401733105392404",
    "402114105350101", "402913084285400",
})

_HYDROATLAS_FILE = "attributes_hydroATLAS.csv"
_NLDAS2_FILE = "attributes_nldas2_climate.csv"
_CHECKSUM_FILENAME = "source_attributes_v001_checksums.sha256"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _norm_staid(s: object) -> str:
    try:
        return f"{int(float(str(s).strip())):08d}"
    except (ValueError, TypeError):
        return str(s).strip().zfill(8)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_staid_col(df: pd.DataFrame) -> str:
    for candidate in ("STAID", "staid"):
        if candidate in df.columns:
            return candidate
    log.error("No STAID column found. Columns: %s", list(df.columns[:10]))
    sys.exit(1)


def _decode_column_sentinels(series: pd.Series, colname: str, counts: dict[str, int]) -> pd.Series:
    """Replace exact sentinel value(s) for ``colname`` with NaN; pass through unchanged
    otherwise. Only columns present in ``_SENTINEL_VALUES_BY_COLUMN`` are touched. Fails
    loud if the column contains a non-blank value that doesn't coerce to numeric (a
    sentinel-mapped column is expected to be numeric-with-sentinels, not mixed-schema).
    Records the replacement count in ``counts`` (0 included, not omitted)."""
    sentinels = _SENTINEL_VALUES_BY_COLUMN.get(colname)
    if not sentinels:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    nonnumeric_mask = numeric.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if nonnumeric_mask.any():
        offending = sorted(series[nonnumeric_mask].astype(str).unique())[:10]
        log.error(
            "Column '%s' is mapped for sentinel decoding (values %s) but contains "
            "non-numeric value(s), refusing to silently proceed. Offending raw values "
            "(up to 10): %s",
            colname, sorted(sentinels), offending,
        )
        sys.exit(1)
    sentinel_mask = numeric.isin(sentinels)
    counts[colname] = int(sentinel_mask.sum())
    decoded = series.copy()
    decoded[sentinel_mask] = np.nan
    return decoded


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source-dir", required=True,
                   help="Source attribute mirror directory (29 files).")
    p.add_argument("--manifest", required=True,
                   help="Stage 1 basin manifest CSV (STAID, final_training_status).")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for the derived matrix + manifest + provenance.")
    p.add_argument("--matrix-name", default=MATRIX_NAME_DEFAULT,
                   help=f"Base filename for outputs (default: {MATRIX_NAME_DEFAULT}).")
    p.add_argument("--require-checksums", dest="require_checksums",
                   action="store_true", default=True,
                   help="Fail if the source checksum file is missing or a checksum mismatches "
                        "(default: on).")
    p.add_argument("--no-require-checksums", dest="require_checksums",
                   action="store_false",
                   help="Do not require/verify the source checksum file (local dry runs only).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite out-dir contents if the matrix file already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs, run the full classification, print the plan; write nothing.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Source verification
# ---------------------------------------------------------------------------

def _verify_source_dir(source_dir: Path, require_checksums: bool) -> tuple[list[Path], list[Path], Path | None]:
    """Return (gageii_csvs, other_csvs=[hydroatlas, nldas2], checksum_path or None)."""
    if not source_dir.is_dir():
        log.error("Source directory does not exist: %s", source_dir)
        sys.exit(1)

    gageii_csvs = sorted(source_dir.glob("attributes_gageii_*.csv"))
    hydroatlas = source_dir / _HYDROATLAS_FILE
    nldas2 = source_dir / _NLDAS2_FILE
    for fp in (hydroatlas, nldas2):
        if not fp.exists():
            log.error("Required source file missing: %s", fp)
            sys.exit(1)
    if not gageii_csvs:
        log.error("No attributes_gageii_*.csv files found under %s", source_dir)
        sys.exit(1)

    # Guard against accidental raw-data files/dirs in the mirror (this
    # directory must contain only the 29 curated source files).
    unexpected = []
    for entry in sorted(source_dir.iterdir()):
        if entry.is_dir():
            unexpected.append(entry.name)
            continue
        if entry.suffix.lower() in (".csv", ".xlsx", ".sha256", ".md"):
            continue
        unexpected.append(entry.name)
    if unexpected:
        log.warning("Unexpected non-source entries in source-dir (review manually): %s", unexpected)

    checksum_path = source_dir / _CHECKSUM_FILENAME
    if not checksum_path.exists():
        msg = f"Checksum file not found: {checksum_path}"
        if require_checksums:
            log.error(msg + " (use --no-require-checksums to bypass for local dry runs)")
            sys.exit(1)
        log.warning(msg + " -- proceeding WITHOUT checksum verification (--no-require-checksums).")
        checksum_path = None

    if checksum_path is not None:
        _verify_checksums(source_dir, checksum_path)

    return gageii_csvs, [hydroatlas, nldas2], checksum_path


def _verify_checksums(source_dir: Path, checksum_path: Path) -> None:
    log.info("Verifying source checksums against %s ...", checksum_path.name)
    entries: list[tuple[str, str]] = []
    with open(checksum_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                log.error("Malformed checksum line: %r", line)
                sys.exit(1)
            digest, fname = parts
            fname = fname.lstrip("*").strip()
            entries.append((digest.lower(), fname))

    mismatches: list[str] = []
    missing: list[str] = []
    for digest, fname in entries:
        fp = source_dir / fname
        if not fp.exists():
            missing.append(fname)
            continue
        actual = _sha256(fp)
        if actual.lower() != digest:
            mismatches.append(fname)

    if missing:
        log.error("Files listed in checksum file but missing on disk: %s", missing)
    if mismatches:
        log.error("Checksum MISMATCH for: %s", mismatches)
    if missing or mismatches:
        sys.exit(1)
    log.info("Checksums verified OK for %d files.", len(entries))


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)
    df = pd.read_csv(manifest_path, dtype=str)
    if "STAID" not in df.columns:
        log.error("Manifest missing STAID column. Columns: %s", list(df.columns))
        sys.exit(1)
    df = df.copy()
    df["gauge_id"] = df["STAID"].apply(_norm_staid)
    dupes = df["gauge_id"].duplicated().sum()
    if dupes:
        log.error("Manifest has %d duplicate gauge_id(s) after normalization.", dupes)
        sys.exit(1)
    return df.set_index("gauge_id")


# ---------------------------------------------------------------------------
# Per-file loading + classification
# ---------------------------------------------------------------------------

class _FileResult:
    def __init__(self, name: str):
        self.name = name
        self.model_input = pd.DataFrame()
        self.split_support = pd.DataFrame()
        self.diagnostic = pd.DataFrame()
        self.diagnostic_record_network_qa = pd.DataFrame()
        self.deferred_ambiguous = pd.DataFrame()
        self.categorical_deferred = pd.DataFrame()
        self.dropped: dict[str, list[str]] = {
            "duplicate": [], "admin_text": [], "admin_numeric_id": [],
            "per_year_raw_dropped": [],
        }
        self.binary_encoded: list[str] = []
        self.per_year_reduced: list[str] = []
        self.coverage_missing: list[str] = []
        self.sentinel_replacement_counts: dict[str, int] = {}


def _classify_columns(name: str, cols: list[str]) -> dict[str, str]:
    """Return {column: role} for every non-ID column in this file."""
    roles: dict[str, str] = {}
    dup_drop = _DUPLICATE_DROP.get(name, set())
    drop_re = _PER_YEAR_DROP.get(name)
    reduce_spec = _PER_YEAR_REDUCE.get(name)

    for c in cols:
        if c in dup_drop:
            roles[c] = "duplicate"
        elif c in _ADMIN_DROP_TEXT:
            roles[c] = "admin_text"
        elif c in _ADMIN_DROP_NUMERIC_ID:
            roles[c] = "admin_numeric_id"
        elif c in _DIAGNOSTIC_RECORD_NETWORK_QA:
            # Checked before _BINARY_FLAGS: the 5 sparse-membership flags
            # (HCDN_2009/HBN36/OLD_HCDN/NSIP_SENTINEL/ACTIVE09) are a subset
            # of this diagnostic set and must land here, not in model_input.
            roles[c] = "diagnostic_record_network_qa"
        elif c in _BINARY_FLAGS:
            roles[c] = "binary_flag"
        elif c in _DEFERRED_AMBIGUOUS:
            roles[c] = "deferred_ambiguous"
        elif c in _CATEGORICAL_DEFER_EXPLICIT or c.endswith(_CATEGORICAL_DEFER_SUFFIXES):
            roles[c] = "categorical_deferred"
        elif c in _GEO_SPLIT_SUPPORT:
            roles[c] = "split_support"
        elif c in _DIAGNOSTIC_LATLON:
            roles[c] = "diagnostic_latlon"
        elif drop_re and re.match(drop_re, c):
            roles[c] = "per_year_raw_dropped"
        elif reduce_spec and re.match(reduce_spec[0], c):
            roles[c] = "per_year_reduce_source"
        else:
            roles[c] = "candidate_model_input"
    return roles


def _load_and_classify(path: Path, stage1_ids: list[str]) -> _FileResult:
    name = path.name
    log.info("Loading %s ...", name)
    raw = pd.read_csv(path, sep=None, engine="python", dtype=str)
    id_col = _find_staid_col(raw)
    raw = raw.copy()
    raw["gauge_id"] = raw[id_col].apply(_norm_staid)
    dupes = raw["gauge_id"].duplicated().sum()
    if dupes:
        log.error("%s: %d duplicate gauge_id rows after normalization.", name, dupes)
        sys.exit(1)
    raw = raw.set_index("gauge_id")
    if id_col != "gauge_id":
        raw = raw.drop(columns=[id_col])

    result = _FileResult(name)
    is_hydroatlas = name == _HYDROATLAS_FILE

    present_ids = [i for i in stage1_ids if i in raw.index]
    missing_ids = [i for i in stage1_ids if i not in raw.index]

    if is_hydroatlas:
        observed_gap = frozenset(missing_ids)
        if observed_gap != _EXPECTED_HYDROATLAS_GAP_STAIDS:
            log.error(
                "HydroATLAS coverage gap changed since the 2K-G-F/2K-G-F-B audit. "
                "Expected exactly %s missing, observed %s missing. "
                "This is a mandatory build/audit gate (see docs/stage1_static_attribute_matrix_plan.md "
                "sec 4/9) -- refusing to silently proceed. Update the expected-gap policy "
                "deliberately if this change is understood, then rerun.",
                sorted(_EXPECTED_HYDROATLAS_GAP_STAIDS), sorted(observed_gap),
            )
            sys.exit(1)
        log.info(
            "HydroATLAS gap gate: observed gap matches expected 5-basin gap exactly. "
            "Retaining those basins with NaN HydroATLAS columns + hydroatlas_coverage_flag=0."
        )
        result.coverage_missing = sorted(missing_ids)
        # Reindex to the full Stage 1 universe; missing basins become all-NaN rows.
        sub = raw.reindex(stage1_ids)
    else:
        if missing_ids:
            log.error("%s: %d Stage 1 basin(s) missing (unexpected -- audit found 100%% coverage): %s",
                      name, len(missing_ids), missing_ids[:10])
            sys.exit(1)
        sub = raw.loc[stage1_ids]

    # Sentinel decoding happens before role classification and before the
    # missingness calculation, so decoded NaNs correctly inflate missingness
    # for the affected columns (2026-07-20 static-attribute semantic
    # correction). Only columns explicitly listed in
    # _SENTINEL_VALUES_BY_COLUMN are touched; every other column, including
    # unrelated ones containing the same literal sentinel number, is
    # untouched.
    for c in sub.columns:
        if c in _SENTINEL_VALUES_BY_COLUMN:
            sub[c] = _decode_column_sentinels(sub[c], c, result.sentinel_replacement_counts)

    non_id_cols = [c for c in sub.columns]
    roles = _classify_columns(name, non_id_cols)
    reduce_spec = _PER_YEAR_REDUCE.get(name)

    model_input_cols: list[str] = []
    for c, role in roles.items():
        if role == "duplicate":
            result.dropped["duplicate"].append(c)
        elif role == "admin_text":
            result.dropped["admin_text"].append(c)
        elif role == "admin_numeric_id":
            result.dropped["admin_numeric_id"].append(c)
        elif role == "per_year_raw_dropped":
            result.dropped["per_year_raw_dropped"].append(c)
        elif role == "diagnostic_record_network_qa":
            if c in _BINARY_FLAGS:
                enc = sub[c].notna() & (sub[c].astype(str).str.strip() != "")
                result.diagnostic_record_network_qa[c] = enc.astype("int8")
                result.binary_encoded.append(c)
            else:
                result.diagnostic_record_network_qa[c] = pd.to_numeric(sub[c], errors="coerce")
        elif role == "binary_flag":
            enc = sub[c].notna() & (sub[c].astype(str).str.strip() != "")
            result.model_input[c] = enc.astype("int8")
            result.binary_encoded.append(c)
        elif role == "deferred_ambiguous":
            result.deferred_ambiguous[c] = pd.to_numeric(sub[c], errors="coerce")
        elif role == "categorical_deferred":
            result.categorical_deferred[c] = sub[c]
        elif role == "split_support":
            result.split_support[c] = sub[c]
        elif role == "diagnostic_latlon":
            result.diagnostic[c] = pd.to_numeric(sub[c], errors="coerce")
        elif role == "per_year_reduce_source":
            continue  # handled in bulk below
        elif role == "candidate_model_input":
            coerced = pd.to_numeric(sub[c], errors="coerce")
            # The 90% "reliably numeric" gate is a schema-drift safety net for
            # columns with no explicit classification. Sentinel-mapped columns
            # already went through an explicit, fail-loud numeric validation
            # in _decode_column_sentinels() -- their post-decode missingness
            # can legitimately exceed 10% (that's the point: it lets the
            # dynamic >20% high-missingness filter in build() decide their
            # fate, rather than this coarse gate hard-failing the whole build).
            if c not in _SENTINEL_VALUES_BY_COLUMN and coerced.notna().mean() < 0.90:
                log.error(
                    "%s: column '%s' is not reliably numeric (only %.1f%% coerces) and is not "
                    "classified as admin/categorical/flag -- refusing to silently include or drop. "
                    "Add an explicit classification for this column.",
                    name, c, 100 * coerced.notna().mean(),
                )
                sys.exit(1)
            result.model_input[c] = coerced
        else:
            log.error("%s: column '%s' has unrecognized role '%s'.", name, c, role)
            sys.exit(1)

    if reduce_spec is not None:
        pattern, prefix, unit = reduce_spec
        year_cols = [c for c in non_id_cols if re.match(pattern, c)]
        if year_cols:
            block = sub[year_cols].apply(pd.to_numeric, errors="coerce")
            result.model_input[f"{prefix}_mean_{unit}"] = block.mean(axis=1, skipna=True)
            result.model_input[f"{prefix}_std_{unit}"] = block.std(axis=1, skipna=True)
            result.per_year_reduced.extend(year_cols)

    # Safety net: nothing that looks ID-like should have landed in model_input.
    id_like_hits = [c for c in result.model_input.columns if ID_LIKE_PATTERN.search(c)]
    if id_like_hits:
        log.error("%s: ID-like column name(s) leaked into model_input: %s", name, id_like_hits)
        sys.exit(1)

    return result


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    gageii_csvs, other_csvs, checksum_path = _verify_source_dir(source_dir, args.require_checksums)
    manifest = _load_manifest(manifest_path)
    stage1_ids = sorted(manifest.index.tolist())
    log.info("Stage 1 basin universe: %d (TRAIN_CORE=%d, TRAIN_SOFT_KEEP=%d)",
              len(stage1_ids),
              (manifest["final_training_status"] == "TRAIN_CORE").sum(),
              (manifest["final_training_status"] == "TRAIN_SOFT_KEEP").sum())

    all_files = gageii_csvs + other_csvs
    log.info("Merging %d source files ...", len(all_files))

    results: list[_FileResult] = [_load_and_classify(p, stage1_ids) for p in all_files]

    model_input = pd.DataFrame(index=stage1_ids)
    split_support = pd.DataFrame(index=stage1_ids)
    diagnostic = pd.DataFrame(index=stage1_ids)
    diagnostic_record_network_qa = pd.DataFrame(index=stage1_ids)
    deferred_ambiguous = pd.DataFrame(index=stage1_ids)
    categorical_deferred = pd.DataFrame(index=stage1_ids)
    hydroatlas_flag = pd.Series(1, index=stage1_ids, dtype="int8", name="hydroatlas_coverage_flag")

    dropped_all: dict[str, list[str]] = {"duplicate": [], "admin_text": [], "admin_numeric_id": [],
                                           "per_year_raw_dropped": []}
    binary_encoded_all: list[str] = []
    per_year_reduced_all: list[str] = []
    column_source: dict[str, str] = {}
    sentinel_replacement_counts_all: dict[str, int] = {}

    model_input_parts = [model_input]
    split_support_parts = [split_support]
    diagnostic_parts = [diagnostic]
    diagnostic_record_network_qa_parts = [diagnostic_record_network_qa]
    deferred_ambiguous_parts = [deferred_ambiguous]
    categorical_deferred_parts = [categorical_deferred]
    seen_model_input_cols: set[str] = set()

    for r in results:
        collisions = seen_model_input_cols & set(r.model_input.columns)
        if collisions:
            log.error("Column name collision in model_input: %s (from %s, already present).",
                      sorted(collisions), r.name)
            sys.exit(1)
        seen_model_input_cols.update(r.model_input.columns)
        if not r.model_input.empty:
            model_input_parts.append(r.model_input)
        if not r.split_support.empty:
            split_support_parts.append(r.split_support)
        if not r.diagnostic.empty:
            diagnostic_parts.append(r.diagnostic)
        if not r.diagnostic_record_network_qa.empty:
            diagnostic_record_network_qa_parts.append(r.diagnostic_record_network_qa)
        if not r.deferred_ambiguous.empty:
            deferred_ambiguous_parts.append(r.deferred_ambiguous)
        if not r.categorical_deferred.empty:
            categorical_deferred_parts.append(r.categorical_deferred)
        for col in r.model_input.columns:
            column_source[col] = r.name
        for col in r.split_support.columns:
            column_source[col] = r.name
        for col in r.diagnostic.columns:
            column_source[col] = r.name
        for col in r.diagnostic_record_network_qa.columns:
            column_source[col] = r.name
        for col in r.deferred_ambiguous.columns:
            column_source[col] = r.name
        for col in r.categorical_deferred.columns:
            column_source[col] = r.name
        for k in dropped_all:
            dropped_all[k].extend(f"{r.name}:{c}" for c in r.dropped[k])
        binary_encoded_all.extend(f"{r.name}:{c}" for c in r.binary_encoded)
        per_year_reduced_all.extend(f"{r.name}:{c}" for c in r.per_year_reduced)
        for col, n in r.sentinel_replacement_counts.items():
            sentinel_replacement_counts_all[f"{r.name}:{col}"] = n
        if r.coverage_missing:
            hydroatlas_flag.loc[r.coverage_missing] = 0

    model_input = pd.concat(model_input_parts, axis=1)
    split_support = pd.concat(split_support_parts, axis=1)
    diagnostic = pd.concat(diagnostic_parts, axis=1)
    diagnostic_record_network_qa = pd.concat(diagnostic_record_network_qa_parts, axis=1)
    deferred_ambiguous = pd.concat(deferred_ambiguous_parts, axis=1)
    categorical_deferred = pd.concat(categorical_deferred_parts, axis=1)

    # ---- dynamic near-constant / high-missingness exclusion (model_input only) ----
    near_constant_excluded: list[str] = []
    high_missing_excluded: list[str] = []
    for col in list(model_input.columns):
        nunique = model_input[col].nunique(dropna=True)
        missing_frac = model_input[col].isna().mean()
        if nunique <= _NEAR_CONSTANT_MAX_NUNIQUE:
            near_constant_excluded.append(col)
            model_input.drop(columns=[col], inplace=True)
        elif missing_frac > _HIGH_MISSING_THRESHOLD:
            high_missing_excluded.append(col)
            model_input.drop(columns=[col], inplace=True)

    if near_constant_excluded:
        log.warning("Excluded %d near-constant model_input column(s): %s",
                    len(near_constant_excluded), near_constant_excluded)
    if high_missing_excluded:
        log.warning("Excluded %d high-missingness (>%.0f%%) model_input column(s): %s",
                    len(high_missing_excluded), 100 * _HIGH_MISSING_THRESHOLD, high_missing_excluded)

    # ---- assemble final matrix ----
    matrix = pd.concat(
        [model_input, split_support, diagnostic, diagnostic_record_network_qa,
         deferred_ambiguous, categorical_deferred, hydroatlas_flag],
        axis=1,
    )
    matrix.index.name = "gauge_id"
    matrix = matrix.join(manifest[["final_training_status"]])

    role_map: dict[str, str] = {c: "model_input" for c in model_input.columns}
    role_map.update({c: "split_support" for c in split_support.columns})
    role_map.update({c: "diagnostic_latlon" for c in diagnostic.columns})
    role_map.update({c: "diagnostic_record_network_qa" for c in diagnostic_record_network_qa.columns})
    role_map.update({c: "deferred_ambiguous" for c in deferred_ambiguous.columns})
    role_map.update({c: "categorical_deferred" for c in categorical_deferred.columns})
    role_map["hydroatlas_coverage_flag"] = "flag"
    role_map["final_training_status"] = "flag"

    log.info("Final matrix: %d rows x %d columns (%d model_input).",
              len(matrix), len(matrix.columns), len(model_input.columns))

    if args.dry_run:
        log.info("[dry-run] Would write matrix + manifest + provenance to %s. Nothing written.", out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / f"{args.matrix_name}.parquet"
    manifest_json_path = out_dir / f"{args.matrix_name}_column_manifest.json"
    provenance_path = out_dir / f"{args.matrix_name}_provenance.json"

    if matrix_path.exists() and not args.force:
        log.error("%s already exists (use --force to overwrite).", matrix_path)
        sys.exit(1)

    matrix.to_parquet(matrix_path)
    matrix_sha256 = _sha256(matrix_path)

    column_manifest = {
        "matrix_name": args.matrix_name,
        "created_utc": CREATED_UTC,
        "n_rows": len(matrix),
        "n_columns": len(matrix.columns),
        "columns": {
            c: {"role": role_map.get(c, "id" if c == "gauge_id" else "unknown"),
                "source_file": column_source.get(c)}
            for c in matrix.columns
        },
    }
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(column_manifest, f, indent=2)

    provenance = {
        "script": SCRIPT_NAME,
        "created_utc": CREATED_UTC,
        "source_dir": str(source_dir),
        "checksum_file_verified": checksum_path is not None,
        "manifest_path": str(manifest_path),
        "stage1_basin_count": len(stage1_ids),
        "matrix_path": str(matrix_path),
        "matrix_sha256": matrix_sha256,
        "n_model_input_columns": len(model_input.columns),
        "n_split_support_columns": len(split_support.columns),
        "n_diagnostic_latlon_columns": len(diagnostic.columns),
        "n_diagnostic_record_network_qa_columns": len(diagnostic_record_network_qa.columns),
        "n_deferred_ambiguous_columns": len(deferred_ambiguous.columns),
        "n_categorical_deferred_columns": len(categorical_deferred.columns),
        "hydroatlas_gap": {
            "expected_missing_staids": sorted(_EXPECTED_HYDROATLAS_GAP_STAIDS),
            "policy": "retain with NaN HydroATLAS columns + hydroatlas_coverage_flag=0 "
                      "(option b of the mandatory gate); build fails if observed gap differs",
            "basins_flagged_missing": int((hydroatlas_flag == 0).sum()),
        },
        "dropped_columns": dropped_all,
        "binary_flags_encoded": binary_encoded_all,
        "per_year_columns_reduced_to_summary_stats": per_year_reduced_all,
        "per_year_columns_dropped_raw_native_summary_exists": dropped_all["per_year_raw_dropped"],
        "near_constant_excluded_model_input": near_constant_excluded,
        "high_missing_excluded_model_input": high_missing_excluded,
        "near_constant_max_nunique": _NEAR_CONSTANT_MAX_NUNIQUE,
        "high_missing_threshold": _HIGH_MISSING_THRESHOLD,
        "sentinel_decoding": {
            "algorithm_id": _SENTINEL_DECODE_ALGORITHM_ID,
            "sentinel_values_by_column": {
                c: sorted(v) for c, v in _SENTINEL_VALUES_BY_COLUMN.items()
            },
            "replacement_counts_by_source_qualified_column": sentinel_replacement_counts_all,
            "total_values_replaced": sum(sentinel_replacement_counts_all.values()),
        },
    }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    log.info("Wrote matrix       -> %s (sha256=%s)", matrix_path, matrix_sha256)
    log.info("Wrote col manifest -> %s", manifest_json_path)
    log.info("Wrote provenance   -> %s", provenance_path)


def main() -> None:
    args = _parse_args()
    build(args)


if __name__ == "__main__":
    main()
