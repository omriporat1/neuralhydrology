#!/usr/bin/env python3
"""Audit the Stage 1 derived static-attribute matrix (Milestone 2K-G-F-B).

Independently checks the output of scripts/build_stage1_static_attribute_matrix.py
against the Stage 1 basin manifest and the documented column-classification
policy in docs/stage1_static_attribute_matrix_plan.md. Read-only: never
modifies the matrix or source files.

Usage:
  python scripts/audit_stage1_static_attribute_matrix.py \\
    --matrix-dir /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/ \\
    --matrix-name stage1_static_attributes_v001 \\
    --manifest   config/stage1_initial_training_basin_manifest.csv

Exit code 0 = PASS (no errors; warnings allowed), 1 = FAIL (>=1 error).
Writes stage1_static_attributes_v001_audit_summary.md into --matrix-dir.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_NAME = Path(__file__).name
CREATED_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# Must match scripts/build_stage1_static_attribute_matrix.py policy exactly.
_GEO_SPLIT_SUPPORT = {"STATE", "HUC02"}
# Direct gauge / basin-centroid coordinates -- diagnostic only, never model_input
# (2026-07-20 static-attribute semantic correction; see docs/decision_log.md).
_DIAGNOSTIC_LATLON = {"LAT_GAGE", "LNG_GAGE", "LAT_CENT", "LONG_CENT"}

# Gauge-record history / gauge-network membership / boundary-processing QA
# metadata -- diagnostic only, never model_input (same correction).
_DIAGNOSTIC_RECORD_NETWORK_QA = {
    "FLOWYRS_1900_2009", "FLOWYRS_1950_2009", "FLOWYRS_1990_2009",
    "FLOW_PCT_EST_VALUES", "BASIN_BOUNDARY_CONFIDENCE",
    "ACTIVE09", "HBN36", "HCDN_2009", "OLD_HCDN", "NSIP_SENTINEL",
    "PCT_DIFF_NWIS", "NWIS_DRAIN_SQKM",
}

# Deferred/ambiguous fields -- never model_input in the first Stage 1 baseline
# (same correction).
_DEFERRED_AMBIGUOUS = {"lka_pc_use"}

# The 8 infrastructure-distance sentinel columns: must be excluded from
# model_input via the ordinary high-missingness mechanism (>20%) after
# sentinel decoding, never by hand-classification. If any survive as
# model_input, the sentinel-decode/exclusion pipeline is broken.
_RAW_INFRA_DISTANCE_COLUMNS = {
    "RAW_DIS_NEAREST_DAM", "RAW_AVG_DIS_ALLDAMS",
    "RAW_DIS_NEAREST_MAJ_DAM", "RAW_AVG_DIS_ALL_MAJ_DAMS",
    "RAW_DIS_NEAREST_CANAL", "RAW_AVG_DIS_ALLCANALS",
    "RAW_DIS_NEAREST_MAJ_NPDES", "RAW_AVG_DIS_ALL_MAJ_NPDES",
}

# Fields explicitly retained as model_input despite the semantic correction
# (2026-07-20; see docs/decision_log.md) -- positive-checked below.
_RETAINED_MODEL_INPUT_FIELDS = {"PERHOR", "STRAHLER_MAX", "dor_pc_pva", "dis_m3_pyr", "run_mm_syr"}

# Explicit per-column sentinel-value maps, mirrored independently from
# scripts/build_stage1_static_attribute_matrix.py::_SENTINEL_VALUES_BY_COLUMN.
# Only these exact (column, value) pairs are checked -- not a blanket
# negative-value rejection.
_SENTINEL_VALUES_BY_COLUMN: dict[str, set[float]] = {
    "RAW_DIS_NEAREST_DAM": {-999.0},
    "RAW_AVG_DIS_ALLDAMS": {-999.0},
    "RAW_DIS_NEAREST_MAJ_DAM": {-999.0},
    "RAW_AVG_DIS_ALL_MAJ_DAMS": {-999.0},
    "RAW_DIS_NEAREST_CANAL": {-999.0},
    "RAW_AVG_DIS_ALLCANALS": {-999.0},
    "RAW_DIS_NEAREST_MAJ_NPDES": {-999.0},
    "RAW_AVG_DIS_ALL_MAJ_NPDES": {-999.0},
    "NWIS_DRAIN_SQKM": {-9999.0},
    "PCT_DIFF_NWIS": {-9999.0},
    "PERHOR": {-9999.0},
    "STRAHLER_MAX": {-99.0},
}

_EXPECTED_HYDROATLAS_GAP_STAIDS = frozenset({
    "393109104464500", "394839104570300", "401733105392404",
    "402114105350101", "402913084285400",
})
ID_LIKE_PATTERN = re.compile(
    r"(CODE|FIPS|REACHCODE|SUID|_SOURCE$|_ID$|_id_smj$|_cl_smj$|COMID|SITENO|GAGEID|^HUC\d+$)",
    re.IGNORECASE,
)
_NEAR_CONSTANT_MAX_NUNIQUE = 1
_HIGH_MISSING_THRESHOLD = 0.20

_issues: list[str] = []
_warns: list[str] = []
_oks: list[str] = []


def err(msg: str, basin: str = "") -> None:
    tag = f"[{basin}] " if basin else ""
    _issues.append(f"{tag}{msg}")


def warn(msg: str, basin: str = "") -> None:
    tag = f"[{basin}] " if basin else ""
    _warns.append(f"{tag}{msg}")


def ok(msg: str) -> None:
    _oks.append(msg)


def chk(label: str, passed: bool, detail: str = "", basin: str = "") -> None:
    if passed:
        ok(f"{label}" + (f" ({detail})" if detail else ""))
    else:
        err(f"{label} FAILED" + (f" -- {detail}" if detail else ""), basin=basin)


def _norm_staid(s: object) -> str:
    try:
        return f"{int(float(str(s).strip())):08d}"
    except (ValueError, TypeError):
        return str(s).strip().zfill(8)


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix-dir", required=True)
    p.add_argument("--matrix-name", default="stage1_static_attributes_v001")
    p.add_argument("--manifest", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    matrix_dir = Path(args.matrix_dir)
    matrix_path = matrix_dir / f"{args.matrix_name}.parquet"
    manifest_json_path = matrix_dir / f"{args.matrix_name}_column_manifest.json"
    provenance_path = matrix_dir / f"{args.matrix_name}_provenance.json"

    if not matrix_path.exists():
        err(f"Matrix file not found: {matrix_path}")
        _finish(matrix_dir, args.matrix_name)
        return
    if not manifest_json_path.exists():
        err(f"Column manifest not found: {manifest_json_path}")
    if not provenance_path.exists():
        err(f"Provenance file not found: {provenance_path}")

    df = pd.read_parquet(matrix_path)
    chk("Matrix loads", True, f"{len(df)} rows x {len(df.columns)} cols")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        err(f"Basin manifest not found: {manifest_path}")
        _finish(matrix_dir, args.matrix_name)
        return
    manifest = pd.read_csv(manifest_path, dtype=str)
    manifest["gauge_id"] = manifest["STAID"].apply(_norm_staid)
    expected_ids = set(manifest["gauge_id"])

    # ---- 1. row count / basin coverage ----
    actual_ids = set(df.index.astype(str))
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    chk("Row count matches Stage 1 basin manifest", len(df) == len(expected_ids),
        f"matrix={len(df)}, manifest={len(expected_ids)}")
    chk("All Stage 1 basins present", not missing, f"{len(missing)} missing: {sorted(missing)[:10]}")
    chk("No unexpected extra basins", not extra, f"{len(extra)} extra: {sorted(extra)[:10]}")

    # ---- 2. duplicate STAIDs ----
    dup_count = df.index.duplicated().sum()
    chk("No duplicate gauge_id in matrix index", dup_count == 0, f"{dup_count} duplicates")

    # ---- column manifest / roles ----
    col_roles: dict[str, str] = {}
    if manifest_json_path.exists():
        with open(manifest_json_path, "r", encoding="utf-8") as f:
            col_manifest = json.load(f)
        col_roles = {c: v["role"] for c, v in col_manifest.get("columns", {}).items()}
        ok(f"Column manifest loaded ({len(col_roles)} columns described)")
    else:
        warn("Column manifest missing -- role-based checks will use heuristics only")

    _non_model_input_fallback = (
        _GEO_SPLIT_SUPPORT | _DIAGNOSTIC_LATLON | _DIAGNOSTIC_RECORD_NETWORK_QA | _DEFERRED_AMBIGUOUS
    )
    model_input_cols = [c for c, r in col_roles.items() if r == "model_input"] or \
        [c for c in df.columns if c not in _non_model_input_fallback
         and c not in ("hydroatlas_coverage_flag", "final_training_status")]

    # ---- 3. missingness per variable ----
    high_missing = []
    for c in model_input_cols:
        if c not in df.columns:
            continue
        frac = df[c].isna().mean()
        if frac > _HIGH_MISSING_THRESHOLD:
            high_missing.append((c, round(float(frac), 3)))
    if high_missing:
        for c, frac in high_missing:
            if c == "hydroatlas_coverage_flag" or frac <= 0.0018:  # ~5/2843 HydroATLAS gap tolerance
                continue
            warn(f"model_input column '{c}' has {frac:.1%} missingness (>{_HIGH_MISSING_THRESHOLD:.0%})")
    ok(f"Missingness scan complete over {len(model_input_cols)} model_input columns")

    # ---- missingness per basin ----
    if model_input_cols:
        per_basin_missing = df[model_input_cols].isna().mean(axis=1)
        worst = per_basin_missing.sort_values(ascending=False).head(5)
        for gid, frac in worst.items():
            if frac > 0.5:
                warn(f"basin has {frac:.1%} of model_input columns missing", basin=str(gid))
        ok("Per-basin missingness scan complete")

    # ---- 4. numeric ranges (sanity, not hard bounds) ----
    range_issues = []
    for c in model_input_cols:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        col = df[c].dropna()
        if col.empty:
            continue
        if not (col.abs() < 1e13).all():
            range_issues.append(c)
    # Note: HydroATLAS basin-integrated economic aggregates (e.g. gdp_ud_usu,
    # upstream-summed GDP in USD) legitimately reach into the trillions for
    # large basins -- 1e13 is calibrated to allow that while still catching
    # genuine unit/overflow errors.
    chk("No implausibly large (>1e13) model_input values", not range_issues, str(range_issues[:10]))

    # ---- 5. constant / near-constant columns ----
    const_cols = []
    for c in model_input_cols:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=True) <= _NEAR_CONSTANT_MAX_NUNIQUE:
            const_cols.append(c)
    chk("No constant/near-constant model_input columns", not const_cols, str(const_cols[:10]))

    # ---- 6. duplicate columns (by value, not just name) ----
    dup_col_pairs = []
    seen: dict[tuple, str] = {}
    for c in model_input_cols:
        if c not in df.columns:
            continue
        try:
            key = tuple(pd.util.hash_pandas_object(df[c], index=False))
        except TypeError:
            continue
        if key in seen:
            dup_col_pairs.append((seen[key], c))
        else:
            seen[key] = c
    chk("No duplicate-value model_input column pairs", not dup_col_pairs, str(dup_col_pairs[:10]))

    # ---- 7. categorical / non-numeric leakage into model_input ----
    non_numeric = [c for c in model_input_cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    chk("No non-numeric columns in model_input", not non_numeric, str(non_numeric))
    id_like_leak = [c for c in model_input_cols if ID_LIKE_PATTERN.search(c)]
    chk("No ID/code-like column names in model_input", not id_like_leak, str(id_like_leak))

    # ---- 8. STATE/HUC02/lat/lon excluded from model_input ----
    leaked_split_support = sorted(_GEO_SPLIT_SUPPORT & set(model_input_cols))
    leaked_latlon = sorted(_DIAGNOSTIC_LATLON & set(model_input_cols))
    chk("STATE/HUC02 excluded from model_input", not leaked_split_support, str(leaked_split_support))
    chk("Direct coordinate fields excluded from model_input", not leaked_latlon, str(leaked_latlon))
    chk("STATE/HUC02 present in matrix (as split_support)", _GEO_SPLIT_SUPPORT.issubset(df.columns) or
        any(c in df.columns for c in _GEO_SPLIT_SUPPORT))
    chk("Direct coordinate fields present in matrix (as diagnostic)", _DIAGNOSTIC_LATLON.issubset(df.columns) or
        any(c in df.columns for c in _DIAGNOSTIC_LATLON))

    # ---- 8b. record/network/QA + deferred-ambiguous fields excluded from model_input
    # (2026-07-20 static-attribute semantic correction; see docs/decision_log.md) ----
    leaked_record_qa = sorted(_DIAGNOSTIC_RECORD_NETWORK_QA & set(model_input_cols))
    chk("Gauge-record/network/QA fields excluded from model_input", not leaked_record_qa,
        str(leaked_record_qa))
    leaked_deferred = sorted(_DEFERRED_AMBIGUOUS & set(model_input_cols))
    chk("lka_pc_use excluded from model_input", not leaked_deferred, str(leaked_deferred))

    # ---- 8c. the 8 infrastructure-distance columns must NOT survive as model_input
    # (they must be excluded through the ordinary high-missingness mechanism after
    # sentinel decoding, never by hand-classification -- this proves the policy is
    # operating correctly) ----
    leaked_raw_infra = sorted(_RAW_INFRA_DISTANCE_COLUMNS & set(model_input_cols))
    chk("Infrastructure-distance RAW_* columns excluded from model_input", not leaked_raw_infra,
        str(leaked_raw_infra))

    # ---- 8d. no literal mapped sentinel value remains in any model_input column ----
    sentinel_leaks = []
    for c, sentinels in _SENTINEL_VALUES_BY_COLUMN.items():
        if c not in model_input_cols or c not in df.columns:
            continue
        n_hit = int(df[c].isin(sentinels).sum())
        if n_hit:
            sentinel_leaks.append(f"{c}: {n_hit} value(s) still equal to sentinel(s) {sorted(sentinels)}")
    chk("No literal sentinel values remain in model_input columns", not sentinel_leaks, str(sentinel_leaks))

    # ---- 8e. positive checks: fields explicitly retained as model_input ----
    for c in sorted(_RETAINED_MODEL_INPUT_FIELDS):
        if c not in col_roles:
            warn(f"retained field '{c}' not present in column manifest -- cannot verify role")
            continue
        chk(f"'{c}' remains model_input", col_roles.get(c) == "model_input",
            f"role={col_roles.get(c)!r}")
    for c in ("PERHOR", "STRAHLER_MAX"):
        if c not in df.columns or c not in _SENTINEL_VALUES_BY_COLUMN:
            continue
        n_hit = int(df[c].isin(_SENTINEL_VALUES_BY_COLUMN[c]).sum())
        chk(f"'{c}' has no remaining sentinel value(s)", n_hit == 0, f"{n_hit} found")

    # ---- 8f. column-manifest / matrix column consistency ----
    if manifest_json_path.exists():
        manifest_cols = set(col_roles.keys())
        matrix_cols = set(df.columns) | {"gauge_id"}  # gauge_id is the index, not a data column
        manifest_only = sorted(manifest_cols - matrix_cols)
        matrix_only = sorted(set(df.columns) - manifest_cols)
        chk("Column manifest matches matrix columns exactly", not manifest_only and not matrix_only,
            f"manifest_only={manifest_only[:10]}, matrix_only={matrix_only[:10]}")

    # ---- 9. HydroATLAS 5-basin gap handling ----
    if "hydroatlas_coverage_flag" not in df.columns:
        err("hydroatlas_coverage_flag column missing from matrix")
    else:
        flagged_missing = set(df.index[df["hydroatlas_coverage_flag"] == 0].astype(str))
        chk("HydroATLAS coverage flag matches expected 5-basin gap exactly",
            flagged_missing == _EXPECTED_HYDROATLAS_GAP_STAIDS,
            f"flagged={sorted(flagged_missing)}")
        # for flagged basins, HydroATLAS-sourced model_input cols should be NaN
        hydroatlas_cols = [c for c, src in col_manifest.get("columns", {}).items()
                            if col_manifest["columns"][c].get("source_file") == "attributes_hydroATLAS.csv"
                            and col_manifest["columns"][c]["role"] == "model_input"] if manifest_json_path.exists() else []
        if hydroatlas_cols and flagged_missing:
            not_nan = []
            for gid in flagged_missing:
                if gid not in df.index:
                    continue
                row = df.loc[gid, hydroatlas_cols]
                if row.notna().any():
                    not_nan.append(gid)
            chk("Flagged-missing basins have NaN HydroATLAS model_input columns", not not_nan,
                f"non-NaN found for: {not_nan[:5]}")

    # ---- 10. final matrix checksum ----
    if provenance_path.exists():
        with open(provenance_path, "r", encoding="utf-8") as f:
            prov = json.load(f)
        recorded = prov.get("matrix_sha256")
        actual = _sha256(matrix_path)
        chk("Matrix checksum matches provenance record", recorded == actual,
            f"recorded={recorded}, actual={actual}")
    else:
        warn("No provenance file -- cannot verify matrix checksum")

    _finish(matrix_dir, args.matrix_name)


def _finish(matrix_dir: Path, matrix_name: str) -> None:
    passed = len(_issues) == 0
    lines = [
        f"# Stage 1 Static Attribute Matrix Audit Summary",
        "",
        f"- Script: {SCRIPT_NAME}",
        f"- Created (UTC): {CREATED_UTC}",
        f"- Matrix dir: {matrix_dir}",
        f"- Matrix name: {matrix_name}",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        f"- Errors: {len(_issues)}, Warnings: {len(_warns)}, OK checks: {len(_oks)}",
        "",
        "## Errors",
    ]
    lines += [f"- {m}" for m in _issues] if _issues else ["- (none)"]
    lines += ["", "## Warnings"]
    lines += [f"- {m}" for m in _warns] if _warns else ["- (none)"]
    lines += ["", "## OK"]
    lines += [f"- {m}" for m in _oks] if _oks else ["- (none)"]

    print("\n".join(lines))

    if matrix_dir.exists():
        summary_path = matrix_dir / f"{matrix_name}_audit_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote audit summary -> {summary_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
