#!/usr/bin/env python
"""Minimal human visual sanity check for a Stage 1 split candidate (I-A4).

This is a small, single-purpose QC script, not a reusable visualization
framework. It answers one question: does the seeded split candidate show
any obvious geographic clustering, missing region, or severe area/aridity
imbalance between non-California development and spatial-holdout basins?
It does not judge PASS/FAIL itself -- that is a human review step.

Usage:
    python scripts/generate_stage1_baseline_split_qc.py \\
        --assignment-csv tmp/stage1_baseline_splits_v001_candidate/split_assignment.csv \\
        --attributes-parquet <path to stage1_static_attributes_v001.parquet> \\
        --out-dir tmp/stage1_baseline_splits_v001_qc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.baseline.staid import normalize_staid

REQUIRED_ASSIGNMENT_COLUMNS = {"STAID", "split_role", "STATE"}
REQUIRED_MATRIX_COLUMNS = {"LAT_GAGE", "LNG_GAGE", "STATE", "DRAIN_SQKM", "ari_ix_uav"}
EXPECTED_ROLES = {
    "development_train",
    "spatial_holdout_nonca",
    "california_finetune_train",
    "california_holdout",
}

DEV_ROLE = "development_train"
HOLDOUT_ROLE = "spatial_holdout_nonca"
CA_TRAIN_ROLE = "california_finetune_train"
CA_HOLDOUT_ROLE = "california_holdout"


class SplitQcError(ValueError):
    pass


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assignment-csv", required=True, help="Path to the candidate's split_assignment.csv")
    p.add_argument("--attributes-parquet", required=True, help="Path to stage1_static_attributes_v001.parquet")
    p.add_argument("--out-dir", required=True, help="Output directory for PNGs + summary (must not already exist unless --force)")
    p.add_argument("--force", action="store_true", help="Allow writing into an existing --out-dir")
    return p.parse_args(argv)


def _normalize_staid_series(series: pd.Series, source: str) -> pd.Series:
    try:
        return series.map(normalize_staid)
    except (TypeError, ValueError) as exc:
        raise SplitQcError(f"{source}: STAID normalization failed: {exc}") from exc


def load_assignment(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SplitQcError(f"assignment CSV not found: {path}")
    df = pd.read_csv(path, dtype=str)
    missing = REQUIRED_ASSIGNMENT_COLUMNS - set(df.columns)
    if missing:
        raise SplitQcError(f"assignment CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["STAID"] = _normalize_staid_series(df["STAID"], "assignment CSV")
    if df["STAID"].duplicated().any():
        dupes = df.loc[df["STAID"].duplicated(), "STAID"].tolist()
        raise SplitQcError(f"assignment CSV has duplicate STAID rows: {dupes}")
    present_roles = set(df["split_role"].unique())
    missing_roles = EXPECTED_ROLES - present_roles
    if missing_roles:
        raise SplitQcError(f"assignment CSV is missing expected split_role values: {sorted(missing_roles)}")
    return df


def load_matrix(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SplitQcError(f"attributes parquet not found: {path}")
    df = pd.read_parquet(path)
    if df.index.name == "gauge_id":
        df = df.reset_index()
    elif "gauge_id" not in df.columns:
        raise SplitQcError("attributes parquet has neither a 'gauge_id' index nor column")
    missing = REQUIRED_MATRIX_COLUMNS - set(df.columns)
    if missing:
        raise SplitQcError(f"attributes parquet missing required columns: {sorted(missing)}")
    if pd.api.types.is_numeric_dtype(df["gauge_id"]):
        raise SplitQcError("attributes parquet gauge_id column is numeric; STAIDs must be read as strings to preserve leading zeros")
    df = df.copy()
    df["gauge_id"] = _normalize_staid_series(df["gauge_id"], "attributes parquet")
    if df["gauge_id"].duplicated().any():
        raise SplitQcError("attributes parquet has duplicate gauge_id rows")
    return df.rename(columns={"gauge_id": "STAID"})


def build_joined_frame(assignment_df: pd.DataFrame, matrix_df: pd.DataFrame) -> pd.DataFrame:
    matrix_cols = matrix_df[["STAID", "LAT_GAGE", "LNG_GAGE", "DRAIN_SQKM", "ari_ix_uav"]]
    merged = assignment_df.merge(matrix_cols, on="STAID", how="left", validate="one_to_one", indicator=True)
    unmatched = merged.loc[merged["_merge"] != "both", "STAID"].tolist()
    if unmatched:
        raise SplitQcError(f"{len(unmatched)} assignment STAIDs did not join one-to-one with the matrix: {unmatched[:10]}")
    return merged.drop(columns=["_merge"])


def prepare_out_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists():
        if not force:
            raise SplitQcError(f"--out-dir already exists: {out_dir} (pass --force to overwrite into it)")
    else:
        out_dir.mkdir(parents=True)


def plot_conus_overview(joined: pd.DataFrame, out_path: Path) -> None:
    dev = joined[joined["split_role"] == DEV_ROLE]
    holdout = joined[joined["split_role"] == HOLDOUT_ROLE]
    ca = joined[joined["split_role"].isin([CA_TRAIN_ROLE, CA_HOLDOUT_ROLE])]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ca["LNG_GAGE"], ca["LAT_GAGE"], s=8, c="lightgray", alpha=0.6, label=f"California (n={len(ca)})")
    ax.scatter(dev["LNG_GAGE"], dev["LAT_GAGE"], s=8, c="tab:blue", alpha=0.6, label=f"{DEV_ROLE} (n={len(dev)})")
    ax.scatter(holdout["LNG_GAGE"], holdout["LAT_GAGE"], s=14, c="tab:red", alpha=0.8, label=f"{HOLDOUT_ROLE} (n={len(holdout)})")
    ax.set_xlabel("Longitude (LNG_GAGE)")
    ax.set_ylabel("Latitude (LAT_GAGE)")
    ax.set_title("CONUS overview: non-CA development vs. spatial holdout (California muted)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_california_overview(joined: pd.DataFrame, out_path: Path) -> None:
    train = joined[joined["split_role"] == CA_TRAIN_ROLE]
    holdout = joined[joined["split_role"] == CA_HOLDOUT_ROLE]

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.scatter(train["LNG_GAGE"], train["LAT_GAGE"], s=14, c="tab:blue", alpha=0.7, label=f"{CA_TRAIN_ROLE} (n={len(train)})")
    ax.scatter(holdout["LNG_GAGE"], holdout["LAT_GAGE"], s=24, c="tab:red", alpha=0.9, label=f"{CA_HOLDOUT_ROLE} (n={len(holdout)})")
    ax.set_xlabel("Longitude (LNG_GAGE)")
    ax.set_ylabel("Latitude (LAT_GAGE)")
    ax.set_title("California overview: fine-tune train vs. holdout")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _ecdf(values: pd.Series):
    sorted_vals = values.sort_values().to_numpy()
    n = len(sorted_vals)
    y = (pd.RangeIndex(1, n + 1)).to_numpy() / n
    return sorted_vals, y


def plot_drainage_area_comparison(joined: pd.DataFrame, out_path: Path) -> None:
    dev = joined.loc[joined["split_role"] == DEV_ROLE, "DRAIN_SQKM"].dropna()
    holdout = joined.loc[joined["split_role"] == HOLDOUT_ROLE, "DRAIN_SQKM"].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    x, y = _ecdf(dev)
    ax.plot(x, y, c="tab:blue", label=f"{DEV_ROLE} (n={len(dev)})")
    x, y = _ecdf(holdout)
    ax.plot(x, y, c="tab:red", label=f"{HOLDOUT_ROLE} (n={len(holdout)})")
    ax.set_xscale("log")
    ax.set_xlabel("DRAIN_SQKM (log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Non-CA drainage area: development vs. holdout")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_aridity_comparison(joined: pd.DataFrame, out_path: Path) -> tuple[int, int]:
    nonca = joined[joined["split_role"].isin([DEV_ROLE, HOLDOUT_ROLE])]
    missing_mask = nonca["ari_ix_uav"].isna()
    excluded_dev = int((missing_mask & (nonca["split_role"] == DEV_ROLE)).sum())
    excluded_holdout = int((missing_mask & (nonca["split_role"] == HOLDOUT_ROLE)).sum())

    dev = nonca.loc[(nonca["split_role"] == DEV_ROLE) & ~missing_mask, "ari_ix_uav"]
    holdout = nonca.loc[(nonca["split_role"] == HOLDOUT_ROLE) & ~missing_mask, "ari_ix_uav"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x, y = _ecdf(dev)
    ax.plot(x, y, c="tab:blue", label=f"{DEV_ROLE} (n={len(dev)})")
    x, y = _ecdf(holdout)
    ax.plot(x, y, c="tab:red", label=f"{HOLDOUT_ROLE} (n={len(holdout)})")
    ax.set_xlabel("ari_ix_uav")
    ax.set_ylabel("Empirical CDF")
    total_excluded = excluded_dev + excluded_holdout
    ax.set_title(f"Non-CA aridity: development vs. holdout ({total_excluded} missing-aridity basins excluded)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return excluded_dev, excluded_holdout


def write_summary(out_dir: Path, joined: pd.DataFrame, excluded_dev: int, excluded_holdout: int, filenames: dict) -> Path:
    role_counts = joined["split_role"].value_counts().to_dict()
    lines = [
        "# Stage 1 baseline split candidate -- visual QC summary (I-A4)",
        "",
        "This is a machine-generated visual aid only. It does not declare a",
        "PASS/FAIL verdict -- human review of the plots below is required.",
        "",
        "## Role counts plotted",
        "",
    ]
    for role in sorted(EXPECTED_ROLES):
        lines.append(f"- `{role}`: {role_counts.get(role, 0)}")
    lines += [
        "",
        "## Missing-aridity exclusion (aridity comparison plot only)",
        "",
        f"- excluded from `{DEV_ROLE}`: {excluded_dev}",
        f"- excluded from `{HOLDOUT_ROLE}`: {excluded_holdout}",
        f"- total excluded: {excluded_dev + excluded_holdout}",
        "- these basins are not imputed; they are omitted from this plot only.",
        "",
        "## Output files",
        "",
    ]
    for name in filenames.values():
        lines.append(f"- `{name}`")
    lines += [
        "",
        "## Human review questions",
        "",
        "- Is the non-California holdout broadly distributed across CONUS?",
        "- Is the California holdout reasonably distributed within California?",
        "- Do development and holdout area distributions broadly overlap?",
        "- Do development and holdout aridity distributions broadly overlap?",
        "- Is any visible difference severe enough to invalidate the split?",
        "",
    ]
    out_path = out_dir / "visual_qc_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        assignment_path = Path(args.assignment_csv)
        matrix_path = Path(args.attributes_parquet)
        out_dir = Path(args.out_dir)

        assignment_df = load_assignment(assignment_path)
        matrix_df = load_matrix(matrix_path)
        joined = build_joined_frame(assignment_df, matrix_df)
        prepare_out_dir(out_dir, args.force)

        filenames = {
            "conus": "01_conus_overview.png",
            "california": "02_california_overview.png",
            "drainage_area": "03_drainage_area_comparison.png",
            "aridity": "04_aridity_comparison.png",
        }

        plot_conus_overview(joined, out_dir / filenames["conus"])
        plot_california_overview(joined, out_dir / filenames["california"])
        plot_drainage_area_comparison(joined, out_dir / filenames["drainage_area"])
        excluded_dev, excluded_holdout = plot_aridity_comparison(joined, out_dir / filenames["aridity"])

        for name in filenames.values():
            produced = out_dir / name
            if not produced.is_file():
                raise SplitQcError(f"expected PNG was not produced: {produced}")

        summary_path = write_summary(out_dir, joined, excluded_dev, excluded_holdout, filenames)

    except SplitQcError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    print(f"wrote 4 PNGs + summary under: {out_dir}")
    for name in filenames.values():
        print(f"  {out_dir / name}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
