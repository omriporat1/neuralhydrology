#!/usr/bin/env python3
"""Export v001 basin STAID list from the target package manifest.

Reads the `basins` key from stage1_target_package_v001/manifest.json and
writes a one-column CSV (`STAID`) suitable for use as --basin-manifest in
extract_stage1_forcing_chunk.py and --basin-list in build_stage1_basin_weights.py.

Usage (on h2o):
    python scripts/export_v001_basin_list.py

    # With explicit paths:
    python scripts/export_v001_basin_list.py \\
        --manifest /data42/omrip/Flash-NH/tmp/stage1_target_package_v001/manifest.json \\
        --out      /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv

    # Overwrite existing output:
    python scripts/export_v001_basin_list.py --force

    # Check only — print count and first/last STAIDs without writing:
    python scripts/export_v001_basin_list.py --dry-run

Validation:
    - Checks that the manifest contains a "basins" list.
    - Validates expected count (default 2,752) — exits 1 on mismatch.
    - Zero-pads all STAIDs to 8 characters.
    - Verifies uniqueness (no duplicates).
    - Refuses to overwrite existing output unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = (
    "/data42/omrip/Flash-NH/tmp/stage1_target_package_v001/manifest.json"
)
DEFAULT_OUT = (
    "/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv"
)
DEFAULT_EXPECTED_COUNT = 2752


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export v001 basin STAID list from target package manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=(
            f"Path to stage1_target_package_v001/manifest.json "
            f"(default: {DEFAULT_MANIFEST})"
        ),
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=(
            f"Output CSV path (default: {DEFAULT_OUT}). "
            "Must not exist unless --force is given."
        ),
    )
    p.add_argument(
        "--expected-count",
        type=int,
        default=DEFAULT_EXPECTED_COUNT,
        dest="expected_count",
        help=f"Expected number of basins (default: {DEFAULT_EXPECTED_COUNT}). "
             "Script exits 1 if the manifest count does not match.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print validation results and first/last STAIDs without writing anything.",
    )
    return p.parse_args()


def _pad_staid(raw: str) -> str:
    return str(raw).strip().zfill(8)


def main() -> int:
    args = _parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)

    # -----------------------------------------------------------------------
    # Load manifest
    # -----------------------------------------------------------------------

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        print(
            "  Expected the stage1_target_package_v001 manifest on h2o at:\n"
            f"  {DEFAULT_MANIFEST}\n"
            "  If the package is at a different path, use --manifest <path>.",
            file=sys.stderr,
        )
        return 1

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:
        print(f"ERROR: Cannot parse manifest: {exc}", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------------
    # Extract basin list
    # -----------------------------------------------------------------------

    raw_basins = manifest.get("basins")
    if raw_basins is None:
        available = list(manifest.keys())
        print(
            f"ERROR: Manifest has no 'basins' key.\n"
            f"  Available keys: {available}",
            file=sys.stderr,
        )
        return 1

    if not isinstance(raw_basins, list):
        print(
            f"ERROR: 'basins' is not a list (got {type(raw_basins).__name__}).",
            file=sys.stderr,
        )
        return 1

    staids = [_pad_staid(s) for s in raw_basins]

    # -----------------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------------

    errors: list[str] = []

    # Count check
    n = len(staids)
    if n != args.expected_count:
        errors.append(
            f"Count mismatch: manifest has {n} basins, expected {args.expected_count}. "
            "Use --expected-count to override if this is intentional."
        )

    # Uniqueness check
    seen: set[str] = set()
    dupes: list[str] = []
    for s in staids:
        if s in seen:
            dupes.append(s)
        seen.add(s)
    if dupes:
        errors.append(f"Duplicate STAIDs ({len(dupes)}): {dupes[:10]}")

    # Format check: all digits, length >= 8.
    # Standard USGS site IDs are 8 digits; some non-standard IDs (coordinate-
    # based or Alaska/PR sites) can be 9-15 digits — these are valid and must
    # not be truncated or rejected. zfill(8) is a no-op for IDs longer than 8.
    bad_format = [s for s in staids if not s.isdigit() or len(s) < 8]
    if bad_format:
        errors.append(
            f"{len(bad_format)} STAIDs are non-numeric or shorter than 8 digits after padding: "
            f"{bad_format[:5]}"
        )
    # Informational: count non-standard-length IDs
    nonstandard = [s for s in staids if len(s) != 8]
    if nonstandard:
        print(
            f"Info: {len(nonstandard)} STAIDs have non-standard length "
            f"(not 8 digits) — treated as valid long-form IDs."
        )

    # -----------------------------------------------------------------------
    # Dry-run / report
    # -----------------------------------------------------------------------

    print(f"Manifest:        {manifest_path}")
    print(f"basins count:    {n}")
    print(f"expected count:  {args.expected_count}")
    print(f"unique STAIDs:   {len(seen)}")
    print(f"First 5:  {staids[:5]}")
    print(f"Last  5:  {staids[-5:]}")

    if errors:
        print("\nVALIDATION ERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("Validation: OK")

    if args.dry_run:
        print("\nDRY RUN — no file written.")
        return 0

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------

    if out_path.exists() and not args.force:
        print(
            f"\nERROR: Output already exists: {out_path}\n"
            "  Use --force to overwrite, or --dry-run to check without writing.",
            file=sys.stderr,
        )
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("STAID\n")
        for s in staids:
            fh.write(f"{s}\n")

    # Verify written file
    written_lines = out_path.read_text(encoding="utf-8").splitlines()
    n_rows = len(written_lines) - 1  # exclude header
    if n_rows != n:
        print(f"ERROR: Written {n_rows} rows but expected {n}", file=sys.stderr)
        return 1

    print(f"\nWritten: {out_path}")
    print(f"  Rows (excluding header): {n_rows}")
    print(f"  First STAID: {staids[0]}   Last STAID: {staids[-1]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
