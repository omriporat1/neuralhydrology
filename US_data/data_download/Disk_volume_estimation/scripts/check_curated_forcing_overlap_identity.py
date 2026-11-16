#!/usr/bin/env python
"""Check that two curated forcing products agree exactly over their overlap window.

Used for the WY2026 curated product (docs/wy2026_data_plan.md §9 L5): its
2025-10-01..2025-12-31 rows are built from the same monthly chunks as the v001
product, so they must be bit-identical to the v001 rows for those hours. A
difference means the period arguments changed how a month is built.

For every basin Parquet in --new-dir/time_series, the rows inside
[--window-start, --window-end] are compared with the same rows of the basin's
--ref-dir/time_series Parquet using DataFrame.equals (same index, columns,
dtypes and values; NaN matches NaN). Only counts and STAIDs are printed or
written; no forcing values are reported.

Exit codes: 0 all compared basins identical; 1 any mismatch or missing
reference basin; 2 argument or input error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new-dir", required=True, help="Product dir under test.")
    p.add_argument("--ref-dir", required=True, help="Reference product dir (e.g. v001).")
    p.add_argument("--window-start", required=True, help="First overlap hour, UTC.")
    p.add_argument("--window-end", required=True, help="Last overlap hour, UTC.")
    p.add_argument("--out-json", default=None,
                   help="Write a counts-only result JSON here (default: "
                        "<new-dir>/overlap_identity_check.json).")
    return p.parse_args(argv)


def _utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def compare_basin(new_path: Path, ref_path: Path,
                  start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Return 'identical', 'mismatch', 'empty_window' or 'missing_ref'."""
    if not ref_path.exists():
        return "missing_ref"
    new = pd.read_parquet(new_path)
    ref = pd.read_parquet(ref_path)
    new_w = new.loc[(new.index >= start) & (new.index <= end)]
    ref_w = ref.loc[(ref.index >= start) & (ref.index <= end)]
    expected = len(pd.date_range(start, end, freq="h"))
    if len(new_w) != expected or len(ref_w) != expected:
        return "empty_window" if len(new_w) == 0 or len(ref_w) == 0 else "mismatch"
    return "identical" if new_w.equals(ref_w) else "mismatch"


def main(argv: list[str] | None = None) -> int:
    args    = _parse_args(argv)
    new_ts  = Path(args.new_dir) / "time_series"
    ref_ts  = Path(args.ref_dir) / "time_series"
    start   = _utc(args.window_start)
    end     = _utc(args.window_end)
    if end < start:
        print(f"ERROR: window end {end} is before start {start}", file=sys.stderr)
        return 2
    if not new_ts.is_dir() or not ref_ts.is_dir():
        print(f"ERROR: missing time_series dir under {args.new_dir} or {args.ref_dir}",
              file=sys.stderr)
        return 2

    files = sorted(new_ts.glob("*.parquet"))
    if not files:
        print(f"ERROR: no basin Parquets in {new_ts}", file=sys.stderr)
        return 2

    status: dict[str, list[str]] = {}
    for f in files:
        s = compare_basin(f, ref_ts / f.name, start, end)
        status.setdefault(s, []).append(f.stem)

    n_ok   = len(status.get("identical", []))
    result = {
        "new_dir":       str(args.new_dir),
        "ref_dir":       str(args.ref_dir),
        "window_start":  start.isoformat(),
        "window_end":    end.isoformat(),
        "window_hours":  len(pd.date_range(start, end, freq="h")),
        "n_compared":    len(files),
        "counts":        {k: len(v) for k, v in sorted(status.items())},
        "non_identical_staids": sorted(
            s for k, v in status.items() if k != "identical" for s in v
        ),
        "pass":          n_ok == len(files),
    }
    out_json = Path(args.out_json) if args.out_json else Path(args.new_dir) / "overlap_identity_check.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"Overlap identity {result['window_start']} .. {result['window_end']}: "
          f"{n_ok}/{len(files)} basins identical; counts={result['counts']}")
    if result["non_identical_staids"]:
        print(f"Non-identical STAIDs (first 20): {result['non_identical_staids'][:20]}")
    print(f"Wrote {out_json}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
