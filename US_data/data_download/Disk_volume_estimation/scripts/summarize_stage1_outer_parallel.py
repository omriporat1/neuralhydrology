#!/usr/bin/env python3
"""General outer-parallelism benchmark summary for Flash-NH Stage 1.

Reads the manifests, hourly CSVs, and parent timing written by
bench_stage1_outer_parallel_h2o.sh, validates each chunk, computes
per-chunk timing medians, and projects the effective full-period
wall-clock duration.

Chunks are discovered automatically from subdirectories of the bench
output root that contain a manifests/ directory.

Usage:
    python scripts/summarize_stage1_outer_parallel.py <bench_output_root>

Example:
    /data42/omrip/Flash-NH/envs/flashnh-stage1/bin/python \\
        scripts/summarize_stage1_outer_parallel.py \\
        /data42/omrip/Flash-NH/tmp/stage1_bench/outer_parallel_rtma_48h_dw6_x3 \\
        2>&1 | tee /data42/omrip/Flash-NH/tmp/stage1_bench/\\
outer_parallel_rtma_48h_dw6_x3/logs/sanity_output.txt

Outputs written under <bench_output_root>/logs/:
    summary.csv

Outputs written under <bench_output_root>/<chunk_label>/manifests/:
    <label>_validation_checks.csv  (one per chunk)
"""

import csv
import json
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Validation constants (fixed for all Stage 1 RTMA 48-hour benchmarks)
# ---------------------------------------------------------------------------

EXPECTED_HOURS = 48
EXPECTED_ROWS  = 48 * 2752 * 11   # 1,453,056  (hours x basins x RTMA vars)
FULL_HOURS     = 45720             # total hours in the v001 forcing period

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_bench_base():
    if len(sys.argv) != 2:
        name = Path(sys.argv[0]).name
        sys.exit(
            f"Usage: {name} <bench_output_root>\n"
            f"  e.g. {name} /data42/omrip/Flash-NH/tmp/stage1_bench/"
            "outer_parallel_rtma_48h_dw6_x3"
        )
    p = Path(sys.argv[1])
    if not p.is_dir():
        sys.exit(f"ERROR: bench output root not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Timing file parsing
# ---------------------------------------------------------------------------


def parse_timing(log_dir):
    """Return dict of key->value from parent_timing.txt (all string values)."""
    timing_path = log_dir / "parent_timing.txt"
    if not timing_path.exists():
        print(f"WARNING: {timing_path} not found — benchmark may not be complete.")
        return {}
    result = {}
    for line in timing_path.read_text().splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


# ---------------------------------------------------------------------------
# Chunk discovery
# ---------------------------------------------------------------------------


def discover_chunks(bench_base):
    """Scan bench_base for subdirs containing manifests/; return sorted list of
    (chunk_label, chunk_dir) pairs. Skips the logs/ directory."""
    found = []
    for d in sorted(bench_base.iterdir()):
        if not d.is_dir() or d.name == "logs":
            continue
        mfst_dir = d / "manifests"
        if not mfst_dir.is_dir():
            continue
        for mfst in sorted(mfst_dir.glob("*_manifest.json")):
            label = mfst.name[: -len("_manifest.json")]
            found.append((label, d))
    return found


# ---------------------------------------------------------------------------
# Per-chunk helpers
# ---------------------------------------------------------------------------


def _median(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col, "").strip()]
    return statistics.median(vals) if vals else float("nan")


def _read_hourly_csv(path):
    if not path.exists():
        print(f"  WARNING: hourly CSV not found: {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _write_validation_csv(path, checks):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "actual", "expected", "result"])
        w.writerows(checks)


# ---------------------------------------------------------------------------
# Per-chunk analysis
# ---------------------------------------------------------------------------


def analyse_chunk(label, chunk_dir):
    """Read manifest + hourly CSV for one chunk. Returns a summary dict or None."""
    mfst_path = chunk_dir / "manifests" / f"{label}_manifest.json"
    csv_path  = chunk_dir / "manifests" / f"{label}_hourly_runtime_and_volume.csv"
    vc_path   = chunk_dir / "manifests" / f"{label}_validation_checks.csv"

    if not mfst_path.exists():
        print(f"\n[{label}] MISSING manifest — chunk may not have finished.")
        return None

    mfst       = json.loads(mfst_path.read_text())
    prod       = mfst["product_summary"]["rtma_conus_aws_2p5km"]
    ok_h       = prod["successful_hours"]
    tot_r      = prod["total_output_rows"]
    chunk_wall = mfst["wall_clock_seconds"]
    all_pass   = mfst["all_pass"]
    git_commit = mfst.get("git_commit", "?")
    val        = mfst.get("validation", {})

    hr_rows   = _read_hourly_csv(csv_path)
    dl_md     = _median(hr_rows, "download_time_s")
    dec_md    = _median(hr_rows, "decode_time_s")
    ext_md    = _median(hr_rows, "extraction_time_s")
    wr_md     = _median(hr_rows, "write_time_s")
    total_md  = _median(hr_rows, "total_processing_time_s")
    rows_ok   = tot_r == EXPECTED_ROWS
    hours_ok  = ok_h == EXPECTED_HOURS

    print(f"""
=== {label} ===
  all_pass             : {all_pass}
  successful_hours     : {ok_h}/{EXPECTED_HOURS}  {"PASS" if hours_ok else "FAIL"}
  actual_rows          : {tot_r}  (expected {EXPECTED_ROWS})  {"PASS" if rows_ok else "FAIL"}
  chunk_wall_s         : {chunk_wall:.1f}
  download_median_s    : {dl_md:.3f}
  decode_median_s      : {dec_md:.3f}
  extraction_median_s  : {ext_md:.4f}
  write_median_s       : {wr_md:.5f}
  total_median_s       : {total_md:.3f}
  git_commit           : {git_commit[:12]}""")

    def _vc(key):
        actual = val.get(key, "")
        return (str(actual), "True", "PASS" if actual is True else "FAIL")

    checks = [
        ("all_pass",                       str(all_pass),  "True",
         "PASS" if all_pass else "FAIL"),
        ("successful_hours",               str(ok_h),      str(EXPECTED_HOURS),
         "PASS" if hours_ok else "FAIL"),
        ("actual_rows",                    str(tot_r),     str(EXPECTED_ROWS),
         "PASS" if rows_ok else "FAIL"),
        ("rtma_extracted_hours_gt_zero",   *_vc("rtma_extracted_hours_gt_zero")),
        ("rtma_10wdir_absent",             *_vc("rtma_10wdir_absent")),
        ("rtma_orog_absent",               *_vc("rtma_orog_absent")),
        ("rtma_at_least_8_variables",      *_vc("rtma_at_least_8_variables")),
        ("rtma_no_all_null_weighted_mean", *_vc("rtma_no_all_null_weighted_mean")),
        ("rtma_parquet_written",           *_vc("rtma_parquet_written")),
        ("combined_parquet_written",       *_vc("combined_parquet_written")),
    ]
    _write_validation_csv(vc_path, checks)
    all_vc = all(c[3] == "PASS" for c in checks)
    print(f"  validation CSV       : {vc_path.name}  "
          f"({'all PASS' if all_vc else 'SOME FAIL — check CSV'})")

    return {
        "label":            label,
        "all_pass":         all_pass,
        "successful_hours": ok_h,
        "expected_hours":   EXPECTED_HOURS,
        "actual_rows":      tot_r,
        "expected_rows":    EXPECTED_ROWS,
        "rows_match":       rows_ok,
        "chunk_wall_s":     chunk_wall,
        "dl_median_s":      round(dl_md,   3),
        "dec_median_s":     round(dec_md,  3),
        "ext_median_s":     round(ext_md,  4),
        "wr_median_s":      round(wr_md,   5),
        "total_median_s":   round(total_md, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    bench_base = _resolve_bench_base()
    log_dir    = bench_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timing
    timing      = parse_timing(log_dir)
    parent_wall_str = timing.get("parent_wall_seconds", "")
    parent_wall = int(parent_wall_str) if parent_wall_str.isdigit() else None

    # Chunk exits keyed by label (keys like "exit_outer-x3-a")
    exits = {k[5:]: int(v) for k, v in timing.items() if k.startswith("exit_")}

    # Discover and analyse chunks
    chunks = discover_chunks(bench_base)
    if not chunks:
        sys.exit(f"ERROR: no chunk manifests found under {bench_base}")

    summary_rows = []
    for label, chunk_dir in chunks:
        row = analyse_chunk(label, chunk_dir)
        if row is not None:
            row["exit_code"] = exits.get(label, "?")
            summary_rows.append(row)

    n_chunks = len(summary_rows)

    # Effective projection
    print(f"""
=== CONCURRENT PAIR (N={n_chunks}) ===""")
    for label, _ in chunks:
        print(f"  exit_{label:<22}: {exits.get(label, '?')}")
    print(f"  parent_wall_seconds  : {parent_wall}")

    proj = None
    flag = "UNKNOWN"
    if parent_wall is not None and n_chunks > 0:
        proj = FULL_HOURS * parent_wall / (n_chunks * EXPECTED_HOURS) / 86400
        if proj <= 3.0:
            flag = "STRONG GREEN — throughput on target; consider launching full period"
        elif proj <= 4.0:
            flag = "USEFUL GREEN — within acceptable range; plan full-period launch"
        elif proj <= 6.0:
            flag = "YELLOW — partial scaling; discuss before proceeding"
        else:
            flag = "RED — scaling insufficient; do not increase concurrency"
        print(f"  proj_effective_days  : {proj:.3f}")
        print(f"  flag                 : {flag}")
        print(f"  formula              : {FULL_HOURS} x {parent_wall}"
              f" / ({n_chunks}x{EXPECTED_HOURS}) / 86400")
    else:
        print("  proj_effective_days  : UNKNOWN")

    # Write summary CSV
    if summary_rows:
        summary_rows[0]["parent_wall_s"]       = parent_wall
        summary_rows[0]["n_chunks"]            = n_chunks
        summary_rows[0]["proj_effective_days"] = round(proj, 3) if proj is not None else ""
        summary_rows[0]["flag"]                = flag

        cols = list(summary_rows[0].keys())
        out_csv = log_dir / "summary.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row.get(k, "") for k in cols})
        print(f"\nSummary CSV : {out_csv}")

    # Overall result
    both_pass = (
        n_chunks == len(chunks)
        and all(r["all_pass"] for r in summary_rows)
        and all(r["rows_match"] for r in summary_rows)
        and all(r["successful_hours"] == EXPECTED_HOURS for r in summary_rows)
    )
    print()
    if not both_pass:
        print("RESULT: FAIL — one or more chunks did not pass all checks.")
        return 1
    print("RESULT: PASS — all chunks complete and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
