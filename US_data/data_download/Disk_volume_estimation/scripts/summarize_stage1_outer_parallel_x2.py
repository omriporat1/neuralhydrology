#!/usr/bin/env python3
"""Outer-parallelism x2 benchmark summary for Flash-NH Stage 1.

Reads the manifests, hourly CSVs, and parent timing written by
bench_stage1_outer_parallel_x2_h2o.sh, then:
  - confirms each chunk passed all quality checks
  - computes per-chunk download/decode/extraction/write medians
  - writes per-chunk validation CSVs
  - computes effective full-period projection
  - writes summary_outer_x2.csv

Run on h2o after the benchmark script completes:

    /data42/omrip/Flash-NH/envs/flashnh-stage1/bin/python \\
        scripts/summarize_stage1_outer_parallel_x2.py \\
        2>&1 | tee /data42/omrip/Flash-NH/tmp/stage1_bench/\\
outer_parallel_rtma_48h_dw8_x2/logs/sanity_output.txt

Outputs (all under BENCH_BASE/):
  logs/summary_outer_x2.csv
  chunk_a/manifests/outer-x2-a_validation_checks.csv
  chunk_b/manifests/outer-x2-b_validation_checks.csv
"""

import csv
import json
import re
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCH_BASE  = Path("/data42/omrip/Flash-NH/tmp/stage1_bench/outer_parallel_rtma_48h_dw8_x2")
LOG_DIR     = BENCH_BASE / "logs"

EXPECTED_HOURS = 48
EXPECTED_ROWS  = 48 * 2752 * 11   # 1,453,056
FULL_HOURS     = 45720
N_CONCURRENT   = 2

CHUNKS = [
    ("outer-x2-a", BENCH_BASE / "chunk_a"),
    ("outer-x2-b", BENCH_BASE / "chunk_b"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median(rows: list[dict], col: str) -> float:
    vals = [float(r[col]) for r in rows if r.get(col, "").strip()]
    return statistics.median(vals) if vals else float("nan")


def _parse_timing() -> dict[str, int | None]:
    timing_path = LOG_DIR / "parent_timing.txt"
    if not timing_path.exists():
        print(f"WARNING: {timing_path} not found — benchmark may not have finished.")
        return {"parent_wall_seconds": None, "exit_a": None, "exit_b": None}
    txt = timing_path.read_text()
    def grab(pat):
        m = re.search(pat, txt)
        return int(m.group(1)) if m else None
    return {
        "parent_wall_seconds": grab(r"parent_wall_seconds=(\d+)"),
        "exit_a":              grab(r"exit_a=(\d+)"),
        "exit_b":              grab(r"exit_b=(\d+)"),
    }


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def _read_hourly_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"WARNING: hourly CSV not found: {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _write_validation_csv(path: Path, checks: list[tuple[str, str, str, str]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "actual", "expected", "result"])
        w.writerows(checks)


# ---------------------------------------------------------------------------
# Per-chunk analysis
# ---------------------------------------------------------------------------

def analyse_chunk(label: str, out_dir: Path) -> dict | None:
    mfst_path = out_dir / "manifests" / f"{label}_manifest.json"
    csv_path  = out_dir / "manifests" / f"{label}_hourly_runtime_and_volume.csv"
    vc_path   = out_dir / "manifests" / f"{label}_validation_checks.csv"

    try:
        mfst = _read_manifest(mfst_path)
    except FileNotFoundError as e:
        print(f"\n[{label}] MISSING manifest — chunk may have failed.\n  {e}")
        return None

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
    rows_match = (tot_r == EXPECTED_ROWS)

    print(f"""
=== {label} ===
  all_pass             : {all_pass}
  successful_hours     : {ok_h}/{EXPECTED_HOURS}  {"PASS" if ok_h == EXPECTED_HOURS else "FAIL"}
  actual_rows          : {tot_r}  (expected {EXPECTED_ROWS})  {"PASS" if rows_match else "FAIL"}
  chunk_wall_s         : {chunk_wall:.1f}
  download_median_s    : {dl_md:.3f}
  decode_median_s      : {dec_md:.3f}
  extraction_median_s  : {ext_md:.4f}
  write_median_s       : {wr_md:.5f}
  total_median_s       : {total_md:.3f}
  git_commit           : {git_commit[:12]}""")

    def _vc(key: str) -> tuple[str, str, str]:
        actual = val.get(key, "")
        return str(actual), "True", "PASS" if actual is True else "FAIL"

    checks: list[tuple[str, str, str, str]] = [
        ("all_pass",                       str(all_pass),    "True", "PASS" if all_pass else "FAIL"),
        ("successful_hours",               str(ok_h),        str(EXPECTED_HOURS), "PASS" if ok_h == EXPECTED_HOURS else "FAIL"),
        ("actual_rows",                    str(tot_r),       str(EXPECTED_ROWS),  "PASS" if rows_match else "FAIL"),
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
    print(f"  validation CSV       : {vc_path.name}  ({'all PASS' if all_vc else 'SOME FAIL — check CSV'})")

    return {
        "label":            label,
        "all_pass":         all_pass,
        "successful_hours": ok_h,
        "expected_hours":   EXPECTED_HOURS,
        "actual_rows":      tot_r,
        "expected_rows":    EXPECTED_ROWS,
        "rows_match":       rows_match,
        "chunk_wall_s":     chunk_wall,
        "dl_median_s":      round(dl_md,  3),
        "dec_median_s":     round(dec_md, 3),
        "ext_median_s":     round(ext_md, 4),
        "wr_median_s":      round(wr_md,  5),
        "total_median_s":   round(total_md, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    timing = _parse_timing()
    parent_wall = timing["parent_wall_seconds"]
    exit_a      = timing["exit_a"]
    exit_b      = timing["exit_b"]

    summary_rows: list[dict] = []
    for label, out_dir in CHUNKS:
        row = analyse_chunk(label, out_dir)
        if row is not None:
            summary_rows.append(row)

    # ---------- Effective projection ----------
    print(f"""
=== CONCURRENT PAIR ===
  exit_a               : {exit_a}
  exit_b               : {exit_b}
  parent_wall_seconds  : {parent_wall}""")

    proj: float | None = None
    flag = "UNKNOWN"
    if parent_wall is not None:
        proj = FULL_HOURS * parent_wall / (N_CONCURRENT * EXPECTED_HOURS) / 86400
        if proj <= 4.0:
            flag = "GREEN — recommend x3 benchmark"
        elif proj <= 6.0:
            flag = "YELLOW — partial scaling, discuss"
        else:
            flag = "RED — do not increase concurrency"
        print(f"  proj_effective_days  : {proj:.3f}")
        print(f"  flag                 : {flag}")
        print(f"  formula              : {FULL_HOURS} x {parent_wall} / ({N_CONCURRENT}x{EXPECTED_HOURS}) / 86400")
    else:
        print("  proj_effective_days  : UNKNOWN (parent_wall not found)")

    # ---------- Write summary CSV ----------
    if summary_rows:
        summary_rows[0]["parent_wall_s"]        = parent_wall
        summary_rows[0]["exit_a"]               = exit_a
        summary_rows[0]["exit_b"]               = exit_b
        summary_rows[0]["proj_effective_days"]  = round(proj, 3) if proj is not None else ""
        summary_rows[0]["flag"]                 = flag

        cols = list(summary_rows[0].keys())
        out_csv = LOG_DIR / "summary_outer_x2.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row.get(k, "") for k in cols})
        print(f"\nSummary CSV : {out_csv}")

    # ---------- Exit code ----------
    both_pass = (
        len(summary_rows) == 2
        and all(r["all_pass"] for r in summary_rows)
        and all(r["rows_match"] for r in summary_rows)
        and all(r["successful_hours"] == EXPECTED_HOURS for r in summary_rows)
    )
    if not both_pass:
        print("\nRESULT: FAIL — one or both chunks did not pass all checks.")
        return 1
    print("\nRESULT: PASS — both chunks complete and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
