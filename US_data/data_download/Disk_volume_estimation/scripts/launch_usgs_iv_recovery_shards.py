"""
Flash-NH Stage 1 — Sharded USGS IV Recovery Launcher
=====================================================

Splits a STAID manifest into shards and launches one recovery subprocess per
shard in parallel, each calling recover_usgs_iv_full_period_hourly.py with its
own --staids-file and --out-dir.

Shard assignment strategy: contiguous sequential blocks by row order.
STAIDs 0..k -> shard_00, k+1..2k -> shard_01, etc.  This keeps
geographically/numerically adjacent basins together, making partial re-runs
and per-shard debugging straightforward.  Block sizes differ by at most 1
(ceil/floor via divmod).

Each shard subprocess writes its own canonical NCs and per-station logs under
<out-root>/shard_XX/.  The launcher captures aggregate subprocess output to
<out-root>/shard_XX/logs/recovery.log via a drain thread (prevents pipe
deadlock regardless of output volume).

Usage
-----
  # Dry-run — no downloads, confirms shard manifests and commands
  python scripts/launch_usgs_iv_recovery_shards.py \\
      --out-root /data42/omrip/Flash-NH/tmp/stage1_full_2843 \\
      --n-shards 4 \\
      --dry-run

  # Full 2,843-basin run with 4 shards (recommended first run)
  python scripts/launch_usgs_iv_recovery_shards.py \\
      --out-root /data42/omrip/Flash-NH/tmp/stage1_full_2843 \\
      --n-shards 4 \\
      --force

  # Re-run only shard 02 after a partial failure (pass its shard manifest)
  python scripts/recover_usgs_iv_full_period_hourly.py \\
      --staids-file /data42/omrip/Flash-NH/tmp/stage1_full_2843/manifests/shard_02.csv \\
      --out-dir     /data42/omrip/Flash-NH/tmp/stage1_full_2843/shard_02 \\
      --force
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RECOVERY_SCRIPT = pathlib.Path(__file__).resolve().parent / "recover_usgs_iv_full_period_hourly.py"
DEFAULT_MANIFEST = REPO_ROOT / "config" / "stage1_initial_training_basin_manifest.csv"
DEFAULT_START = "2020-10-14T00:00:00Z"
DEFAULT_END   = "2025-12-31T23:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "UNKNOWN"


def load_manifest(path: pathlib.Path) -> list[str]:
    """
    Read STAID column from a CSV file.

    zfill(8) is applied to all IDs — this zero-pads IDs shorter than 8 chars
    and leaves longer IDs unchanged (zfill never truncates).
    Duplicate STAIDs trigger a warning; the first occurrence is kept.
    """
    if not path.exists():
        print(f"ERROR: --staids-file not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    if "STAID" not in df.columns:
        print(
            f"ERROR: manifest must contain a 'STAID' column; "
            f"found columns: {df.columns.tolist()}",
            file=sys.stderr,
        )
        sys.exit(1)
    raw_ids = [str(s).strip() for s in df["STAID"].dropna() if str(s).strip()]
    padded = [s.zfill(8) for s in raw_ids]

    seen: dict[str, bool] = {}
    dupes: list[str] = []
    for s in padded:
        if s in seen:
            dupes.append(s)
        else:
            seen[s] = True
    if dupes:
        preview = dupes[:5]
        suffix = "..." if len(dupes) > 5 else ""
        print(
            f"WARNING: {len(dupes)} duplicate STAID(s) in manifest; "
            f"de-duplicating (keeping first occurrence): {preview}{suffix}",
            file=sys.stderr,
        )
    staids = list(seen.keys())
    if not staids:
        print("ERROR: manifest resulted in an empty STAID list.", file=sys.stderr)
        sys.exit(1)
    return staids


def split_into_shards(staids: list[str], n_shards: int) -> list[list[str]]:
    """
    Split STAIDs into n_shards contiguous sequential blocks.
    Block sizes differ by at most 1 (divmod distribution).
    Empty shards (when n_shards > len(staids)) are omitted.
    """
    n = len(staids)
    base, rem = divmod(n, n_shards)
    shards: list[list[str]] = []
    idx = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        if size > 0:
            shards.append(staids[idx : idx + size])
            idx += size
    return shards


def drain_to_log(proc: subprocess.Popen, log_path: pathlib.Path) -> None:
    """Thread target: drain subprocess stdout+stderr into a shard log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Split a STAID manifest into shards and run parallel USGS IV "
            "recovery subprocesses for Flash-NH Stage 1."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--staids-file", type=pathlib.Path, default=DEFAULT_MANIFEST,
        metavar="PATH",
        help=(
            "CSV with a STAID column. "
            f"Default: config/{DEFAULT_MANIFEST.name}"
        ),
    )
    p.add_argument(
        "--out-root", type=pathlib.Path, required=True,
        metavar="PATH",
        help="Root output directory. Shard outputs go to <out-root>/shard_XX/.",
    )
    p.add_argument(
        "--n-shards", type=int, default=4,
        metavar="INT",
        help="Number of parallel shard subprocesses. Default: 4.",
    )
    p.add_argument(
        "--start", type=str, default=DEFAULT_START,
        help="Period start (ISO 8601 UTC). Default: %(default)s",
    )
    p.add_argument(
        "--end", type=str, default=DEFAULT_END,
        help="Period end (ISO 8601 UTC). Default: %(default)s",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Pass --force to each shard (overwrite existing canonical NCs).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Create shard manifests and run recovery dry-runs; no network calls.",
    )
    p.add_argument(
        "--python", type=str, default=sys.executable,
        metavar="PATH",
        help="Python interpreter for shard subprocesses. Default: current interpreter.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    launch_start = datetime.now(timezone.utc)
    generated_utc = launch_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    git_hash = git_commit_hash()

    staids = load_manifest(args.staids_file)
    n_total = len(staids)

    n_shards = min(args.n_shards, n_total)
    if n_shards != args.n_shards:
        print(
            f"NOTE: --n-shards {args.n_shards} exceeds STAID count {n_total}; "
            f"using {n_shards} shard(s).",
            file=sys.stderr,
        )

    shard_lists = split_into_shards(staids, n_shards)

    out_root = args.out_root
    manifests_dir = out_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFlash-NH Stage 1 — Sharded USGS IV Recovery Launcher")
    print("=" * 60)
    print(f"  Manifest:     {args.staids_file}")
    print(f"  Total STAIDs: {n_total:,}")
    print(f"  Shards:       {n_shards}")
    print(f"  Period:       {args.start}  ->  {args.end}")
    print(f"  Out root:     {out_root}")
    print(f"  Dry-run:      {args.dry_run}")
    print(f"  Force:        {args.force}")
    print(f"  Git:          {git_hash[:12]}")
    print("=" * 60)

    # Build shard records and write shard manifests
    shard_records: list[dict] = []
    for i, shard_staids in enumerate(shard_lists):
        shard_id = f"{i:02d}"
        shard_dir = out_root / f"shard_{shard_id}"
        shard_csv = manifests_dir / f"shard_{shard_id}.csv"
        shard_log = shard_dir / "logs" / "recovery.log"

        pd.DataFrame({"STAID": shard_staids}).to_csv(
            shard_csv, index=False, lineterminator="\n"
        )

        cmd: list[str] = [
            args.python,
            str(RECOVERY_SCRIPT),
            "--staids-file", str(shard_csv),
            "--out-dir",     str(shard_dir),
            "--start",       args.start,
            "--end",         args.end,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.force:
            cmd.append("--force")

        shard_records.append({
            "shard_id":     shard_id,
            "n_staids":     len(shard_staids),
            "shard_csv":    str(shard_csv),
            "shard_dir":    str(shard_dir),
            "shard_log":    str(shard_log),
            "cmd":          cmd,
            "cmd_str":      " ".join(shlex.quote(str(c)) for c in cmd),
            "pid":          None,
            "return_code":  None,
            "wall_clock_s": None,
            "_t_start":     None,
        })

        print(f"\n  shard_{shard_id}: {len(shard_staids):,} STAIDs")
        print(f"    manifest: {shard_csv}")
        print(f"    out-dir:  {shard_dir}")
        print(f"    log:      {shard_log}")

    print(f"\nLaunching {n_shards} shard(s) in parallel...\n")

    # Launch all subprocesses and start drain threads immediately
    procs: list[subprocess.Popen] = []
    drain_threads: list[threading.Thread] = []

    for rec in shard_records:
        proc = subprocess.Popen(
            rec["cmd"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        rec["pid"] = proc.pid
        rec["_t_start"] = time.time()
        procs.append(proc)

        t = threading.Thread(
            target=drain_to_log,
            args=(proc, pathlib.Path(rec["shard_log"])),
            daemon=True,
        )
        t.start()
        drain_threads.append(t)

        print(
            f"  [shard_{rec['shard_id']}] PID={proc.pid}  "
            f"{rec['n_staids']:,} STAIDs  log -> {rec['shard_log']}"
        )

    print()

    # Wait for each subprocess in launch order.
    # Drain threads running in parallel prevent pipe-buffer deadlock.
    for proc, rec in zip(procs, shard_records):
        rc = proc.wait()
        elapsed = round(time.time() - rec["_t_start"], 2)
        rec["return_code"] = rc
        rec["wall_clock_s"] = elapsed
        status = "PASS" if rc == 0 else f"FAIL (rc={rc})"
        print(
            f"  [shard_{rec['shard_id']}] {status} — {elapsed:.1f}s — "
            f"log: {rec['shard_log']}"
        )

    # Give drain threads up to 30 s to flush remaining buffered output
    for t in drain_threads:
        t.join(timeout=30)

    launch_end = datetime.now(timezone.utc)
    total_elapsed = (launch_end - launch_start).total_seconds()
    n_pass = sum(1 for rec in shard_records if rec["return_code"] == 0)
    n_fail = len(shard_records) - n_pass
    any_fail = n_fail > 0

    print(f"\n{'=' * 60}")
    print(f"Launcher complete — {total_elapsed:.1f}s total")
    print(f"  PASS: {n_pass}   FAIL: {n_fail}")
    for rec in shard_records:
        print(
            f"  shard_{rec['shard_id']}: rc={rec['return_code']}  "
            f"({rec['n_staids']:,} STAIDs, {rec['wall_clock_s']:.1f}s)"
        )
    print("=" * 60)

    # Write JSON summary
    summary = {
        "generated_utc":      generated_utc,
        "launch_end_utc":     launch_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_wall_clock_s": round(total_elapsed, 2),
        "git_commit":         git_hash,
        "staids_file":        str(args.staids_file),
        "n_total_staids":     n_total,
        "n_shards":           n_shards,
        "start":              args.start,
        "end":                args.end,
        "dry_run":            args.dry_run,
        "force":              args.force,
        "n_pass":             n_pass,
        "n_fail":             n_fail,
        "shards": [
            {
                "shard_id":     rec["shard_id"],
                "n_staids":     rec["n_staids"],
                "return_code":  rec["return_code"],
                "wall_clock_s": rec["wall_clock_s"],
                "pid":          rec["pid"],
                "cmd":          rec["cmd_str"],
                "shard_csv":    rec["shard_csv"],
                "shard_dir":    rec["shard_dir"],
                "shard_log":    rec["shard_log"],
            }
            for rec in shard_records
        ],
    }
    json_path = out_root / "launcher_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Write Markdown summary
    result_label = (
        "**PASS**" if not any_fail
        else f"**FAIL** — {n_fail} of {n_shards} shard(s) failed"
    )
    md_rows = "\n".join(
        f"| shard_{rec['shard_id']} | {rec['n_staids']:,} "
        f"| {rec['return_code']} "
        f"| {rec['wall_clock_s']:.1f} "
        f"| {rec['pid']} "
        f"| `{rec['shard_log']}` |"
        for rec in shard_records
    )
    md_content = f"""# Flash-NH Stage 1 Launcher Summary

| Item | Value |
|---|---|
| Generated UTC | {generated_utc} |
| Git commit | `{git_hash[:12]}` |
| Manifest | `{args.staids_file}` |
| Total STAIDs | {n_total:,} |
| Shards | {n_shards} |
| Period | `{args.start}` -> `{args.end}` |
| Dry-run | {args.dry_run} |
| Total wall clock | {total_elapsed:.1f} s |
| Result | {result_label} |

## Shard Detail

| Shard | STAIDs | Return code | Wall clock (s) | PID | Log |
|---|---|---|---|---|---|
{md_rows}
"""
    md_path = out_root / "launcher_summary.md"
    md_path.write_text(md_content, encoding="utf-8")

    print(f"\nSummary written:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
