"""Shared manifest and provenance utilities for Flash-NH pipeline runs.

Every significant pipeline run should call write_run_manifest() to produce:
  - run_command.txt   — the exact command that was executed
  - git_commit.txt    — current HEAD SHA (or "unknown")
  - config_snapshot.yaml — frozen copy of the effective config
  - manifest.json     — full provenance payload (machine-readable)
  - summary.json      — compact pass/fail summary (machine-readable)
  - summary.md        — human-readable run summary

Usage:
    from src.pipeline.provenance import write_run_manifest
    manifest = write_run_manifest(
        run_dir=output_path / "09_manifests" / "my_run",
        run_command="python scripts/my_script.py ...",
        config_dict=config_to_dict(cfg),
        input_paths={"basin_status": str(status_file)},
        output_paths={"pilot_manifest": str(out_csv)},
        validation_results={"manifest_written": True, "n_basins": 50},
    )
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


def git_commit_hash() -> Optional[str]:
    """Return the current HEAD commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def write_run_manifest(
    run_dir: Path,
    *,
    run_command: Optional[str] = None,
    config_dict: Optional[dict[str, Any]] = None,
    input_paths: Optional[dict[str, Any]] = None,
    output_paths: Optional[dict[str, Any]] = None,
    validation_results: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write standard provenance files to run_dir and return the manifest dict.

    Args:
        run_dir: Directory to write provenance files into (created if absent).
        run_command: The command string executed; defaults to sys.argv joined.
        config_dict: Serializable config snapshot.
        input_paths: Named input paths (for reproducibility tracing).
        output_paths: Named output paths produced by this run.
        validation_results: Dict of check_name -> bool pass/fail results.
        extra: Additional key/value pairs merged into manifest.json.
        dry_run: If True, skip writing files and return the manifest dict only.

    Returns:
        The manifest dict (same content as manifest.json).
    """
    run_dir = Path(run_dir)
    ts = datetime.now(timezone.utc).isoformat()
    git_hash = git_commit_hash()
    cmd = run_command or " ".join(sys.argv)

    manifest: dict[str, Any] = {
        "run_timestamp_utc": ts,
        "git_commit": git_hash,
        "run_command": cmd,
        "dry_run": dry_run,
        "config": config_dict or {},
        "inputs": input_paths or {},
        "outputs": output_paths or {},
        "validation": validation_results or {},
    }
    if extra:
        manifest.update(extra)

    if dry_run:
        return manifest

    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_command.txt").write_text(cmd + "\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text((git_hash or "unknown") + "\n", encoding="utf-8")

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    if config_dict:
        with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(config_dict, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    summary_compact: dict[str, Any] = {
        "run_timestamp_utc": ts,
        "git_commit": git_hash,
        "dry_run": dry_run,
        "validation": validation_results or {},
        "overall": "PASS" if _all_pass(validation_results) else "FAIL",
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_compact, fh, indent=2, default=str)

    _write_summary_md(run_dir, manifest)

    return manifest


def _all_pass(validation: Optional[dict[str, Any]]) -> bool:
    if not validation:
        return True
    return all(bool(v) for v in validation.values() if isinstance(v, bool))


def _write_summary_md(run_dir: Path, manifest: dict[str, Any]) -> None:
    validation = manifest.get("validation") or {}
    overall = "PASS" if _all_pass(validation) else "FAIL"

    lines = [
        "# Run Summary",
        "",
        f"**Overall:** {overall}",
        f"**Timestamp (UTC):** {manifest['run_timestamp_utc']}",
        f"**Git commit:** `{manifest.get('git_commit') or 'unknown'}`",
        f"**Dry run:** {manifest.get('dry_run', False)}",
        "",
        "## Run Command",
        "",
        "```",
        manifest.get("run_command", ""),
        "```",
        "",
    ]

    if validation:
        lines += ["## Validation", ""]
        for k, v in validation.items():
            if isinstance(v, bool):
                icon = "PASS" if v else "FAIL"
                lines.append(f"- **{k}**: {icon}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    outputs = manifest.get("outputs") or {}
    if outputs:
        lines += ["## Outputs", ""]
        for k, v in outputs.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    inputs = manifest.get("inputs") or {}
    if inputs:
        lines += ["## Inputs", ""]
        for k, v in inputs.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
