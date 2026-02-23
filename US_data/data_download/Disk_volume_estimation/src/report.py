from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.estimate import EstimateResult


def humanize_bytes(value: int | None) -> str:
	if value is None:
		return "n/a"
	units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
	size = float(value)
	for unit in units:
		if size < 1024:
			return f"{size:,.2f} {unit}"
		size /= 1024
	return f"{size:,.2f} EiB"


def write_csv(results: Iterable[EstimateResult], report_dir: Path) -> Path:
	report_dir.mkdir(parents=True, exist_ok=True)
	rows = [asdict(result) for result in results]
	path = report_dir / "disk_volume_estimate.csv"
	pd.DataFrame(rows).to_csv(path, index=False)
	return path


def write_json(results: Iterable[EstimateResult], report_dir: Path, metadata: dict) -> Path:
	report_dir.mkdir(parents=True, exist_ok=True)
	payload = {
		"generated_at": datetime.utcnow().isoformat() + "Z",
		"metadata": metadata,
		"sources": [asdict(result) for result in results],
	}
	path = report_dir / "disk_volume_estimate.json"
	path.write_text(json.dumps(payload, indent=2))
	return path


def print_table(results: Iterable[EstimateResult]) -> None:
	console = Console()
	table = Table(title="Disk Volume Estimate")
	table.add_column("Source")
	table.add_column("RAW-HOT")
	table.add_column("DERIVED-HOT")
	table.add_column("RAW-COLD")
	table.add_column("PEAK-LOCAL")
	table.add_column("Notes")

	for result in results:
		table.add_row(
			result.source,
			humanize_bytes(result.raw_hot_bytes),
			humanize_bytes(result.derived_hot_bytes),
			humanize_bytes(result.raw_cold_bytes),
			humanize_bytes(result.peak_local_bytes),
			result.notes,
		)

	console.print(table)
