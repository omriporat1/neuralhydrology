"""Cheap, real (non-NH, non-torch) global sample-support audit for the
Sequence-Length-A candidate set (``seq_length`` in ``{12, 24, 48, 72}``).

Answers one narrow question before any real training is launched: does a
larger ``seq_length`` (more required history per sample) meaningfully shrink
the number of globally usable training/validation/test timeline positions,
once the real signed-off MRMS forcing-gap inventory (Milestone 2K-E,
Policy B hard-exclusion; see ``src.baseline.gap_mask_io``/``validity_mask``)
is accounted for? Uses exactly the already-committed, already-tested
``src.baseline.validity_mask.compute_validity_mask`` primitive against the
REAL local copy of the Milestone 2K-E gap inventory
(``tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/
fullperiod_missing_hour_products.csv``) -- no synthetic data.

Deliberately GLOBAL, not per-basin: the MRMS/RTMA gap inventory this module
consumes is shared across every basin (an archive-level gap, not a basin-
level one; see ``gap_mask_io`` module docstring), so
``compute_history_valid``'s window-validity result is identical for every
basin at a given ``seq_length`` -- there is no need to loop over 2,557
basins or touch any real per-basin time-series/NetCDF file to answer this
question. This intentionally does NOT account for basin-specific qobs
NaNs/short station records (a separate, heavier, per-basin concern -- see
the module docstring of ``validity_mask.py``); it characterizes only the
forcing-gap-driven, basin-independent floor on sample support.

CAVEAT (recorded in the printed report, not silently dropped): the local
gap-inventory CSV's ``valid_time_utc`` coverage ends 2024-11-15 -- it does
not include the 2025 test period or December 2024. Any forcing gap in that
uncovered tail is invisible to this audit; the printed report says so
explicitly via ``gap_inventory_coverage_end_utc`` and
``gap_inventory_covers_full_timeline``.

Prints one JSON report to stdout. Read-only: writes nothing, calls no NH/
torch/W&B code, submits no Slurm job.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPO_WORKDIR = Path(__file__).resolve().parent.parent
if str(_REPO_WORKDIR) not in sys.path:
    sys.path.insert(0, str(_REPO_WORKDIR))

from src.baseline.gap_mask_io import (
    load_missing_hour_products,
    select_gap_timestamps,
    validate_gap_timestamps_against_timeline,
)
from src.baseline.validity_mask import compute_validity_mask

_DEFAULT_GAP_INVENTORY_PATH = (
    _REPO_WORKDIR / "tmp" / "stage1_forcing_fullperiod_postrun_audit_20260624T060504Z"
    / "fullperiod_missing_hour_products.csv"
)

# Matches the committed six-run pilot policy's lead_hours and the
# Sequence-Length-A closure script's candidate seq_lengths (the full closed
# {12, 24, 48, 72} set).
_LEAD_HOURS = 6
_SEQ_LENGTHS = (12, 24, 48, 72)

# Matches the real generated Sequence-Length-A pilot config's temporal split
# dates (see config/stage1_lead06_pilot_v001.yaml / the generated
# config.yaml's train_start_date/validation_start_date/test_end_date).
_SPLITS = {
    "train": ("2020-10-14", "2023-12-31 23:00"),
    "validation": ("2024-01-01", "2024-12-31 23:00"),
    "test": ("2025-01-01", "2025-12-31 23:00"),
}


def audit(gap_inventory_path=_DEFAULT_GAP_INVENTORY_PATH) -> dict:
    timeline = pd.date_range(_SPLITS["train"][0], _SPLITS["test"][1], freq="h")

    df = load_missing_hour_products(gap_inventory_path)
    mrms_gap_timestamps = select_gap_timestamps(df)  # Policy B default: MRMS-only.
    bad_hour_mask = validate_gap_timestamps_against_timeline(
        mrms_gap_timestamps, timeline, on_out_of_range="ignore"
    )
    gap_inventory_coverage_end_utc = str(pd.to_datetime(df["valid_time_utc"]).max())

    per_seq_length = {}
    for seq_length in _SEQ_LENGTHS:
        result = compute_validity_mask(timeline, bad_hour_mask, seq_length, _LEAD_HOURS)
        valid_series = pd.Series(result.combined_valid, index=timeline)
        per_split = {}
        for name, (start, end) in _SPLITS.items():
            sub = valid_series.loc[start:end]
            per_split[name] = {
                "n_valid": int(sub.sum()),
                "n_total": int(len(sub)),
                "fraction_valid": float(sub.mean()),
            }
        per_seq_length[str(seq_length)] = {
            "n_timeline": result.n_timeline,
            "n_bad_hours": result.n_bad_hours,
            "n_history_valid": result.n_history_valid,
            "n_boundary_valid": result.n_boundary_valid,
            "n_combined_valid": result.n_combined_valid,
            "fraction_of_timeline_valid": result.n_combined_valid / result.n_timeline,
            "per_split": per_split,
        }

    return {
        "schema_name": "stage1_seq_length_a_sample_support_audit",
        "schema_version": 1,
        "lead_hours": _LEAD_HOURS,
        "seq_lengths_audited": list(_SEQ_LENGTHS),
        "timeline_start_utc": str(timeline[0]),
        "timeline_end_utc": str(timeline[-1]),
        "n_timeline_hours": len(timeline),
        "gap_inventory_path": str(gap_inventory_path),
        "gap_product": "mrms_qpe_1h_pass1 (Policy B default: MRMS-only hard-exclusion)",
        "n_mrms_gap_timestamps_total": len(mrms_gap_timestamps),
        "n_mrms_gap_timestamps_within_timeline": int(bad_hour_mask.sum()),
        "gap_inventory_coverage_end_utc": gap_inventory_coverage_end_utc,
        "gap_inventory_covers_full_timeline": gap_inventory_coverage_end_utc >= str(timeline[-1]),
        "per_seq_length": per_seq_length,
    }


def main() -> None:
    report = audit()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
