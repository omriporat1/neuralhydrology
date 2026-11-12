"""Tests for the additive --wy-chunks option of the USGS IV recovery script.

Network-free: only the chunk-table helpers and the shard launcher's command
construction (via --dry-run plumbing is not exercised; we check the argparse
surface and pure helpers).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd
import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rec = _load("recover_usgs_iv_full_period_hourly")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_legacy_default_unchanged_for_legacy_period():
    start, end = _ts("2020-10-14T00:00:00"), _ts("2025-12-31T23:00:00")
    rec.check_chunk_table_covers_period(rec.WY_CHUNKS, start, end)
    assert rec.compute_active_chunks(start, end) == rec.compute_active_chunks(
        start, end, rec.WY_CHUNKS
    )
    labels = [c[0] for c in rec.compute_active_chunks(start, end)]
    assert labels == ["WY2021", "WY2022", "WY2023", "WY2024", "WY2025", "WY2026"]


def test_derived_matches_legacy_windows_inside_legacy_period():
    # Derived bounds give the same request windows as the legacy table, except
    # that the first chunk keeps the 15-min head buffer (the legacy table's
    # chunk bound clips it away when a period starts on a WY boundary).
    start, end = _ts("2021-10-01T00:00:00"), _ts("2025-12-31T23:00:00")
    legacy = rec.compute_active_chunks(start, end, rec.WY_CHUNKS)
    derived = rec.compute_active_chunks(start, end, rec.derive_wy_chunks(start, end))
    assert legacy[1:] == derived[1:]
    assert legacy[0] == ("WY2022", "2021-10-01T00:00:00Z", "2022-09-30T23:59:59Z")
    assert derived[0] == ("WY2022", "2021-09-30T23:45:00Z", "2022-09-30T23:59:59Z")


def test_legacy_table_rejects_wy2026_full_year():
    start, end = _ts("2025-10-01T00:00:00"), _ts("2026-09-30T23:00:00")
    with pytest.raises(SystemExit):
        rec.check_chunk_table_covers_period(rec.WY_CHUNKS, start, end)


def test_derived_wy2026_full_year_single_chunk():
    start, end = _ts("2025-10-01T00:00:00"), _ts("2026-09-30T23:00:00")
    table = rec.derive_wy_chunks(start, end)
    rec.check_chunk_table_covers_period(table, start, end)
    active = rec.compute_active_chunks(start, end, table)
    assert active == [("WY2026", "2025-09-30T23:45:00Z", "2026-09-30T23:15:00Z")]


def test_derived_jan_sep_2026_append_window():
    start, end = _ts("2026-01-01T00:00:00"), _ts("2026-09-30T23:00:00")
    table = rec.derive_wy_chunks(start, end)
    active = rec.compute_active_chunks(start, end, table)
    assert active == [("WY2026", "2025-12-31T23:45:00Z", "2026-09-30T23:15:00Z")]


def test_derived_spans_multiple_water_years():
    start, end = _ts("2025-09-15T00:00:00"), _ts("2026-10-05T00:00:00")
    labels = [c[0] for c in rec.derive_wy_chunks(start, end)]
    assert labels == ["WY2025", "WY2026", "WY2027"]


def test_recover_cli_default_is_legacy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--staids", "01234567"])
    assert rec.parse_args().wy_chunks == "legacy"
    monkeypatch.setattr(sys, "argv", ["x", "--staids", "01234567", "--wy-chunks", "derived"])
    assert rec.parse_args().wy_chunks == "derived"


_FULL = [("WY2026", "2025-09-30T23:45:00Z", "2026-09-30T23:15:00Z")]
_APPEND = [("WY2026", "2025-12-31T23:45:00Z", "2026-09-30T23:15:00Z")]


def test_window_lock_dry_run_checks_but_does_not_write(tmp_path):
    rec.enforce_derived_window_lock(tmp_path, _FULL, dry_run=True)
    assert not (tmp_path / rec.WINDOW_LOCK_NAME).exists()


def test_window_lock_written_then_same_window_accepted(tmp_path):
    rec.enforce_derived_window_lock(tmp_path, _FULL, dry_run=False)
    assert (tmp_path / rec.WINDOW_LOCK_NAME).exists()
    rec.enforce_derived_window_lock(tmp_path, _FULL, dry_run=False)  # resume: OK


def test_window_lock_rejects_different_window(tmp_path):
    rec.enforce_derived_window_lock(tmp_path, _APPEND, dry_run=False)
    with pytest.raises(SystemExit):
        rec.enforce_derived_window_lock(tmp_path, _FULL, dry_run=False)


@pytest.mark.parametrize("artifact", ["canonical/01234567_hourly.nc",
                                      "raw_cache/01234567/01234567_WY2026.parquet"])
def test_window_lock_rejects_unlocked_nonempty_out_dir(tmp_path, artifact):
    # e.g. the legacy full-period root: derived mode (with or without --force)
    # must never write into it.
    p = tmp_path / artifact
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    with pytest.raises(SystemExit):
        rec.enforce_derived_window_lock(tmp_path, _FULL, dry_run=True)


def test_launcher_cli_passthrough_default_legacy(monkeypatch):
    launch = _load("launch_usgs_iv_recovery_shards")
    monkeypatch.setattr(sys, "argv", ["x", "--out-root", "o"])
    assert launch.parse_args().wy_chunks == "legacy"
    monkeypatch.setattr(sys, "argv", ["x", "--out-root", "o", "--wy-chunks", "derived"])
    assert launch.parse_args().wy_chunks == "derived"
