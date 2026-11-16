"""Tests for the additive period arguments of the curated forcing builder and auditor,
and for the overlap identity check (docs/wy2026_data_plan.md §3 C1/C2, §9 L5).

Network-free. The end-to-end tests build a tiny synthetic product (2 basins, a
2-month non-default period) from synthetic monthly chunk Parquets in tmp_path.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bld = _load("build_stage1_curated_forcing_fullperiod")
aud = _load("audit_stage1_curated_forcing_basin_parquets")
ovl = _load("check_curated_forcing_overlap_identity")

WY26 = [
    "--period-start", "2025-10-01T00:00:00Z", "--period-end", "2026-09-30T23:00:00Z",
    "--expected-hours", "8760", "--expected-mrms-gaps", "0", "--expected-rtma-gaps", "1",
    "--product-name", "stage1_basin_hourly_forcings_wy2026_v001",
]
BASE = ["--forcing-root", "unused", "--out-dir", "unused"]

# Synthetic product: 2025-11-01T00Z .. 2025-12-31T23Z (1,464 h), one RTMA gap.
SYN_START, SYN_END = "2025-11-01T00:00:00Z", "2025-12-31T23:00:00Z"
SYN_HOURS = 1464
SYN_GAP = "2025-12-15T05:00:00Z"
SYN_NAME = "synthetic_period_product"
SYN_STAIDS = ["01000001", "01000002"]
_RTMA_SRC = ["2t", "2d", "2sh", "sp", "10u", "10v", "tcc", "vis", "gust", "ceil"]


# ---------------------------------------------------------------------------
# Builder: period resolution
# ---------------------------------------------------------------------------

def test_builder_default_period_is_v001():
    p = bld._resolve_period(bld._parse_args(BASE))
    assert p["is_default"] is True
    assert p["months"] == bld._ALL_MONTHS and len(p["months"]) == 63
    assert p["expected_hours"] == 45_720
    assert (p["expected_mrms_gaps"], p["expected_rtma_gaps"]) == (136, 2)
    assert p["product_name"] == bld.PRODUCT_NAME
    total = sum(len(bld._month_index(m, p["start"], p["end"])) for m in p["months"])
    assert total == 45_720
    # The wrapper keeps the 2020-10 partial month.
    assert bld._stage1_month_index("2020-10")[0] == pd.Timestamp("2020-10-14", tz="UTC")


def test_builder_wy2026_period():
    p = bld._resolve_period(bld._parse_args(BASE + WY26))
    assert p["is_default"] is False
    assert p["months"][0] == "2025-10" and p["months"][-1] == "2026-09"
    assert len(p["months"]) == 12
    total = sum(len(bld._month_index(m, p["start"], p["end"])) for m in p["months"])
    assert total == 8760 == p["expected_hours"]


@pytest.mark.parametrize("argv, msg", [
    (["--expected-hours", "8760"], "only accepted together"),
    (["--product-name", "x"], "only accepted together"),
    (WY26[:4], "also requires"),
    ([a if a != "stage1_basin_hourly_forcings_wy2026_v001" else
      "stage1_basin_hourly_forcings_v001" for a in WY26], "reserved"),
    ([a if a != "8760" else "8759" for a in WY26], "does not match"),
    ([a if a != "2026-09-30T23:00:00Z" else "2026-09-30T23:30:00Z" for a in WY26],
     "whole UTC hour"),
    ([a if a != "2026-09-30T23:00:00Z" else "2025-09-30T23:00:00Z" for a in WY26],
     "not after"),
])
def test_builder_period_errors(argv, msg):
    with pytest.raises(ValueError, match=msg):
        bld._resolve_period(bld._parse_args(BASE + argv))


def test_builder_bad_period_exits_2(tmp_path):
    with pytest.raises(SystemExit) as e:
        bld.main(["--forcing-root", str(tmp_path), "--out-dir", str(tmp_path / "o"),
                  "--expected-hours", "8760"])
    assert e.value.code == 2


# ---------------------------------------------------------------------------
# Builder: period lock
# ---------------------------------------------------------------------------

def _wy26_period():
    return bld._resolve_period(bld._parse_args(BASE + WY26))


def test_period_lock_refuses_v001_like_dir(tmp_path):
    (tmp_path / "dataset_config.json").write_text("{}")
    with pytest.raises(ValueError, match="no period_lock.json"):
        bld._check_period_lock(tmp_path, _wy26_period(), write=True)
    assert not (tmp_path / "period_lock.json").exists()


def test_period_lock_refuses_nonempty_time_series(tmp_path):
    (tmp_path / "time_series").mkdir()
    (tmp_path / "time_series" / "x.parquet").write_bytes(b"")
    with pytest.raises(ValueError):
        bld._check_period_lock(tmp_path, _wy26_period(), write=True)


@pytest.mark.parametrize("make", [
    lambda d: (d / "01000001.parquet").write_bytes(b"x"),         # a v001 time_series dir
    lambda d: (d / "notes.txt").write_text("x"),                   # any unknown file
    lambda d: ((d / "sub").mkdir(), (d / "sub" / "f").write_text("x")),
])
def test_period_lock_refuses_any_occupied_dir(tmp_path, make):
    make(tmp_path)
    with pytest.raises(ValueError, match="no period_lock.json"):
        bld._check_period_lock(tmp_path, _wy26_period(), write=True)
    assert not (tmp_path / "period_lock.json").exists()


def test_period_lock_tolerates_build_log_and_empty_dirs(tmp_path):
    (tmp_path / "build.log").write_text("launcher log")
    (tmp_path / "time_series").mkdir()
    bld._check_period_lock(tmp_path, _wy26_period(), write=True)
    assert (tmp_path / "period_lock.json").exists()


def test_period_lock_written_then_resume_passes_and_mismatch_refused(tmp_path):
    p = _wy26_period()
    bld._check_period_lock(tmp_path, p, write=False)
    assert not (tmp_path / "period_lock.json").exists()      # dry-run writes nothing
    bld._check_period_lock(tmp_path, p, write=True)
    assert json.loads((tmp_path / "period_lock.json").read_text())["n_hours"] == 8760
    (tmp_path / "dataset_config.json").write_text("{}")
    bld._check_period_lock(tmp_path, p, write=True)           # resume, same period
    other = dict(p, product_name="other_name")
    with pytest.raises(ValueError, match="refusing to mix periods"):
        bld._check_period_lock(tmp_path, other, write=True)


def test_period_lock_noop_for_default(tmp_path):
    (tmp_path / "dataset_config.json").write_text("{}")
    bld._check_period_lock(tmp_path, bld._resolve_period(bld._parse_args(BASE)), write=True)
    assert not (tmp_path / "period_lock.json").exists()


def test_period_lock_refuses_default_build_into_locked_dir(tmp_path):
    bld._check_period_lock(tmp_path, _wy26_period(), write=True)
    with pytest.raises(ValueError, match="refusing a default"):
        bld._check_period_lock(tmp_path, bld._resolve_period(bld._parse_args(BASE)),
                               write=True)


# ---------------------------------------------------------------------------
# Synthetic end-to-end: builder -> auditor
# ---------------------------------------------------------------------------

def _write_chunks(root: pathlib.Path, months=("2025-11", "2025-12")) -> None:
    gap = pd.Timestamp(SYN_GAP)
    rng = np.random.default_rng(0)
    for month in months:
        idx = bld._month_index(month, pd.Timestamp(SYN_START), pd.Timestamp(SYN_END))
        rows = []
        for staid in SYN_STAIDS:
            for t in idx:
                rows.append((staid, bld._MRMS_PRODUCT_ID, "tp", t, float(rng.random())))
                if t == gap:
                    continue
                for v in _RTMA_SRC:
                    rows.append((staid, bld._RTMA_PRODUCT_ID, v, t, float(rng.random())))
        df = pd.DataFrame(rows, columns=["STAID", "product", "variable",
                                         "valid_time_utc", "weighted_mean"])
        out = root / "chunks" / month
        out.mkdir(parents=True)
        df.to_parquet(out / f"combined_{month}.parquet", index=False)


def _builder_argv(root, out):
    return ["--forcing-root", str(root), "--out-dir", str(out),
            "--staids", *SYN_STAIDS,
            "--period-start", SYN_START, "--period-end", SYN_END,
            "--expected-hours", str(SYN_HOURS), "--expected-mrms-gaps", "0",
            "--expected-rtma-gaps", "1", "--product-name", SYN_NAME]


def _auditor_argv(out, gap=SYN_GAP):
    return ["--product-dir", str(out), "--full-period",
            "--period-start", SYN_START, "--period-end", SYN_END,
            "--expected-rows", str(SYN_HOURS), "--expected-mrms-gaps", "0",
            "--expected-rtma-gaps", "1", "--known-rtma-gap-timestamps", gap,
            "--product-name", SYN_NAME, "--expected-basins", "2"]


@pytest.fixture(scope="module")
def built_product(tmp_path_factory):
    root = tmp_path_factory.mktemp("forcing_root")
    _write_chunks(root)
    out = root / "product"
    bld.main(_builder_argv(root, out))
    return root, out


def test_builder_writes_period_product(built_product):
    _, out = built_product
    cfg = json.loads((out / "dataset_config.json").read_text())
    assert cfg["product_name"] == SYN_NAME
    assert cfg["period_start_utc"] == "2025-11-01T00:00:00+00:00"
    assert cfg["period_end_utc"] == "2025-12-31T23:00:00+00:00"
    assert cfg["n_hours_expected"] == SYN_HOURS
    assert cfg["months_included"] == ["2025-11", "2025-12"]
    prov = json.loads((out / "run_provenance.json").read_text())
    assert prov["period_args"]["product_name"] == SYN_NAME
    assert json.loads((out / "period_lock.json").read_text())["n_hours"] == SYN_HOURS
    df = pd.read_parquet(out / "time_series" / f"{SYN_STAIDS[0]}.parquet")
    assert len(df) == SYN_HOURS
    assert int(df["rtma_gap"].sum()) == 1 and int(df["mrms_qpe_1h_mm_gap"].sum()) == 0


def test_auditor_passes_on_period_product(built_product):
    _, out = built_product
    with pytest.raises(SystemExit) as e:
        aud.main(_auditor_argv(out))
    assert e.value.code == 0
    # audit_summary.md is written in the locale encoding (pre-existing); compare bytes.
    summary = (out / "audit_summary.md").read_bytes()
    assert SYN_NAME.encode() in summary
    assert b"2025-11-01" in summary and b"2025-12-15 05:00" in summary.replace(b"T", b" ")


def test_auditor_wrong_known_gap_fails(built_product):
    # Same gap count, wrong location: gated for non-default periods.
    _, out = built_product
    with pytest.raises(SystemExit) as e:
        aud.main(_auditor_argv(out, gap="2025-12-15T06:00:00Z"))
    assert e.value.code == 1


def test_builder_refuses_other_period_into_same_dir(built_product):
    root, out = built_product
    argv = _builder_argv(root, out)
    argv[argv.index(SYN_NAME)] = "different_name"
    with pytest.raises(SystemExit) as e:
        bld.main(argv)
    assert e.value.code == 2


def test_builder_default_build_into_period_dir_refused(built_product):
    # A v001-mode run (no period args) must not relabel the period product as v001.
    root, out = built_product
    before = (out / "dataset_config.json").read_bytes()
    with pytest.raises(SystemExit) as e:
        bld.main(["--forcing-root", str(root), "--out-dir", str(out),
                  "--staids", *SYN_STAIDS])
    assert e.value.code == 2
    assert (out / "dataset_config.json").read_bytes() == before


def test_builder_missing_month_refused_then_full_build(tmp_path):
    root = tmp_path / "root"
    _write_chunks(root, months=("2025-11",))
    out = root / "product"
    with pytest.raises(SystemExit) as e:
        bld.main(_builder_argv(root, out))
    assert e.value.code == 1
    assert not (out / "time_series").exists() or not any((out / "time_series").iterdir())
    # Once the missing chunk exists, the same command builds the full period.
    _write_chunks(root, months=("2025-12",))
    bld.main(_builder_argv(root, out))
    df = pd.read_parquet(out / "time_series" / f"{SYN_STAIDS[0]}.parquet")
    assert len(df) == SYN_HOURS


def test_builder_resume_refuses_short_parquet(built_product, tmp_path):
    root, out = built_product
    copy = tmp_path / "product"
    shutil.copytree(out, copy)
    p = copy / "time_series" / f"{SYN_STAIDS[0]}.parquet"
    pd.read_parquet(p).iloc[:-24].to_parquet(p)
    with pytest.raises(SystemExit) as e:
        bld.main(_builder_argv(root, copy))
    assert e.value.code == 1


@pytest.mark.parametrize("config_edit", [
    lambda c: {},
    lambda c: dict(c, product_name="stage1_basin_hourly_forcings_v001"),
    lambda c: dict(c, period_start_utc="2025-10-01T00:00:00+00:00"),
    lambda c: dict(c, n_hours_expected=8760),
])
def test_auditor_rejects_wrong_dataset_config(built_product, tmp_path, config_edit):
    _, out = built_product
    copy = tmp_path / "product"
    shutil.copytree(out, copy)
    cfg_path = copy / "dataset_config.json"
    cfg_path.write_text(json.dumps(config_edit(json.loads(cfg_path.read_text()))))
    with pytest.raises(SystemExit) as e:
        aud.main(_auditor_argv(copy))
    assert e.value.code == 1


# ---------------------------------------------------------------------------
# Auditor: period resolution
# ---------------------------------------------------------------------------

def _aud_args(extra):
    return aud._parse_args(["--product-dir", "x", "--full-period"] + extra)


def test_auditor_default_period_is_v001():
    p = aud._resolve_full_period(_aud_args([]))
    assert p["is_default"] is True
    assert len(p["full_index"]) == 45_720
    assert (p["expected_mrms_gaps"], p["expected_rtma_gaps"]) == (136, 2)
    assert p["known_rtma_gaps"] == list(aud._KNOWN_RTMA_GAP_TIMESTAMPS)
    assert p["product_name"] == aud.PRODUCT_NAME


def test_auditor_wy2026_period():
    p = aud._resolve_full_period(_aud_args([
        "--period-start", "2025-10-01T00:00:00Z", "--period-end", "2026-09-30T23:00:00Z",
        "--expected-rows", "8760", "--expected-mrms-gaps", "0", "--expected-rtma-gaps", "1",
        "--known-rtma-gap-timestamps", "2026-09-23T19:00:00Z",
        "--product-name", "stage1_basin_hourly_forcings_wy2026_v001",
    ]))
    assert p["is_default"] is False and len(p["full_index"]) == 8760
    assert p["known_rtma_gaps"] == [pd.Timestamp("2026-09-23T19:00:00Z")]


_AUD_OK = ["--period-start", SYN_START, "--period-end", SYN_END,
           "--expected-rows", str(SYN_HOURS), "--expected-mrms-gaps", "0",
           "--expected-rtma-gaps", "1", "--known-rtma-gap-timestamps", SYN_GAP,
           "--product-name", SYN_NAME]


def _replace(argv, old, new):
    return [new if a == old else a for a in argv]


@pytest.mark.parametrize("argv, msg", [
    (_AUD_OK[:4], "also requires"),
    (_replace(_AUD_OK, str(SYN_HOURS), "1463"), "does not match"),
    (_replace(_AUD_OK, SYN_GAP, "2026-01-01T00:00:00Z"), "outside the period"),
    (_replace(_AUD_OK, SYN_GAP, "2025-12-15T05:30:00Z"), "off the hour"),
    (_replace(_AUD_OK, "1", "2"), "known RTMA gap hours but"),
])
def test_auditor_period_errors(argv, msg):
    with pytest.raises(ValueError, match=msg):
        aud._resolve_full_period(_aud_args(argv))


def test_auditor_period_args_need_full_period(tmp_path):
    with pytest.raises(SystemExit) as e:
        aud.main(["--product-dir", str(tmp_path), "--product-name", "x"])
    assert e.value.code == 2


def test_check_known_rtma_gaps():
    idx = pd.date_range("2025-12-15T00:00:00Z", periods=10, freq="h")
    df = pd.DataFrame({"valid_time_utc": idx, "rtma_gap": [False] * 10})
    df.loc[5, "rtma_gap"] = True
    assert aud._check_known_rtma_gaps("s", df, [idx[5]]) is True
    assert aud._check_known_rtma_gaps("s", df, [idx[6]]) is False
    assert aud._check_known_rtma_gaps("s", df, []) is False


# ---------------------------------------------------------------------------
# Overlap identity check
# ---------------------------------------------------------------------------

def _product(dirpath: pathlib.Path, frames: dict[str, pd.DataFrame]) -> pathlib.Path:
    ts = dirpath / "time_series"
    ts.mkdir(parents=True)
    for staid, df in frames.items():
        df.to_parquet(ts / f"{staid}.parquet")
    return dirpath


def _frame(start, periods, seed):
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC", name="valid_time_utc")
    vals = np.random.default_rng(seed).random(periods).astype("float32")
    return pd.DataFrame({"mrms_qpe_1h_mm": vals, "rtma_gap": np.zeros(periods, bool)},
                        index=idx)


def test_overlap_identity_pass_and_mismatch(tmp_path):
    ref_a = _frame("2025-09-01", 24 * 61, 1)     # Sep + Oct
    ref_b = _frame("2025-09-01", 24 * 61, 2)
    new_a = ref_a.loc["2025-10-01":].copy()       # Oct onwards, identical rows
    new_b = ref_b.loc["2025-10-01":].copy()
    ref = _product(tmp_path / "ref", {"A": ref_a, "B": ref_b})
    new = _product(tmp_path / "new", {"A": new_a, "B": new_b})
    win = ["--window-start", "2025-10-01T00:00:00Z", "--window-end", "2025-10-31T23:00:00Z"]
    assert ovl.main(["--new-dir", str(new), "--ref-dir", str(ref)] + win) == 0
    res = json.loads((new / "overlap_identity_check.json").read_text())
    assert res["pass"] is True and res["counts"] == {"identical": 2}

    new_b.iloc[3, 0] = np.float32(new_b.iloc[3, 0] + 1.0)
    new_b.to_parquet(new / "time_series" / "B.parquet")
    (ref / "time_series" / "A.parquet").unlink()
    assert ovl.main(["--new-dir", str(new), "--ref-dir", str(ref)] + win) == 1
    res = json.loads((new / "overlap_identity_check.json").read_text())
    assert res["counts"] == {"mismatch": 1, "missing_ref": 1}
    assert res["non_identical_staids"] == ["A", "B"]


def test_overlap_identity_input_errors(tmp_path):
    assert ovl.main(["--new-dir", str(tmp_path / "n"), "--ref-dir", str(tmp_path / "r"),
                     "--window-start", "2025-10-01", "--window-end", "2025-10-02"]) == 2
    assert ovl.main(["--new-dir", str(tmp_path), "--ref-dir", str(tmp_path),
                     "--window-start", "2025-10-02", "--window-end", "2025-10-01"]) == 2
