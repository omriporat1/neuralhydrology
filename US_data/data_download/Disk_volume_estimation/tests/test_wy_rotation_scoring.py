"""Checkpoint-year scorer of the water-year rotation (``src/baseline/wy_rotation_scoring.py``).

Synthetic fixtures only: every run directory, sealing receipt, results pickle
and package NetCDF is written here by the test. No real rotation run,
prediction, results pickle or score is opened; no sealed scope is read.

The fixture is a cycle-1 run family on a five-basin population:

* ``01010101``, ``02020202`` -- distinct observations, flood-eligible;
* ``03030303`` -- tied observations (10 % of the hours at the maximum), so the
  D5(b) tie cap makes it flood-ineligible;
* ``04040404`` -- in the results, but no package NetCDF: the sealed scorer
  excludes it from the area derivation (spec G3);
* ``05050505`` -- missing from the results, explained by a zero admissible
  target count in the score-blind evidence (spec G1).

So 2 of 5 basins are eligible and the low-eligibility flag is set. The
expected NSE, Q98 threshold and Q98-normalised RMSE are recomputed here with
plain numpy from the synthetic arrays, independently of the scorer.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from src.baseline import nh_raw_space_evaluation as nre
from src.baseline import wy_rotation_runs as rot
from src.baseline import wy_rotation_scoring as sc
from src.baseline import wy_rotation_split as wy
from src.baseline import wy_rotation_tracking as trk
from src.baseline.wandb_tracking import TrackingError, TrackingRun

ROOT = Path(__file__).resolve().parents[1]
TARGET = sc.TARGET
POP = ["01010101", "02020202", "03030303", "04040404", "05050505"]
AREAS = {"01010101": 120.0, "02020202": 35.5, "03030303": 800.0}  # 04040404 has no NetCDF
MISSING = "05050505"
NO_NC = "04040404"
SCORED = sorted(AREAS)
N_HOURS = 3000
NAN_OBS_01 = (5, 17, 400)
LAMBDAS = (0.5, 2.0, 8.0)
EPOCHS = (1, 2, 3)
# (below-Q98 noise, flood-hour noise) per (lambda, epoch): the NSE and flood picks
# land on different candidates by construction.
SCHEDULE = {
    (0.0, 1): (0.05, 0.60), (0.0, 2): (0.30, 0.05), (0.0, 3): (0.20, 0.20),
    (0.5, 1): (0.25, 0.30), (0.5, 2): (0.10, 0.40), (0.5, 3): (0.15, 0.10),
    (2.0, 1): (0.04, 0.50), (2.0, 2): (0.20, 0.08), (2.0, 3): (0.12, 0.25),
    (8.0, 1): (0.30, 0.02), (8.0, 2): (0.20, 0.30), (8.0, 3): (0.35, 0.15),
}


# --------------------------------------------------------------------------- fixture builders

def _safe_read_run_config(run_dir):
    path = Path(run_dir) / "config.yml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


@pytest.fixture(scope="module", autouse=True)
def _no_ruamel():
    # The local venv has no ruamel.yaml (NH's reader); the run config is plain YAML.
    mp = pytest.MonkeyPatch()
    mp.setattr(nre, "_read_run_config", _safe_read_run_config)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def contract_path(tmp_path_factory) -> Path:
    return wy.write_split_contract(tmp_path_factory.mktemp("contract") / "split_contract_v001.json",
                                   wy.build_split_contract())


def _obs(basin: str) -> np.ndarray:
    """Observed mm/h at the results' issue dates (NaN = not admitted)."""
    rng = np.random.default_rng(int(basin[:2]))
    if basin == "03030303":
        return np.where(np.arange(N_HOURS) % 10 == 0, 1.0, 0.1)
    obs = rng.lognormal(0.0, 0.9, N_HOURS) * 0.1
    if basin == "01010101":
        obs[list(NAN_OBS_01)] = np.nan
    return obs


def _flood_region(obs: np.ndarray) -> np.ndarray:
    return obs >= np.nanquantile(obs, 0.98)


def _sim(basin: str, lam: float, epoch: int) -> np.ndarray:
    obs = _obs(basin)
    ka, kf = SCHEDULE[(lam, epoch)]
    rng = np.random.default_rng(1000 + int(basin[:2]))
    z_below, z_flood = rng.standard_normal(N_HOURS), rng.standard_normal(N_HOURS)
    flood = _flood_region(obs)
    sim = np.where(flood, obs * (1.0 + kf * z_flood), obs * (1.0 + ka * z_below))
    return np.where(np.isfinite(obs), sim, 0.3)  # NH emits a value at every issue hour


def _write_package(data_dir: Path, basins) -> None:
    """Package NetCDFs in cycle 1's TRAINING years (the sealed area derivation
    reads only there): target[t] = 3.6 * qobs[t + 6] / area."""
    ts = data_dir / "time_series"
    ts.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2022-12-01 00:00", periods=400, freq="h")
    for basin in basins:
        rng = np.random.default_rng(int(basin[:2]) + 50)
        q = 1.0 + 5.0 * rng.random(len(dates))
        target = np.full(len(dates), np.nan)
        target[:-6] = 3.6 * q[6:] / AREAS[basin]
        xr.Dataset({TARGET: (("date",), target), "qobs_m3s": (("date",), q)},
                   coords={"date": dates}).to_netcdf(ts / f"{basin}.nc")


def _dataset(dates, obs, sim) -> xr.Dataset:
    return xr.Dataset({f"{TARGET}_obs": (("date", "time_step"), obs.reshape(-1, 1)),
                       f"{TARGET}_sim": (("date", "time_step"), sim.reshape(-1, 1))},
                      coords={"date": pd.DatetimeIndex(dates), "time_step": [0]})


def _write_receipts(nh_run_dir: Path, sealing, n_basins: int) -> None:
    for period in ("train", "validation"):
        receipt = dict(sealing.receipt(), period=period)
        if period == "train":
            receipt.update(n_basins=n_basins, n_train_targets_masked_outside_period=0)
        path = nre.split_sealing_receipt_path(nh_run_dir, period)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")


def make_run(root: Path, contract_path: Path, *, name: str, lam: float = 0.0, epochs=EPOCHS, data_dir=None,
             results_basins=None, obs_edit=None, sim_edit=None, missing=(MISSING,), counts=None,
             status="COMPLETE", cycle=1, is_canary=False, config_epochs=None) -> Path:
    """One synthetic COMPLETE rotation run; returns its config dir (receipt +
    score-blind evidence). ``obs_edit``/``sim_edit`` are ``f(basin, epoch, arr)``.
    The run's budget is ``len(epochs)`` (receipt and config) unless
    ``config_epochs`` overrides the config's."""
    nh_run_dir = root / name / "runs" / "exp_1911_120000"
    nh_run_dir.mkdir(parents=True)
    if data_dir is None:
        data_dir = root / "package"
        if not data_dir.is_dir():
            _write_package(data_dir, AREAS)
    config = {"dataset": "flashnh", nre.SPLIT_CONTRACT_CONFIG_KEY: contract_path.as_posix(),
              nre.ROTATION_CYCLE_CONFIG_KEY: cycle, "validate_n_random_basins": wy.BASIN_POPULATION_COUNT,
              "validate_every": None, "data_dir": Path(data_dir).as_posix(),
              "epochs": len(epochs) if config_epochs is None else config_epochs}
    if lam:
        config["flashnh_magnitude_weight_lambda"] = lam
    (nh_run_dir / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    sealing = nre.read_run_rotation_sealing(nh_run_dir)
    basins = list(results_basins) if results_basins is not None else [b for b in POP if b not in missing]
    _write_receipts(nh_run_dir, sealing, len(basins))
    start = pd.to_datetime(sealing.nh_dates["validation_start_date"], format="%d/%m/%Y")
    dates = pd.date_range(start, periods=N_HOURS, freq="h")
    for epoch in epochs:
        results = {}
        for b in basins:
            obs = _obs(b) if b in AREAS or b == NO_NC else _obs("02020202")
            sim = _sim(b, lam, epoch) if b in AREAS or b == NO_NC else _sim("02020202", lam, epoch)
            if obs_edit:
                obs = obs_edit(b, epoch, obs.copy())
            if sim_edit:
                sim = sim_edit(b, epoch, sim.copy())
            results[b] = {"1h": {"xr": _dataset(dates, obs, sim)}}
        path = rot.results_pickle_path(nh_run_dir, epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(results, fh)
    config_dir = root / name / "config"
    config_dir.mkdir(parents=True)
    run_id = f"wyrot-c{cycle}-{name}"
    (config_dir / sc.RECEIPT_NAME).write_text(json.dumps(
        {"status": status, "run_id": run_id, "cycle": cycle, "epochs": len(epochs), "is_canary": is_canary,
         "nh_run_dir": str(nh_run_dir), "slurm_job_id": "900", "git_commit": "c0ffee1"}), encoding="utf-8")
    (config_dir / sc.EVIDENCE_NAME).write_text(json.dumps(
        {"passed": True, "missing_basins": list(missing),
         "missing_basin_admissible_target_counts": counts if counts is not None else {b: 0 for b in missing}}),
        encoding="utf-8")
    return config_dir


STD_FLOOD = ("01010101", "02020202")  # the standard fixture's flood-eligible basins


def agreeing_table(config_dir, out, flood=STD_FLOOD) -> str:
    """A published round-E table (one per cycle folder ``out``) for the run's
    checkpoint year that agrees with the facts of the standard fixture:
    flood-eligible ``flood``, 04 area-excluded, 05 explained-missing."""
    try:
        cycle_no = json.loads((Path(config_dir) / sc.RECEIPT_NAME).read_text(encoding="utf-8"))["cycle"]
    except (OSError, ValueError, KeyError, TypeError):
        cycle_no = 1
    water_year = {c.cycle: c.checkpoint_wy for c in wy.ROTATION_TABLE_V001}.get(cycle_no, 2022)
    rows = [{"STAID": b, "water_year": water_year, "eligible": b != MISSING, "flood_eligible": b in flood,
             "reasons": [] if b in flood else [sc.NOT_SCORED_MISSING if b == MISSING else sc.INELIGIBLE_TOO_FEW]}
            for b in POP]
    return sc.write_eligibility_table(_elig_rows(rows), Path(out).parent / f"elig_{Path(out).name}",
                                      provenance={"note": "synthetic"})["csv"]


def score(config_dir, out, budget=len(EPOCHS), flood=STD_FLOOD):
    """Score at a test campaign budget (the production default is 12), with an
    agreeing published round-E table (required, delta review D1)."""
    return sc.score_run_checkpoint_year(config_dir=config_dir, population=POP, cycle_output_dir=out,
                                        budget_epochs=budget, eligibility_csv=agreeing_table(config_dir, out, flood))


def spath(out, s) -> Path:
    """The written score file of run score ``s`` in cycle folder ``out``."""
    return Path(out) / s["run_id"] / sc.RUN_SCORES_NAME


def picks(cycle_no, scores, out, budget=len(EPOCHS)):
    return sc.make_arm_picks(cycle=cycle_no, run_scores_paths=[spath(out, s) for s in scores],
                             cycle_output_dir=out, budget_epochs=budget)


# --------------------------------------------------------------------------- independent expectations

def _nse(o, s):
    return 1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def expected_basin(basin, lam, epoch):
    obs, sim = _obs(basin), _sim(basin, lam, epoch)
    keep = np.isfinite(obs)
    o, s = obs[keep], sim[keep]
    thr = float(np.quantile(o, 0.98, method="linear"))
    flood = o >= thr
    return {"nse": _nse(o, s), "q98": thr, "n_flood": int(flood.sum()), "n_admitted": int(keep.sum()),
            "q98_nrmse": float(np.sqrt(np.mean((s[flood] - o[flood]) ** 2)) / thr)}


def expected_curve(lam, epoch):
    rows = {b: expected_basin(b, lam, epoch) for b in SCORED}
    eligible = [b for b in SCORED if rows[b]["n_flood"] >= 50 and rows[b]["n_flood"] / rows[b]["n_admitted"] <= 0.05]
    return {"median_nse": float(np.median([rows[b]["nse"] for b in SCORED])),
            "median_q98_nrmse": float(np.median([rows[b]["q98_nrmse"] for b in eligible])),
            "eligible": eligible}


def expected_pick(lams, rule):
    key = "median_nse" if rule == "nse" else "median_q98_nrmse"
    cands = [(expected_curve(lam, e)[key], e, lam) for lam in lams for e in EPOCHS]
    best = max(c[0] for c in cands) if rule == "nse" else min(c[0] for c in cands)
    return min((e, lam) for v, e, lam in cands if v == best)


# --------------------------------------------------------------------------- the scored cycle (shared)

@pytest.fixture(scope="module")
def cycle(tmp_path_factory, contract_path):
    root = tmp_path_factory.mktemp("cycle1")
    out = root / "scores"
    dirs = {"base": make_run(root, contract_path, name="base")}
    for lam in LAMBDAS:
        dirs[lam] = make_run(root, contract_path, name=f"magw-l{lam:g}", lam=lam)
    scores = {"base": score(dirs["base"], out)}
    facts_after_base = (out / sc.FACTS_NAME).read_bytes()
    for lam in LAMBDAS:
        scores[lam] = score(dirs[lam], out)
    return {"root": root, "out": out, "dirs": dirs, "scores": scores, "facts_after_base": facts_after_base}


# --------------------------------------------------------------------------- spec identity

def test_scorer_pins_the_frozen_spec():
    assert sc.verify_spec_sha256() == sc.SPEC_SHA256


def test_a_changed_spec_is_refused(tmp_path):
    spec = tmp_path / sc.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_bytes((ROOT / sc.SPEC_RELATIVE_PATH).read_bytes() + b"\n")
    with pytest.raises(sc.WyRotationScoringError, match="expected the frozen"):
        sc.verify_spec_sha256(tmp_path)
    with pytest.raises(sc.WyRotationScoringError, match="not found"):
        sc.verify_spec_sha256(tmp_path / "elsewhere")


def test_scorer_file_names_equal_the_runners():
    spec = importlib.util.spec_from_file_location("run_wy_rotation_cycle", ROOT / "scripts" / "run_wy_rotation_cycle.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert (runner.RECEIPT_NAME, runner.EVIDENCE_NAME) == (sc.RECEIPT_NAME, sc.EVIDENCE_NAME)
    assert runner._archive_name(runner.RECEIPT_NAME, "123") == sc.archived_receipt_name("123")
    assert runner._RESUMABLE_STATUSES == sc._RESUMABLE_STATUSES


# --------------------------------------------------------------------------- D5(b) eligibility

@pytest.mark.parametrize("n_admitted, q98, n_flood, reasons", [
    (3000, 1.0, 60, []),
    (1000, 1.0, 50, []),                                      # exactly 50 flood hours, exactly 5 %
    (3000, 1.0, 49, [sc.INELIGIBLE_TOO_FEW]),
    (1000, 1.0, 51, [sc.INELIGIBLE_TIE_CAP]),                 # 5.1 % > 5 % cap
    (3000, 0.0, 60, [sc.INELIGIBLE_Q98_NOT_POSITIVE]),
    (100, 0.0, 100, [sc.INELIGIBLE_Q98_NOT_POSITIVE, sc.INELIGIBLE_TIE_CAP]),
    (0, float("nan"), 0, [sc.INELIGIBLE_NO_ADMITTED]),        # filtered first (spec §2d)
])
def test_flood_eligibility_rules(n_admitted, q98, n_flood, reasons):
    assert sc.flood_eligibility(n_admitted=n_admitted, q98_threshold=q98, n_flood=n_flood) == reasons


def _series(basin, obs, sim=None):
    dates = pd.date_range("2021-10-01", periods=len(obs), freq="h").values.astype("datetime64[ns]")
    sim = obs if sim is None else sim
    return sc.SealedBasinSeries(basin_id=basin, freq="1h", area_km2=1.0, dates=dates, obs_m3s=obs, sim_m3s=sim,
                                n_total=len(obs), n_admitted=len(obs), n_sim_nonfinite_at_admitted=0,
                                sealed_metrics={})


def test_flood_facts_filter_an_empty_basin_before_the_quantile():
    empty = sc.SealedBasinSeries(basin_id="b0", freq="1h", area_km2=1.0, dates=np.array([], "datetime64[ns]"),
                                 obs_m3s=np.array([]), sim_m3s=np.array([]), n_total=10, n_admitted=0,
                                 n_sim_nonfinite_at_admitted=0, sealed_metrics={})
    good = _series("b1", np.linspace(1.0, 100.0, 3000))
    _, record = sc.build_flood_facts({"b0": empty, "b1": good}, population=["b0", "b1", "b2"],
                                     not_scored={"b2": sc.NOT_SCORED_AREA})
    assert record["basins"]["b0"]["reasons"] == [sc.INELIGIBLE_NO_ADMITTED]
    assert record["basins"]["b2"]["reasons"] == [sc.NOT_SCORED_AREA]
    assert record["eligible_basins"] == ["b1"] and record["n_eligible"] == 1
    assert record["low_eligibility_flag"] is True  # 1/3 < 0.5
    with pytest.raises(sc.WyRotationScoringError, match="exactly the development population"):
        sc.build_flood_facts({"b1": good}, population=["b1", "b9"], not_scored={})


# --------------------------------------------------------------------------- end to end on the synthetic cycle

def test_facts_come_from_the_baseline_observations(cycle):
    facts = json.loads((cycle["out"] / sc.FACTS_NAME).read_text(encoding="utf-8"))
    assert (cycle["out"] / sc.FACTS_NAME).read_bytes() == cycle["facts_after_base"]  # arm 2 never rewrote it
    assert facts["source_run_id"] == "wyrot-c1-base" and facts["source_epoch"] == 1 and facts["cycle"] == 1
    assert facts["spec_sha256"] == sc.SPEC_SHA256
    assert facts["missing_explained"] == [MISSING] and facts["area_excluded"] == [NO_NC]
    assert facts["n_population"] == 5 and facts["n_scored"] == 3 and facts["n_eligible"] == 2
    assert facts["eligible_basins"] == ["01010101", "02020202"]
    assert facts["low_eligibility_flag"] is True and facts["eligible_fraction"] == pytest.approx(0.4)
    b = facts["basins"]
    assert b["03030303"]["reasons"] == [sc.INELIGIBLE_TIE_CAP] and b["03030303"]["n_flood"] == 300
    assert b[NO_NC]["reasons"] == [sc.NOT_SCORED_AREA] and b[MISSING]["reasons"] == [sc.NOT_SCORED_MISSING]
    assert b["01010101"]["n_admitted"] == N_HOURS - len(NAN_OBS_01)
    for basin in SCORED:
        exp = expected_basin(basin, 0.0, 1)
        assert b[basin]["n_flood"] == exp["n_flood"]
        # threshold in m^3/s = mm/h threshold * area / 3.6 (the area is derived, not given)
        assert b[basin]["q98_threshold"] == pytest.approx(exp["q98"] * AREAS[basin] / 3.6, rel=1e-9)


def test_curves_match_an_independent_computation(cycle):
    for key, lam in [("base", 0.0)] + [(lam, lam) for lam in LAMBDAS]:
        s = cycle["scores"][key]
        assert s["arm"] == ("base" if lam == 0 else f"magw-l{lam:g}") and s["lambda"] == lam
        assert s["epochs"] == list(EPOCHS) and s["period"] == "validation"
        assert s["gates"]["G3_area_excluded"] == [NO_NC] and s["gates"]["G1_missing_explained"] == [MISSING]
        for e in EPOCHS:
            exp, got = expected_curve(lam, e), s["curves"][str(e)]
            assert got["median_nse"] == pytest.approx(exp["median_nse"], rel=1e-9, abs=1e-12)
            assert got["median_q98_nrmse"] == pytest.approx(exp["median_q98_nrmse"], rel=1e-9)
            assert (got["n_basins_scored"], got["n_nse_finite"], got["n_flood_scored"]) == (3, 3, 2)


def test_per_basin_files_hold_every_d1_metric(cycle):
    s = cycle["scores"]["base"]
    run_out = cycle["out"] / s["run_id"]
    for e in EPOCHS:
        meta = s["per_basin_files"][str(e)]
        path = run_out / meta["file"]
        assert sc.sha256_file(path) == meta["sha256"]
        rows = {r["basin_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["rows"]}
        assert sorted(rows) == SCORED
        for basin in SCORED:
            exp, row = expected_basin(basin, 0.0, e), rows[basin]
            assert row["nse"] == pytest.approx(exp["nse"], rel=1e-9)
            assert row["q98_nrmse"] == pytest.approx(exp["q98_nrmse"], rel=1e-9)
            for k in ("kge", "kge_r", "kge_alpha", "kge_beta", "pbias", "high_flow_nse", "q98_rel_volume_bias",
                      "peak_rel_error", "pbias_below_q98"):
                assert row[k] is not None, (basin, k)
        assert rows["03030303"]["flood_eligible"] is False and rows["01010101"]["flood_eligible"] is True


def test_results_pickles_are_unchanged_and_hashed(cycle):
    s = cycle["scores"]["base"]
    for e in EPOCHS:
        assert s["results_sha256"][str(e)] == sc.sha256_file(rot.results_pickle_path(Path(s["nh_run_dir"]), e))


def test_picks_follow_d3_d6_d13(cycle, tmp_path):
    out = cycle["out"]
    base = picks(1, [cycle["scores"]["base"]], out)
    magw = picks(1, [cycle["scores"][lam] for lam in LAMBDAS], out)
    for rec, lams in ((base, [0.0]), (magw, list(LAMBDAS))):
        assert rec["budget_epochs"] == len(EPOCHS)
        for rule in sc.PICK_RULES:
            p = rec["picks"][rule]
            assert (p["epoch"], p["lambda"]) == expected_pick(lams, rule), (rec["family"], rule)
            assert p["n_candidates"] == 3 * len(lams)
            assert p["budget_edge"] is (p["epoch"] in sc.budget_edge_epochs(len(EPOCHS)))
        assert rec["flood_low_eligibility_flag"] is True and rec["flood_n_eligible"] == 2
        for r in rec["runs"]:  # R4: each run's score file and its bytes' sha256
            f = out / r["run_id"] / sc.RUN_SCORES_NAME
            assert r["scores_file"] == f"{r['run_id']}/{sc.RUN_SCORES_NAME}"  # relative to the cycle folder
            assert r["scores_sha256"] == sc.sha256_file(f)
    assert base["picks"]["nse"]["epoch"] != base["picks"]["flood"]["epoch"]  # the fixture's design
    assert base["picks"]["nse"]["strength_edge"] is False and base["per_lambda_picks"] == {}
    assert sorted(magw["per_lambda_picks"]) == ["0.5", "2", "8"]
    for lam in LAMBDAS:
        for rule in sc.PICK_RULES:
            assert magw["per_lambda_picks"][f"{lam:g}"][rule]["epoch"] == expected_pick([lam], rule)[0]
    assert magw["picks"]["flood"]["lambda"] == 8.0 and magw["picks"]["flood"]["strength_edge"] is True
    assert {r["run_id"] for r in magw["runs"]} == {cycle["scores"][lam]["run_id"] for lam in LAMBDAS}
    # idempotent rerun; a different record for the same arm is refused
    assert picks(1, [cycle["scores"]["base"]], out) == base
    copied = _copy_cycle_out(cycle, tmp_path)
    f = spath(copied, cycle["scores"]["base"])
    tampered = json.loads(f.read_text(encoding="utf-8"))
    tampered["curves"]["3"]["median_nse"] = 0.99999
    f.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(sc.WyRotationScoringError, match="already exists with different content"):
        picks(1, [cycle["scores"]["base"]], copied)


def _copy_cycle_out(cycle, tmp_path) -> Path:
    """A copy of the cycle's facts, round-E reconciliation, base score file and
    base pick record (if any)."""
    import shutil
    dst = tmp_path / "out_copy"
    dst.mkdir()
    shutil.copy2(cycle["out"] / sc.FACTS_NAME, dst / sc.FACTS_NAME)
    shutil.copy2(cycle["out"] / sc.RECONCILIATION_NAME, dst / sc.RECONCILIATION_NAME)
    rid = cycle["scores"]["base"]["run_id"]
    (dst / rid).mkdir()
    shutil.copy2(spath(cycle["out"], cycle["scores"]["base"]), dst / rid / sc.RUN_SCORES_NAME)
    rec = cycle["out"] / sc.PICKS_TEMPLATE.format(family="base")
    if rec.exists():
        shutil.copy2(rec, dst / rec.name)
    return dst


def test_arm_picks_refuse_an_incomplete_or_mixed_arm(cycle, tmp_path):
    s, out = cycle["scores"], cycle["out"]
    with pytest.raises(sc.WyRotationScoringError, match="expected"):
        picks(1, [s[0.5], s[2.0]], out)
    with pytest.raises(sc.WyRotationScoringError, match="mix arm families"):
        picks(1, [s["base"], s[0.5]], out)
    with pytest.raises(sc.WyRotationScoringError, match="cycle/spec differ"):
        picks(2, [s["base"]], out)
    with pytest.raises(sc.WyRotationScoringError, match="given twice"):
        picks(1, [s["base"], s["base"]], out)
    copied = _copy_cycle_out(cycle, tmp_path)
    f = spath(copied, s["base"])
    other = json.loads(f.read_text(encoding="utf-8"))
    other["facts_sha256"] = "0" * 64
    f.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(sc.WyRotationScoringError, match="other Q98 facts"):
        picks(1, [s["base"]], copied)


def test_arm_picks_read_only_the_cycles_written_score_files(cycle, tmp_path):
    """R4: picks take score-file paths in the cycle folder, never in-memory dicts."""
    s, out = cycle["scores"], cycle["out"]
    for bad in ({"a": 1}, str(spath(out, s["base"])), spath(out, s["base"]), [], ()):
        with pytest.raises(sc.WyRotationScoringError, match="non-empty sequence of score-file paths"):
            sc.make_arm_picks(cycle=1, run_scores_paths=bad, cycle_output_dir=out, budget_epochs=3)
    with pytest.raises(sc.WyRotationScoringError, match="pass paths, not in-memory score dicts"):
        sc.make_arm_picks(cycle=1, run_scores_paths=[s["base"]], cycle_output_dir=out, budget_epochs=3)
    with pytest.raises(sc.WyRotationScoringError, match="not found"):
        sc.make_arm_picks(cycle=1, run_scores_paths=[tmp_path / "nope.json"], cycle_output_dir=out,
                          budget_epochs=3)
    stray = tmp_path / "elsewhere" / s["base"]["run_id"] / sc.RUN_SCORES_NAME  # a byte-identical copy
    stray.parent.mkdir(parents=True)
    stray.write_bytes(spath(out, s["base"]).read_bytes())
    with pytest.raises(sc.WyRotationScoringError, match="is not the score file"):
        sc.make_arm_picks(cycle=1, run_scores_paths=[stray], cycle_output_dir=out, budget_epochs=3)


@pytest.mark.parametrize("budget", [2, 4, 12])
def test_arm_picks_check_every_run_against_the_campaign_budget(cycle, budget):
    """R2: a run scored at another budget never enters a pick."""
    with pytest.raises(sc.WyRotationScoringError, match="are not the campaign budget"):
        picks(1, [cycle["scores"][lam] for lam in LAMBDAS], cycle["out"], budget=budget)


def test_a_run_is_scored_once(cycle):
    with pytest.raises(sc.WyRotationScoringError, match="a run is scored once"):
        score(cycle["dirs"]["base"], cycle["out"])


# --------------------------------------------------------------------------- order and gates

def test_arm2_before_the_baseline_is_refused(tmp_path, contract_path):
    d = make_run(tmp_path, contract_path, name="magw-l2", lam=2.0, epochs=(1,))
    with pytest.raises(sc.WyRotationScoringError, match="score the cycle's baseline run first"):
        score(d, tmp_path / "scores", budget=1)
    assert not (tmp_path / "scores").exists()


# --------------------------------------------------------------------------- Q1: resume chain

def with_resume_chain(config_dir: Path, commits=("aaa1111", "bbb2222"), defects=None) -> list[str]:
    """Make the run's receipt the last job of a resume chain: earlier FAILED
    jobs 101, 102, ... with ``commits``, archived as the runner does.
    ``defects`` maps a job id to the runner's ``stopped_as_defect`` text.
    Returns the chain's job ids, oldest first."""
    receipt_path = config_dir / sc.RECEIPT_NAME
    current = json.loads(receipt_path.read_text(encoding="utf-8"))
    link, jobs = {}, []
    for i, commit in enumerate(commits):
        job = str(101 + i)
        path = config_dir / sc.archived_receipt_name(job)
        marker = {"stopped_as_defect": defects[job]} if defects and job in defects else {}
        path.write_text(json.dumps({"status": "FAILED", "run_id": current["run_id"], "slurm_job_id": job,
                                    "git_commit": commit, **marker, **link}), encoding="utf-8")
        link = {"resumed_from_job": job, "previous_status": "FAILED", "previous_receipt_sha256": sc.sha256_file(path),
                "previous_git_commit": commit}
        jobs.append(job)
    current.update(slurm_job_id=str(101 + len(commits)), **link)
    receipt_path.write_text(json.dumps(current), encoding="utf-8")
    return jobs + [current["slurm_job_id"]]


def test_a_score_file_records_the_training_commit_of_every_job_in_the_resume_chain(tmp_path, contract_path):
    """Q1 (Omri, 2026-11-19): every job of the chain, in order, not only the last."""
    plain = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    s = score(plain, tmp_path / "scores", budget=1)
    assert s["training_commits"] == ["c0ffee1"] and [j["slurm_job_id"] for j in s["resume_chain"]] == ["900"]
    resumed = make_run(tmp_path / "r", contract_path, name="base", epochs=(1,))
    jobs = with_resume_chain(resumed)
    s = score(resumed, tmp_path / "r" / "scores", budget=1)
    assert s["training_commits"] == ["aaa1111", "bbb2222", "c0ffee1"]
    assert [j["slurm_job_id"] for j in s["resume_chain"]] == jobs == ["101", "102", "103"]
    assert [j["status"] for j in s["resume_chain"]] == ["FAILED", "FAILED", "COMPLETE"]
    assert s["resume_chain"][0]["receipt_sha256"] == sc.sha256_file(resumed / sc.archived_receipt_name("101"))
    assert s == sc.load_run_scores(spath(tmp_path / "r" / "scores", s))


@pytest.mark.parametrize("damage, match", [
    ("missing", "is missing"),
    ("tampered", "sha256 differs"),
    ("commit", "does not match the job, run, status and commit"),
    ("no_commit", "no recorded training commit"),
])
def test_a_broken_resume_chain_is_refused_before_any_score_file(tmp_path, contract_path, damage, match):
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    with_resume_chain(d)
    oldest = d / sc.archived_receipt_name("101")
    if damage == "missing":
        oldest.unlink()
    elif damage == "tampered":
        oldest.write_bytes(oldest.read_bytes() + b" ")
    else:  # rewrite job 102's citation of 101 consistently, so only the commit check can catch it
        rec = json.loads(oldest.read_text(encoding="utf-8"))
        rec["git_commit"] = "ddd4444" if damage == "commit" else None
        oldest.write_text(json.dumps(rec), encoding="utf-8")
        middle_path = d / sc.archived_receipt_name("102")
        middle = json.loads(middle_path.read_text(encoding="utf-8"))
        middle["previous_receipt_sha256"] = sc.sha256_file(oldest)
        if damage == "no_commit":
            middle["previous_git_commit"] = None
        middle_path.write_text(json.dumps(middle), encoding="utf-8")
        current = json.loads((d / sc.RECEIPT_NAME).read_text(encoding="utf-8"))
        current["previous_receipt_sha256"] = sc.sha256_file(middle_path)
        (d / sc.RECEIPT_NAME).write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(sc.WyRotationScoringError, match=match):
        score(d, tmp_path / "scores", budget=1)
    assert not list((tmp_path / "scores").rglob(sc.RUN_SCORES_NAME))


def test_a_defect_stop_in_the_resume_chain_shows_in_the_score_file(tmp_path, contract_path):
    """Delta review D5(a): each chain entry carries the job's stopped_as_defect marker."""
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    marker = ("PermissionError(13, 'Permission denied'): another job may own this run directory; inspect it "
              "(sacct/squeue, file owners) before any resume")
    with_resume_chain(d, defects={"102": marker})
    s = score(d, tmp_path / "scores", budget=1)
    assert [j["stopped_as_defect"] for j in s["resume_chain"]] == [None, marker, None]
    assert s == sc.load_run_scores(spath(tmp_path / "scores", s))
    sc.unlock_stage_b(config_dir=d, run_scores_path=spath(tmp_path / "scores", s), budget_epochs=1)
    plain = make_run(tmp_path / "p", contract_path, name="base", epochs=(1,))
    assert score(plain, tmp_path / "p" / "scores", budget=1)["resume_chain"][0]["stopped_as_defect"] is None


def test_stage_b_refuses_when_the_resume_chain_changed_after_scoring(tmp_path, contract_path):
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    with_resume_chain(d)
    s = score(d, tmp_path / "scores", budget=1)
    sc.unlock_stage_b(config_dir=d, run_scores_path=spath(tmp_path / "scores", s), budget_epochs=1)
    (d / sc.archived_receipt_name("101")).unlink()
    with pytest.raises(sc.WyRotationScoringError, match="is missing"):
        sc.unlock_stage_b(config_dir=d, run_scores_path=spath(tmp_path / "scores", s), budget_epochs=1)


def test_only_a_complete_run_is_scored(tmp_path, contract_path):
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,), status="FAILED")
    with pytest.raises(sc.WyRotationScoringError, match="only a COMPLETE run"):
        score(d, tmp_path / "scores", budget=1)


def test_a_canary_run_is_never_scored(tmp_path, contract_path):
    """R2: a canary receipt is refused, even at the canary's own epoch count."""
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,), is_canary=True)
    with pytest.raises(sc.WyRotationScoringError, match="is_canary=True"):
        score(d, tmp_path / "scores", budget=1)
    assert not (tmp_path / "scores").exists()


@pytest.mark.parametrize("budget", [1, 3])
def test_only_a_run_at_the_campaign_budget_is_scored(tmp_path, contract_path, budget):
    """R2: receipt epochs and the run's cycle config must both equal the budget."""
    short = make_run(tmp_path, contract_path, name="base", epochs=(1, 2))
    with pytest.raises(sc.WyRotationScoringError, match=f"a scored run must have the campaign budget of {budget}"):
        score(short, tmp_path / "scores", budget=budget)
    cfg = make_run(tmp_path, contract_path, name="base-cfg", epochs=tuple(range(1, budget + 1)), config_epochs=12)
    with pytest.raises(sc.WyRotationScoringError, match="cycle config has epochs=12"):
        score(cfg, tmp_path / "scores", budget=budget)
    assert not (tmp_path / "scores").exists()


@pytest.mark.parametrize("bad", [0, -3, True, 3.0, "12", None])
def test_an_invalid_budget_is_refused(tmp_path, bad):
    with pytest.raises(sc.WyRotationScoringError, match="budget_epochs must be a positive int"):
        sc.score_run_checkpoint_year(config_dir=tmp_path, population=POP, cycle_output_dir=tmp_path,
                                     budget_epochs=bad)


def test_the_budget_has_one_source():
    """R2: the scorer's default budget is the cycle config's epochs, not a second literal."""
    assert sc.DEFAULT_BUDGET_EPOCHS is rot.CAMPAIGN_EPOCHS and rot._BASE_EXPECTED["epochs"] == rot.CAMPAIGN_EPOCHS
    assert sc.BUDGET_EDGE_EPOCHS == sc.budget_edge_epochs(rot.CAMPAIGN_EPOCHS) == (11, 12)
    assert sc.budget_edge_epochs(3) == (2, 3) and sc.budget_edge_epochs(1) == (1,)


def test_g1_refuses_a_missing_basin_without_a_zero_count(tmp_path, contract_path):
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,), counts={MISSING: 4})
    with pytest.raises(sc.WyRotationScoringError, match="G1: missing basin"):
        score(d, tmp_path / "scores", budget=1)


def test_g1_refuses_results_outside_the_population(tmp_path, contract_path):
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,),
                 results_basins=[b for b in POP if b != MISSING] + ["06060606"])
    with pytest.raises(sc.WyRotationScoringError, match="G1 epoch 1: results keys"):
        score(d, tmp_path / "scores", budget=1)
    assert not (tmp_path / "scores").exists()


def test_g2_fails_after_the_facts_and_before_any_score_file(tmp_path, contract_path):
    def nan_sim(basin, epoch, sim):
        if basin == "02020202" and epoch == 2:
            sim[100] = np.nan  # an admitted hour (basin 02 has no NaN observation)
        return sim
    d = make_run(tmp_path, contract_path, name="base", epochs=(1, 2), sim_edit=nan_sim)
    out = tmp_path / "scores"
    with pytest.raises(sc.WyRotationScoringError, match="G2 epoch 2"):
        score(d, out, budget=2)
    # the facts were written from epoch 1 before any score; no score file exists
    assert sorted(p.name for p in out.iterdir()) == [sc.FACTS_NAME]
    # R3: arm 2 is never scored against facts whose baseline run failed a gate
    m = make_run(tmp_path, contract_path, name="magw-l2", lam=2.0, epochs=(1, 2))
    with pytest.raises(sc.WyRotationScoringError, match="which has no score file"):
        score(m, out, budget=2)
    assert sorted(p.name for p in out.iterdir()) == [sc.FACTS_NAME]


def test_facts_bind_to_the_cycles_baseline_run(tmp_path, contract_path):
    """R3: another baseline run cannot adopt the facts; arm 2 needs the baseline
    score file that cites this exact facts file and the same explained-missing set."""
    out = tmp_path / "scores"
    score(make_run(tmp_path, contract_path, name="base", epochs=(1,)), out, budget=1)
    other_base = make_run(tmp_path, contract_path, name="base-rerun", epochs=(1,))
    with pytest.raises(sc.WyRotationScoringError, match="cannot adopt it"):
        score(other_base, out, budget=1)
    all_basins = make_run(tmp_path, contract_path, name="magw-l0.5", lam=0.5, epochs=(1,), missing=(),
                          results_basins=POP)
    with pytest.raises(sc.WyRotationScoringError, match="G1: explained-missing basins differ"):
        score(all_basins, out, budget=1)
    facts = out / sc.FACTS_NAME
    facts.write_bytes(facts.read_bytes() + b"\n")  # same JSON, other bytes: not the file the baseline cites
    m = make_run(tmp_path, contract_path, name="magw-l2", lam=2.0, epochs=(1,))
    with pytest.raises(sc.WyRotationScoringError, match="baseline score file that cites"):
        score(m, out, budget=1)
    assert not (out / "wyrot-c1-magw-l2").exists() and not (out / "wyrot-c1-base-rerun").exists()


def test_g3_refuses_a_different_area_exclusion_across_arms(tmp_path, contract_path):
    out = tmp_path / "scores"
    score(make_run(tmp_path, contract_path, name="base", epochs=(1,)), out, budget=1)
    narrow = tmp_path / "package_without_02"
    _write_package(narrow, ["01010101", "03030303"])
    d = make_run(tmp_path, contract_path, name="magw-l2", lam=2.0, epochs=(1,), data_dir=narrow)
    with pytest.raises(sc.WyRotationScoringError, match="G3 epoch 1"):
        score(d, out, budget=1)
    assert not (out / "wyrot-c1-magw-l2").exists()


def test_g4_refuses_arm2_observations_that_differ_from_the_facts(tmp_path, contract_path):
    out = tmp_path / "scores"
    score(make_run(tmp_path, contract_path, name="base", epochs=(1,)), out, budget=1)

    def nudge(basin, epoch, obs):
        if basin == "02020202":
            obs[1234] = obs[1234] * (1 + 1e-12)
        return obs
    d = make_run(tmp_path, contract_path, name="magw-l2", lam=2.0, epochs=(1,), obs_edit=nudge)
    with pytest.raises(sc.WyRotationScoringError, match="G4 .*02020202"):
        score(d, out, budget=1)
    assert not (out / "wyrot-c1-magw-l2").exists()


def test_g4_within_a_run_refuses_observations_that_change_between_epochs(tmp_path, contract_path):
    def drift(basin, epoch, obs):
        if basin == "01010101" and epoch == 2:
            obs[50] = np.nan  # one admitted hour fewer in epoch 2
        return obs
    d = make_run(tmp_path, contract_path, name="base", epochs=(1, 2), obs_edit=drift)
    with pytest.raises(sc.WyRotationScoringError, match="G4 .*epoch 2"):
        score(d, tmp_path / "scores", budget=2)


def test_unsealed_run_is_refused(tmp_path):
    with pytest.raises(sc.WyRotationScoringError, match="not a split-contract run"):
        sc.sealed_period_series(run_dir=tmp_path, period="validation", epoch=1, package_root=tmp_path,
                                basin_ids=["01010101"])


# --------------------------------------------------------------------------- pick rule units (D3, D6, D7, round C)

def _c(epoch, lam, value):
    return {"epoch": epoch, "lambda": lam, "value": value}


def test_pick_ties_go_to_the_earliest_epoch_then_the_smallest_lambda():
    cands = [_c(5, 2.0, 0.7), _c(3, 8.0, 0.7), _c(3, 0.5, 0.7), _c(1, 0.5, 0.6)]
    p = sc.pick_checkpoint(cands, rule="nse")
    assert (p["epoch"], p["lambda"], p["n_tied"]) == (3, 0.5, 3)
    p = sc.pick_checkpoint(cands, rule="flood")
    assert (p["epoch"], p["lambda"], p["n_tied"]) == (1, 0.5, 1)


def test_non_finite_values_never_win():
    cands = [_c(1, 0.0, float("nan")), _c(2, 0.0, None), _c(3, 0.0, float("inf")), _c(4, 0.0, 0.2)]
    assert sc.pick_checkpoint(cands, rule="nse")["epoch"] == 4
    assert sc.pick_checkpoint([_c(1, 0.0, float("-inf")), _c(2, 0.0, 0.5)], rule="flood")["epoch"] == 2
    with pytest.raises(sc.WyRotationScoringError, match="no candidate has a finite value"):
        sc.pick_checkpoint([_c(1, 0.0, float("nan"))], rule="flood")
    with pytest.raises(sc.WyRotationScoringError, match="unknown pick rule"):
        sc.pick_checkpoint([_c(1, 0.0, 1.0)], rule="kge")


@pytest.mark.parametrize("epoch, lam, budget, strength", [
    (10, 2.0, False, False), (11, 2.0, True, False), (12, 0.5, True, True), (1, 8.0, False, True),
    (12, 0.0, True, False)])
def test_budget_and_strength_edge_flags(epoch, lam, budget, strength):
    p = sc.pick_checkpoint([_c(epoch, lam, 1.0)], rule="nse")
    assert (p["budget_edge"], p["strength_edge"]) == (budget, strength)


@pytest.mark.parametrize("budget, epoch, edge", [(3, 1, False), (3, 2, True), (3, 3, True), (5, 3, False),
                                                 (5, 4, True)])
def test_budget_edge_follows_the_budget_parameter(budget, epoch, edge):
    assert sc.pick_checkpoint([_c(epoch, 0.0, 1.0)], rule="nse", budget_epochs=budget)["budget_edge"] is edge


def _rec(family, cycle, nse=(1, 0.0), flood=(1, 0.0)):
    def p(epoch, lam):
        return {"epoch": epoch, "lambda": lam, "budget_edge": epoch in sc.BUDGET_EDGE_EPOCHS}
    return {"family": family, "cycle": cycle, "picks": {"nse": p(*nse), "flood": p(*flood)}}


def test_cross_cycle_labels_need_two_of_five_cycles():
    one = [_rec("base", 1, nse=(12, 0.0)), _rec("base", 2), _rec("base", 3)]
    two = one + [_rec("base", 4, nse=(11, 0.0))]
    assert sc.cross_cycle_flags(one)["base"]["nse"]["possibly_budget_limited"] is False
    flags = sc.cross_cycle_flags(two)["base"]
    assert flags["nse"] == {"n_cycles": 4, "budget_edge_cycles": [1, 4], "possibly_budget_limited": True}
    assert flags["flood"]["possibly_budget_limited"] is False and "strength_edge_cycles" not in flags["flood"]
    magw = [_rec("magw", 1, flood=(3, 8.0)), _rec("magw", 2, flood=(3, 0.5)), _rec("magw", 3, flood=(3, 2.0))]
    assert sc.cross_cycle_flags(magw)["magw"]["flood"]["possibly_too_narrow"] is False  # one at each edge
    magw.append(_rec("magw", 4, flood=(2, 8.0)))
    f = sc.cross_cycle_flags(magw)["magw"]["flood"]
    assert f["strength_edge_cycles"] == {"0.5": [2], "8": [1, 4]} and f["possibly_too_narrow"] is True
    with pytest.raises(sc.WyRotationScoringError, match="two pick records"):
        sc.cross_cycle_flags([_rec("base", 1), _rec("base", 1)])


# --------------------------------------------------------------------------- W&B stage B unlock and stage C lock

def _null_tracker(run_id):
    run = TrackingRun(backend="null", max_artifact_reference_bytes=1024, run_identity={"flashnh_run_id": run_id},
                      wandb_run_id=trk.wandb_run_id(run_id))
    return trk.RotationTracker(run, layout={})


def _stage_b(cycle, key="base"):
    s = cycle["scores"][key]
    path = cycle["out"] / s["run_id"] / sc.RUN_SCORES_NAME
    return s, path, sc.unlock_stage_b(config_dir=cycle["dirs"][key], run_scores_path=path, budget_epochs=3)


def test_stage_b_logs_curves_then_picks_into_the_allow_listed_keys(cycle):
    s, _, unlock = _stage_b(cycle)
    assert unlock.stage == "B" and unlock.flashnh_run_id == s["run_id"] and unlock.cycle == 1
    rec = picks(1, [s], cycle["out"])
    tracker = _null_tracker(s["run_id"])
    sc.log_checkpoint_year(tracker, unlock, s, rec)
    logged = tracker.run.resource_metrics
    epochs = [(e, m) for e, m in logged if e > 0]
    assert [e for e, _ in epochs] == list(EPOCHS)
    for e, m in epochs:
        assert set(m) == {f"ckpt/{k}" for k in trk.STAGE_B_EPOCH_KEYS} | {"epoch"}
        assert m["ckpt/median_nse"] == s["curves"][str(e)]["median_nse"]
    (summary,) = [m for e, m in logged if e == 0]
    assert set(summary) == {f"ckpt/{k}" for k in trk.STAGE_B_SUMMARY_KEYS}
    assert summary["ckpt/pick_nse_epoch"] == rec["picks"]["nse"]["epoch"] and summary["ckpt/flood_low_eligibility_flag"] is True


def test_stage_b_refuses_another_run_and_changed_evidence(cycle, tmp_path):
    s, path, unlock = _stage_b(cycle)
    with pytest.raises(TrackingError, match="unlock is for"):
        _null_tracker("wyrot-c1-other").log_stage(unlock, {"ckpt/median_nse": 0.5}, epoch=1)
    with pytest.raises(sc.WyRotationScoringError, match="this run's stage-B unlock"):
        sc.log_checkpoint_year(_null_tracker(s["run_id"]), unlock, cycle["scores"][0.5])
    forged = trk.StageUnlock(stage="B", flashnh_run_id=s["run_id"], cycle=1,
                             evidence=((str(path), "0" * 64),), derivation=unlock.derivation)
    with pytest.raises(sc.WyRotationScoringError, match="changed or vanished"):
        _null_tracker(s["run_id"]).log_stage(forged, {"ckpt/median_nse": 0.5}, epoch=1)
    # R1: real evidence is not enough; the unlock is re-derived from its derivation
    no_derivation = trk.StageUnlock(stage="B", flashnh_run_id=s["run_id"], cycle=1, evidence=unlock.evidence,
                                    derivation=())
    with pytest.raises(sc.WyRotationScoringError, match="derivation has keys"):
        _null_tracker(s["run_id"]).log_stage(no_derivation, {"ckpt/median_nse": 0.5}, epoch=1)
    at_production_budget = dict(unlock.derivation, budget_epochs="12")  # the run was scored at 3
    wrong = trk.StageUnlock(stage="B", flashnh_run_id=s["run_id"], cycle=1, evidence=unlock.evidence,
                            derivation=tuple(sorted(at_production_budget.items())))
    with pytest.raises(sc.WyRotationScoringError, match="stage B precondition does not hold"):
        _null_tracker(s["run_id"]).log_stage(wrong, {"ckpt/median_nse": 0.5}, epoch=1)
    other_cycle = trk.StageUnlock(stage="B", flashnh_run_id=s["run_id"], cycle=2, evidence=unlock.evidence,
                                  derivation=unlock.derivation)
    with pytest.raises(sc.WyRotationScoringError, match="differs from a fresh derivation"):
        _null_tracker(s["run_id"]).log_stage(other_cycle, {"ckpt/median_nse": 0.5}, epoch=1)
    with pytest.raises(TrackingError, match="needs a StageUnlock"):
        _null_tracker(s["run_id"]).log_stage({"stage": "B"}, {"ckpt/median_nse": 0.5}, epoch=1)


def test_stage_b_unlock_needs_a_complete_run_with_its_own_scores(cycle, tmp_path, contract_path):
    s, path, _ = _stage_b(cycle)
    with pytest.raises(sc.WyRotationScoringError, match="another run"):
        sc.unlock_stage_b(config_dir=cycle["dirs"][0.5], run_scores_path=path, budget_epochs=3)
    with pytest.raises(sc.WyRotationScoringError, match="campaign budget"):
        sc.unlock_stage_b(config_dir=cycle["dirs"]["base"], run_scores_path=path, budget_epochs=12)
    failed = make_run(tmp_path, contract_path, name="base", epochs=(1,), status="FAILED")
    with pytest.raises(sc.WyRotationScoringError, match="COMPLETE campaign run"):
        sc.unlock_stage_b(config_dir=failed, run_scores_path=path, budget_epochs=3)
    canary = make_run(tmp_path, contract_path, name="canary", epochs=(1,), is_canary=True)
    with pytest.raises(sc.WyRotationScoringError, match="COMPLETE campaign run"):
        sc.unlock_stage_b(config_dir=canary, run_scores_path=path, budget_epochs=3)


def test_stage_b_evidence_changed_after_the_unlock_stops_the_mirror(cycle, tmp_path, contract_path):
    out = tmp_path / "scores"
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    s = score(d, out, budget=1)
    path = out / s["run_id"] / sc.RUN_SCORES_NAME
    unlock = sc.unlock_stage_b(config_dir=d, run_scores_path=path, budget_epochs=1)
    path.write_bytes(path.read_bytes() + b" ")
    tracker = _null_tracker(s["run_id"])
    with pytest.raises(sc.WyRotationScoringError, match="changed or vanished"):
        sc.log_checkpoint_year(tracker, unlock, s)
    assert tracker.run.resource_metrics == []


@pytest.mark.parametrize("stage, kind, key", [
    ("B", "epoch", "ckpt/median_kge"),            # not allow-listed
    ("B", "epoch", "assess/nse_pick/median_nse"),
    ("B", "summary", "train/avg_total_loss"),
    ("C", "epoch", "assess/nse_pick/median_nse"),  # stage C has no per-epoch keys
    ("C", "summary", "assess/best/median_nse"),
    ("C", "summary", "ckpt/median_nse"),
    ("D", "summary", "wy2026/median_nse"),         # stage D is never open
    ("A", "summary", "train/avg_total_loss"),      # stage A goes through log()/set_summary()
])
def test_stage_keys_outside_the_allow_list_are_refused(stage, kind, key):
    with pytest.raises(TrackingError):
        trk.check_stage_keys({key: 1.0}, stage, kind=kind)


def test_stage_a_logging_still_refuses_score_keys():
    for key in ("ckpt/median_nse", "assess/nse_pick/median_nse", "wy2026/median_nse"):
        with pytest.raises(TrackingError):
            trk.check_metric_keys({key: 1.0})


def test_stage_c_stays_locked_until_both_arms_have_both_picks(tmp_path, contract_path):
    out = tmp_path / "scores"
    base_dir = make_run(tmp_path, contract_path, name="base", epochs=(1, 2))
    base = score(base_dir, out, budget=2)
    with pytest.raises(sc.WyRotationScoringError, match="stage C locked: no recorded base picks"):
        sc.unlock_stage_c(cycle=1, cycle_output_dir=out, flashnh_run_id=base["run_id"], budget_epochs=2)
    picks(1, [base], out, budget=2)
    with pytest.raises(sc.WyRotationScoringError, match="stage C locked: no recorded magw picks"):
        sc.unlock_stage_c(cycle=1, cycle_output_dir=out, flashnh_run_id=base["run_id"], budget_epochs=2)
    magw = [score(make_run(tmp_path, contract_path, name=f"magw-l{lam:g}", lam=lam, epochs=(1, 2)), out, budget=2)
            for lam in LAMBDAS]
    picks(1, magw, out, budget=2)
    with pytest.raises(sc.WyRotationScoringError, match="not the base pick record of cycle 2"):
        sc.unlock_stage_c(cycle=2, cycle_output_dir=out, flashnh_run_id=base["run_id"], budget_epochs=2)
    with pytest.raises(sc.WyRotationScoringError, match="at the campaign budget 12"):
        sc.unlock_stage_c(cycle=1, cycle_output_dir=out, flashnh_run_id=base["run_id"])
    with pytest.raises(sc.WyRotationScoringError, match="not a run of the recorded picks"):
        sc.unlock_stage_c(cycle=1, cycle_output_dir=out, flashnh_run_id="wyrot-c1-other", budget_epochs=2)
    unlock = sc.unlock_stage_c(cycle=1, cycle_output_dir=out, flashnh_run_id=magw[1]["run_id"], budget_epochs=2)
    assert unlock.stage == "C" and unlock.flashnh_run_id == magw[1]["run_id"]
    tracker = _null_tracker(magw[1]["run_id"])
    tracker.set_stage_summary(unlock, {"assess/nse_pick/median_nse": 0.5, "assess/flood_pick/epoch": 2})
    with pytest.raises(TrackingError):
        tracker.log_stage(unlock, {"assess/nse_pick/median_nse": 0.5}, epoch=1)
    with pytest.raises(TrackingError):
        tracker.set_stage_summary(unlock, {"ckpt/median_nse": 0.5})


@pytest.mark.parametrize("damage, match", [
    ("scores_changed", "is missing or differs from the sha256"),
    ("scores_missing", "is missing or differs from the sha256"),
    ("scores_file_moved", "does not cite its score file"),
    ("reconciliation_changed", "cites no, or another, round-E reconciliation"),
    ("reconciliation_missing", "checkpoint_eligibility_reconciliation.json is missing"),
])
def test_stage_c_rehashes_every_cited_score_file_and_the_reconciliation(tmp_path, contract_path, damage, match):
    """Delta review D3: stage C refuses once a score file a pick record cites, or
    the cycle's round-E reconciliation, is not what the picks were made from."""
    out = tmp_path / "scores"
    base = score(make_run(tmp_path, contract_path, name="base", epochs=(1,)), out, budget=1)
    magw = [score(make_run(tmp_path, contract_path, name=f"magw-l{lam:g}", lam=lam, epochs=(1,)), out, budget=1)
            for lam in LAMBDAS]
    base_rec, magw_rec = picks(1, [base], out, budget=1), picks(1, magw, out, budget=1)
    reconciliation = out / sc.RECONCILIATION_NAME
    assert base_rec["round_e_reconciliation_sha256"] == magw_rec["round_e_reconciliation_sha256"] == \
        sc.sha256_file(reconciliation)
    unlock = sc.unlock_stage_c(cycle=1, cycle_output_dir=out, flashnh_run_id=base["run_id"], budget_epochs=1)
    assert (str(reconciliation), sc.sha256_file(reconciliation)) in unlock.evidence
    target = spath(out, magw[2])
    if damage == "scores_changed":
        target.write_bytes(target.read_bytes() + b" ")
    elif damage == "scores_missing":
        target.unlink()
    elif damage == "scores_file_moved":
        path = out / sc.PICKS_TEMPLATE.format(family="magw")
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["runs"][2]["scores_file"] = f"../elsewhere/{sc.RUN_SCORES_NAME}"
        path.write_text(json.dumps(rec), encoding="utf-8")
    elif damage == "reconciliation_changed":
        reconciliation.write_bytes(reconciliation.read_bytes() + b" ")
    else:
        reconciliation.unlink()
    with pytest.raises(sc.WyRotationScoringError, match=match):
        sc.unlock_stage_c(cycle=1, cycle_output_dir=out, flashnh_run_id=base["run_id"], budget_epochs=1)
    with pytest.raises(sc.WyRotationScoringError):  # the earlier unlock no longer re-verifies either
        sc.reverify_unlock(unlock)


def test_stage_c_refuses_a_pick_record_without_a_recorded_pick(tmp_path, contract_path):
    out = tmp_path / "scores"
    base = score(make_run(tmp_path, contract_path, name="base", epochs=(1,)), out, budget=1)
    rec = picks(1, [base], out, budget=1)
    broken = dict(rec, family="magw", picks={"nse": rec["picks"]["nse"], "flood": {}})
    (out / sc.PICKS_TEMPLATE.format(family="magw")).write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(sc.WyRotationScoringError, match="no recorded flood pick"):
        sc.unlock_stage_c(cycle=1, cycle_output_dir=out, flashnh_run_id=base["run_id"], budget_epochs=1)


# --------------------------------------------------------------------------- score-blind eligibility table

ELIG_DATES = pd.date_range("2020-10-14 00:00", "2025-10-31 23:00", freq="h")
ELIG_OBS_TIME = ELIG_DATES + pd.Timedelta(hours=6)
ELIG_WY = np.where(ELIG_OBS_TIME.month >= 10, ELIG_OBS_TIME.year + 1, ELIG_OBS_TIME.year)
ELIG_YEARS = (2021, 2022, 2023, 2024, 2025)
CAP = pd.Timestamp("2025-09-30 23:00")
E_FULL, E_GAP23, E_TIED, E_DRY, E_SHORT22, E_NO_NC, E_EDGE_IN, E_EDGE_OUT = (
    "11000001", "11000002", "11000003", "11000004", "11000005", "11000006", "11000007", "11000008")
ELIG_BASINS = [E_FULL, E_GAP23, E_TIED, E_DRY, E_SHORT22, E_NO_NC, E_EDGE_IN, E_EDGE_OUT]


def _elig_target(basin: str) -> np.ndarray:
    """Synthetic target (mm/h) at each issue hour: row t holds the observation at t + 6 h."""
    n = len(ELIG_DATES)
    rng = np.random.default_rng(int(basin[-2:]) + 7)
    v = rng.lognormal(0.0, 1.0, n) * 0.05
    i = ELIG_DATES.get_loc
    if basin == E_GAP23:
        v[ELIG_WY == 2023] = np.nan                      # WY2023 explained-missing
    elif basin == E_TIED:
        v[:] = 0.5                                       # Q98 > 0 but every hour >= Q98: tie cap
    elif basin == E_DRY:
        v[:] = 0.0
        v[rng.random(n) < 0.01] = 1.0                    # Q98 == 0
    elif basin == E_SHORT22:
        idx = np.flatnonzero(ELIG_WY == 2022)
        v[idx[1000:]] = np.nan                           # 1,000 admitted hours: ~20 flood hours
    elif basin == E_EDGE_IN:
        v[ELIG_WY == 2025] = np.nan
        v[i(pd.Timestamp("2025-09-30 17:00"))] = 0.3     # obs at 23:00, the last admissible hour
        v[ELIG_OBS_TIME > CAP] = 1e9                     # obs past the cap: never read
    elif basin == E_EDGE_OUT:
        v[ELIG_WY == 2025] = np.nan
        v[ELIG_OBS_TIME > CAP] = 0.4                     # obs past the cap only
        for h in range(18, 24):                          # issue in WY2024, obs in WY2025: not admissible
            v[i(pd.Timestamp(f"2024-09-30 {h:02d}:00"))] = 0.4
    return v


def _write_elig_package(data_dir: Path, transform=None) -> Path:
    ts = data_dir / "time_series"
    ts.mkdir(parents=True, exist_ok=True)
    for basin in ELIG_BASINS:
        if basin == E_NO_NC:
            continue
        v = _elig_target(basin)
        if transform is not None:
            v = transform(basin, v.copy())
        q = np.full(len(v), 1.0)
        xr.Dataset({TARGET: (("date",), v), "qobs_m3s": (("date",), q)},
                   coords={"date": ELIG_DATES}).to_netcdf(ts / f"{basin}.nc")
    return data_dir


@pytest.fixture(scope="module")
def elig_pkg(tmp_path_factory) -> Path:
    return _write_elig_package(tmp_path_factory.mktemp("elig_pkg"))


@pytest.fixture(scope="module")
def elig_table(elig_pkg, contract_path):
    return sc.build_eligibility_table(split_contract=contract_path, data_dir=elig_pkg, basins=ELIG_BASINS)


def _expected_elig_row(basin: str, wy: int) -> dict:
    """Independent expectation: plain pandas/numpy on the synthetic arrays."""
    if basin == E_NO_NC:
        return {"eligible": False, "flood_eligible": False, "reason": sc.NOT_SCORED_NO_SERIES}
    s = pd.Series(_elig_target(basin), index=ELIG_DATES)
    first = max(pd.Timestamp(f"{wy - 1}-10-01 00:00"), pd.Timestamp("2020-10-14 00:00"))
    last_issue = pd.Timestamp(f"{wy}-09-30 23:00") - pd.Timedelta(hours=6)
    v = s.loc[first:last_issue].to_numpy()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"eligible": False, "flood_eligible": False, "reason": sc.NOT_SCORED_MISSING}
    q98 = np.quantile(v, 0.98)
    n_flood = int((v >= q98).sum())
    reasons = []
    if not q98 > 0:
        reasons.append("q98_not_positive")
    if n_flood < 50:
        reasons.append("too_few_flood_hours")
    if n_flood / v.size > 0.05:
        reasons.append("flood_hour_fraction_above_cap")
    return {"eligible": True, "flood_eligible": not reasons, "reason": ";".join(reasons)}


def _row_view(rows) -> dict:
    return {(r["STAID"], r["water_year"]): {"eligible": r["eligible"], "flood_eligible": r["flood_eligible"],
                                            "reason": ";".join(r["reasons"])} for r in rows}


def test_eligibility_table_matches_an_independent_computation(elig_table):
    rows = elig_table["rows"]
    assert [(r["STAID"], r["water_year"]) for r in rows] == [(b, wy) for b in sorted(ELIG_BASINS) for wy in ELIG_YEARS]
    got = _row_view(rows)
    for basin in ELIG_BASINS:
        for wy in ELIG_YEARS:
            assert got[(basin, wy)] == _expected_elig_row(basin, wy), (basin, wy)
    # the fixture reaches every outcome
    assert got[(E_FULL, 2023)] == {"eligible": True, "flood_eligible": True, "reason": ""}
    assert got[(E_GAP23, 2023)]["reason"] == sc.NOT_SCORED_MISSING
    assert got[(E_TIED, 2022)]["reason"] == sc.INELIGIBLE_TIE_CAP
    assert sc.INELIGIBLE_Q98_NOT_POSITIVE in got[(E_DRY, 2024)]["reason"]
    assert got[(E_SHORT22, 2022)]["reason"] == sc.INELIGIBLE_TOO_FEW
    assert got[(E_SHORT22, 2021)]["flood_eligible"]
    assert got[(E_NO_NC, 2025)]["reason"] == sc.NOT_SCORED_NO_SERIES != sc.NOT_SCORED_AREA
    edge_in = next(r for r in rows if (r["STAID"], r["water_year"]) == (E_EDGE_IN, 2025))
    assert edge_in["eligible"] and edge_in["n_admitted"] == 1 and not edge_in["flood_eligible"]
    assert got[(E_EDGE_OUT, 2025)]["reason"] == sc.NOT_SCORED_MISSING
    per = elig_table["per_year"]
    assert per["2023"]["n_basins"] == len(ELIG_BASINS)
    assert per["2023"]["n_eligible"] == sum(got[(b, 2023)]["eligible"] for b in ELIG_BASINS)
    assert per["2021"]["checkpoint_window_utc"] == ["2020-10-01T00:00:00", "2021-09-30T23:00:00"]
    assert per["2021"]["rotation_cycle"] == 5 and per["2022"]["rotation_cycle"] == 1
    assert per["2025"]["low_eligibility_flag"] is (per["2025"]["n_flood_eligible"] / len(ELIG_BASINS) < 0.5)


def test_eligibility_never_reads_outside_the_year_window_or_past_the_cap(tmp_path, contract_path, elig_table):
    def past_cap(basin, v):
        v[ELIG_OBS_TIME > CAP] = -1.0e9
        return v

    def outside_2023(basin, v):
        v[ELIG_WY != 2023] = 7.0e9
        return v

    base = _row_view(elig_table["rows"])
    capped = sc.build_eligibility_table(split_contract=contract_path, basins=ELIG_BASINS,
                                        data_dir=_write_elig_package(tmp_path / "cap", past_cap))
    assert _row_view(capped["rows"]) == base
    only23 = sc.build_eligibility_table(split_contract=contract_path, basins=ELIG_BASINS,
                                        data_dir=_write_elig_package(tmp_path / "y23", outside_2023))
    view = _row_view(only23["rows"])
    assert {k: v for k, v in view.items() if k[1] == 2023} == {k: v for k, v in base.items() if k[1] == 2023}
    assert view != base  # the other years did change: the scramble was real


def test_eligibility_shares_the_scorers_definitions(monkeypatch, tmp_path, elig_pkg, contract_path, elig_table):
    # eligible <=> the explained-missing count of the run gate is > 0 (same reader)
    sealings = sc._eligibility_sealings(contract_path)
    got = _row_view(elig_table["rows"])
    with_nc = [b for b in ELIG_BASINS if b != E_NO_NC]
    for wy, sealing in sealings.items():
        run_dir = tmp_path / f"run_wy{wy}"
        run_dir.mkdir()
        (run_dir / "config.yml").write_text(yaml.safe_dump({"data_dir": str(elig_pkg)}), encoding="utf-8")
        counts = rot.count_admissible_checkpoint_targets(run_dir, with_nc, sealing=sealing, target=TARGET)
        assert {b: counts[b] > 0 for b in with_nc} == {b: got[(b, wy)]["eligible"] for b in with_nc}
    # flood eligibility goes through the scorer's one D5(b) function
    monkeypatch.setattr(sc, "flood_eligibility", lambda **kw: ["sentinel_rule"])
    patched = sc.build_eligibility_table(split_contract=contract_path, data_dir=elig_pkg, basins=[E_FULL])
    assert {r["water_year"]: (r["flood_eligible"], r["reasons"]) for r in patched["rows"]} == {
        wy: (False, ["sentinel_rule"]) for wy in ELIG_YEARS}
    dates = np.array(pd.date_range("2023-01-01", periods=100, freq="h"), dtype="datetime64[ns]")
    s = sc.SealedBasinSeries(basin_id=E_FULL, freq="1h", area_km2=1.0, dates=dates, obs_m3s=np.linspace(1, 2, 100),
                             sim_m3s=np.ones(100), n_total=100, n_admitted=100, n_sim_nonfinite_at_admitted=0,
                             sealed_metrics={})
    _, record = sc.build_flood_facts({E_FULL: s}, population=[E_FULL], not_scored={})
    assert record["basins"][E_FULL]["reasons"] == ["sentinel_rule"]


def test_eligibility_opens_no_run_output(monkeypatch, contract_path, elig_pkg):
    def refuse(*a, **k):
        raise AssertionError("the eligibility table must not open run output")

    for name in ("load_period_results", "raw_space_metrics_for_run_period", "sealed_period_series",
                 "score_run_checkpoint_year"):
        monkeypatch.setattr(sc, name, refuse)
    monkeypatch.setattr(rot, "evaluate_all_epochs", refuse)
    table = sc.build_eligibility_table(split_contract=contract_path, data_dir=elig_pkg, basins=[E_FULL, E_NO_NC])
    assert len(table["rows"]) == 10


def test_eligibility_refuses_an_unpinned_contract_and_bad_basins(tmp_path, contract_path, elig_pkg):
    tampered = json.loads(contract_path.read_text(encoding="utf-8"))
    tampered["observation_upper_bound_utc"] = "2026-09-30T23:00:00Z"
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(nre.RawSpaceEvaluationError):
        sc.build_eligibility_table(split_contract=bad, data_dir=elig_pkg, basins=[E_FULL])
    for basins in ([], [E_FULL, E_FULL]):
        with pytest.raises(sc.WyRotationScoringError, match="duplicates"):
            sc.build_eligibility_table(split_contract=contract_path, data_dir=elig_pkg, basins=basins)


def test_eligibility_csv_sha256_manifest_and_write_once(tmp_path, elig_table):
    out = tmp_path / "out"
    written = sc.write_eligibility_table(elig_table, out, provenance={"note": "synthetic"})
    csv = Path(written["csv"])
    data = csv.read_bytes()
    assert data.splitlines()[0] == b"STAID,water_year,eligible,flood_eligible,reason"
    assert b"\r" not in data
    assert Path(written["sha256"]).read_text(encoding="utf-8") == f"{sc.sha256_file(csv)}  {csv.name}\n"
    # parsed the way the panel freeze reads it: string ids, empty reasons kept, True/False
    df = pd.read_csv(csv, dtype={"STAID": str, "reason": str}, keep_default_na=False)
    assert list(df.columns) == list(sc.ELIGIBILITY_COLUMNS) and len(df) == 40
    assert set(df["eligible"].astype(str)) <= {"True", "False"}
    assert not (df["flood_eligible"] & ~df["eligible"]).any()
    assert df.duplicated(["STAID", "water_year"]).sum() == 0
    assert set(df["water_year"]) == set(ELIG_YEARS)
    assert df.loc[(df.STAID == E_FULL) & (df.water_year == 2023), "reason"].item() == ""
    manifest = json.loads(Path(written["manifest"]).read_text(encoding="utf-8"))
    assert manifest["csv_sha256"] == written["csv_sha256"] and manifest["spec_sha256"] == sc.SPEC_SHA256
    assert "reference-panel exclusion input" in manifest["use"] and "reconcile_eligibility" in manifest["use"]
    assert "published round-E" in manifest["use"]
    assert manifest["per_year"] == json.loads(json.dumps(elig_table["per_year"]))
    # an identical rerun is a no-op; different content is refused
    assert sc.write_eligibility_table(elig_table, out, provenance={"note": "synthetic"}) == written
    changed = copy.deepcopy(elig_table)
    changed["rows"][0]["eligible"] = not changed["rows"][0]["eligible"]
    changed["rows"][0]["flood_eligible"] = False
    with pytest.raises(sc.WyRotationScoringError, match="written once"):
        sc.write_eligibility_table(changed, out, provenance={"note": "synthetic"})
    assert csv.read_bytes() == data


def test_eligibility_csv_refuses_flood_eligible_but_not_eligible():
    row = {"STAID": E_FULL, "water_year": 2023, "eligible": False, "flood_eligible": True, "reasons": []}
    with pytest.raises(sc.WyRotationScoringError, match="flood-eligible but not eligible"):
        sc.eligibility_csv_bytes([row])


def test_eligibility_only_command_line(monkeypatch, tmp_path, contract_path, elig_pkg, capsys):
    monkeypatch.setattr(wy, "verify_basin_population", lambda contract, root: list(ELIG_BASINS))
    out = tmp_path / "elig_out"
    argv = ["--eligibility-only", "--split-contract", str(contract_path), "--data-dir", str(elig_pkg),
            "--out-dir", str(out), "--project-root", str(tmp_path)]
    assert sc.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert Path(report["csv"]).name == sc.ELIGIBILITY_TABLE_NAME and (out / sc.ELIGIBILITY_MANIFEST_NAME).is_file()
    assert set(report["per_year"]) == {str(y) for y in ELIG_YEARS}
    with pytest.raises(SystemExit):  # the only mode must be named
        sc.main([a for a in argv if a != "--eligibility-only"])
    with pytest.raises(SystemExit):  # R8: the target is fixed, not an option
        sc.main(argv + ["--target", "qobs_m3s"])
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    with pytest.raises(sc.WyRotationScoringError, match="outside the project root"):
        sc.main(argv[:-4] + ["--out-dir", str(outside), "--project-root", str(tmp_path)])
    assert not outside.exists()


# --------------------------------------------------------------------------- R5: table vs cycle facts reconciliation

def _elig_rows(rows):
    return {"rows": rows, "per_year": {}, "contract_checksum_sha256": "0" * 64}


def test_reconciliation_records_every_difference_between_table_and_facts(cycle, tmp_path):
    """Cycle 1 (checkpoint WY2022) facts: scored 01/02/03, flood-eligible 01/02,
    04 area-excluded, 05 explained-missing. The table row for each basin is set
    to produce one kind of difference; WY2023 rows must be ignored."""
    import shutil
    out = tmp_path / "cycle_out"
    out.mkdir()
    shutil.copy2(cycle["out"] / sc.FACTS_NAME, out / sc.FACTS_NAME)

    def row(staid, wy, eligible, flood, reasons=()):
        return {"STAID": staid, "water_year": wy, "eligible": eligible, "flood_eligible": flood,
                "reasons": list(reasons)}
    rows = [row("01010101", 2022, True, True),
            row("02020202", 2022, True, False, [sc.INELIGIBLE_TIE_CAP]),  # flood in the facts only
            row("03030303", 2022, True, True),                              # flood in the table only
            row(NO_NC, 2022, True, True),                                   # the area exclusion
            row(MISSING, 2022, True, False, [sc.INELIGIBLE_TOO_FEW]),       # eligible, not scored, not area
            row("09090909", 2022, False, False, [sc.NOT_SCORED_NO_SERIES]),
            row("01010101", 2023, False, False, [sc.NOT_SCORED_MISSING])]
    rows.sort(key=lambda r: (r["STAID"], r["water_year"]))
    written = sc.write_eligibility_table(_elig_rows(rows), tmp_path / "elig", provenance={"note": "synthetic"})
    rec = sc.reconcile_eligibility(cycle_output_dir=out, eligibility_csv=written["csv"])
    assert (rec["cycle"], rec["water_year"], rec["spec_sha256"]) == (1, 2022, sc.SPEC_SHA256)
    assert rec["facts_sha256"] == sc.sha256_file(out / sc.FACTS_NAME) and rec["facts_source_run_id"] == "wyrot-c1-base"
    assert rec["eligibility_csv_sha256"] == written["csv_sha256"]
    assert rec["basins_only_in_table"] == ["09090909"] and rec["basins_only_in_facts"] == []
    assert rec["table_eligible_not_scored"] == {"area_excluded": [NO_NC], "other": [MISSING]}
    assert rec["scored_not_table_eligible"] == []
    assert rec["flood_eligible_table_only"] == ["03030303", NO_NC]
    assert rec["flood_eligible_facts_only"] == ["02020202"]
    assert rec["counts"] == {"table_eligible": 5, "scored": 3, "table_flood_eligible": 3, "facts_flood_eligible": 2}
    # Q3: published = table flood-eligible and scored (01, 03): 2, as in the facts, but a basin outside the
    # facts' population is in the table, so the two do not agree
    published = {k: rec["published_round_e"][k] for k in ("n_eligible", "eligible_fraction", "facts_agree")}
    assert published == {"n_eligible": 2, "eligible_fraction": 2 / 5, "facts_agree": False}
    on_disk = json.loads((out / sc.RECONCILIATION_NAME).read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(rec))
    # written once: an identical rerun is a no-op, another table is refused
    assert sc.reconcile_eligibility(cycle_output_dir=out, eligibility_csv=written["csv"]) == rec
    rows2 = [dict(r, eligible=False, flood_eligible=False) if r["STAID"] == MISSING else r for r in rows]
    other = sc.write_eligibility_table(_elig_rows(rows2), tmp_path / "elig2", provenance={})
    with pytest.raises(sc.WyRotationScoringError, match="written once"):
        sc.reconcile_eligibility(cycle_output_dir=out, eligibility_csv=other["csv"])


def _cycle1_table(tmp_path, *, flood_02: bool) -> Path:
    """A WY2022 table for a fresh cycle-1 baseline (facts: scored 01/02/03,
    flood-eligible 01/02, 04 area-excluded, 05 explained-missing)."""
    def row(staid, eligible, flood, reasons=()):
        return {"STAID": staid, "water_year": 2022, "eligible": eligible, "flood_eligible": flood,
                "reasons": list(reasons)}
    rows = [row("01010101", True, True),
            row("02020202", True, flood_02, () if flood_02 else [sc.INELIGIBLE_TOO_FEW]),
            row("03030303", True, False, [sc.INELIGIBLE_TOO_FEW]),
            row(NO_NC, True, True),                       # area-excluded by the scorer: not in the count
            row(MISSING, False, False, [sc.NOT_SCORED_MISSING])]
    return sc.write_eligibility_table(_elig_rows(rows), tmp_path / "elig", provenance={"note": "synthetic"})["csv"]


def test_round_e_scoring_agrees_with_the_published_table(tmp_path, contract_path):
    """Q3 (Omri, 2026-11-19): the table is the published count; the facts agree."""
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    out = tmp_path / "scores"
    s = sc.score_run_checkpoint_year(config_dir=d, population=POP, cycle_output_dir=out, budget_epochs=1,
                                     eligibility_csv=_cycle1_table(tmp_path, flood_02=True))
    rec = json.loads((out / sc.RECONCILIATION_NAME).read_text(encoding="utf-8"))
    assert rec["published_round_e"]["facts_agree"] is True and rec["published_round_e"]["n_eligible"] == 2
    facts = json.loads((out / sc.FACTS_NAME).read_text(encoding="utf-8"))
    assert (facts["n_eligible"], facts["eligible_fraction"]) == (2, 2 / 5)
    assert s["round_e_reconciliation_sha256"] == sc.sha256_file(out / sc.RECONCILIATION_NAME)


def test_scoring_without_the_published_table_is_refused_before_any_file(tmp_path, contract_path):
    """Delta review D1 (enforces Omri's Q3): no table, no facts and no score file."""
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    out = tmp_path / "scores"
    with pytest.raises(sc.WyRotationScoringError, match="eligibility table .* is required"):
        sc.score_run_checkpoint_year(config_dir=d, population=POP, cycle_output_dir=out, budget_epochs=1)
    assert not out.exists()


ROUND_E_DAMAGE = [
    ("null", "cites no round-E reconciliation"),
    ("other", "cites another round-E reconciliation"),
    ("missing", "checkpoint_eligibility_reconciliation.json is missing"),
]


def _damage_round_e(out: Path, s, damage: str) -> None:
    """``null``: the score file cites no reconciliation (as a pre-D1 score
    without a table did); ``other``: the cycle's reconciliation is not the one
    the score file cites; ``missing``: the cycle has none."""
    reconciliation = out / sc.RECONCILIATION_NAME
    if damage == "null":
        path = spath(out, s)
        body = json.loads(path.read_text(encoding="utf-8"))
        body["round_e_reconciliation_sha256"] = None
        path.write_text(json.dumps(body), encoding="utf-8")
    elif damage == "other":
        reconciliation.write_bytes(reconciliation.read_bytes() + b" ")
    else:
        reconciliation.unlink()


@pytest.mark.parametrize("damage, match", ROUND_E_DAMAGE)
def test_picks_refuse_a_score_file_without_the_cycles_round_e_reconciliation(tmp_path, contract_path, damage,
                                                                            match):
    """Delta review D1."""
    out = tmp_path / "scores"
    s = score(make_run(tmp_path, contract_path, name="base", epochs=(1,)), out, budget=1)
    _damage_round_e(out, s, damage)
    with pytest.raises(sc.WyRotationScoringError, match=match):
        picks(1, [s], out, budget=1)
    assert not (out / sc.PICKS_TEMPLATE.format(family="base")).exists()


@pytest.mark.parametrize("damage, match", ROUND_E_DAMAGE)
def test_stage_b_refuses_a_score_file_without_the_cycles_round_e_reconciliation(tmp_path, contract_path, damage,
                                                                               match):
    """Delta review D1."""
    out = tmp_path / "scores"
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    s = score(d, out, budget=1)
    unlock = sc.unlock_stage_b(config_dir=d, run_scores_path=spath(out, s), budget_epochs=1)
    assert (str(out / sc.RECONCILIATION_NAME), sc.sha256_file(out / sc.RECONCILIATION_NAME)) in unlock.evidence
    _damage_round_e(out, s, damage)
    with pytest.raises(sc.WyRotationScoringError, match=match):
        sc.unlock_stage_b(config_dir=d, run_scores_path=spath(out, s), budget_epochs=1)
    with pytest.raises(sc.WyRotationScoringError):
        sc.reverify_unlock(unlock)


def test_round_e_disagreement_is_refused_before_any_score_file(tmp_path, contract_path):
    d = make_run(tmp_path, contract_path, name="base", epochs=(1,))
    out = tmp_path / "scores"
    with pytest.raises(sc.WyRotationScoringError, match="round E: .* disagree with the published table count"):
        sc.score_run_checkpoint_year(config_dir=d, population=POP, cycle_output_dir=out, budget_epochs=1,
                                     eligibility_csv=_cycle1_table(tmp_path, flood_02=False))
    assert (out / sc.FACTS_NAME).is_file() and not list(out.rglob(sc.RUN_SCORES_NAME))
    assert not list(out.rglob("per_basin_epoch*.json"))
    rec = json.loads((out / sc.RECONCILIATION_NAME).read_text(encoding="utf-8"))
    assert rec["published_round_e"]["facts_agree"] is False and rec["published_round_e"]["n_eligible"] == 1
    assert rec["flood_eligible_facts_only"] == ["02020202"]


def test_reconciliation_refuses_a_table_that_does_not_match_its_sidecar(cycle, tmp_path):
    import shutil
    out = tmp_path / "cycle_out"
    out.mkdir()
    shutil.copy2(cycle["out"] / sc.FACTS_NAME, out / sc.FACTS_NAME)
    rows = [{"STAID": "01010101", "water_year": 2022, "eligible": True, "flood_eligible": True, "reasons": []}]
    written = sc.write_eligibility_table(_elig_rows(rows), tmp_path / "elig", provenance={})
    csv = Path(written["csv"])
    csv.write_bytes(csv.read_bytes().replace(b"True,True", b"True,False"))
    with pytest.raises(sc.WyRotationScoringError, match="does not match its sidecar"):
        sc.reconcile_eligibility(cycle_output_dir=out, eligibility_csv=csv)
    assert not (out / sc.RECONCILIATION_NAME).exists()
    (out / sc.FACTS_NAME).unlink()
    with pytest.raises(sc.WyRotationScoringError, match="cannot read Q98 facts"):
        sc.reconcile_eligibility(cycle_output_dir=out, eligibility_csv=csv)


# --------------------------------------------------------------------------- R7: write-once against a concurrent writer

def _no_tmp_left(folder: Path) -> bool:
    return not [p for p in folder.iterdir() if ".tmp" in p.name]


def test_write_once_refuses_a_racing_writer_with_other_bytes(monkeypatch, tmp_path):
    target = tmp_path / "f.json"

    def racing_link(src, dst):
        Path(dst).write_bytes(b"theirs")  # another writer won between the exists-check and the link
        raise FileExistsError(dst)
    monkeypatch.setattr(sc.os, "link", racing_link)
    with pytest.raises(sc.WyRotationScoringError, match="already exists with different content"):
        sc._write_bytes_once(target, b"ours")
    assert target.read_bytes() == b"theirs" and _no_tmp_left(tmp_path)


def test_write_once_accepts_a_racing_writer_with_the_same_bytes(monkeypatch, tmp_path):
    target = tmp_path / "f.json"

    def racing_link(src, dst):
        Path(dst).write_bytes(b"same")
        raise FileExistsError(dst)
    monkeypatch.setattr(sc.os, "link", racing_link)
    assert sc._write_bytes_once(target, b"same") == sc.hashlib.sha256(b"same").hexdigest()
    assert _no_tmp_left(tmp_path)


def test_write_once_falls_back_to_an_exclusive_create_without_hard_links(monkeypatch, tmp_path):
    def no_links(src, dst):
        raise OSError("hard links not supported")
    monkeypatch.setattr(sc.os, "link", no_links)
    target = tmp_path / "f.json"
    assert sc._write_bytes_once(target, b"data") == sc.hashlib.sha256(b"data").hexdigest()
    assert target.read_bytes() == b"data" and _no_tmp_left(tmp_path)
    assert sc._write_bytes_once(target, b"data") == sc.hashlib.sha256(b"data").hexdigest()  # idempotent
    with pytest.raises(sc.WyRotationScoringError, match="already exists with different content"):
        sc._write_bytes_once(target, b"other")
    real_open = sc.os.open

    def racing_open(path, flags, *a):
        if flags & sc.os.O_EXCL and not Path(path).exists():
            Path(path).write_bytes(b"theirs")
        return real_open(path, flags, *a)
    monkeypatch.setattr(sc.os, "open", racing_open)
    with pytest.raises(sc.WyRotationScoringError, match="already exists with different content"):
        sc._write_bytes_once(tmp_path / "g.json", b"ours")
    assert (tmp_path / "g.json").read_bytes() == b"theirs" and _no_tmp_left(tmp_path)
