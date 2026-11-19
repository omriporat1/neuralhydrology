"""Checkpoint-year scorer of the water-year rotation (spec v001, D1-D7, D13).

``docs/wy_rotation_metric_spec_v001.md`` (frozen; its full sha256 is pinned in
:data:`SPEC_SHA256`) fixes the metrics, the two checkpoint picks and the gates.
This module implements, for the cycle's checkpoint year (NH period
``validation``) only:

* the per-basin series seam of spec §0, by composition (no fork): the qualified
  sealed scorer :func:`nh_seed_evaluation.raw_space_metrics_for_run_period`
  runs first (sealing receipts, readable-window check, training-window area,
  raw-space NSE); the same results pickle is then reread and each scored basin's
  obs/sim converted again with :func:`convert_period_to_raw_space` and the
  area that call derived. The recomputed :func:`raw_space_metrics` must equal
  the sealed call's values exactly, otherwise the scorer stops;
* per (cycle, period) Q98 facts and flood eligibility D4/D5(b), from the
  baseline run's observations only, written before any per-epoch score file;
* gates G1-G4, checked before the per-epoch flood diagnostics of each epoch;
* per-basin metrics D1 and the 12-epoch curves of median NSE and median
  Q98-normalised RMSE;
* the two picks per arm (D3, D6, D13) with the budget-edge (D7) and
  strength-edge (review round C) flags, per-lambda own picks (round D), and the
  cross-cycle >= 2/5 labels;
* the stage-B unlock (checkpoint-year W&B scores) and the stage-C lock
  (assessment-year scores open only after both picks of every arm of the cycle
  are recorded). Nothing here scores an assessment year or WY2026.

Flash-NH files are authoritative; W&B only mirrors them
(:func:`log_checkpoint_year`). Every gate failure raises
:class:`WyRotationScoringError`; nothing is repaired silently.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import nh_raw_space_evaluation as nre
from . import wy_rotation_tracking as trk
from .nh_raw_space_evaluation import convert_period_to_raw_space, raw_space_metrics
from .nh_seed_evaluation import load_period_results, period_results_path, raw_space_metrics_for_run_period
from .wy_rotation_runs import CAMPAIGN_EPOCHS
from .stage1_v2_12plus12_hydrological_consumer import (
    HIGH_FLOW_QUANTILE,
    MIN_HIGH_FLOW_NSE_SAMPLES,
    compute_q98_configuration_basin_diagnostics,
    derive_canonical_basin_q98_facts,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORER_VERSION = "wy_rotation_scoring_v001"
SPEC_RELATIVE_PATH = "docs/wy_rotation_metric_spec_v001.md"
SPEC_SHA256 = "157ece673ed50cfa6dbc2c5e46dfe1ee724b0ca713fcd01a2373e7d3549fa898"

CHECKPOINT_PERIOD = "validation"  # NH period that holds the cycle's checkpoint year
ASSESSMENT_PERIOD = "test"        # never scored here (stage C)
TARGET = "qobs_mm_per_h_lead06"
LEAD_HOURS = 6

# Runner file names (scripts/run_wy_rotation_cycle.py; equality is a test).
RECEIPT_NAME = "wy_rotation_run_receipt.json"
EVIDENCE_NAME = "wy_rotation_score_blind_evidence.json"

FACTS_NAME = "checkpoint_flood_facts.json"
RUN_SCORES_NAME = "checkpoint_scores.json"
PICKS_TEMPLATE = "checkpoint_picks_{family}.json"

# D5(b) and D2/E
MIN_FLOOD_HOURS = MIN_HIGH_FLOW_NSE_SAMPLES  # 50
MAX_FLOOD_HOUR_FRACTION = 0.05
MIN_ELIGIBLE_FRACTION = 0.5
# D7, D11, round C. The campaign budget has one source: the frozen cycle
# config facts in wy_rotation_runs (the runner refuses a cycle config with any
# other epoch count). Every scored run must have exactly this many epochs.
#: Default campaign budget (epochs per run); tests pass a smaller ``budget_epochs``.
DEFAULT_BUDGET_EPOCHS = CAMPAIGN_EPOCHS
LAMBDA_GRID = (0.5, 2.0, 8.0)
STRENGTH_EDGE_LAMBDAS = (0.5, 8.0)
CROSS_CYCLE_FLAG_MIN = 2  # >= 2 of 5 cycles
N_CYCLES = 5
# Arm families that must each hold a recorded pick before stage C opens.
REQUIRED_ARM_FAMILIES = ("base", "magw")
PICK_RULES = ("nse", "flood")

INELIGIBLE_NO_ADMITTED = "no_admitted_hours"
INELIGIBLE_Q98_NOT_POSITIVE = "q98_not_positive"
INELIGIBLE_TOO_FEW = "too_few_flood_hours"
INELIGIBLE_TIE_CAP = "flood_hour_fraction_above_cap"
NOT_SCORED_MISSING = "not_scored_explained_missing"
NOT_SCORED_AREA = "not_scored_area_excluded"
# Eligibility table only: the package has no NetCDF for the basin (no series to read).
NOT_SCORED_NO_SERIES = "not_scored_no_package_series"
RECONCILIATION_NAME = "checkpoint_eligibility_reconciliation.json"

_SEALED_METRIC_KEYS = ("n_samples", "nse", "kge", "kge_r", "kge_alpha", "kge_beta", "rmse", "mae", "pearson_r",
                       "bias", "pbias")


class WyRotationScoringError(ValueError):
    """A scoring gate, identity or ordering violation."""


# ---------------------------------------------------------------------------- small helpers

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_spec_sha256(project_root=None) -> str:
    """Return the spec sha256, or raise unless the frozen spec is byte-identical."""
    path = Path(project_root or _PROJECT_ROOT) / SPEC_RELATIVE_PATH
    if not path.is_file():
        raise WyRotationScoringError(f"metric spec {path} not found")
    digest = sha256_file(path)
    if digest != SPEC_SHA256:
        raise WyRotationScoringError(f"metric spec {path} has sha256 {digest}, expected the frozen {SPEC_SHA256}")
    return digest


def _same_value(a, b) -> bool:
    """Exact equality, NaN equal to NaN (bit-for-bit up to the sign of zero)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return _finite_or_none(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _dump(obj) -> bytes:
    return (json.dumps(_jsonable(obj), indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_bytes_once(path: Path, data: bytes) -> str:
    """Create ``path`` holding ``data``, atomically against a concurrent writer:
    the bytes go to a private temp file that is then hard-linked to the final
    name, which fails if the name exists (no exists-check/replace window). If
    hard links are unsupported, ``O_CREAT | O_EXCL`` on the final name is used.
    An existing file must hold the same bytes (an idempotent rerun), otherwise
    refuse. Returns the sha256 of ``data``."""
    path = Path(path)
    digest = hashlib.sha256(data).hexdigest()

    def same_or_refuse() -> str:
        if path.read_bytes() != data:
            raise WyRotationScoringError(f"{path} already exists with different content; scores, picks and "
                                         "tables are written once and never replaced")
        return digest

    if path.exists():
        return same_or_refuse()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            return same_or_refuse()
        except OSError:  # no hard links here: exclusive create on the final name
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
            except FileExistsError:
                return same_or_refuse()
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
    finally:
        tmp.unlink(missing_ok=True)
    return digest


def _write_new(path: Path, obj) -> str:
    """Create ``path`` with ``obj`` as JSON, written once (:func:`_write_bytes_once`)."""
    return _write_bytes_once(path, _dump(obj))


def budget_edge_epochs(budget_epochs: int) -> tuple[int, ...]:
    """D7: a pick at the last two epochs of the budget is at the budget edge
    (epochs 11 and 12 for the campaign budget of 12; epochs start at 1)."""
    b = _check_budget(budget_epochs)
    return tuple(e for e in (b - 1, b) if e >= 1)


def _check_budget(budget_epochs) -> int:
    if isinstance(budget_epochs, bool) or not isinstance(budget_epochs, int) or budget_epochs < 1:
        raise WyRotationScoringError(f"budget_epochs must be a positive int, got {budget_epochs!r}")
    return budget_epochs


BUDGET_EDGE_EPOCHS = budget_edge_epochs(CAMPAIGN_EPOCHS)


def _read_json(path, what: str) -> dict:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WyRotationScoringError(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WyRotationScoringError(f"{what} {path} is not a JSON object")
    return value


def _obs_time_iso(issue_date, lead_hours: int) -> str:
    t = np.datetime64(issue_date, "s") + np.timedelta64(int(lead_hours), "h")
    return str(np.datetime_as_string(t, unit="s"))


def series_fingerprint(dates: np.ndarray, obs_m3s: np.ndarray) -> str:
    """sha256 over the admitted issue dates (int64 ns) and observed m^3/s
    (float64) bytes: equal fingerprints mean bit-identical series (G4)."""
    d = np.ascontiguousarray(np.asarray(dates).astype("datetime64[ns]").view("int64"))
    o = np.ascontiguousarray(np.asarray(obs_m3s, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(d.size).encode())
    h.update(d.tobytes())
    h.update(o.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------- sealed series seam (spec §0)

@dataclass(frozen=True)
class SealedBasinSeries:
    basin_id: str
    freq: str
    area_km2: float
    dates: np.ndarray      # admitted issue dates (observation time = issue + lead)
    obs_m3s: np.ndarray    # admitted observations
    sim_m3s: np.ndarray    # simulations at the admitted hours
    n_total: int
    n_admitted: int
    n_sim_nonfinite_at_admitted: int
    sealed_metrics: Mapping[str, Any]  # the sealed call's raw_space_metrics values


def sealed_period_series(*, run_dir, period: str, epoch: int, package_root, basin_ids: Sequence[str],
                         target_variable: str = TARGET, lead_hours: int = LEAD_HOURS) -> dict[str, Any]:
    """Per-basin admitted dates/obs/sim of one sealed run period and epoch.

    Runs the qualified sealed scorer first (it raises on any sealing problem),
    then rereads the same pickle and rebuilds each scored basin's admitted
    series with the area that call derived. Raises unless the rebuilt counts
    and every raw-space metric equal the sealed call's values exactly.
    Returns ``{"sealed": <sealed result without per_basin>, "series":
    {basin_id: SealedBasinSeries}, "results_keys": sorted keys}``."""
    if nre.read_run_rotation_sealing(run_dir) is None:
        raise WyRotationScoringError(f"{run_dir} is not a split-contract run; the scorer reads sealed runs only")
    sealed = raw_space_metrics_for_run_period(run_dir=run_dir, period=period, epoch=epoch, package_root=package_root,
                                              target_variable=target_variable, lead_hours=lead_hours,
                                              basin_ids=list(basin_ids), compute_pooled=False)
    if not sealed.get("split_contract"):
        raise WyRotationScoringError(f"{run_dir} {period}/epoch{epoch}: the sealed call returned no split contract")
    results = load_period_results(run_dir, period, epoch)
    obs_key, sim_key = f"{target_variable}_obs", f"{target_variable}_sim"
    series: dict[str, SealedBasinSeries] = {}
    for row in sealed["per_basin"]:
        basin_id, freq = row["basin_id"], row["freq"]
        if basin_id in series:
            raise WyRotationScoringError(f"basin {basin_id}: more than one frequency in the results; the rotation "
                                         "scores one hourly series per basin")
        ds = results[basin_id][freq]["xr"]
        obs = np.asarray(ds[obs_key].values, dtype=np.float64).reshape(-1)
        sim = np.asarray(ds[sim_key].values, dtype=np.float64).reshape(-1)
        dates = np.asarray(ds["date"].values).astype("datetime64[ns]")
        if not (obs.size == sim.size == dates.size):
            raise WyRotationScoringError(f"basin {basin_id}: {obs.size} obs, {sim.size} sim, {dates.size} dates")
        conv = convert_period_to_raw_space(obs, sim, row["area_km2"])
        mask = conv.admitted_mask
        if (conv.n_total, conv.n_admitted, conv.n_sim_nonfinite_at_admitted) != (
                row["n_total"], row["n_admitted"], row["n_sim_nonfinite_at_admitted"]):
            raise WyRotationScoringError(f"basin {basin_id}: reread counts differ from the sealed call")
        obs_adm, sim_adm = conv.obs_m3s[mask], conv.sim_m3s[mask]
        recomputed = raw_space_metrics(obs_adm, sim_adm)
        sealed_metrics = {k: row[k] for k in _SEALED_METRIC_KEYS}
        diff = [k for k in _SEALED_METRIC_KEYS if not _same_value(recomputed[k], sealed_metrics[k])]
        if diff:
            raise WyRotationScoringError(f"basin {basin_id}: reread metrics {diff} differ from the sealed call")
        series[basin_id] = SealedBasinSeries(
            basin_id=basin_id, freq=freq, area_km2=float(row["area_km2"]), dates=dates[mask], obs_m3s=obs_adm,
            sim_m3s=sim_adm, n_total=conv.n_total, n_admitted=conv.n_admitted,
            n_sim_nonfinite_at_admitted=conv.n_sim_nonfinite_at_admitted, sealed_metrics=sealed_metrics)
    head = {k: v for k, v in sealed.items() if k != "per_basin"}
    return {"sealed": head, "series": series, "results_keys": sorted(results)}


# ---------------------------------------------------------------------------- Q98 facts and eligibility (D4, D5)

def flood_eligibility(*, n_admitted: int, q98_threshold: float, n_flood: int) -> list[str]:
    """D5(b) reasons a basin-year is flood-ineligible (empty list = eligible)."""
    if n_admitted <= 0:
        return [INELIGIBLE_NO_ADMITTED]
    reasons = []
    if not (q98_threshold > 0.0):
        reasons.append(INELIGIBLE_Q98_NOT_POSITIVE)
    if n_flood < MIN_FLOOD_HOURS:
        reasons.append(INELIGIBLE_TOO_FEW)
    if n_flood / n_admitted > MAX_FLOOD_HOUR_FRACTION:
        reasons.append(INELIGIBLE_TIE_CAP)
    return reasons


def basin_flood_facts(basin_id: str, dates: np.ndarray, admitted_obs: np.ndarray, *, lead_hours: int = LEAD_HOURS):
    """Q98 facts and D5(b) eligibility of one basin-year from its admitted
    (finite) observed series: the single definition shared by the scorer's
    facts (:func:`build_flood_facts`) and the score-blind eligibility table
    (:func:`build_eligibility_table`). Returns ``(facts or None, entry)``;
    ``entry`` holds ``q98_threshold``, ``n_flood``, ``reasons`` and
    ``eligible`` (and the peak fields when the series is not empty)."""
    n_admitted = int(np.asarray(admitted_obs).size)
    if n_admitted == 0:  # the helper raises on an empty series: filter first (spec §2d)
        return None, {"q98_threshold": None, "n_flood": 0, "reasons": [INELIGIBLE_NO_ADMITTED], "eligible": False}
    f = derive_canonical_basin_q98_facts(basin_id=basin_id, date=dates, obs_m3s=admitted_obs)
    reasons = flood_eligibility(n_admitted=n_admitted, q98_threshold=f.q98_threshold, n_flood=f.n_high_flow)
    return f, {"q98_threshold": float(f.q98_threshold), "n_flood": int(f.n_high_flow),
               "flood_hour_fraction": f.n_high_flow / n_admitted,
               "observed_peak_value_m3s": float(f.observed_peak_value),
               "observed_peak_index": int(f.observed_peak_index),
               "observed_peak_obs_time": _obs_time_iso(dates[f.observed_peak_index], lead_hours),
               "reasons": reasons, "eligible": not reasons}


def build_flood_facts(series: Mapping[str, SealedBasinSeries], *, population: Sequence[str],
                      not_scored: Mapping[str, str], lead_hours: int = LEAD_HOURS):
    """Q98 facts (threshold, flood mask, peak) and eligibility of every scored
    basin, from observations only. Returns ``(facts_by_basin, record)``:
    canonical facts objects for the in-process diagnostics and the JSON record
    (counts, per-basin reasons, series fingerprints, low-eligibility flag)."""
    facts_by_basin = {}
    basins: dict[str, dict] = {}
    for basin_id in sorted(series):
        s = series[basin_id]
        if s.n_admitted != s.obs_m3s.size:
            raise WyRotationScoringError(f"basin {basin_id}: {s.n_admitted} admitted hours, {s.obs_m3s.size} values")
        f, flood = basin_flood_facts(basin_id, s.dates, s.obs_m3s, lead_hours=lead_hours)
        if f is not None:
            facts_by_basin[basin_id] = f
        basins[basin_id] = {"n_admitted": s.n_admitted, "series_sha256": series_fingerprint(s.dates, s.obs_m3s),
                            **flood}
    for basin_id, reason in sorted(not_scored.items()):
        basins[basin_id] = {"n_admitted": None, "reasons": [reason], "eligible": False}
    if set(basins) != set(population):
        raise WyRotationScoringError("flood facts do not cover exactly the development population")
    n_eligible = sum(1 for e in basins.values() if e["eligible"])
    reason_counts: dict[str, int] = {}
    for e in basins.values():
        for r in e["reasons"]:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    fraction = n_eligible / len(population) if population else 0.0
    record = {
        "n_population": len(population),
        "n_scored": len(series),
        "n_eligible": n_eligible,
        "eligible_fraction": fraction,
        "low_eligibility_flag": fraction < MIN_ELIGIBLE_FRACTION,
        "ineligible_reason_counts": reason_counts,
        "eligible_basins": sorted(b for b, e in basins.items() if e["eligible"]),
        "basins": basins,
        "rules": {"quantile": HIGH_FLOW_QUANTILE, "quantile_method": "linear", "mask": "obs >= threshold",
                  "min_flood_hours": MIN_FLOOD_HOURS, "max_flood_hour_fraction": MAX_FLOOD_HOUR_FRACTION,
                  "min_eligible_fraction": MIN_ELIGIBLE_FRACTION},
    }
    return facts_by_basin, record


def check_series_against_facts(series: Mapping[str, SealedBasinSeries], facts_record: Mapping[str, Any], *,
                               what: str) -> None:
    """G4 (and the within-run epoch identity): the scored basins and each
    basin's admitted dates + observations are bit-identical to the series the
    facts were built from. Checked before any flood diagnostic is computed."""
    stored = {b: e for b, e in facts_record["basins"].items() if e.get("series_sha256") is not None}
    if set(series) != set(stored):
        raise WyRotationScoringError(f"G4 {what}: scored basins differ from the Q98 facts' basins "
                                     f"(only here: {sorted(set(series) - set(stored))[:5]}, only in facts: "
                                     f"{sorted(set(stored) - set(series))[:5]})")
    bad = sorted(b for b, s in series.items()
                 if series_fingerprint(s.dates, s.obs_m3s) != stored[b]["series_sha256"])
    if bad:
        raise WyRotationScoringError(f"G4 {what}: admitted dates/observations differ from the Q98 facts for "
                                     f"{len(bad)} basin(s), first {bad[:5]}")


def rebuild_facts(series: Mapping[str, SealedBasinSeries], facts_record: Mapping[str, Any]) -> dict:
    """Canonical facts objects for a run whose series passed
    :func:`check_series_against_facts`: rebuilt from the bit-identical
    observations and checked equal to the stored threshold, flood count and
    peak (the baseline's facts reused for arm 2, G4)."""
    out = {}
    for basin_id, s in series.items():
        e = facts_record["basins"][basin_id]
        if s.n_admitted == 0:
            continue
        f = derive_canonical_basin_q98_facts(basin_id=basin_id, date=s.dates, obs_m3s=s.obs_m3s)
        if (float(f.q98_threshold), int(f.n_high_flow), int(f.observed_peak_index)) != (
                e["q98_threshold"], e["n_flood"], e["observed_peak_index"]):
            raise WyRotationScoringError(f"basin {basin_id}: rebuilt Q98 facts differ from the stored facts")
        out[basin_id] = f
    return out


# ---------------------------------------------------------------------------- per-basin and per-epoch scores (D1)

def score_basin(s: SealedBasinSeries, facts, *, eligible: bool, trial_id: str) -> dict[str, Any]:
    m = s.sealed_metrics
    row = {"basin_id": s.basin_id, "n_admitted": s.n_admitted, "area_km2": s.area_km2,
           "nse": m["nse"], "kge": m["kge"], "kge_r": m["kge_r"], "kge_alpha": m["kge_alpha"],
           "kge_beta": m["kge_beta"], "pbias": m["pbias"], "flood_eligible": bool(eligible),
           "n_flood": 0, "q98_nrmse": float("nan"), "high_flow_nse": float("nan"),
           "q98_rel_volume_bias": float("nan"), "peak_rel_error": float("nan"), "pbias_below_q98": float("nan")}
    if facts is None:
        return row
    d = compute_q98_configuration_basin_diagnostics(trial_id=trial_id, facts=facts, date=s.dates,
                                                    obs_m3s=s.obs_m3s, sim_m3s=s.sim_m3s)
    below = ~facts.high_flow_mask
    row.update(n_flood=int(d.n_high_flow), q98_nrmse=d.q98_normalized_rmse,
               high_flow_nse=d.high_flow_nse if d.high_flow_nse_available else float("nan"),
               q98_rel_volume_bias=d.relative_volume_bias, peak_rel_error=d.observed_peak_time_magnitude_error,
               pbias_below_q98=raw_space_metrics(s.obs_m3s[below], s.sim_m3s[below])["pbias"])
    return row


def epoch_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Median NSE over basins with finite NSE; median Q98-nRMSE over eligible
    basins with a finite value (D3)."""
    nse = np.array([r["nse"] for r in rows], dtype=np.float64)
    nse = nse[np.isfinite(nse)]
    fl = np.array([r["q98_nrmse"] for r in rows if r["flood_eligible"]], dtype=np.float64)
    fl = fl[np.isfinite(fl)]
    return {"n_basins_scored": len(rows), "n_nse_finite": int(nse.size),
            "median_nse": float(np.median(nse)) if nse.size else float("nan"),
            "n_flood_scored": int(fl.size),
            "median_q98_nrmse": float(np.median(fl)) if fl.size else float("nan")}


# ---------------------------------------------------------------------------- one run's checkpoint year

_RESUMABLE_STATUSES = ("FAILED", "STARTED")  # the runner's: only these are ever resumed (and archived)


def archived_receipt_name(job: str) -> str:
    """The runner's archive name of job ``job``'s receipt (``_archive_name`` in
    ``scripts/run_wy_rotation_cycle.py``; a test keeps the two equal)."""
    stem, suffix = RECEIPT_NAME.rsplit(".", 1)
    return f"{stem}.job{job}.{suffix}"


def resume_chain(config_dir, receipt: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    """Every job of the run's resume chain, oldest first, ending with the job
    that wrote the current receipt (Omri, 2026-11-19, gate-4 Q1: a score file
    records the training commit of every job in the chain, in order). Each
    earlier job is read from its archived receipt, which must be the bytes the
    newer receipt cites (``previous_receipt_sha256``), carry the commit it cites
    (``previous_git_commit``), be that job's own (``slurm_job_id``), belong to
    the same run and be FAILED or STARTED. A missing, changed or inconsistent
    archive, a loop, or a job without a recorded commit is refused. Each entry
    carries the job's ``stopped_as_defect`` marker (None when absent), so a
    score file shows that a defect stop happened in the chain (delta review
    D5(a))."""
    config_dir = Path(config_dir)
    receipt_path = config_dir / RECEIPT_NAME
    current = dict(receipt) if receipt is not None else _read_json(receipt_path, "run receipt")

    def entry(rec, sha):
        commit = rec.get("git_commit")
        if not isinstance(commit, str) or not commit:
            raise WyRotationScoringError(f"resume chain of {config_dir}: job {rec.get('slurm_job_id')!r} has no "
                                         "recorded training commit")
        return {"slurm_job_id": rec.get("slurm_job_id"), "git_commit": commit, "receipt_sha256": sha,
                "status": rec.get("status"), "stopped_as_defect": rec.get("stopped_as_defect")}

    chain = [entry(current, sha256_file(receipt_path))]
    seen: set[str] = set()
    newer = current
    while newer.get("resumed_from_job") is not None:
        job = str(newer["resumed_from_job"])
        if job in seen:
            raise WyRotationScoringError(f"resume chain of {config_dir}: job {job} appears twice")
        seen.add(job)
        path = config_dir / archived_receipt_name(job)
        if not path.is_file():
            raise WyRotationScoringError(f"resume chain of {config_dir}: the receipt resumes job {job}, but "
                                         f"{path.name} is missing")
        sha = sha256_file(path)
        if sha != newer.get("previous_receipt_sha256"):
            raise WyRotationScoringError(f"resume chain: {path.name} is not the receipt the newer job cites "
                                         "(sha256 differs)")
        older = _read_json(path, "archived run receipt")
        recorded = "none" if older.get("slurm_job_id") is None else str(older["slurm_job_id"])
        if (recorded != job or older.get("run_id") != current.get("run_id")
                or older.get("status") not in _RESUMABLE_STATUSES
                or older.get("status") != newer.get("previous_status")
                or older.get("git_commit") != newer.get("previous_git_commit")):
            raise WyRotationScoringError(f"resume chain: {path.name} does not match the job, run, status and "
                                         "commit the newer receipt cites")
        chain.append(entry(older, sha))
        newer = older
    return chain[::-1]


def _load_run_inputs(config_dir: Path, population: Sequence[str], budget_epochs: int) -> dict[str, Any]:
    receipt_path, evidence_path = config_dir / RECEIPT_NAME, config_dir / EVIDENCE_NAME
    receipt = _read_json(receipt_path, "run receipt")
    if receipt.get("status") != "COMPLETE":
        raise WyRotationScoringError(f"{receipt_path}: status {receipt.get('status')!r}; only a COMPLETE run "
                                     "(training info recorded, score-blind evidence passed) is scored")
    if receipt.get("is_canary") is not False:
        raise WyRotationScoringError(f"{receipt_path}: is_canary={receipt.get('is_canary')!r}; a canary (or a "
                                     "receipt that does not say it is a campaign run) is never scored")
    if receipt.get("epochs") != budget_epochs:
        raise WyRotationScoringError(f"{receipt_path}: {receipt.get('epochs')!r} epochs; a scored run must have "
                                     f"the campaign budget of {budget_epochs}")
    evidence = _read_json(evidence_path, "score-blind evidence")
    if evidence.get("passed") is not True:
        raise WyRotationScoringError(f"{evidence_path}: score-blind evidence did not pass")
    missing = list(evidence.get("missing_basins") or [])
    counts = evidence.get("missing_basin_admissible_target_counts") or {}
    unexplained = sorted(b for b in missing if counts.get(b) != 0)
    if unexplained:  # G1 (11-17 erratum): missing only with 0 admissible targets, independently counted
        raise WyRotationScoringError(f"G1: missing basin(s) without a zero admissible-target count: {unexplained}")
    if not set(missing) <= set(population):
        raise WyRotationScoringError("G1: evidence lists missing basins outside the population")
    nh_run_dir = Path(receipt["nh_run_dir"])
    config = nre._read_run_config(nh_run_dir)
    cycle = int(receipt["cycle"])
    if int(config.get(nre.ROTATION_CYCLE_CONFIG_KEY, -1)) != cycle:
        raise WyRotationScoringError(f"{nh_run_dir}: config cycle differs from the receipt's cycle {cycle}")
    if config.get("epochs") != budget_epochs:  # the cycle config the run trained with
        raise WyRotationScoringError(f"{nh_run_dir}: cycle config has epochs={config.get('epochs')!r}, not the "
                                     f"campaign budget {budget_epochs}")
    arm, lam = trk._arm(config)
    return {"receipt": receipt, "receipt_sha256": sha256_file(receipt_path), "evidence_sha256":
            sha256_file(evidence_path), "missing": missing, "nh_run_dir": nh_run_dir, "config": config,
            "cycle": cycle, "arm": arm, "lam": lam, "epochs": list(range(1, budget_epochs + 1)),
            "package_root": config.get("data_dir"), "resume_chain": resume_chain(config_dir, receipt)}


def _check_facts_binding(facts_path: Path, facts_record: Mapping[str, Any], *, inp: Mapping[str, Any],
                         run_id: str, out_root: Path) -> None:
    """R3 (gate-4 review): the cycle's facts file must come from the cycle's
    baseline run. A baseline run may reuse only the facts it wrote itself (a
    rerun after a failed gate); an arm-2 run needs the baseline's score file,
    which exists only after all its epochs passed G1-G4, and that score file
    must cite this exact facts file. The explained-missing set must equal the
    facts' (11-17 erratum)."""
    if int(facts_record["cycle"]) != inp["cycle"]:
        raise WyRotationScoringError(f"{facts_path} belongs to cycle {facts_record['cycle']}")
    source = str(facts_record.get("source_run_id"))
    if inp["arm"] == "base":
        if source != run_id:
            raise WyRotationScoringError(f"{facts_path} was written by baseline run {source!r}; baseline run "
                                         f"{run_id!r} cannot adopt it")
    else:
        base_scores = out_root / source / RUN_SCORES_NAME
        if not base_scores.is_file():
            raise WyRotationScoringError(f"{facts_path} comes from baseline run {source!r}, which has no score file "
                                         f"{base_scores} (it failed a gate or is unscored); arm 2 is not scored "
                                         "against it")
        base = load_run_scores(base_scores)
        if (base.get("arm") != "base" or base.get("run_id") != source or int(base.get("cycle", -1)) != inp["cycle"]
                or base.get("facts_sha256") != sha256_file(facts_path)):
            raise WyRotationScoringError(f"{base_scores} is not the cycle-{inp['cycle']} baseline score file that "
                                         f"cites {facts_path}")
    if sorted(inp["missing"]) != list(facts_record["missing_explained"]):
        raise WyRotationScoringError(f"G1: explained-missing basins differ from those of the facts file "
                                     f"{facts_path} (the 11-17 erratum requires one set per cycle)")


def score_run_checkpoint_year(*, config_dir, population: Sequence[str], cycle_output_dir, project_root=None,
                              target_variable: str = TARGET, lead_hours: int = LEAD_HOURS,
                              budget_epochs: int = DEFAULT_BUDGET_EPOCHS, eligibility_csv=None) -> dict[str, Any]:
    """Score every epoch of one COMPLETE campaign rotation run on its
    checkpoint year. Canary receipts and runs whose receipt or cycle config
    has another epoch count than ``budget_epochs`` (the campaign budget) are
    refused.

    The baseline run of a cycle must be scored first: its first epoch's
    observations give the cycle's Q98 facts and eligibility, written to
    ``cycle_output_dir/checkpoint_flood_facts.json`` before any score file.
    An arm-2 run reuses that file after the G4 check, and only once the
    baseline that wrote it has its score file (:func:`_check_facts_binding`).
    Score files are written only after every epoch passed G1-G4; nothing is
    overwritten.

    ``eligibility_csv`` (Omri, 2026-11-19, gate-4 Q3; required, delta review
    D1): the score-blind eligibility table, built from observations alone
    before any score file exists, is the formally published round-E eligible
    count. Without it nothing is scored (refused before any facts or score
    file is written). The cycle is reconciled against the facts
    (:func:`reconcile_eligibility`) before any score file is written, and the
    run is refused (no score file) unless the facts' ``n_eligible`` and
    ``eligible_fraction`` equal the reconciled published count."""
    spec_sha = verify_spec_sha256(project_root)
    budget_epochs = _check_budget(budget_epochs)
    if eligibility_csv is None:
        raise WyRotationScoringError("round E (Omri 2026-11-19, gate-4 Q3): the published score-blind eligibility "
                                     "table (eligibility_csv) is required; nothing is scored without it")
    config_dir, out_root = Path(config_dir), Path(cycle_output_dir)
    inp = _load_run_inputs(config_dir, population, budget_epochs)
    if inp["package_root"] is None:
        raise WyRotationScoringError(f"{inp['nh_run_dir']}: config has no data_dir")
    run_id = inp["receipt"]["run_id"]
    run_out = out_root / run_id
    if (run_out / RUN_SCORES_NAME).exists():
        raise WyRotationScoringError(f"{run_out / RUN_SCORES_NAME} exists; a run is scored once")
    missing = set(inp["missing"])
    expected = [b for b in population if b not in missing]
    facts_path = out_root / FACTS_NAME
    facts_record = _read_json(facts_path, "Q98 facts") if facts_path.exists() else None
    if inp["arm"] != "base" and facts_record is None:
        raise WyRotationScoringError(f"{facts_path} missing: score the cycle's baseline run first (spec G4: Q98 "
                                     "facts come from the baseline run and are reused for arm 2)")
    if facts_record is not None:
        _check_facts_binding(facts_path, facts_record, inp=inp, run_id=run_id, out_root=out_root)

    facts_by_basin = None
    area_excluded_ref = None if facts_record is None else facts_record["area_excluded"]
    per_epoch: dict[int, dict] = {}
    epoch_rows: dict[int, list] = {}
    pickle_sha: dict[int, str] = {}
    for epoch in inp["epochs"]:
        pkl = period_results_path(inp["nh_run_dir"], CHECKPOINT_PERIOD, epoch)
        before = sha256_file(pkl)
        got = sealed_period_series(run_dir=inp["nh_run_dir"], period=CHECKPOINT_PERIOD, epoch=epoch,
                                   package_root=inp["package_root"], basin_ids=expected,
                                   target_variable=target_variable, lead_hours=lead_hours)
        series = got["series"]
        # G1: the results hold exactly the population minus the explained-missing set
        if set(got["results_keys"]) != set(expected):
            raise WyRotationScoringError(f"G1 epoch {epoch}: results keys differ from population minus the "
                                         "explained-missing basins")
        area_excluded = sorted(e["basin_id"] for e in got["sealed"]["area_derivation_excluded"])
        if set(series) | set(area_excluded) != set(expected):
            raise WyRotationScoringError(f"G1 epoch {epoch}: scored + area-excluded basins differ from expected")
        # G3: one area-exclusion list for all epochs (and arms, through the facts file)
        if area_excluded_ref is None:
            area_excluded_ref = area_excluded
        elif area_excluded != list(area_excluded_ref):
            raise WyRotationScoringError(f"G3 epoch {epoch}: area-exclusion list {area_excluded} differs from "
                                         f"{list(area_excluded_ref)}")
        # G2: finite predictions at every admitted hour
        nonfinite = sorted(b for b, s in series.items() if s.n_sim_nonfinite_at_admitted)
        if nonfinite:
            raise WyRotationScoringError(f"G2 epoch {epoch}: non-finite predictions at admitted hours in "
                                         f"{len(nonfinite)} basin(s), first {nonfinite[:5]}")
        if facts_record is None:  # baseline, first epoch: facts + eligibility from observations, written first
            not_scored = {b: NOT_SCORED_MISSING for b in missing}
            not_scored.update({b: NOT_SCORED_AREA for b in area_excluded})
            facts_by_basin, body = build_flood_facts(series, population=population, not_scored=not_scored,
                                                     lead_hours=lead_hours)
            facts_record = {"schema": "flashnh_wy_rotation_checkpoint_flood_facts", "schema_version": 1,
                            "scorer_version": SCORER_VERSION, "spec_sha256": spec_sha, "cycle": inp["cycle"],
                            "period": CHECKPOINT_PERIOD, "source_run_id": run_id, "source_epoch": epoch,
                            "source_results_sha256": before, "missing_explained": sorted(missing),
                            "area_excluded": area_excluded, **body}
            _write_new(facts_path, facts_record)
        # G4 / epoch identity, before any flood diagnostic
        check_series_against_facts(series, facts_record, what=f"{run_id} epoch {epoch}")
        if facts_by_basin is None:
            facts_by_basin = rebuild_facts(series, facts_record)
        eligible = facts_record["basins"]
        rows = [score_basin(series[b], facts_by_basin.get(b), eligible=eligible[b]["eligible"],
                            trial_id=f"{run_id}/epoch{epoch:03d}") for b in sorted(series)]
        if sha256_file(pkl) != before:
            raise WyRotationScoringError(f"{pkl} changed while it was scored")
        pickle_sha[epoch] = before
        epoch_rows[epoch] = rows
        per_epoch[epoch] = epoch_summary(rows)

    # Q3: after the facts exist, before any score file
    rec = reconcile_eligibility(cycle_output_dir=out_root, eligibility_csv=eligibility_csv,
                                project_root=project_root)
    published = rec["published_round_e"]
    if not published["facts_agree"]:
        raise WyRotationScoringError(
            f"round E: the facts' n_eligible/eligible_fraction ({facts_record['n_eligible']}, "
            f"{facts_record['eligible_fraction']}) disagree with the published table count "
            f"({published['n_eligible']}, {published['eligible_fraction']}); see "
            f"{out_root / RECONCILIATION_NAME}. No score file is written")
    reconciliation_sha = sha256_file(out_root / RECONCILIATION_NAME)

    files = {}
    for epoch, rows in epoch_rows.items():
        name = f"per_basin_epoch{epoch:03d}.json"
        files[str(epoch)] = {"file": name, "sha256": _write_new(run_out / name, {"run_id": run_id, "epoch": epoch,
                                                                                 "rows": rows})}
    summary = {
        "schema": "flashnh_wy_rotation_checkpoint_scores", "schema_version": 1, "scorer_version": SCORER_VERSION,
        "spec_sha256": spec_sha, "run_id": run_id, "cycle": inp["cycle"], "arm": inp["arm"], "lambda": inp["lam"],
        "period": CHECKPOINT_PERIOD, "nh_run_dir": str(inp["nh_run_dir"]), "receipt_sha256": inp["receipt_sha256"],
        "evidence_sha256": inp["evidence_sha256"], "facts_sha256": sha256_file(facts_path),
        "resume_chain": inp["resume_chain"], "training_commits": [j["git_commit"] for j in inp["resume_chain"]],
        "round_e_reconciliation_sha256": reconciliation_sha,
        "budget_epochs": budget_epochs, "epochs": inp["epochs"], "results_sha256": {str(e): s for e, s in pickle_sha.items()},
        "per_basin_files": files,
        "gates": {"G1_missing_explained": sorted(missing), "G1_n_scored_or_area_excluded": len(expected),
                  "G2_nonfinite_predictions": 0, "G3_area_excluded": list(area_excluded_ref),
                  "G4_identical_to_facts": True},
        "curves": {str(e): s for e, s in per_epoch.items()},
    }
    _write_new(run_out / RUN_SCORES_NAME, summary)
    return summary


def load_run_scores(path) -> dict[str, Any]:
    """Read a run's checkpoint score summary; refuse a different spec."""
    s = _read_json(path, "run checkpoint scores")
    if s.get("spec_sha256") != SPEC_SHA256 or s.get("period") != CHECKPOINT_PERIOD:
        raise WyRotationScoringError(f"{path}: not a checkpoint-year score file of spec {SPEC_SHA256[:12]}")
    return s


# ---------------------------------------------------------------------------- picks (D3, D6, D7, D13, rounds C/D)

def pick_checkpoint(candidates: Sequence[Mapping[str, Any]], *, rule: str,
                    budget_epochs: int = DEFAULT_BUDGET_EPOCHS) -> dict[str, Any]:
    """One pick among ``candidates`` (dicts with ``epoch``, ``lambda``,
    ``value``). ``rule`` "nse" takes the highest value, "flood" the lowest.
    Non-finite values never win. An exact float64 tie goes to the earliest
    epoch, then the smallest lambda (D6). ``budget_edge`` is D7 for
    ``budget_epochs``."""
    edge = budget_edge_epochs(budget_epochs)
    if rule not in PICK_RULES:
        raise WyRotationScoringError(f"unknown pick rule {rule!r}")
    finite = [c for c in candidates if c["value"] is not None and math.isfinite(float(c["value"]))]
    if not finite:
        raise WyRotationScoringError(f"{rule} pick: no candidate has a finite value")
    values = [float(c["value"]) for c in finite]
    best = max(values) if rule == "nse" else min(values)
    tied = [c for c in finite if float(c["value"]) == best]
    win = min(tied, key=lambda c: (int(c["epoch"]), float(c["lambda"])))
    lam = float(win["lambda"])
    return {"rule": rule, "epoch": int(win["epoch"]), "lambda": lam, "value": best, "n_candidates": len(candidates),
            "n_tied": len(tied), "budget_edge": int(win["epoch"]) in edge,
            "strength_edge": lam > 0 and lam in STRENGTH_EDGE_LAMBDAS}


_CURVE_KEY = {"nse": "median_nse", "flood": "median_q98_nrmse"}


def _cycle_reconciliation_sha(out_root: Path, what: str) -> str:
    """The sha256 of the cycle's round-E reconciliation file (delta review D1)."""
    path = Path(out_root) / RECONCILIATION_NAME
    if not path.is_file():
        raise WyRotationScoringError(f"{what}: {path} is missing; a cycle without the published round-E "
                                     "reconciliation has no usable score file")
    return sha256_file(path)


def _check_round_e(scores: Mapping[str, Any], reconciliation_sha: str, what: str) -> None:
    """Refuse a score file that cites no, or another, round-E reconciliation (D1)."""
    cited = scores.get("round_e_reconciliation_sha256")
    if cited is None:
        raise WyRotationScoringError(f"{what}: the score file cites no round-E reconciliation (scored without "
                                     "the published eligibility table)")
    if cited != reconciliation_sha:
        raise WyRotationScoringError(f"{what}: the score file cites another round-E reconciliation than the "
                                     f"cycle's {RECONCILIATION_NAME}")


def _load_cycle_run_scores(path, out_root: Path) -> tuple[dict[str, Any], str]:
    """A written score file of this cycle's output folder and its sha256 (R4)."""
    if isinstance(path, Mapping):
        raise WyRotationScoringError("picks read the written score files: pass paths, not in-memory score dicts")
    path = Path(path)
    if not path.is_file():
        raise WyRotationScoringError(f"run score file {path} not found")
    digest = sha256_file(path)
    s = load_run_scores(path)
    canonical = out_root / str(s.get("run_id")) / RUN_SCORES_NAME
    if path.resolve() != canonical.resolve():
        raise WyRotationScoringError(f"{path} is not the score file {canonical} of run {s.get('run_id')!r} in "
                                     f"the cycle output folder {out_root}")
    if sha256_file(path) != digest:
        raise WyRotationScoringError(f"{path} changed while it was read")
    return s, digest


def make_arm_picks(*, cycle: int, run_scores_paths: Sequence, cycle_output_dir, project_root=None,
                   budget_epochs: int = DEFAULT_BUDGET_EPOCHS) -> dict[str, Any]:
    """Both picks of one arm of one cycle, written once to
    ``checkpoint_picks_<family>.json``, from the runs' written score files
    (``<cycle_output_dir>/<run_id>/checkpoint_scores.json``; each file's
    sha256 is recorded). Every run must have exactly ``budget_epochs`` epochs
    (the campaign budget). The baseline arm has one run (lambda 0, 12
    candidates); arm 2 has one run per lambda of :data:`LAMBDA_GRID`
    (36 candidates), plus each lambda's own 12-epoch picks (round D,
    descriptive only)."""
    spec_sha = verify_spec_sha256(project_root)
    budget_epochs = _check_budget(budget_epochs)
    out_root = Path(cycle_output_dir)
    if isinstance(run_scores_paths, (str, os.PathLike, Mapping)) or not run_scores_paths:
        raise WyRotationScoringError("run_scores_paths must be a non-empty sequence of score-file paths")
    loaded = [_load_cycle_run_scores(p, out_root) for p in run_scores_paths]
    run_scores = [s for s, _ in loaded]
    scores_sha = {s["run_id"]: d for s, d in loaded}
    if len(scores_sha) != len(loaded):
        raise WyRotationScoringError("the same run's score file was given twice")
    budget_list = list(range(1, budget_epochs + 1))
    families = {"base" if s["arm"] == "base" else "magw" for s in run_scores}
    if len(families) != 1:
        raise WyRotationScoringError(f"run scores mix arm families {sorted(families)}")
    family = families.pop()
    lambdas = sorted(float(s["lambda"]) for s in run_scores)
    want = [0.0] if family == "base" else sorted(LAMBDA_GRID)
    if lambdas != want:
        raise WyRotationScoringError(f"arm {family}: run lambdas {lambdas}, expected {want}")
    facts_path = out_root / FACTS_NAME
    facts_sha = sha256_file(facts_path)
    facts = _read_json(facts_path, "Q98 facts")
    reconciliation_sha = _cycle_reconciliation_sha(out_root, f"arm picks of cycle {cycle}")
    for s in run_scores:
        if int(s["cycle"]) != int(cycle) or s.get("spec_sha256") != spec_sha:
            raise WyRotationScoringError(f"{s['run_id']}: cycle/spec differ from cycle {cycle}, spec {spec_sha[:12]}")
        if s["facts_sha256"] != facts_sha:
            raise WyRotationScoringError(f"{s['run_id']}: scored against other Q98 facts than {facts_path}")
        _check_round_e(s, reconciliation_sha, s["run_id"])
        if (list(s["epochs"]) != budget_list or s.get("budget_epochs") != budget_epochs
                or sorted(int(e) for e in s["curves"]) != budget_list):
            raise WyRotationScoringError(f"{s['run_id']}: epochs {list(s['epochs'])} are not the campaign budget "
                                         f"1..{budget_epochs}")
        if list(s["gates"]["G3_area_excluded"]) != list(facts["area_excluded"]):
            raise WyRotationScoringError(f"G3 {s['run_id']}: area-exclusion list differs across arms")

    def candidates(rule, subset):
        return [{"epoch": int(e), "lambda": float(s["lambda"]), "value": c[_CURVE_KEY[rule]], "run_id": s["run_id"]}
                for s in subset for e, c in s["curves"].items()]

    picks = {}
    for rule in PICK_RULES:
        p = pick_checkpoint(candidates(rule, run_scores), rule=rule, budget_epochs=budget_epochs)
        p["run_id"] = next(s["run_id"] for s in run_scores if float(s["lambda"]) == p["lambda"])
        picks[rule] = p
    per_lambda = {}
    if family == "magw":
        for s in run_scores:
            per_lambda[f"{float(s['lambda']):g}"] = {
                rule: pick_checkpoint(candidates(rule, [s]), rule=rule, budget_epochs=budget_epochs)
                for rule in PICK_RULES}
    record = {
        "schema": "flashnh_wy_rotation_checkpoint_picks", "schema_version": 1, "scorer_version": SCORER_VERSION,
        "spec_sha256": spec_sha, "cycle": int(cycle), "family": family, "period": CHECKPOINT_PERIOD,
        "budget_epochs": budget_epochs,
        "facts_sha256": facts_sha, "round_e_reconciliation_sha256": reconciliation_sha,
        "flood_low_eligibility_flag": bool(facts["low_eligibility_flag"]),
        "flood_n_eligible": facts["n_eligible"], "flood_eligible_fraction": facts["eligible_fraction"],
        "runs": [{"run_id": s["run_id"], "lambda": float(s["lambda"]), "scores_file": f"{s['run_id']}/"
                  f"{RUN_SCORES_NAME}", "scores_sha256": scores_sha[s["run_id"]],
                  "results_sha256": s["results_sha256"]} for s in run_scores],
        "picks": picks, "per_lambda_picks": per_lambda,
    }
    path = out_root / PICKS_TEMPLATE.format(family=family)
    record["file"] = path.name
    _write_new(path, record)
    return record


def cross_cycle_flags(pick_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """D7 and round C across cycles: per arm family and pick rule, the cycles
    flagged budget edge (>= 2 -> "possibly budget-limited") and, for arm 2,
    the cycles at each strength edge (>= 2 at one edge -> "possibly too
    narrow"). Labels only; nothing is rerun."""
    out: dict[str, Any] = {}
    seen = set()
    for rec in pick_records:
        key = (rec["family"], int(rec["cycle"]))
        if key in seen:
            raise WyRotationScoringError(f"two pick records for {key}")
        seen.add(key)
    for family in sorted({r["family"] for r in pick_records}):
        recs = [r for r in pick_records if r["family"] == family]
        out[family] = {}
        for rule in PICK_RULES:
            budget = sorted(int(r["cycle"]) for r in recs if r["picks"][rule]["budget_edge"])
            entry: dict[str, Any] = {"n_cycles": len(recs), "budget_edge_cycles": budget,
                                     "possibly_budget_limited": len(budget) >= CROSS_CYCLE_FLAG_MIN}
            if family == "magw":
                edges = {f"{lam:g}": sorted(int(r["cycle"]) for r in recs
                                            if r["picks"][rule]["lambda"] == lam) for lam in STRENGTH_EDGE_LAMBDAS}
                entry["strength_edge_cycles"] = edges
                entry["possibly_too_narrow"] = any(len(v) >= CROSS_CYCLE_FLAG_MIN for v in edges.values())
            out[family][rule] = entry
    return out


# ---------------------------------------------------------------------------- W&B stage unlocks (B, C)

def _derivation(**args) -> tuple:
    """The issuer's arguments as sorted (name, str-or-None) pairs: what
    :func:`reverify_unlock` calls the issuer with again."""
    return tuple(sorted((k, None if v is None else str(v)) for k, v in args.items()))


def unlock_stage_b(*, config_dir, run_scores_path, project_root=None,
                   budget_epochs: int = DEFAULT_BUDGET_EPOCHS) -> trk.StageUnlock:
    """Stage B (checkpoint-year scores) opens for one run only when the spec is
    the frozen one, the run is a COMPLETE campaign run with passed score-blind
    evidence (stage A's training info came first) and its checkpoint score
    file exists in the run's own folder and cites these receipt and evidence
    bytes, the campaign budget and the cycle's current round-E
    reconciliation (delta review D1)."""
    budget_epochs = _check_budget(budget_epochs)
    verify_spec_sha256(project_root)
    config_dir, scores_path = Path(config_dir), Path(run_scores_path)
    receipt_path, evidence_path = config_dir / RECEIPT_NAME, config_dir / EVIDENCE_NAME
    receipt = _read_json(receipt_path, "run receipt")
    if receipt.get("status") != "COMPLETE" or receipt.get("is_canary") is not False:
        raise WyRotationScoringError("stage B: the run receipt is not that of a COMPLETE campaign run")
    if _read_json(evidence_path, "score-blind evidence").get("passed") is not True:
        raise WyRotationScoringError("stage B: score-blind evidence did not pass")
    scores = load_run_scores(scores_path)
    if scores.get("run_id") != receipt.get("run_id"):
        raise WyRotationScoringError("stage B: score file belongs to another run")
    if scores_path.name != RUN_SCORES_NAME or scores_path.parent.name != scores["run_id"]:
        raise WyRotationScoringError(f"stage B: {scores_path} is not <cycle output>/{scores['run_id']}/"
                                     f"{RUN_SCORES_NAME}")
    if (scores.get("receipt_sha256") != sha256_file(receipt_path)
            or scores.get("evidence_sha256") != sha256_file(evidence_path)
            or int(scores.get("cycle", -1)) != int(receipt["cycle"]) or scores.get("budget_epochs") != budget_epochs):
        raise WyRotationScoringError("stage B: the score file does not cite this run's receipt, evidence, cycle "
                                     "and campaign budget")
    if scores.get("resume_chain") != resume_chain(config_dir, receipt):
        raise WyRotationScoringError("stage B: the score file's resume chain is not the run's (an archived "
                                     "receipt is missing or changed)")
    out_root = scores_path.parent.parent
    _check_round_e(scores, _cycle_reconciliation_sha(out_root, "stage B"), "stage B")
    spec_path = Path(project_root or _PROJECT_ROOT) / SPEC_RELATIVE_PATH
    files = (spec_path, receipt_path, evidence_path, scores_path, out_root / RECONCILIATION_NAME)
    return trk.StageUnlock(stage="B", flashnh_run_id=receipt["run_id"], cycle=int(receipt["cycle"]),
                           evidence=tuple((str(p), sha256_file(p)) for p in files),
                           derivation=_derivation(config_dir=config_dir, run_scores_path=scores_path,
                                                  project_root=project_root, budget_epochs=budget_epochs))


def unlock_stage_c(*, cycle: int, cycle_output_dir, flashnh_run_id: str, project_root=None,
                   budget_epochs: int = DEFAULT_BUDGET_EPOCHS) -> trk.StageUnlock:
    """Stage C (assessment-year scores) lock: opens only when a pick record
    with both picks exists for every arm family in
    :data:`REQUIRED_ARM_FAMILIES`, for this cycle, the frozen spec and the
    campaign budget, and the run is one of the recorded runs. Every score file
    a pick record cites must still have its recorded sha256 and cite the
    cycle's current round-E reconciliation, as must the pick record (delta
    review D3). Nothing in this build scores an assessment year; the lock is
    implemented and tested only."""
    budget_epochs = _check_budget(budget_epochs)
    spec_sha = verify_spec_sha256(project_root)
    out_root = Path(cycle_output_dir)
    reconciliation_sha = _cycle_reconciliation_sha(out_root, "stage C locked")
    evidence = [(str(Path(project_root or _PROJECT_ROOT) / SPEC_RELATIVE_PATH), spec_sha),
                (str(out_root / RECONCILIATION_NAME), reconciliation_sha)]
    run_ids = set()
    for family in REQUIRED_ARM_FAMILIES:
        path = out_root / PICKS_TEMPLATE.format(family=family)
        if not path.is_file():
            raise WyRotationScoringError(f"stage C locked: no recorded {family} picks for cycle {cycle} ({path})")
        rec = _read_json(path, "pick record")
        if rec.get("spec_sha256") != spec_sha or int(rec.get("cycle", -1)) != int(cycle) or rec.get(
                "family") != family or rec.get("budget_epochs") != budget_epochs:
            raise WyRotationScoringError(f"stage C locked: {path} is not the {family} pick record of cycle {cycle} "
                                         f"at the campaign budget {budget_epochs}")
        for rule in PICK_RULES:
            p = (rec.get("picks") or {}).get(rule) or {}
            if not isinstance(p.get("epoch"), int):
                raise WyRotationScoringError(f"stage C locked: {path} has no recorded {rule} pick")
        if rec.get("round_e_reconciliation_sha256") != reconciliation_sha:
            raise WyRotationScoringError(f"stage C locked: {path} cites no, or another, round-E reconciliation "
                                         f"than the cycle's {RECONCILIATION_NAME}")
        runs = rec.get("runs")
        if not isinstance(runs, list) or not runs:
            raise WyRotationScoringError(f"stage C locked: {path} records no runs")
        for r in runs:  # D3: every cited score file still has its recorded bytes
            run_id = r.get("run_id")
            if not isinstance(run_id, str) or r.get("scores_file") != f"{run_id}/{RUN_SCORES_NAME}" or not \
                    isinstance(r.get("scores_sha256"), str):
                raise WyRotationScoringError(f"stage C locked: {path} run {run_id!r} does not cite its score file "
                                             f"<run_id>/{RUN_SCORES_NAME} and its sha256")
            spath = out_root / r["scores_file"]
            if not spath.is_file() or sha256_file(spath) != r["scores_sha256"]:
                raise WyRotationScoringError(f"stage C locked: {spath} is missing or differs from the sha256 the "
                                             f"{family} pick record cites")
            s = load_run_scores(spath)
            if s.get("run_id") != run_id:
                raise WyRotationScoringError(f"stage C locked: {spath} is the score file of another run")
            _check_round_e(s, reconciliation_sha, f"stage C locked: {run_id}")
            run_ids.add(run_id)
        evidence.append((str(path), sha256_file(path)))
    if flashnh_run_id not in run_ids:
        raise WyRotationScoringError(f"stage C: {flashnh_run_id} is not a run of the recorded picks")
    return trk.StageUnlock(stage="C", flashnh_run_id=flashnh_run_id, cycle=int(cycle), evidence=tuple(evidence),
                           derivation=_derivation(cycle=int(cycle), cycle_output_dir=out_root,
                                                  flashnh_run_id=flashnh_run_id, project_root=project_root,
                                                  budget_epochs=budget_epochs))


_DERIVATION_KEYS = {"B": ("budget_epochs", "config_dir", "project_root", "run_scores_path"),
                    "C": ("budget_epochs", "cycle", "cycle_output_dir", "flashnh_run_id", "project_root")}


def reverify_unlock(unlock: trk.StageUnlock) -> None:
    """Called by the tracker before every stage-B/C send. Raise unless every
    file behind ``unlock`` still has its recorded sha256 and the stage's
    precondition still holds: the issuer (:func:`unlock_stage_b` /
    :func:`unlock_stage_c`) is called again with the unlock's ``derivation``
    and must return an identical unlock. A hand-built unlock (spec-only
    evidence, a made-up derivation) is refused."""
    if not isinstance(unlock, trk.StageUnlock):
        raise WyRotationScoringError("not a StageUnlock")
    for path, digest in unlock.evidence:
        if not Path(path).is_file() or sha256_file(path) != digest:
            raise WyRotationScoringError(f"stage {unlock.stage} unlock evidence {path} changed or vanished")
    keys = _DERIVATION_KEYS.get(unlock.stage)
    if keys is None:
        raise WyRotationScoringError(f"stage {unlock.stage!r} has no unlock")
    try:
        args = dict(unlock.derivation)
    except (TypeError, ValueError) as exc:
        raise WyRotationScoringError(f"stage {unlock.stage} unlock has a malformed derivation") from exc
    if tuple(sorted(args)) != keys or len(args) != len(unlock.derivation):
        raise WyRotationScoringError(f"stage {unlock.stage} unlock derivation has keys {sorted(args)}, expected "
                                     f"{list(keys)}")
    try:
        if unlock.stage == "B":
            again = unlock_stage_b(config_dir=args["config_dir"], run_scores_path=args["run_scores_path"],
                                   project_root=args["project_root"], budget_epochs=int(args["budget_epochs"]))
        else:
            again = unlock_stage_c(cycle=int(args["cycle"]), cycle_output_dir=args["cycle_output_dir"],
                                   flashnh_run_id=args["flashnh_run_id"], project_root=args["project_root"],
                                   budget_epochs=int(args["budget_epochs"]))
    except (WyRotationScoringError, TypeError, ValueError, KeyError) as exc:
        raise WyRotationScoringError(f"stage {unlock.stage} precondition does not hold: {exc}") from exc
    if again != unlock:
        raise WyRotationScoringError(f"stage {unlock.stage} unlock differs from a fresh derivation by "
                                     f"unlock_stage_{unlock.stage.lower()}; it was not issued for this evidence")


def log_checkpoint_year(tracker: "trk.RotationTracker", unlock: trk.StageUnlock, run_scores: Mapping[str, Any],
                        picks: Optional[Mapping[str, Any]] = None) -> None:
    """Mirror one run's checkpoint-year curves (per epoch) and, when given,
    its arm's pick record into W&B stage B."""
    if unlock.stage != "B" or unlock.flashnh_run_id != run_scores["run_id"]:
        raise WyRotationScoringError("log_checkpoint_year needs this run's stage-B unlock")
    for e, c in sorted(run_scores["curves"].items(), key=lambda kv: int(kv[0])):
        tracker.log_stage(unlock, {f"ckpt/{k}": c[k] for k in trk.STAGE_B_EPOCH_KEYS}, epoch=int(e))
    facts_summary = {"ckpt/n_area_excluded": len(run_scores["gates"]["G3_area_excluded"]),
                     "ckpt/n_missing_explained": len(run_scores["gates"]["G1_missing_explained"])}
    if picks is not None:
        if run_scores["run_id"] not in {r["run_id"] for r in picks["runs"]}:
            raise WyRotationScoringError("pick record does not include this run")
        facts_summary.update({"ckpt/n_flood_eligible": picks["flood_n_eligible"],
                              "ckpt/flood_eligible_fraction": picks["flood_eligible_fraction"],
                              "ckpt/flood_low_eligibility_flag": picks["flood_low_eligibility_flag"]})
        for rule in PICK_RULES:
            p = picks["picks"][rule]
            facts_summary.update({f"ckpt/pick_{rule}_epoch": p["epoch"], f"ckpt/pick_{rule}_lambda": p["lambda"],
                                  f"ckpt/pick_{rule}_value": p["value"],
                                  f"ckpt/pick_{rule}_budget_edge": p["budget_edge"],
                                  f"ckpt/pick_{rule}_strength_edge": p["strength_edge"]})
    tracker.set_stage_summary(unlock, facts_summary)


# ---------------------------------------------------------------------------- score-blind eligibility table

#: Column layout consumed by the reference-panel freeze (``--eligibility <csv>``).
ELIGIBILITY_COLUMNS = ("STAID", "water_year", "eligible", "flood_eligible", "reason")
ELIGIBILITY_TABLE_NAME = "wy_rotation_eligibility_v001.csv"
ELIGIBILITY_MANIFEST_NAME = "wy_rotation_eligibility_v001.manifest.json"
REASON_SEPARATOR = ";"


def _eligibility_sealings(split_contract, *, project_root=None) -> dict:
    """Water year -> the sealing of the cycle whose checkpoint year it is,
    resolved through the same pinned-contract path the sealed loader uses."""
    import pandas as pd

    from .wy_rotation_split import DEVELOPMENT_WATER_YEARS, ROTATION_TABLE_V001, water_year_target_window

    out = {}
    for cycle in ROTATION_TABLE_V001:
        sealing = nre.parse_rotation_sealing({nre.SPLIT_CONTRACT_CONFIG_KEY: str(split_contract),
                                              nre.ROTATION_CYCLE_CONFIG_KEY: cycle.cycle},
                                             project_root=project_root)
        wy = cycle.checkpoint_wy
        windows = sealing.readable_windows(CHECKPOINT_PERIOD)
        year = water_year_target_window(wy)
        if len(windows) != 1 or (windows[0].first, windows[0].last) != (year.first, year.last):
            raise WyRotationScoringError(f"cycle {cycle.cycle}: checkpoint windows {windows} are not WY{wy}")
        if pd.Timestamp(windows[0].last) > pd.Timestamp(sealing.upper_bound):
            raise WyRotationScoringError(f"WY{wy} window ends after the observation upper bound")
        out[wy] = sealing
    if sorted(out) != sorted(DEVELOPMENT_WATER_YEARS):
        raise WyRotationScoringError(f"checkpoint years {sorted(out)} are not the development years")
    return out


def basin_year_eligibility(staid: str, package_path, *, sealing, target: str = TARGET) -> dict[str, Any]:
    """One row of the eligibility table, from observations only: the basin's
    target series in the checkpoint-year window of ``sealing`` (read with
    ``wy_rotation_runs.read_admissible_period_targets``, the reader behind the
    explained-missing count) and :func:`basin_flood_facts` (the scorer's D5(b)
    definition) on its finite values.

    * no package NetCDF -> not eligible (``not_scored_no_package_series``;
      the table does not claim an area-derivation outcome);
    * 0 admissible finite targets -> explained-missing, undefined, not eligible
      (``not_scored_explained_missing``);
    * otherwise eligible; ``flood_eligible`` and the reasons are D5(b)'s."""
    import pandas as pd

    from .wy_rotation_runs import read_admissible_period_targets

    row: dict[str, Any] = {"STAID": staid, "n_admitted": 0, "q98_threshold": None, "n_flood": 0}
    if package_path is None:
        return {**row, "eligible": False, "flood_eligible": False, "reasons": [NOT_SCORED_NO_SERIES]}
    dates, values = read_admissible_period_targets(package_path, sealing=sealing, target=target,
                                                   period=CHECKPOINT_PERIOD)
    finite = np.isfinite(values)
    if not finite.any():
        return {**row, "eligible": False, "flood_eligible": False, "reasons": [NOT_SCORED_MISSING]}
    obs_times = dates[finite] + np.timedelta64(int(sealing.lead_hours), "h")
    (window,) = sealing.readable_windows(CHECKPOINT_PERIOD)
    lo, hi = np.datetime64(pd.Timestamp(window.first), "ns"), np.datetime64(pd.Timestamp(window.last), "ns")
    cap = np.datetime64(pd.Timestamp(sealing.upper_bound), "ns")
    if (obs_times < lo).any() or (obs_times > hi).any() or (obs_times > cap).any():
        raise WyRotationScoringError(f"basin {staid}: an admitted observation lies outside the year window")
    _, flood = basin_flood_facts(staid, dates[finite], values[finite], lead_hours=int(sealing.lead_hours))
    return {**row, "n_admitted": int(finite.sum()), "q98_threshold": flood["q98_threshold"],
            "n_flood": flood["n_flood"], "eligible": True, "flood_eligible": bool(flood["eligible"]),
            "reasons": list(flood["reasons"])}


def build_eligibility_table(*, split_contract, data_dir, basins: Sequence[str], project_root=None,
                            target: str = TARGET) -> dict[str, Any]:
    """The hydrograph reference panel's exclusion input and, reconciled with
    each cycle's facts (:func:`reconcile_eligibility`, ``published_round_e``),
    the formally published round-E eligible count (Omri, 2026-11-19, gate-4
    Q3): score-blind eligibility of every ``basins`` x WY2021-2025 basin-year,
    from package observations only (no run directory, prediction or score is
    opened). Each year is read in its own checkpoint-year window, never past
    the contract's 2025-09-30 23:00 UTC bound. Returns ``{"rows": [...],
    "per_year": {...}, "contract_checksum_sha256": ...}``; rows are sorted by
    (STAID, water_year).

    The table's eligible set is a superset of the scorer's: it cannot see the
    sealed area-derivation exclusions or NH's forcing-gap filter, and Q98 ties
    at float rounding can differ. :func:`reconcile_eligibility` records those
    differences per cycle once the cycle's facts exist, before the panel is
    used."""
    from .wy_rotation_runs import package_series_path

    basins = [str(b) for b in basins]
    if not basins or len(set(basins)) != len(basins):
        raise WyRotationScoringError("eligibility basins must be a non-empty list without duplicates")
    sealings = _eligibility_sealings(split_contract, project_root=project_root)
    paths = {b: package_series_path(data_dir, b) for b in basins}
    rows = []
    for staid in sorted(basins):
        for wy in sorted(sealings):
            rows.append({"water_year": wy,
                         **basin_year_eligibility(staid, paths[staid], sealing=sealings[wy], target=target)})
    per_year = {}
    for wy in sorted(sealings):
        yr = [r for r in rows if r["water_year"] == wy]
        reason_counts: dict[str, int] = {}
        for r in yr:
            for reason in r["reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        n_flood = sum(r["flood_eligible"] for r in yr)
        (window,) = sealings[wy].readable_windows(CHECKPOINT_PERIOD)
        per_year[str(wy)] = {"n_basins": len(yr), "n_eligible": sum(r["eligible"] for r in yr),
                             "n_flood_eligible": n_flood, "flood_eligible_fraction": n_flood / len(yr),
                             "low_eligibility_flag": n_flood / len(yr) < MIN_ELIGIBLE_FRACTION,
                             "reason_counts": reason_counts, "rotation_cycle": sealings[wy].cycle,
                             "checkpoint_window_utc": [window.first.isoformat(), window.last.isoformat()]}
    return {"rows": rows, "per_year": per_year,
            "contract_checksum_sha256": next(iter(sealings.values())).contract_checksum_sha256}


def eligibility_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """The table as CSV bytes: exactly :data:`ELIGIBILITY_COLUMNS`, LF line
    ends, ``True``/``False``, reasons joined by ``;`` (empty = eligible for both)."""
    lines = [",".join(ELIGIBILITY_COLUMNS)]
    for r in rows:
        reason = REASON_SEPARATOR.join(r["reasons"])
        if "," in reason or "," in r["STAID"]:
            raise WyRotationScoringError(f"basin {r['STAID']}: a comma in a CSV field")
        if r["flood_eligible"] and not r["eligible"]:
            raise WyRotationScoringError(f"basin {r['STAID']}: flood-eligible but not eligible")
        lines.append(f"{r['STAID']},{int(r['water_year'])},{bool(r['eligible'])},{bool(r['flood_eligible'])},"
                     f"{reason}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_eligibility_table(table: Mapping[str, Any], out_dir, *, provenance: Mapping[str, Any]) -> dict[str, str]:
    """Write the CSV, ``<csv>.sha256`` (sha256sum format) and a manifest with
    the per-year counts and provenance. Each file is written once: an identical
    rerun is a no-op, a different one raises. Returns the paths and the CSV sha256."""
    out_dir = Path(out_dir)
    csv_path = out_dir / ELIGIBILITY_TABLE_NAME
    digest = _write_bytes_once(csv_path, eligibility_csv_bytes(table["rows"]))
    sha_path = csv_path.with_name(csv_path.name + ".sha256")
    _write_bytes_once(sha_path, f"{digest}  {csv_path.name}\n".encode("utf-8"))
    manifest = {"scorer_version": SCORER_VERSION, "spec_sha256": SPEC_SHA256, "csv": csv_path.name,
                "csv_sha256": digest, "columns": list(ELIGIBILITY_COLUMNS),
                "use": "hydrograph reference-panel exclusion input, and the formally published round-E "
                       "eligible count once reconciled with each cycle's checkpoint_flood_facts.json "
                       "(reconcile_eligibility, published_round_e; Omri 2026-11-19, gate-4 Q3). The facts' "
                       "n_eligible/eligible_fraction must equal it; reconcile before the panel is used.",
                "contract_checksum_sha256": table["contract_checksum_sha256"], "per_year": table["per_year"],
                "inputs": "package observations (target variable only) inside each year's checkpoint window; "
                          "no run directory, prediction or score",
                "flood_units": "target mm/h; D5(b) is scale-free (the scorer's facts use m^3/s = mm/h * area / 3.6)",
                "provenance": dict(provenance)}
    manifest_path = out_dir / ELIGIBILITY_MANIFEST_NAME
    _write_new(manifest_path, manifest)
    return {"csv": str(csv_path), "sha256": str(sha_path), "manifest": str(manifest_path), "csv_sha256": digest}


def _read_eligibility_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse a table written by :func:`write_eligibility_table`, after checking
    its bytes against the ``.sha256`` sidecar."""
    data = csv_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    sidecar = csv_path.with_name(csv_path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != f"{digest}  {csv_path.name}\n":
        raise WyRotationScoringError(f"{csv_path}: sha256 does not match its sidecar {sidecar}")
    lines = data.decode("utf-8").split("\n")
    if lines[-1] != "" or lines[0] != ",".join(ELIGIBILITY_COLUMNS):
        raise WyRotationScoringError(f"{csv_path}: not an eligibility table (header {lines[0]!r})")
    rows, flags = [], {"True": True, "False": False}
    for line in lines[1:-1]:
        f = line.split(",")
        if len(f) != len(ELIGIBILITY_COLUMNS) or f[2] not in flags or f[3] not in flags:
            raise WyRotationScoringError(f"{csv_path}: malformed row {line!r}")
        rows.append({"STAID": f[0], "water_year": int(f[1]), "eligible": flags[f[2]],
                     "flood_eligible": flags[f[3]], "reasons": [r for r in f[4].split(REASON_SEPARATOR) if r]})
    return rows


def reconcile_eligibility(*, cycle_output_dir, eligibility_csv, project_root=None) -> dict[str, Any]:
    """Required step before the reference panel is used: diff the table's row
    for this cycle's checkpoint year against the cycle's scorer facts
    (``checkpoint_flood_facts.json``) and write the differences once, as
    ``checkpoint_eligibility_reconciliation.json`` in ``cycle_output_dir``.

    Expected differences are the scorer's area-derivation exclusions (eligible
    in the table, not scored) and rare Q98 tie flips. The record is the
    evidence; it reads the observation-derived facts and the table only, never
    a score file.

    ``published_round_e`` (Omri, 2026-11-19, gate-4 Q3) is the formally
    published round-E eligible count: the table's flood-eligible basins of
    this year that the scorer scores (not area-excluded, not explained-
    missing), over the population size, the facts' denominator. ``facts_agree``
    says whether the facts' ``n_eligible``/``eligible_fraction`` equal it, with
    the same basin population on both sides. :func:`score_run_checkpoint_year`
    refuses on disagreement when it is given the table."""
    from .wy_rotation_split import ROTATION_TABLE_V001

    spec_sha = verify_spec_sha256(project_root)
    out_root, csv_path = Path(cycle_output_dir), Path(eligibility_csv)
    facts_path = out_root / FACTS_NAME
    facts = _read_json(facts_path, "Q98 facts")
    if facts.get("spec_sha256") != spec_sha:
        raise WyRotationScoringError(f"{facts_path} was not written under the frozen spec")
    cycles = {c.cycle: c.checkpoint_wy for c in ROTATION_TABLE_V001}
    cycle = facts.get("cycle")
    if cycle not in cycles:
        raise WyRotationScoringError(f"{facts_path}: unknown cycle {cycle!r}")
    wy = cycles[cycle]
    rows = [r for r in _read_eligibility_csv(csv_path) if r["water_year"] == wy]
    table = {r["STAID"]: r for r in rows}
    if len(table) != len(rows):
        raise WyRotationScoringError(f"{csv_path}: duplicate basins in WY{wy}")
    population = set(facts["basins"])
    missing, area = set(facts["missing_explained"]), set(facts["area_excluded"])
    scored = population - missing - area
    flood = set(facts["eligible_basins"])
    t_eligible = {b for b, r in table.items() if r["eligible"]}
    t_flood = {b for b, r in table.items() if r["flood_eligible"]}
    eligible_not_scored = t_eligible - scored
    record = {
        "schema": "flashnh_wy_rotation_eligibility_reconciliation", "schema_version": 1,
        "scorer_version": SCORER_VERSION, "spec_sha256": spec_sha, "cycle": cycle, "water_year": wy,
        "eligibility_csv": csv_path.name, "eligibility_csv_sha256": sha256_file(csv_path),
        "facts_sha256": sha256_file(facts_path), "facts_source_run_id": facts.get("source_run_id"),
        "basins_only_in_table": sorted(set(table) - population),
        "basins_only_in_facts": sorted(population - set(table)),
        "table_eligible_not_scored": {"area_excluded": sorted(eligible_not_scored & area),
                                      "other": sorted(eligible_not_scored - area)},
        "scored_not_table_eligible": sorted(scored - t_eligible),
        "flood_eligible_table_only": sorted(t_flood - flood),
        "flood_eligible_facts_only": sorted(flood - t_flood),
        "counts": {"table_eligible": len(t_eligible), "scored": len(scored), "table_flood_eligible": len(t_flood),
                   "facts_flood_eligible": len(flood)},
    }
    n_published = len(t_flood & scored)
    fraction = n_published / len(population)
    record["published_round_e"] = {
        "n_eligible": n_published, "eligible_fraction": fraction,
        "low_eligibility_flag": fraction < MIN_ELIGIBLE_FRACTION,
        "definition": "table flood-eligible basins of this water year that the scorer scores, over the population",
        "facts_agree": (set(table) == population and facts.get("n_eligible") == n_published
                        and facts.get("eligible_fraction") == fraction)}
    _write_new(out_root / RECONCILIATION_NAME, record)
    return record


def main(argv=None) -> int:
    """Build and write the score-blind eligibility table
    (``python -m src.baseline.wy_rotation_scoring --eligibility-only ...``).
    Scoring has no command line."""
    import argparse

    from .wy_rotation_split import load_split_contract, verify_basin_population

    parser = argparse.ArgumentParser(description="WY-rotation score-blind eligibility table (observations only)")
    parser.add_argument("--eligibility-only", action="store_true", required=True,
                        help="the only mode: observations-only eligibility table, no scoring")
    parser.add_argument("--split-contract", required=True, help="frozen split contract v001 JSON")
    parser.add_argument("--data-dir", required=True, help="package root holding time_series/<STAID>.nc")
    parser.add_argument("--out-dir", required=True, help="output folder (below the project root)")
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir = (out_dir if out_dir.is_absolute() else root / out_dir).resolve()
    if out_dir != root and root not in out_dir.parents:
        raise WyRotationScoringError(f"--out-dir {out_dir} is outside the project root {root}")
    contract_path = Path(args.split_contract)
    contract_path = contract_path if contract_path.is_absolute() else root / contract_path
    basins = verify_basin_population(load_split_contract(contract_path), root)
    table = build_eligibility_table(split_contract=contract_path, data_dir=args.data_dir, basins=basins,
                                    project_root=root, target=TARGET)
    written = write_eligibility_table(table, out_dir, provenance={
        "split_contract": str(contract_path), "data_dir": str(args.data_dir), "target": TARGET,
        "n_basins": len(basins)})
    summary = {wy: {k: v[k] for k in ("n_eligible", "n_flood_eligible", "low_eligibility_flag")}
               for wy, v in table["per_year"].items()}
    print(json.dumps({**written, "per_year": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
