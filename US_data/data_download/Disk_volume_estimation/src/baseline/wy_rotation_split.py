"""Water-year rotation split contract (Stage-1 baseline, decision 2026-11-11).

Status: FROZEN (user decision 2026-11-12, after the gate 2 review;
``docs/decision_log.md``). The contract this module builds is the "frozen split
contract" of gate 2 in the 2026-11-11 entry. No rotation run (training or
evaluation) may start until the loader-sealing increment enforces its readable
windows and observation upper bound, and that increment has been reviewed.

What this module is the authority for
-------------------------------------
- **Water-year bounds.** Water year ``k`` is the set of TARGET VALID TIMES
  ``k-1-10-01 00:00`` .. ``k-09-30 23:00``, naive UTC (the package convention,
  ``nh_dataset.py``). A sample with issue hour ``t`` and lead ``L`` belongs to
  the water year that contains ``t + L``. Observations are never model inputs
  (P19 inputs are MRMS + RTMA only), so forcing look-back may cross any
  boundary; only target observations carry a role.
- **The cycle table** (five cycles over WY2021-WY2025; WY2026 is the protected
  test year and has no role in any cycle).
- **NH period dates** for each cycle: training as a list of slices (contiguous
  training years merged into one slice), validation = the checkpoint year,
  test = the assessment year.
- **Readable observation windows** for each (cycle, NH period): the only
  target-time windows whose observations that period may read. Everything else
  is sealed for that period. (Enforcement in the loaders is a separate change;
  see "Not done here".)
- **Basin population**: the 2,307-basin development list, pinned by path,
  count and sha256.

Boundary convention (issue-hour admission)
------------------------------------------
NH dates have day resolution, and ``FlashNHDataset`` admits a sample only when
both its issue hour and its target hour lie inside one declared slice (concern
B plus the multi-slice leak fix). With day-resolution slices that start on
Oct 1, the first ``L`` target hours of a slice (Oct 1 00:00 .. 0L-1:00) have
their issue hour on Sep 30, outside the slice, so they are never admitted. They
are not leaked or shared -- they keep their single role -- they are simply not
scored or trained on. Merging contiguous training years into one slice removes
the loss at internal boundaries. ``boundary_unadmitted_target_hours`` reports
the exact loss per (cycle, period). The alternative (hour-precise target-window
admission, so that no target hour is lost) was not adopted (user decision
2026-11-12, ``docs/decision_log.md``).

The research timeline starts at ``PACKAGE_FIRST_HOUR`` (2020-10-14 00:00, the
MRMS Pass1 archive start), so WY2021 is partial. Its NH slice starts on that
day rather than on 2020-10-01; the forcing-history filter removes further early
samples. Recorded, not excluded (preflight memo decision 6).

Not done here (next increments, see the preflight memo's adaptation table)
------------------------------------------------------------------------
- loader-side observation sealing by the readable-window lists
  (``nh_dataset`` / ``nh_raw_space_evaluation``);
- support builders per water year (Common-120 rule at L120);
- the new package version and policy pin (``policy.py`` hard-pins the old
  calendar split).

Public API
----------
``water_year_target_window``, ``merge_contiguous_water_years``,
``RotationCycle``, ``ROTATION_TABLE_V001``, ``validate_rotation_table``,
``cycle_nh_dates``, ``readable_observation_windows``,
``admitted_target_windows``, ``boundary_unadmitted_target_hours``,
``build_split_contract``, ``verify_split_contract``, ``write_split_contract``,
``load_split_contract``, ``verify_basin_population``, ``WyRotationSplitError``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.baseline.atomic_shard_store import canonical_json_sha256
from src.baseline.devpop_common120_audit_contract import (
    DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256,
    canonical_membership_sha256,
)

__all__ = [
    "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION",
    "DEVELOPMENT_WATER_YEARS",
    "LEAD_HOURS",
    "NH_PERIODS",
    "OBSERVATION_UPPER_BOUND",
    "PACKAGE_FIRST_HOUR",
    "ROTATION_TABLE_V001",
    "TEST_WATER_YEAR",
    "RotationCycle",
    "TimeWindow",
    "WyRotationSplitError",
    "admitted_target_windows",
    "boundary_unadmitted_target_hours",
    "build_split_contract",
    "cycle_nh_dates",
    "load_split_contract",
    "merge_contiguous_water_years",
    "readable_observation_windows",
    "validate_rotation_table",
    "verify_basin_population",
    "verify_split_contract",
    "water_year_target_window",
    "write_split_contract",
]

CONTRACT_ID = "flashnh_wy_rotation_split_v001"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_STATUS = "FROZEN"

DEVELOPMENT_WATER_YEARS = (2021, 2022, 2023, 2024, 2025)
TEST_WATER_YEAR = 2026
LEAD_HOURS = 6
# First hour of package v002 and of the USGS/MRMS/RTMA products (MRMS Pass1
# archive start; policy.py period.start_utc).
PACKAGE_FIRST_HOUR = pd.Timestamp("2020-10-14 00:00:00")

# No observation later than this hour is readable in any cycle or period: the
# last hour of WY2025. WY2026 starts right after it. The lead-shifted rows of
# 2025-09-30 18:00-23:00 hold WY2026 observations, so loaders must apply this
# bound per observation time, not per row time.
OBSERVATION_UPPER_BOUND = pd.Timestamp("2025-09-30 23:00:00")

BASIN_POPULATION_PATH = "config/stage1_baseline_splits_v001/development_train.txt"
BASIN_POPULATION_COUNT = 2307
# Portable, line-ending-independent identity (sorted unique ids, LF): the same
# constant the Common-120 audit contract pins. A hash of the file bytes would
# differ between LF and CRLF checkouts.
BASIN_POPULATION_MEMBERSHIP_SHA256 = DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256

# NH period name -> rotation role read by that period.
NH_PERIODS = {"train": "train", "validation": "checkpoint", "test": "assessment"}

_NH_DATE_FORMAT = "%d/%m/%Y"
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class WyRotationSplitError(ValueError):
    """Raised for an invalid rotation table, period or contract."""


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeWindow:
    """Inclusive window of hourly timestamps (naive UTC)."""

    first: pd.Timestamp
    last: pd.Timestamp

    def __post_init__(self):
        for name in ("first", "last"):
            ts = getattr(self, name)
            if ts.tzinfo is not None:
                raise WyRotationSplitError(f"window {name} must be naive UTC, got {ts}")
            if ts != ts.floor("h"):
                raise WyRotationSplitError(f"window {name} must be on the hour, got {ts}")
        if self.last < self.first:
            raise WyRotationSplitError(f"window ends before it starts: {self.first} .. {self.last}")

    @property
    def n_hours(self) -> int:
        return int((self.last - self.first) / pd.Timedelta(hours=1)) + 1

    def to_json(self) -> dict:
        return {"first_utc": self.first.strftime(_TS_FORMAT), "last_utc": self.last.strftime(_TS_FORMAT),
                "n_hours": self.n_hours}


def _check_water_year(wy) -> int:
    if isinstance(wy, bool) or not isinstance(wy, int):
        raise WyRotationSplitError(f"water year must be an int, got {wy!r}")
    return wy


def water_year_target_window(wy: int) -> TimeWindow:
    """Target-valid-time window of water year ``wy``: Oct 1 00:00 of ``wy-1``
    through Sep 30 23:00 of ``wy`` (naive UTC)."""
    wy = _check_water_year(wy)
    return TimeWindow(pd.Timestamp(year=wy - 1, month=10, day=1), pd.Timestamp(year=wy, month=9, day=30, hour=23))


def merge_contiguous_water_years(wys) -> tuple:
    """Sorted runs of consecutive water years, e.g. (2021, 2022, 2025) ->
    ((2021, 2022), (2025,)). Duplicates are refused."""
    years = [_check_water_year(w) for w in wys]
    if not years:
        raise WyRotationSplitError("need at least one water year")
    if len(set(years)) != len(years):
        raise WyRotationSplitError(f"duplicate water years: {years}")
    runs = []
    for wy in sorted(years):
        if runs and wy == runs[-1][-1] + 1:
            runs[-1].append(wy)
        else:
            runs.append([wy])
    return tuple(tuple(r) for r in runs)


# ---------------------------------------------------------------------------
# Cycle table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RotationCycle:
    cycle: int
    train_wys: tuple
    checkpoint_wy: int
    assessment_wy: int

    def role_water_years(self, role: str) -> tuple:
        if role == "train":
            return tuple(sorted(self.train_wys))
        if role == "checkpoint":
            return (self.checkpoint_wy,)
        if role == "assessment":
            return (self.assessment_wy,)
        raise WyRotationSplitError(f"unknown role {role!r}")


# Frozen in docs/decision_log.md (2026-11-11). Each development year is
# assessed once, used for checkpoint choice once and trained on three times.
ROTATION_TABLE_V001 = (
    RotationCycle(1, (2023, 2024, 2025), 2022, 2021),
    RotationCycle(2, (2021, 2024, 2025), 2023, 2022),
    RotationCycle(3, (2021, 2022, 2025), 2024, 2023),
    RotationCycle(4, (2021, 2022, 2023), 2025, 2024),
    RotationCycle(5, (2022, 2023, 2024), 2021, 2025),
)


def validate_rotation_table(cycles, *, development_wys=DEVELOPMENT_WATER_YEARS,
                            test_wy: int = TEST_WATER_YEAR) -> tuple:
    """Fail-closed structural checks on a cycle table. Returns it unchanged."""
    cycles = tuple(cycles)
    dev = set(development_wys)
    if test_wy in dev:
        raise WyRotationSplitError(f"test water year {test_wy} is listed as a development year")
    if [c.cycle for c in cycles] != list(range(1, len(cycles) + 1)):
        raise WyRotationSplitError(f"cycles must be numbered 1..n in order, got {[c.cycle for c in cycles]}")
    if len(cycles) != len(dev):
        raise WyRotationSplitError(f"need one cycle per development year ({len(dev)}), got {len(cycles)}")
    for c in cycles:
        roles = list(c.train_wys) + [c.checkpoint_wy, c.assessment_wy]
        for wy in roles:
            _check_water_year(wy)
            if wy == test_wy:
                raise WyRotationSplitError(f"cycle {c.cycle}: protected test year {test_wy} has a rotation role")
            if wy not in dev:
                raise WyRotationSplitError(f"cycle {c.cycle}: {wy} is not a development water year")
        if len(set(roles)) != len(roles):
            raise WyRotationSplitError(f"cycle {c.cycle}: a water year has more than one role: {roles}")
        if set(roles) != dev:
            raise WyRotationSplitError(f"cycle {c.cycle}: roles {sorted(roles)} do not cover {sorted(dev)}")
    for role, attr in (("assessment", "assessment_wy"), ("checkpoint", "checkpoint_wy")):
        used = sorted(getattr(c, attr) for c in cycles)
        if used != sorted(dev):
            raise WyRotationSplitError(f"each development year must be the {role} year exactly once, got {used}")
    return cycles


# ---------------------------------------------------------------------------
# Per-cycle NH dates, readable windows, admission
# ---------------------------------------------------------------------------

def _nh_slices(wys, *, data_first_hour: pd.Timestamp = PACKAGE_FIRST_HOUR) -> list:
    """Day-resolution NH slices (first_day, last_day) for these water years:
    contiguous years merged, start clipped to the first package day."""
    slices = []
    for run in merge_contiguous_water_years(wys):
        first = water_year_target_window(run[0]).first
        last = water_year_target_window(run[-1]).last
        first = max(first, data_first_hour.floor("D"))
        slices.append((first.floor("D"), last.floor("D")))
    return slices


def _cycle(cycles, cycle: int) -> RotationCycle:
    for c in cycles:
        if c.cycle == cycle:
            return c
    raise WyRotationSplitError(f"no cycle {cycle!r}")


def cycle_nh_dates(cycle: RotationCycle, *, data_first_hour: pd.Timestamp = PACKAGE_FIRST_HOUR) -> dict:
    """NH config date keys (DD/MM/YYYY) for one cycle. Training is always a
    list (possibly of one slice); validation and test are single slices."""
    out = {}
    for period, role in NH_PERIODS.items():
        slices = _nh_slices(cycle.role_water_years(role), data_first_hour=data_first_hour)
        starts = [s.strftime(_NH_DATE_FORMAT) for s, _ in slices]
        ends = [e.strftime(_NH_DATE_FORMAT) for _, e in slices]
        if period == "train":
            out["train_start_date"], out["train_end_date"] = starts, ends
        else:
            if len(slices) != 1:
                raise WyRotationSplitError(f"cycle {cycle.cycle}: {period} must be a single slice")
            out[f"{period}_start_date"], out[f"{period}_end_date"] = starts[0], ends[0]
    return out


def readable_observation_windows(cycle: RotationCycle, period: str) -> tuple:
    """Target-time windows whose observations NH period ``period`` of this
    cycle may read (merged where contiguous). Every other observation is
    sealed for that period, including those inside its forcing warm-up."""
    if period not in NH_PERIODS:
        raise WyRotationSplitError(f"unknown NH period {period!r}; expected one of {sorted(NH_PERIODS)}")
    windows = []
    for run in merge_contiguous_water_years(cycle.role_water_years(NH_PERIODS[period])):
        windows.append(TimeWindow(water_year_target_window(run[0]).first, water_year_target_window(run[-1]).last))
    return tuple(windows)


def admitted_target_windows(cycle: RotationCycle, period: str, *, lead_hours: int = LEAD_HOURS,
                            data_first_hour: pd.Timestamp = PACKAGE_FIRST_HOUR) -> tuple:
    """Target-time windows that FlashNHDataset's slice rule can admit for this
    period (issue hour and target hour inside one day-resolution slice), before
    the forcing-history and missing-target filters. Mirrors
    ``nh_dataset._period_slices`` + concern B; tests cross-check it against
    the dataset itself."""
    if period not in NH_PERIODS:
        raise WyRotationSplitError(f"unknown NH period {period!r}")
    lead = pd.Timedelta(hours=lead_hours)
    out = []
    for first_day, last_day in _nh_slices(cycle.role_water_years(NH_PERIODS[period]),
                                          data_first_hour=data_first_hour):
        last_hour = last_day + pd.Timedelta(hours=23)
        out.append(TimeWindow(first_day + lead, last_hour))
    return tuple(out)


def boundary_unadmitted_target_hours(cycle: RotationCycle, period: str, *, lead_hours: int = LEAD_HOURS,
                                     data_first_hour: pd.Timestamp = PACKAGE_FIRST_HOUR) -> dict:
    """Per water year: target hours inside the year (and inside the package
    timeline) that the slice rule cannot admit. Keys are water years."""
    admitted = admitted_target_windows(cycle, period, lead_hours=lead_hours, data_first_hour=data_first_hour)
    out = {}
    for wy in cycle.role_water_years(NH_PERIODS[period]):
        window = water_year_target_window(wy)
        first = max(window.first, data_first_hour)
        hours = pd.date_range(first, window.last, freq="h")
        inside = pd.Series(False, index=hours)
        for w in admitted:
            inside |= (hours >= w.first) & (hours <= w.last)
        out[wy] = int((~inside).sum())
    return out


# ---------------------------------------------------------------------------
# Contract build / verify / IO
# ---------------------------------------------------------------------------

def _contract_body(cycles, *, lead_hours: int, data_first_hour: pd.Timestamp) -> dict:
    cycles = validate_rotation_table(cycles)
    body = {
        "contract_id": CONTRACT_ID,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": CONTRACT_STATUS,
        "decision_ref": "docs/decision_log.md 2026-11-11 (water-year rotation ADOPTED)",
        "conventions": {
            "timezone": "UTC, naive timestamps",
            "water_year": "Oct 1 00:00 of wy-1 through Sep 30 23:00 of wy",
            "role_assignment": "by target valid time (issue hour + lead)",
            "lead_hours": lead_hours,
            "nh_date_resolution": "day; a sample is admitted only if issue and target lie in one slice",
            "contiguous_training_years": "merged into one NH slice",
            "forcing_lookback": "may cross any water-year boundary (observations are not inputs)",
            "observations_outside_readable_windows": "sealed for that period, including warm-up",
            "observation_upper_bound": ("no observation later than observation_upper_bound_utc is readable in "
                                        "any cycle or period, judged by observation time (row time + lead for "
                                        "lead-shifted targets); enforced by the loader-sealing increment, and no "
                                        "rotation run starts before that enforcement exists"),
        },
        "package_first_hour_utc": data_first_hour.strftime(_TS_FORMAT),
        "observation_upper_bound_utc": OBSERVATION_UPPER_BOUND.strftime(_TS_FORMAT),
        "basin_population": {"path": BASIN_POPULATION_PATH, "count": BASIN_POPULATION_COUNT,
                             "membership_sha256": BASIN_POPULATION_MEMBERSHIP_SHA256,
                             "membership_identity": "sha256 of sorted unique basin ids, one per line, LF-terminated "
                                                    "(devpop_common120_audit_contract.canonical_membership_sha256)"},
        "development_water_years": {str(wy): water_year_target_window(wy).to_json()
                                    for wy in DEVELOPMENT_WATER_YEARS},
        "test_water_year": {"water_year": TEST_WATER_YEAR,
                            "target_window": water_year_target_window(TEST_WATER_YEAR).to_json(),
                            "role": "protected; no role in any cycle; scored once per model under a separate contract"},
        "cycles": [],
    }
    for c in cycles:
        for period in NH_PERIODS:
            for w in readable_observation_windows(c, period):
                if w.last > OBSERVATION_UPPER_BOUND:
                    raise WyRotationSplitError(
                        f"cycle {c.cycle} {period}: readable window ends {w.last}, after the observation "
                        f"upper bound {OBSERVATION_UPPER_BOUND}")
        entry = {"cycle": c.cycle, "train_wys": sorted(c.train_wys), "checkpoint_wy": c.checkpoint_wy,
                 "assessment_wy": c.assessment_wy,
                 "nh_dates": cycle_nh_dates(c, data_first_hour=data_first_hour), "periods": {}}
        for period, role in NH_PERIODS.items():
            entry["periods"][period] = {
                "role": role,
                "readable_observation_windows": [w.to_json() for w in readable_observation_windows(c, period)],
                "admitted_target_windows": [w.to_json() for w in admitted_target_windows(
                    c, period, lead_hours=lead_hours, data_first_hour=data_first_hour)],
                "boundary_unadmitted_target_hours": {str(k): v for k, v in boundary_unadmitted_target_hours(
                    c, period, lead_hours=lead_hours, data_first_hour=data_first_hour).items()},
            }
        body["cycles"].append(entry)
    return body


def build_split_contract() -> dict:
    """The v001 contract: ``ROTATION_TABLE_V001``, ``LEAD_HOURS`` and
    ``PACKAGE_FIRST_HOUR``, nothing else. A different table, lead or start is a
    different contract and needs a new ``CONTRACT_ID`` in code, never an edited
    file. (The window helpers above stay parameterised for analysis.)"""
    body = _contract_body(ROTATION_TABLE_V001, lead_hours=LEAD_HOURS, data_first_hour=PACKAGE_FIRST_HOUR)
    body["checksum_sha256"] = canonical_json_sha256(body)
    return body


def _cycles_from_contract(contract: dict) -> tuple:
    try:
        return tuple(RotationCycle(int(c["cycle"]), tuple(int(w) for w in c["train_wys"]),
                                   int(c["checkpoint_wy"]), int(c["assessment_wy"]))
                     for c in contract["cycles"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WyRotationSplitError(f"malformed contract cycles: {exc}") from exc


def verify_split_contract(contract: dict) -> dict:
    """Fail closed unless the contract is exactly the frozen v001 contract:
    its checksum matches its body, its table, lead, first hour and observation
    upper bound equal this module's frozen values, and every other field
    equals what this module derives from them. A file edited and re-checksummed
    is refused however internally consistent it is."""
    if not isinstance(contract, dict):
        raise WyRotationSplitError("contract must be a mapping")
    body = {k: v for k, v in contract.items() if k != "checksum_sha256"}
    if contract.get("checksum_sha256") != canonical_json_sha256(body):
        raise WyRotationSplitError("contract checksum does not match its content")
    if body.get("contract_id") != CONTRACT_ID or body.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise WyRotationSplitError(
            f"unexpected contract identity {body.get('contract_id')!r} v{body.get('schema_version')!r}")
    lead = (body.get("conventions") or {}).get("lead_hours")
    if type(lead) is not int or lead != LEAD_HOURS:
        raise WyRotationSplitError(f"contract lead_hours {lead!r} is not the frozen v001 lead {LEAD_HOURS}")
    expected_first = PACKAGE_FIRST_HOUR.strftime(_TS_FORMAT)
    if body.get("package_first_hour_utc") != expected_first:
        raise WyRotationSplitError(f"contract package_first_hour_utc {body.get('package_first_hour_utc')!r} "
                                   f"is not the frozen v001 value {expected_first!r}")
    expected_bound = OBSERVATION_UPPER_BOUND.strftime(_TS_FORMAT)
    if body.get("observation_upper_bound_utc") != expected_bound:
        raise WyRotationSplitError(f"contract observation_upper_bound_utc "
                                   f"{body.get('observation_upper_bound_utc')!r} is not {expected_bound!r}")
    if _cycles_from_contract(body) != ROTATION_TABLE_V001:
        raise WyRotationSplitError("contract cycle table is not the frozen ROTATION_TABLE_V001")
    rebuilt = _contract_body(ROTATION_TABLE_V001, lead_hours=LEAD_HOURS, data_first_hour=PACKAGE_FIRST_HOUR)
    if rebuilt != body:
        diff = sorted(k for k in set(rebuilt) | set(body) if rebuilt.get(k) != body.get(k))
        raise WyRotationSplitError(f"contract content differs from the frozen v001 derivation: {diff}")
    return contract


def write_split_contract(path, contract: dict) -> Path:
    verify_split_contract(contract)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_split_contract(path) -> dict:
    try:
        contract = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WyRotationSplitError(f"cannot read split contract {path}: {exc}") from exc
    return verify_split_contract(contract)


def verify_basin_population(contract: dict, project_root) -> list:
    """Verify the contract, then read the pinned development basin list and
    check count, uniqueness and the portable membership sha256 (independent of
    LF/CRLF checkout). Returns the basin ids in file order (strings, leading
    zeros kept)."""
    verify_split_contract(contract)
    pop = contract["basin_population"]
    path = Path(project_root) / pop["path"]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WyRotationSplitError(f"cannot read basin population {path}: {exc}") from exc
    basins = [line.strip() for line in text.splitlines() if line.strip()]
    if len(basins) != pop["count"] or len(set(basins)) != len(basins):
        raise WyRotationSplitError(f"{path}: expected {pop['count']} unique basins, got {len(basins)} "
                                   f"({len(set(basins))} unique)")
    if canonical_membership_sha256(basins) != pop["membership_sha256"]:
        raise WyRotationSplitError(f"{path}: membership sha256 does not match the contract")
    return basins


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen water-year rotation split contract v001.")
    parser.add_argument("--out", required=True, help="output JSON path (under the project root)")
    parser.add_argument("--project-root", default=None, help="if given, verify the pinned basin population")
    args = parser.parse_args(argv)
    contract = build_split_contract()
    if args.project_root:
        verify_basin_population(contract, args.project_root)
    path = write_split_contract(args.out, contract)
    print(f"{path}  sha256(content)={contract['checksum_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
