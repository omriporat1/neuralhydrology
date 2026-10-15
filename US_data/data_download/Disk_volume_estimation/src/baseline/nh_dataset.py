"""FlashNHDataset: NeuralHydrology 1.13 integration for Stage 1 validity masks.

Registers (via ``src/baseline/nh_register.py``) a ``GenericDataset`` subclass
that post-filters NH's own lookup table by exact issue timestamp, applying
three separate, explicit validity concerns on top of NH's unmodified sample
construction:

    A. Historical forcing validity -- reuse ``src/baseline/validity_mask.py``
       unchanged (``compute_history_valid``) over a continuous research
       timeline built directly from this dataset instance's own loaded
       ``self._dates`` (see "Research timeline construction" below), so
       cross-period antecedent history is preserved exactly as NH already
       does it.
    B. Period-specific target boundary -- issue_time + lead_hours must stay
       within *this period's own* last real timestamp
       (``self._dates[basin][freq][-1]`` for the dataset instance being
       built), so a training target cannot cross into validation, a
       validation target cannot cross into test, and a test target cannot
       cross beyond the research period. This is intentionally not the same
       check as ``validity_mask.compute_boundary_valid``, which is not
       period-aware.
    C. Basin-specific target availability -- the selected target variable's
       value at the issue timestamp must be non-NaN. Applied uniformly to
       train, validation, and test (NH itself only does this for training).

Do not filter by raw NH ``idx`` arithmetic: every decision resolves the
entry's actual issue timestamp via ``self._dates[basin][freq][idx]`` first.

Timestamp normalization convention (applied everywhere a timestamp is
constructed from a raw value -- gap timestamps, research-timeline bounds,
lookup-entry issue timestamps, and period-end timestamps -- via
``_normalize_timestamp``):
    - timezone-aware timestamps are converted to UTC, then have their
      timezone information stripped;
    - timezone-naive timestamps are treated as already-UTC and left as-is.
All comparisons therefore always happen between normalized, timezone-naive,
UTC-wall-clock timestamps, regardless of whether the gap inventory or NH's
own ``self._dates`` happen to carry timezone info.

Research timeline construction (concern A only): built directly from this
dataset instance's own ``self._dates`` -- the earliest loaded timestamp
across all basins through the latest loaded timestamp across all basins --
rather than from ``cfg`` date strings or an arbitrary safety margin. This is
possible cleanly because ``self._dates`` is already populated by the time
``_apply_flashnh_filters`` runs (immediately after
``super()._create_lookup_table``), and it already reflects whatever
backward warm-up NH itself loaded (e.g. validation's own dates extending
back into train), so no extra margin is ever needed to cover history
look-back, and the timeline never extends past the last real timestamp NH
actually loaded for this instance. The previously-used fallback convention
of "normalized final date + 23 hours" (i.e. treating a date-only bound as
inclusive through the end of that calendar day) is documented here only
because it was considered and is unnecessary for this dataset given the
above; it is not used.

Mask/config discovery (deliberately package-side and NH-config-based, not
environment variables or process-global state):
    - ``lead_hours``: parsed from the trailing ``lead<NN>`` suffix of the
      (single) configured target variable, matching
      ``src/baseline/lead_targets.py``'s ``DEFAULT_VARIABLE_NAME_TEMPLATE``.
    - period boundaries: read directly off the dataset instance's own
      resolved ``self._dates`` (concern B) -- no cross-period config parsing
      needed.
    - archive-gap hours: ``<cfg.data_dir>/masks/gap_timestamps.json``, a flat
      JSON list of ISO timestamp strings (UTC, commonly with a trailing
      ``Z``). This is a deliberate design choice over NaN-detection in the
      loaded dynamic-input arrays: the certified Stage 1 Compact Scientific
      Package (``stage1_compact_scientific_package_v001``, Gate 4 PASS)
      preserves forcing NaNs at gap hours unchanged -- it does not gap-fill
      them -- so ``*_gap`` flag columns and a matching gap-timestamp
      inventory are the authoritative source of truth, and dynamic-input
      NaNs are expected to coincide with them. Earlier MRMS zero-fill /
      RTMA-interpolation gap handling existed only for the retired Smoke
      0/1 technical-validation packages and is not part of this baseline's
      missing-data policy. FlashNHDataset's role here is to hard-exclude any
      sample whose backward warm-up history window intersects a documented
      gap timestamp -- NH's own native sample validation does not perform
      this check during evaluation (``is_train=False``), so this post-filter
      is the sole mechanism protecting validation/test batches. Model-layer
      ``nan_handling_method`` (NeuralHydrology's ``InputLayer`` NaN backstop)
      is not used as the baseline missing-data mechanism and is left unset;
      see ``docs/stage1_baseline_package_implementation_plan.md`` for the
      accepted-findings record of why hard exclusion, not
      ``nan_handling_method``, is the binding policy.

Gap-timestamp strictness: a gap timestamp that falls entirely before or
after this instance's research timeline is assumed to come from a shared
gap inventory covering a wider period than the current run, and is silently
ignored (counted, not dropped without a trace). A gap timestamp whose
wall-clock hour lies *within* the research timeline but does not land
exactly on the hourly grid (e.g. a timezone mismatch or a non-hourly
timestamp) is never silently ignored -- it raises ``FlashNHDatasetError``,
because letting ``validity_mask.bad_hour_mask_from_timestamps``'s
``on_out_of_range="ignore"`` swallow it could silently turn a real,
in-period gap into zero matched bad hours.

Optional fixed common training support (train period only): when the NH
config key ``flashnh_common_support_file`` (registered by
``nh_register.register_flashnh_dataset``) names a checksummed training-support
artifact from ``src/baseline/seqlen_common336_support.py``, the lookup table
left after concerns A/B/C is further restricted to exactly the artifact's
(basin, issue-time) pairs. This is what makes every sequence length L <= the
artifact's ``seq_length_floor`` train on the same pairs. The restriction is
fail-closed: the artifact's identity (lead, target, dynamic inputs, train
dates, gap-artifact sha256, basin set) must match this dataset, and every
support pair must already be admitted naturally -- a support pair missing from
the natural lookup (including a basin NH dropped for having no samples) raises
``FlashNHDatasetError`` rather than training on a smaller set. Validation and
test datasets are never restricted here; common screening support is applied
by the raw-space scorer instead.

Optional observation cutoff (experiment-scoped sealed-period boundary): when
the NH config key ``flashnh_observation_cutoff`` (registered by
``nh_register.register_flashnh_dataset``) is set, each basin NetCDF is opened
lazily and restricted BEFORE any value is read so that no observation later
than the cutoff enters memory: every variable ends at the cutoff, and each
lead-shifted target ``*_lead<NN>`` ends at ``cutoff - NN h`` (NaN-filled after
that, never read) -- see
``nh_raw_space_evaluation.select_through_observation_cutoff``. Without the key,
GenericDataset's whole-file read is unchanged. With it, the ``test`` period
and any period ending after the cutoff are refused. Concern B already removes
every issue time whose lead target lies past the period end, so the lookup
table is the same with or without the cutoff when the cutoff equals the last
loaded period's end.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray
from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.utils.errors import NoEvaluationDataError, NoTrainDataError

from src.baseline.validity_mask import bad_hour_mask_from_timestamps, compute_history_valid

__all__ = ["FlashNHDataset", "FlashNHDatasetError", "FlashNHFilterStats"]

_LEAD_SUFFIX_RE = re.compile(r"lead(\d+)$")
_COMMON_SUPPORT_CONFIG_KEY = "flashnh_common_support_file"
COMMON_SUPPORT_RECEIPT_NAME = "flashnh_common_support_applied.json"

LOGGER = logging.getLogger(__name__)


class FlashNHDatasetError(ValueError):
    """Raised for a missing or malformed Flash-NH mask/config discovery input."""


@dataclass(frozen=True)
class FlashNHFilterStats:
    """Separate, non-overlapping removal counts for one dataset instance.

    ``removed_for_missing_target`` is deliberately distinct from the global
    forcing mask (concern A/B) per the binding I-D1 semantics: basin target
    availability is a separate, later combination layer, not part of
    ``validity_mask.py``.

    ``n_gap_timestamps_loaded``/``_in_range``/``_ignored_outside_range``
    record the gap-inventory classification (see module docstring's
    "Gap-timestamp strictness" section) so a nonempty real-period gap list
    can never silently become zero matched bad hours without a trace.
    """

    period: str
    seq_length: int
    lead_hours: int
    n_before: int
    removed_for_forcing_history: int
    removed_for_period_target_boundary: int
    removed_for_missing_target: int
    n_kept: int
    n_gap_timestamps_loaded: int
    n_gap_timestamps_in_range: int
    n_gap_timestamps_ignored_outside_range: int
    # Fixed common training support (None/0 when not configured).
    common_support_id: str | None = None
    common_support_checksum_sha256: str | None = None
    n_natural_admitted: int | None = None
    removed_for_common_support: int = 0


def _normalize_timestamp(value) -> pd.Timestamp:
    """Normalize any timestamp-like value to a timezone-naive UTC Timestamp.

    Timezone-aware inputs are converted to UTC and then stripped of tzinfo;
    timezone-naive inputs are treated as already being UTC and returned
    unchanged (aside from the ``pd.Timestamp`` conversion itself).
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _parse_lead_hours(target_variables) -> int:
    if not isinstance(target_variables, list) or len(target_variables) != 1:
        raise FlashNHDatasetError(
            f"FlashNHDataset requires exactly one target variable, got {target_variables!r}"
        )
    name = target_variables[0]
    match = _LEAD_SUFFIX_RE.search(name)
    if not match:
        raise FlashNHDatasetError(
            f"target variable {name!r} does not end with a 'lead<NN>' suffix"
        )
    return int(match.group(1))


def _build_research_timeline(dates_by_basin: dict, freq: str) -> pd.DatetimeIndex:
    """Build the shared, continuous hourly research timeline for concern A
    directly from this dataset instance's own already-loaded dates: the
    earliest loaded timestamp across all basins through the latest loaded
    timestamp across all basins. No cfg date parsing, no safety margin."""
    starts = []
    ends = []
    for basin_dates in dates_by_basin.values():
        arr = basin_dates[freq]
        if len(arr) == 0:
            continue
        starts.append(_normalize_timestamp(arr[0]))
        ends.append(_normalize_timestamp(arr[-1]))
    if not starts:
        raise FlashNHDatasetError(
            "no basin dates available to build the Flash-NH research timeline"
        )
    return pd.date_range(start=min(starts), end=max(ends), freq="h")


def _load_gap_timestamps(data_dir) -> list:
    path = Path(data_dir) / "masks" / "gap_timestamps.json"
    if not path.exists():
        raise FlashNHDatasetError(f"gap timestamps artifact not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise FlashNHDatasetError(
            f"{path}: expected a JSON list of ISO timestamp strings, got {type(raw).__name__}"
        )
    return [_normalize_timestamp(v) for v in raw]


def _classify_gap_timestamps(gap_timestamps: list, full_timeline: pd.DatetimeIndex):
    """Split normalized gap timestamps against ``full_timeline``.

    Returns ``(in_range_aligned, n_ignored_outside)``. Raises
    ``FlashNHDatasetError`` immediately for any timestamp that falls inside
    ``[full_timeline[0], full_timeline[-1]]`` but does not land exactly on
    the hourly grid -- see the module docstring's "Gap-timestamp
    strictness" section.
    """
    start, end = full_timeline[0], full_timeline[-1]
    in_range = []
    n_outside = 0
    for ts in gap_timestamps:
        if ts < start or ts > end:
            n_outside += 1
            continue
        if ts not in full_timeline:
            raise FlashNHDatasetError(
                f"gap timestamp {ts} falls within the research timeline "
                f"[{start}, {end}] but does not align to its hourly grid; "
                "check for a timezone mismatch or a non-hourly timestamp"
            )
        in_range.append(ts)
    return in_range, n_outside


def _adapt_temporal_index_to_date(df: pd.DataFrame) -> pd.DataFrame:
    """In-memory package-to-NeuralHydrology temporal-index compatibility boundary.

    NeuralHydrology 1.13's ``BaseDataset._load_or_create_xarray_dataset``
    (basedataset.py) requires the per-basin DataFrame's temporal index to be
    named exactly ``date`` (it does ``df_duplicated.groupby('date')`` and
    later reads ``xr['date']`` on the xarray.Dataset built from this
    DataFrame via ``xarray.Dataset.from_dataframe``), so an unadapted
    non-``date`` index fails with ``KeyError: 'date'``.

    This package's on-disk NetCDF package schema identity determines its
    temporal coordinate name (see ``src/baseline/package_netcdf.py``): the
    frozen, certified ``stage1_compact_scientific_package_v001`` schema uses
    ``time``; the ``stage1_scientific_package_v002`` schema used by future
    scientific package builds uses ``date`` directly. This function is the
    single place both are reconciled to the one in-memory name
    NeuralHydrology requires, per basin, at load time -- it never rewrites
    any on-disk file:

        - index already named ``date``: passed through unchanged (no rename).
        - index named ``time`` (and no stray ``date`` column present): the
          index's ``.name`` metadata is renamed to ``date`` -- values, row
          order, dtypes, and all columns (including gap flags) are untouched.
        - both a temporal index and a same-named-the-other-way stray column
          present (e.g. a ``date`` index alongside a ``time`` column, or vice
          versa), or an index named anything else entirely: raises
          ``FlashNHDatasetError`` rather than guessing.

    Note that this only resolves NeuralHydrology's own structural loading
    requirement. It does not mean stock ``GenericDataset`` sample
    construction reproduces Flash-NH's validity filtering (concerns A/B/C in
    this module's docstring) -- ``FlashNHDataset._apply_flashnh_filters``
    remains required regardless of which package schema was loaded.
    """
    index = df.index
    if isinstance(index, pd.MultiIndex):
        raise FlashNHDatasetError(
            "FlashNHDataset._load_basin_data: expected a single-level temporal "
            f"index, got a MultiIndex with names {index.names!r}"
        )
    name = index.name
    if name == "date":
        if "time" in df.columns:
            raise FlashNHDatasetError(
                "FlashNHDataset._load_basin_data: basin data has both a 'date' "
                "index and a 'time' column; cannot unambiguously adapt to "
                "NeuralHydrology's required 'date' index name"
            )
        return df
    if name == "time":
        if "date" in df.columns:
            raise FlashNHDatasetError(
                "FlashNHDataset._load_basin_data: basin data has both a 'time' "
                "index and a 'date' column; cannot unambiguously adapt to "
                "NeuralHydrology's required 'date' index name"
            )
        return df.rename_axis("date")
    raise FlashNHDatasetError(
        f"FlashNHDataset._load_basin_data: expected temporal index named "
        f"'time' or 'date', got {name!r}"
    )


class FlashNHDataset(GenericDataset):
    """GenericDataset with a Flash-NH validity post-filter on the lookup table."""

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        # Narrowest override point: BaseDataset._load_or_create_xarray_dataset
        # calls self._load_basin_data(basin) once per basin, before any of its
        # date-name-dependent logic (duplicate handling, frequency inference,
        # xarray.Dataset.from_dataframe) runs. Call the parent GenericDataset
        # loader unchanged, then adapt only the temporal index name in memory.
        cutoff = self._observation_cutoff(basin)
        if cutoff is None:
            df = super()._load_basin_data(basin)
        else:
            df = self._load_basin_data_through_cutoff(basin, cutoff)
        return _adapt_temporal_index_to_date(df)

    def _observation_cutoff(self, basin: str):
        """The config's ``flashnh_observation_cutoff`` (``None`` when unset),
        after refusing the test period and any period whose last hour lies
        beyond it (fail closed rather than evaluating NaN-padded hours)."""
        from src.baseline.nh_raw_space_evaluation import (
            OBSERVATION_CUTOFF_CONFIG_KEY, RawSpaceEvaluationError, parse_observation_cutoff,
        )

        try:
            cutoff = parse_observation_cutoff(self.cfg.as_dict().get(OBSERVATION_CUTOFF_CONFIG_KEY))
        except RawSpaceEvaluationError as exc:
            raise FlashNHDatasetError(str(exc)) from exc
        if cutoff is None:
            return None
        if self.period == "test":
            raise FlashNHDatasetError(f"config declares observation cutoff {cutoff}; the test period is refused")
        last_hours = [_normalize_timestamp(d) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
                      for d in self.start_and_end_dates[basin]["end_dates"]]
        if max(last_hours) > cutoff:
            raise FlashNHDatasetError(
                f"{self.period} period for basin {basin} ends {max(last_hours)}, after observation cutoff {cutoff}")
        return cutoff

    def _load_basin_data_through_cutoff(self, basin: str, cutoff: pd.Timestamp) -> pd.DataFrame:
        """GenericDataset's per-basin NetCDF read, restricted lazily before any
        value is read: no variable past ``cutoff`` and no ``*_lead<NN>`` target
        past ``cutoff - NN h`` (NaN there, never read). Same file discovery
        rule as ``genericdataset.load_timeseries``."""
        from src.baseline.nh_raw_space_evaluation import select_through_observation_cutoff

        files_dir = Path(self.cfg.data_dir) / "time_series"
        candidates = [files_dir / f"{basin}{ext}" for ext in (".nc4", ".nc")]
        found = [path for path in candidates if path.is_file()]
        if len(found) != 1:
            raise FlashNHDatasetError(f"expected exactly one NetCDF file for basin {basin} in {files_dir}, "
                                      f"found {[str(p) for p in found]}")
        with xarray.open_dataset(found[0]) as ds:
            coordinate = "date" if "date" in ds.coords else "time"
            return select_through_observation_cutoff(ds, cutoff=cutoff, coordinate=coordinate).to_dataframe()

    def _create_lookup_table(self, xr: xarray.Dataset):
        super()._create_lookup_table(xr)
        self.flashnh_filter_stats = self._apply_flashnh_filters()

    def _apply_flashnh_filters(self) -> FlashNHFilterStats:
        cfg = self.cfg
        freq = self.frequencies[0]
        seq_length = self.seq_len[0]
        lead_hours = _parse_lead_hours(cfg.target_variables)

        gap_timestamps = _load_gap_timestamps(cfg.data_dir)
        full_timeline = _build_research_timeline(self._dates, freq)
        in_range_gaps, n_gap_outside = _classify_gap_timestamps(gap_timestamps, full_timeline)
        bad_hour_mask = bad_hour_mask_from_timestamps(
            full_timeline, in_range_gaps, on_out_of_range="error"
        )
        history_valid = compute_history_valid(full_timeline, bad_hour_mask, seq_length)
        timeline_pos = pd.Series(range(len(full_timeline)), index=full_timeline)

        n_before = len(self.lookup_table)
        removed_history = 0
        removed_boundary = 0
        removed_target = 0
        kept = {}
        new_key = 0

        lead_delta = pd.Timedelta(hours=lead_hours)
        for old_key in sorted(self.lookup_table.keys()):
            basin, indices = self.lookup_table[old_key]
            idx = indices[0]
            dates = self._dates[basin][freq]
            issue_ts = _normalize_timestamp(dates[idx])

            pos = timeline_pos.get(issue_ts)
            if pos is None or not history_valid[pos]:
                removed_history += 1
                continue

            period_end = _normalize_timestamp(dates[-1])
            if issue_ts + lead_delta > period_end:
                removed_boundary += 1
                continue

            target_value = self._y[basin][freq][idx][0]
            if torch.isnan(target_value):
                removed_target += 1
                continue

            kept[new_key] = (basin, indices)
            new_key += 1

        support = None
        n_natural = len(kept)
        support_path = cfg.as_dict().get(_COMMON_SUPPORT_CONFIG_KEY)
        if support_path not in (None, "None") and self.period == "train":
            support = self._load_common_support(Path(support_path), seq_length, lead_hours)
            kept = self._restrict_to_common_support(kept, support)
            self._write_common_support_receipt(Path(support_path), support, seq_length, n_natural, len(kept))

        self.lookup_table = kept
        self.num_samples = len(kept)

        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError

        return FlashNHFilterStats(
            period=self.period,
            seq_length=seq_length,
            lead_hours=lead_hours,
            n_before=n_before,
            removed_for_forcing_history=removed_history,
            removed_for_period_target_boundary=removed_boundary,
            removed_for_missing_target=removed_target,
            n_kept=len(kept),
            n_gap_timestamps_loaded=len(gap_timestamps),
            n_gap_timestamps_in_range=len(in_range_gaps),
            n_gap_timestamps_ignored_outside_range=n_gap_outside,
            common_support_id=None if support is None else support["support_id"],
            common_support_checksum_sha256=None if support is None else support["checksum_sha256"],
            n_natural_admitted=None if support is None else n_natural,
            removed_for_common_support=0 if support is None else n_natural - len(kept),
        )

    def _load_common_support(self, path: Path, seq_length: int, lead_hours: int) -> dict:
        """Load the checksummed training support and verify it describes this
        dataset (fail closed on any mismatch)."""
        from src.baseline.seqlen_common336_support import Common336SupportError, load_training_support

        cfg = self.cfg
        try:
            support = load_training_support(path)
        except Common336SupportError as exc:
            raise FlashNHDatasetError(f"invalid common training support {path}: {exc}") from exc

        def _single_date(value):
            if isinstance(value, list):
                if len(value) != 1:
                    raise FlashNHDatasetError("common training support requires a single train period")
                value = value[0]
            return pd.Timestamp(value)

        gap_path = Path(cfg.data_dir) / "masks" / "gap_timestamps.json"
        checks = {
            "seq_length <= seq_length_floor": seq_length <= support["seq_length_floor"],
            "lead_hours": support["lead_hours"] == lead_hours,
            "target_variable": [support["target_variable"]] == list(cfg.target_variables),
            "dynamic_inputs": isinstance(cfg.dynamic_inputs, list)
            and support["dynamic_inputs"] == list(cfg.dynamic_inputs),
            "date_start": pd.Timestamp(support["date_start"]) == _single_date(cfg.train_start_date),
            "date_end": pd.Timestamp(support["date_end"]) == _single_date(cfg.train_end_date),
            "gap_artifact_sha256": gap_path.is_file()
            and hashlib.sha256(gap_path.read_bytes()).hexdigest() == support["gap_artifact_sha256"],
            "basin set": set(support["basin_ids"]) == set(self.basins),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise FlashNHDatasetError(f"common training support {path} does not match this dataset: {failed}")
        return support

    def _write_common_support_receipt(self, path: Path, support: dict, seq_length: int, n_natural: int,
                                      n_kept: int) -> None:
        """Durable proof, next to NH's own train-data scaler, that this run's
        actual training loader applied the support (independent of logging)."""
        train_dir = getattr(self.cfg, "train_dir", None)
        if train_dir is None:
            return
        receipt = {"support_id": support["support_id"], "checksum_sha256": support["checksum_sha256"],
                   "support_file": str(path), "seq_length": int(seq_length), "n_natural_admitted": int(n_natural),
                   "n_kept": int(n_kept), "removed_for_common_support": int(n_natural - n_kept),
                   "n_basins": len(support["basin_ids"])}
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        (Path(train_dir) / COMMON_SUPPORT_RECEIPT_NAME).write_text(json.dumps(receipt, indent=1, sort_keys=True),
                                                                   encoding="utf-8")

    def _restrict_to_common_support(self, entries: dict, support: dict) -> dict:
        """Keep exactly the support's (basin, issue-time) pairs out of the
        naturally admitted ``entries``; raise if any support pair is absent."""
        from src.baseline.seqlen_common336_support import decode_issue_intervals

        freq = self.frequencies[0]
        member = {}
        for basin, intervals in support["per_basin_intervals"].items():
            if basin not in self._dates:
                continue  # NH dropped the basin; reported by the count check below
            dates = np.asarray(self._dates[basin][freq], dtype="datetime64[ns]")
            member[basin] = np.isin(dates, decode_issue_intervals(intervals))
        restricted = {}
        counts = dict.fromkeys(support["basin_ids"], 0)
        for _, (basin, indices) in sorted(entries.items()):
            if member[basin][indices[0]]:
                restricted[len(restricted)] = (basin, indices)
                counts[basin] += 1
        short = {b: (counts[b], n) for b, n in support["eligible_counts"].items() if counts[b] != n}
        if short:
            raise FlashNHDatasetError(
                f"{len(short)} basin(s) lack naturally admitted samples for common-support pairs "
                f"(kept, expected): {dict(list(sorted(short.items()))[:10])}")
        LOGGER.info("Flash-NH common training support %s (%s): %d of %d naturally admitted samples kept",
                    support["support_id"], support["checksum_sha256"][:12], len(restricted), len(entries))
        return restricted
