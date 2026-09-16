"""RD1-C4-D1 section H: the completeness-gated reducer.

The reducer's entire job is to refuse to produce a partial answer.

A 24x400 diagnostic is only interpretable as a whole. A table silently
assembled from 23 trials, or from 9,599 cells, would look exactly like a
complete one and would invite a conclusion the evidence does not support --
so every completeness condition below is a hard gate, checked before any
output file is written:

* only receipt-qualified completed shards are read at all (a shard without a
  valid matching receipt does not exist as far as this module is concerned);
* exactly the expected trial identities must be present -- no more, no
  fewer, no substitutions;
* exactly 12 Bayesian and 12 random-control trials;
* exactly :data:`~.rd1_c4_observation_diagnostic.EXPECTED_BASIN_COUNT` typed
  cell records per trial;
* the exact trial x basin cross-product, with no duplicate cell, no missing
  cell, no unexpected trial, and no unexpected basin;
* one schema name/version across every shard, and one contract identity and
  one package identity across every shard;
* every component hash re-verified against its receipt.

What it deliberately does NOT require is that cells be ``ok``. Typed error
cells are evidence and are reduced as such -- coverage and the status
distribution are reported, and an incomplete *measurement* is visible rather
than dropped. What it deliberately does NOT do is classify: no threshold is
selected, no pass/fail verdict is emitted, and no scientific conclusion
about the package-versus-pickle relationship is published. That reading
belongs to the user, from this evidence.

The whole output directory is published atomically, the same way a shard is:
every file is built under a private staging directory and the directory is
made visible with a single ``os.replace`` only after every output has been
written and hashed. A crash or kill partway through leaves the staging
directory behind for inspection and no trace at the public output path --
``out_dir`` is never observed half-written, and an existing ``out_dir`` (a
previous complete reduction) is refused rather than silently overwritten.

Outputs::

    cells.parquet                        the 9,600-row typed cell table
    cell_status.csv                      status distribution by trial and overall
    extreme_elements.csv                 every retained extreme record
    ulp_or_source_precision_histogram.csv  summed ordered-bit distance bins
    q98_consequences.csv                 per trial x basin Q98 consequence facts
    summary.json                         coverage, identities, counts
    receipt_index.csv                    one row per consumed shard receipt
    manifest.csv                         every output file with size and hash
    checksums.sha256                     sha256sum-format checksums
    reduction.log                        human-readable completion log
"""
from __future__ import annotations

import csv
import io
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .atomic_shard_store import (
    AtomicShardStore,
    ShardStoreError,
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_json_sha256,
    fsync_dir,
    replace_dir,
    sha256_path,
)
from .rd1_c4_observation_diagnostic import (
    CELL_STATUSES,
    D1_SCHEMA_NAME,
    D1_SCHEMA_VERSION,
    D1_SHARD_FAMILY,
    EXPECTED_BASIN_COUNT,
    EXTREME_KINDS,
)
from .rd1_c4_trial_authentication import EXPECTED_ARM_COUNTS, EXPECTED_TRIAL_COUNT

__all__ = [
    "ReductionError",
    "EXPECTED_TRIAL_COUNT",
    "EXPECTED_ARM_COUNTS",
    "EXPECTED_CELL_COUNT",
    "reduce_observation_diagnostic",
]


class ReductionError(RuntimeError):
    """Raised for any completeness, identity, or integrity failure. The
    reducer writes nothing when it raises."""


EXPECTED_CELL_COUNT = EXPECTED_TRIAL_COUNT * EXPECTED_BASIN_COUNT  # 9,600

_CELL_TABLE_COLUMNS = (
    "trial_id",
    "search_arm",
    "basin_id",
    "status",
    "status_detail",
    "n_compared",
    "n_bitwise_equal",
    "n_unequal",
    "n_nonfinite_package",
    "n_nonfinite_reconstructed",
    "abs_diff_max",
    "abs_diff_mean",
    "abs_diff_rms",
    "abs_diff_median",
    "abs_diff_p50",
    "abs_diff_p90",
    "abs_diff_p99",
    "abs_diff_p99_9",
    "rel_diff_package_reference_max",
    "rel_diff_symmetric_max",
    "source_precision_distance_max",
    "n_source_precision_undefined",
    "n_reconstruction_not_float32_representable",
    "spacing_ratio_max",
    "n_exceeding_provisional_envelope",
    "max_provisional_envelope_exceedance_factor",
    "n_detail_rows",
    "n_extremes_recorded",
    "q98_any_frozen_decision_changes",
    "area_km2",
    "area_relative_mad",
    "n_area_samples",
    "package_basin_sha256",
)

_Q98_COLUMNS = (
    "trial_id",
    "basin_id",
    "quantile",
    "n_admitted",
    "q98_threshold_package",
    "q98_threshold_reconstructed",
    "q98_threshold_abs_diff",
    "q98_threshold_rel_diff",
    "n_high_flow_package",
    "n_high_flow_reconstructed",
    "high_flow_mask_symmetric_difference",
    "high_flow_mask_sha256_package",
    "high_flow_mask_sha256_reconstructed",
    "observed_peak_value_package",
    "observed_peak_value_reconstructed",
    "observed_peak_index_package",
    "observed_peak_index_reconstructed",
    "peak_tie_count_package",
    "peak_tie_count_reconstructed",
    "earliest_tied_peak_date_ns_package",
    "earliest_tied_peak_date_ns_reconstructed",
    "peak_index_equal",
    "peak_timestamp_equal",
    "high_flow_membership_equal",
    "any_frozen_decision_changes",
)

_EXTREME_COLUMNS = (
    "trial_id",
    "basin_id",
    "extreme_kind",
    "support_index",
    "date_ns",
    "package_value_m3s",
    "package_value_float32_bits",
    "reconstructed_value_m3s",
    "abs_diff",
    "rel_diff_package_reference",
    "rel_diff_symmetric",
    "source_precision_distance",
    "source_precision_defined",
    "spacing_ratio",
    "provisional_tolerance",
    "exceeds_provisional_envelope",
    "provisional_envelope_exceedance_factor",
)


def _read_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ReductionError(f"{path}:{line_number}: unparseable cell record: {exc}") from exc
    return records


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping]) -> bytes:
    """Deterministic UTF-8 CSV bytes with ``\\n`` line endings, so the same
    rows always produce the same checksum regardless of platform."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    return buffer.getvalue().encode("utf-8")


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def reduce_observation_diagnostic(
    *,
    store_root,
    out_dir,
    expected_trials: Mapping[str, str],
    expected_basin_ids: Sequence[str],
    expected_trial_count: int = EXPECTED_TRIAL_COUNT,
    expected_arm_counts: Optional[Mapping[str, int]] = None,
    expected_basin_count: int = EXPECTED_BASIN_COUNT,
) -> dict:
    """Reduce every completed D1 trial shard into the compact output set.

    ``expected_trials`` maps each expected ``trial_id`` to its search arm
    (``"bayesian"`` or ``"random_control"``). It is supplied by the caller
    rather than discovered from the store on purpose: a reducer that learns
    which trials to expect from whichever shards happen to exist cannot
    detect a missing trial, which is the single most important thing this
    gate has to catch.

    Raises :class:`ReductionError` -- writing nothing at all -- on any
    completeness, identity, schema, or hash failure.
    """
    store = AtomicShardStore(store_root, D1_SHARD_FAMILY)
    out_dir = Path(out_dir)
    expected_arm_counts = dict(expected_arm_counts or EXPECTED_ARM_COUNTS)
    expected_basin_ids = list(expected_basin_ids)
    expected_basin_set = set(expected_basin_ids)
    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- gate 1: the expected population is itself well formed ------------ #
    if len(expected_trials) != expected_trial_count:
        raise ReductionError(
            f"expected exactly {expected_trial_count} trial identities, was given {len(expected_trials)}"
        )
    arm_counts = Counter(expected_trials.values())
    for arm, count in expected_arm_counts.items():
        if arm_counts.get(arm, 0) != count:
            raise ReductionError(
                f"expected exactly {count} {arm!r} trials, was given {arm_counts.get(arm, 0)}"
            )
    unexpected_arms = set(arm_counts) - set(expected_arm_counts)
    if unexpected_arms:
        raise ReductionError(f"unexpected search arm(s) in the expected population: {sorted(unexpected_arms)}")
    if len(expected_basin_set) != expected_basin_count:
        raise ReductionError(
            f"expected exactly {expected_basin_count} distinct basin ids, was given {len(expected_basin_set)}"
        )

    # -- gate 2: exactly the expected receipt-qualified shards ------------ #
    try:
        completed = set(store.list_completed())
    except ShardStoreError as exc:
        raise ReductionError(f"shard store is not in a reducible state: {exc}") from exc
    missing = sorted(set(expected_trials) - completed)
    unexpected = sorted(completed - set(expected_trials))
    if missing or unexpected:
        raise ReductionError(
            "shard population does not match the expected 24 trials -- refusing to reduce a partial "
            f"cross-product. missing={missing} unexpected={unexpected}"
        )

    cells: list = []
    extremes: list = []
    q98_rows: list = []
    receipt_rows: list = []
    histogram_totals: Counter = Counter()
    seen_cells: set = set()
    schema_seen: set = set()
    contract_identities: set = set()
    package_identities: set = set()

    for trial_id in sorted(expected_trials):
        arm = expected_trials[trial_id]
        try:
            receipt = store.verify(trial_id)  # gate 3: re-hash every component
        except ShardStoreError as exc:
            raise ReductionError(f"{trial_id}: shard failed hash verification: {exc}") from exc

        shard_dir = store.shard_dir(trial_id)
        summary = json.loads((shard_dir / "trial_summary.json").read_text(encoding="utf-8"))
        if summary.get("schema_name") != D1_SCHEMA_NAME or summary.get("schema_version") != D1_SCHEMA_VERSION:
            raise ReductionError(
                f"{trial_id}: shard schema {summary.get('schema_name')}/{summary.get('schema_version')} "
                f"!= {D1_SCHEMA_NAME}/{D1_SCHEMA_VERSION}"
            )
        identity = summary.get("identity", {})
        contract_identities.add(
            (identity.get("contract_id"), identity.get("contract_checksum_sha256"))
        )
        package = identity.get("package", {})
        package_identities.add(
            (
                package.get("package_manifest_sha256"),
                package.get("package_file_checksums_sha256"),
                package.get("package_run_provenance_sha256"),
            )
        )
        if summary.get("search_arm") != arm:
            raise ReductionError(
                f"{trial_id}: shard records search arm {summary.get('search_arm')!r}, expected {arm!r}"
            )

        trial_cells = _read_jsonl(shard_dir / "cells.jsonl")
        if len(trial_cells) != expected_basin_count:
            raise ReductionError(
                f"{trial_id}: shard holds {len(trial_cells)} cell records, expected exactly "
                f"{expected_basin_count}"
            )
        trial_basins = set()
        for cell in trial_cells:
            schema_seen.add((cell.get("schema_name"), cell.get("schema_version")))
            basin_id = cell.get("basin_id")
            if cell.get("trial_id") != trial_id:
                raise ReductionError(
                    f"{trial_id}: cell record claims trial {cell.get('trial_id')!r}"
                )
            if basin_id not in expected_basin_set:
                raise ReductionError(f"{trial_id}: unexpected basin id {basin_id!r} in cell records")
            if basin_id in trial_basins:
                raise ReductionError(f"{trial_id}: duplicate cell record for basin {basin_id!r}")
            if cell.get("status") not in CELL_STATUSES:
                raise ReductionError(
                    f"{trial_id}/{basin_id}: unknown cell status {cell.get('status')!r}"
                )
            trial_basins.add(basin_id)
            key = (trial_id, basin_id)
            if key in seen_cells:
                raise ReductionError(f"duplicate cell {key} across shards")
            seen_cells.add(key)
            row = dict(cell)
            row["search_arm"] = arm
            cells.append(row)
            for label, count in (cell.get("source_precision_histogram") or {}).items():
                histogram_totals[label] += int(count)

        missing_basins = expected_basin_set - trial_basins
        if missing_basins:
            raise ReductionError(
                f"{trial_id}: {len(missing_basins)} expected basins have no cell record, e.g. "
                f"{sorted(missing_basins)[:5]}"
            )

        for record in _read_jsonl(shard_dir / "extremes.jsonl"):
            if record.get("extreme_kind") not in EXTREME_KINDS:
                raise ReductionError(
                    f"{trial_id}: unknown extreme kind {record.get('extreme_kind')!r}"
                )
            extremes.append(record)
        q98_rows.extend(_read_jsonl(shard_dir / "q98_consequences.jsonl"))

        detail_index = summary.get("detail_index", {})
        receipt_rows.append(
            {
                "trial_id": trial_id,
                "search_arm": arm,
                "receipt_created_at_utc": receipt.created_at_utc,
                "identity_sha256": receipt.identity_sha256,
                "content_sha256": receipt.content_sha256,
                "n_components": len(receipt.components),
                "n_cells": len(trial_cells),
                "detail_sha256": detail_index.get("sha256"),
                "detail_n_rows": detail_index.get("n_rows"),
                "detail_size_bytes": detail_index.get("size_bytes"),
                "validation_pickle_sha256": identity.get("validation_pickle_sha256"),
                "contract_checksum_sha256": identity.get("contract_checksum_sha256"),
                "package_manifest_sha256": package.get("package_manifest_sha256"),
                "git_head": identity.get("git_head"),
            }
        )

    # -- gate 4: one schema, one contract, one package across all shards -- #
    if schema_seen != {(D1_SCHEMA_NAME, D1_SCHEMA_VERSION)}:
        raise ReductionError(f"cell records disagree on schema identity: {sorted(schema_seen)}")
    if len(contract_identities) != 1:
        raise ReductionError(
            f"shards disagree on the fixed-support contract identity: {sorted(contract_identities)}"
        )
    if len(package_identities) != 1:
        raise ReductionError(f"shards disagree on the package identity: {sorted(package_identities)}")

    # -- gate 5: the exact cross-product ---------------------------------- #
    expected_keys = {(trial_id, basin_id) for trial_id in expected_trials for basin_id in expected_basin_ids}
    if seen_cells != expected_keys:
        raise ReductionError(
            f"cell set is not the exact {expected_trial_count}x{expected_basin_count} cross-product: "
            f"{len(seen_cells)} cells present, {len(expected_keys)} expected"
        )
    if len(cells) != expected_trial_count * expected_basin_count:
        raise ReductionError(
            f"assembled {len(cells)} cells, expected exactly {expected_trial_count * expected_basin_count}"
        )

    # -- every gate passed: only now is anything written ------------------ #
    #
    # The whole output directory is published the same way a shard is: built
    # in full under a staging directory the public ``out_dir`` path never
    # points at, then made visible in one ``os.replace``. A crash or kill
    # partway through writing leaves only the staging directory (inspectable,
    # never cleaned up automatically -- same non-goal as abandoned shard
    # attempts in :mod:`atomic_shard_store`); ``out_dir`` itself either does
    # not exist yet or is a previous, complete reduction, never a partial one.
    # An existing ``out_dir`` is refused rather than silently overwritten, so
    # a complete prior reduction is never lost without an operator noticing.
    cells.sort(key=lambda row: (row["trial_id"], row["basin_id"]))
    extremes.sort(key=lambda row: (row["trial_id"], row["basin_id"], row["extreme_kind"]))
    q98_rows.sort(key=lambda row: (row["trial_id"], row["basin_id"]))
    receipt_rows.sort(key=lambda row: row["trial_id"])

    if out_dir.exists():
        raise ReductionError(
            f"refusing to reduce into {out_dir}: it already exists. A reduction output directory is "
            "never overwritten, complete or partial -- remove or rename it explicitly first."
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}.attempt.", dir=out_dir.parent))
    written: dict = {}

    written["cells.parquet"] = _write_cells_parquet(staging_dir / "cells.parquet", cells)
    written["cell_status.csv"] = atomic_write_bytes(
        staging_dir / "cell_status.csv", _cell_status_bytes(cells, expected_trials)
    )
    written["extreme_elements.csv"] = atomic_write_bytes(
        staging_dir / "extreme_elements.csv", _csv_bytes(_EXTREME_COLUMNS, extremes)
    )
    written["ulp_or_source_precision_histogram.csv"] = atomic_write_bytes(
        staging_dir / "ulp_or_source_precision_histogram.csv", _histogram_bytes(histogram_totals)
    )
    written["q98_consequences.csv"] = atomic_write_bytes(
        staging_dir / "q98_consequences.csv", _csv_bytes(_Q98_COLUMNS, q98_rows)
    )
    written["receipt_index.csv"] = atomic_write_bytes(
        staging_dir / "receipt_index.csv", _csv_bytes(list(receipt_rows[0].keys()), receipt_rows)
    )

    status_counts = Counter(row["status"] for row in cells)
    ok_cells = [row for row in cells if row["status"] == "ok"]
    summary_payload = {
        "schema_name": D1_SCHEMA_NAME,
        "schema_version": D1_SCHEMA_VERSION,
        "reduced_at_utc": started_utc,
        "n_trials": expected_trial_count,
        "n_basins_per_trial": expected_basin_count,
        "n_cells": len(cells),
        "search_arm_counts": dict(Counter(row["search_arm"] for row in cells)),
        "cell_status_counts": {status: int(status_counts.get(status, 0)) for status in CELL_STATUSES},
        "coverage": {
            "n_cells_compared": len(ok_cells),
            "fraction_cells_compared": len(ok_cells) / len(cells) if cells else None,
            "n_cells_not_compared": len(cells) - len(ok_cells),
            "n_elements_compared": sum(int(row.get("n_compared") or 0) for row in ok_cells),
            "n_elements_bitwise_equal": sum(int(row.get("n_bitwise_equal") or 0) for row in ok_cells),
            "n_elements_unequal": sum(int(row.get("n_unequal") or 0) for row in ok_cells),
            "n_elements_exceeding_provisional_envelope": sum(
                int(row.get("n_exceeding_provisional_envelope") or 0) for row in ok_cells
            ),
            "n_detail_rows": sum(int(row.get("n_detail_rows") or 0) for row in ok_cells),
        },
        "q98_consequences": {
            "n_records": len(q98_rows),
            "n_basins_with_changed_frozen_decision": sum(
                1 for row in q98_rows if row.get("any_frozen_decision_changes")
            ),
            "n_basins_with_changed_membership": sum(
                1 for row in q98_rows if not row.get("high_flow_membership_equal")
            ),
            "n_basins_with_changed_peak_timestamp": sum(
                1 for row in q98_rows if not row.get("peak_timestamp_equal")
            ),
        },
        "source_precision_histogram_totals": dict(sorted(histogram_totals.items())),
        "n_extreme_records": len(extremes),
        "extreme_kind_counts": dict(sorted(Counter(row["extreme_kind"] for row in extremes).items())),
        "contract_identity": sorted(contract_identities)[0],
        "package_identity": sorted(package_identities)[0],
        "receipt_index_sha256": canonical_json_sha256(receipt_rows),
        "provisional_envelope_is_report_only": True,
        "final_rd1_c4_audit_policy": "unresolved",
        "scientific_classification": (
            "not performed -- this reduction reports measurements only and selects no threshold"
        ),
    }
    written["summary.json"] = atomic_write_bytes(
        staging_dir / "summary.json", canonical_json_bytes(summary_payload)
    )
    written["reduction.log"] = atomic_write_bytes(
        staging_dir / "reduction.log", _reduction_log_bytes(summary_payload).encode("utf-8")
    )

    manifest_rows = [
        {
            "relative_path": name,
            "sha256": digest,
            "size_bytes": (staging_dir / name).stat().st_size,
        }
        for name, digest in sorted(written.items())
    ]
    atomic_write_bytes(
        staging_dir / "manifest.csv", _csv_bytes(("relative_path", "sha256", "size_bytes"), manifest_rows)
    )
    atomic_write_bytes(
        staging_dir / "checksums.sha256",
        "".join(f"{row['sha256']}  {row['relative_path']}\n" for row in manifest_rows).encode("utf-8"),
    )
    summary_payload["outputs"] = manifest_rows

    # Made visible in one step. If ``out_dir`` was created by a concurrent
    # attempt since the check above, the replace still refuses to replace a
    # non-empty directory (POSIX and Windows both raise), so two racing
    # reductions can never silently merge or clobber one another; exactly one
    # wins and the other's staging directory is left for inspection.
    replace_dir(staging_dir, out_dir)
    fsync_dir(out_dir.parent)
    return summary_payload


def _write_cells_parquet(path: Path, cells: Sequence[Mapping]) -> str:
    """Write the 9,600-row typed cell table.

    Published atomically like every other output: written to a sibling
    temporary file and renamed, so a reader never sees a half-written table.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {}
    for name in _CELL_TABLE_COLUMNS:
        columns[name] = pa.array([_parquet_value(row.get(name)) for row in cells])
    table = pa.table(columns)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return sha256_path(path)


def _parquet_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _cell_status_bytes(cells: Sequence[Mapping], expected_trials: Mapping[str, str]) -> bytes:
    """Status distribution per trial plus an explicit ``__all__`` total row,
    with one column per status so a zero is visibly zero rather than absent."""
    rows = []
    by_trial: dict = {trial_id: Counter() for trial_id in expected_trials}
    overall: Counter = Counter()
    for cell in cells:
        by_trial[cell["trial_id"]][cell["status"]] += 1
        overall[cell["status"]] += 1
    for trial_id in sorted(expected_trials):
        row = {"trial_id": trial_id, "search_arm": expected_trials[trial_id], "n_cells": sum(by_trial[trial_id].values())}
        row.update({status: by_trial[trial_id].get(status, 0) for status in CELL_STATUSES})
        rows.append(row)
    total = {"trial_id": "__all__", "search_arm": "__all__", "n_cells": len(cells)}
    total.update({status: overall.get(status, 0) for status in CELL_STATUSES})
    rows.append(total)
    return _csv_bytes(("trial_id", "search_arm", "n_cells", *CELL_STATUSES), rows)


def _histogram_bytes(histogram_totals: Mapping[str, int]) -> bytes:
    rows = [
        {"bin": label, "n_elements": int(count)}
        for label, count in sorted(histogram_totals.items(), key=lambda item: _histogram_sort_key(item[0]))
    ]
    return _csv_bytes(("bin", "n_elements"), rows)


def _histogram_sort_key(label: str) -> tuple:
    """Sort the bins by their numeric lower edge, with ``undefined`` last --
    lexicographic order would put ``[1024,inf)`` before ``[2,4)``."""
    if label == "undefined":
        return (1, float("inf"))
    try:
        return (0, float(label.split(",")[0].lstrip("[")))
    except ValueError:
        return (1, float("inf"))


def _reduction_log_bytes(summary: Mapping[str, Any]) -> str:
    coverage = summary["coverage"]
    q98 = summary["q98_consequences"]
    lines = [
        "RD1-C4-D1 observation diagnostic -- reduction",
        f"reduced_at_utc={summary['reduced_at_utc']}",
        "",
        f"trials={summary['n_trials']} basins_per_trial={summary['n_basins_per_trial']} "
        f"cells={summary['n_cells']}",
        "search_arm_counts=" + json.dumps(summary["search_arm_counts"], sort_keys=True),
        "cell_status_counts=" + json.dumps(summary["cell_status_counts"], sort_keys=True),
        "",
        f"cells compared: {coverage['n_cells_compared']} / {summary['n_cells']}",
        f"elements compared: {coverage['n_elements_compared']}",
        f"elements bitwise equal: {coverage['n_elements_bitwise_equal']}",
        f"elements unequal: {coverage['n_elements_unequal']}",
        "elements beyond the PROVISIONAL report-only envelope: "
        f"{coverage['n_elements_exceeding_provisional_envelope']}",
        f"detail rows retained: {coverage['n_detail_rows']}",
        "",
        f"Q98 records: {q98['n_records']}",
        f"basins whose frozen Q98 decision would change: {q98['n_basins_with_changed_frozen_decision']}",
        f"  changed high-flow membership: {q98['n_basins_with_changed_membership']}",
        f"  changed observed-peak timestamp: {q98['n_basins_with_changed_peak_timestamp']}",
        "",
        "The envelope counted above is a REPORTED comparison field, not a pass criterion.",
        "This reduction selects no threshold, classifies nothing, and closes nothing:",
        "the final RD1-C4 audit policy remains unresolved and is the user's decision.",
    ]
    return "\n".join(lines) + "\n"
