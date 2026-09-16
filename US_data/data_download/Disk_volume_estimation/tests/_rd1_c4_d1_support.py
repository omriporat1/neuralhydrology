"""Small synthetic fixtures for the RD1-C4-D1 tests.

Everything here is generated locally into ``tmp_path``. No real Moriah
product, no real package, no real validation pickle, and no sealed scope is
touched by any test that uses these helpers.

The point of the builder is that it produces a *complete, self-consistent*
package in the shape the real producer (``src.baseline.package_builder``)
writes -- the six required metadata files plus ``time_series/<basin>.nc``,
a builder manifest carrying the producer's own schema identity, package
role, basin table, lead targets, timeline and gap-artifact binding, a
closed-world ``manifests/file_checksums.csv`` with the producer's artifact
roles, and a ``run_provenance.json`` carrying the producer's own keys -- and
a fixed-support contract whose three package-identity hashes are the real
SHA-256 of those three payloads. Tests then break exactly one fact at a time
to prove that each identity check is actually load-bearing.

RD1-C4-D1 Correction Pass A: these fixtures were previously shaped to what
the qualifier happened to read rather than to what the producer writes, so
they could not have caught a qualifier that accepted an unknown schema. They
are now producer-shaped, which is why the strict manifest/provenance/checksum
parsing in ``package_identity_qualification`` can be exercised at all.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr

from src.baseline import fixed_support_contract_v2 as fixed
from src.baseline.nh_seed_evaluation import weight_stem
from src.baseline.sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2
from src.baseline.sweep_v1_campaign import MODEL_SEED_A
from src.baseline.sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2,
    DOMAIN_VERSION_V2,
    FIDELITY_ID_V2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
)
from src.baseline.sweep_v2_six_axis_config import V2_METRIC_NAME

LEAD_HOURS = 6
TARGET_VARIABLE = "qobs_mm_per_h_lead06"
AREA_KM2 = 875.5424998122709  # the real 06911000 area, so magnitudes are realistic

# Producer identity, mirrored from src.baseline.package_audit's independent
# redeclaration of the real builder's own schema contract. The fixture
# declares the ``date``-coordinate scientific-v002 NetCDF schema because that
# is the coordinate these fixture datasets actually use.
BUILDER_MANIFEST_SCHEMA_NAME = "stage1_compact_scientific_package_builder_v001"
BUILDER_MANIFEST_SCHEMA_VERSION = 1
BUILDER_MODULE = "src.baseline.package_builder"
NETCDF_SCHEMA_NAME = "stage1_scientific_package_v002"
NETCDF_SCHEMA_VERSION = 2
NETCDF_TIME_COORDINATE = "date"
PACKAGE_ROLE = "stage1_scientific_package"
TIME_SERIES_ROLE = "authoritative_time_series"
GAP_MASK_RELATIVE_PATH = "masks/gap_timestamps.json"
FIXED_ARTIFACT_ROLES = {
    "attributes/attributes.csv": "authoritative_static_attributes",
    "basins/basin_ids.txt": "authoritative_basin_list",
    GAP_MASK_RELATIVE_PATH: "authoritative_gap_mask",
}


def write_fixed_authoritative_artifacts(package_root, basin_ids):
    """Write the three fixed authoritative artifacts the producer always
    emits, so the package has the layout ``package_audit.check_package_layout``
    requires and the checksum manifest can be closed-world."""
    package_root = Path(package_root)
    (package_root / "attributes").mkdir(parents=True, exist_ok=True)
    (package_root / "basins").mkdir(parents=True, exist_ok=True)
    (package_root / "masks").mkdir(parents=True, exist_ok=True)
    ids = list(basin_ids)
    attributes = "gauge_id,area_km2\n" + "".join(f"{b},{AREA_KM2}\n" for b in ids)
    (package_root / "attributes" / "attributes.csv").write_bytes(attributes.encode("utf-8"))
    (package_root / "basins" / "basin_ids.txt").write_bytes(
        ("\n".join(ids) + "\n").encode("utf-8")
    )
    (package_root / GAP_MASK_RELATIVE_PATH).write_bytes(
        json.dumps({basin_id: [] for basin_id in ids}, sort_keys=True).encode("utf-8")
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_package(
    tmp_path,
    basin_ids,
    *,
    n_hours: int = 240,
    seed: int = 7,
    area_km2: float = AREA_KM2,
    package_name: str = "package",
):
    """Write a complete synthetic package and return
    ``(package_root, package_dates, qobs_by_basin)``.

    ``qobs_m3s`` is stored as float32 exactly as the real package does, and
    the derived ``qobs_mm_per_h_lead06`` target is computed in float64 and
    then stored as float32 -- which is precisely the "two independently
    quantized float32 representations" situation D1 exists to measure.
    """
    package_root = Path(tmp_path) / package_name
    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    dates = np.arange(
        np.datetime64("2024-01-01T00", "ns"),
        np.datetime64("2024-01-01T00", "ns") + np.timedelta64(n_hours, "h"),
        np.timedelta64(1, "h"),
    )
    qobs_by_basin = {}
    for index, basin_id in enumerate(basin_ids):
        rng = np.random.default_rng(seed + index)
        qobs_m3s = rng.uniform(0.01, 300.0, size=n_hours).astype(np.float32)
        usable = n_hours - LEAD_HOURS
        target = np.full(n_hours, np.nan, dtype=np.float64)
        target[:usable] = 3.6 * qobs_m3s[LEAD_HOURS:].astype(np.float64) / area_km2
        xr.Dataset(
            {
                "qobs_m3s": ("date", qobs_m3s),
                TARGET_VARIABLE: ("date", target.astype(np.float32)),
            },
            coords={"date": dates},
        ).to_netcdf(ts_dir / f"{basin_id}.nc")
        qobs_by_basin[basin_id] = qobs_m3s
    write_package_manifests(package_root, basin_ids, dates)
    return package_root, dates, qobs_by_basin


def write_package_manifests(package_root, basin_ids, dates, *, manifest_overrides=None,
                            run_provenance_overrides=None, extra_checksum_rows=(),
                            drop_checksum_rows=()):
    """(Re)write the producer-shaped metadata so it describes what is actually
    on disk. Called again by tests that mutate a basin file.

    The override/extra/drop hooks exist so a test can break exactly one
    producer-declared fact (an unknown schema name, a duplicate or
    path-traversing checksum row, a dry-run provenance record) without
    hand-writing a whole package.
    """
    package_root = Path(package_root)
    manifests_dir = package_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_fixed_authoritative_artifacts(package_root, basin_ids)

    rows = []
    for relative, role in sorted(FIXED_ARTIFACT_ROLES.items()):
        payload = (package_root / relative).read_bytes()
        rows.append(
            {
                "relative_path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
                "artifact_role": role,
            }
        )

    entries = []
    for basin_id in basin_ids:
        relative = f"time_series/{basin_id}.nc"
        path = package_root / relative
        payload = path.read_bytes()
        digest = sha256_bytes(payload)
        rows.append(
            {
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": len(payload),
                "artifact_role": TIME_SERIES_ROLE,
            }
        )
        entries.append(
            {
                "basin_id": basin_id,
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": len(payload),
            }
        )

    dropped = set(drop_checksum_rows)
    rows = [row for row in rows if row["relative_path"] not in dropped]
    rows.extend(dict(row) for row in extra_checksum_rows)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["relative_path", "sha256", "size_bytes", "artifact_role"],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    (manifests_dir / "file_checksums.csv").write_bytes(buffer.getvalue().encode("utf-8"))

    gap_payload = (package_root / GAP_MASK_RELATIVE_PATH).read_bytes()
    manifest = {
        "schema_name": BUILDER_MANIFEST_SCHEMA_NAME,
        "schema_version": BUILDER_MANIFEST_SCHEMA_VERSION,
        "package_role": PACKAGE_ROLE,
        "netcdf_package_schema_name": NETCDF_SCHEMA_NAME,
        "netcdf_package_schema_version": NETCDF_SCHEMA_VERSION,
        "netcdf_time_coordinate": NETCDF_TIME_COORDINATE,
        "basin_count": len(list(basin_ids)),
        "basin_ids": list(basin_ids),
        "raw_target_variable": "qobs_m3s",
        "lead_targets": [{"name": TARGET_VARIABLE, "lead_hours": LEAD_HOURS}],
        "timeline": {
            "start": str(np.datetime64(dates[0], "s")),
            "end": str(np.datetime64(dates[-1], "s")),
            "rows": int(len(dates)),
            "frequency": "1h",
        },
        "gap_timestamp_artifact": {
            "relative_path": GAP_MASK_RELATIVE_PATH,
            "sha256": sha256_bytes(gap_payload),
            "count": 0,
        },
        "per_basin_time_series": entries,
    }
    manifest.update(manifest_overrides or {})
    (manifests_dir / "package_manifest.json").write_bytes(
        json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    )

    run_provenance = {
        "builder_module": BUILDER_MODULE,
        "builder_schema_version": BUILDER_MANIFEST_SCHEMA_VERSION,
        "package_schema_name": BUILDER_MANIFEST_SCHEMA_NAME,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "dry_run": False,
        "basin_count": len(list(basin_ids)),
        "qc_csv_enabled": False,
        "netcdf_package_schema_name": NETCDF_SCHEMA_NAME,
        "netcdf_package_schema_version": NETCDF_SCHEMA_VERSION,
        "netcdf_time_coordinate": NETCDF_TIME_COORDINATE,
    }
    run_provenance.update(run_provenance_overrides or {})
    (package_root / "run_provenance.json").write_bytes(
        json.dumps(run_provenance, sort_keys=True, indent=2).encode("utf-8")
    )


def package_identity_hashes(package_root) -> dict:
    package_root = Path(package_root)
    return {
        "package_manifest_sha256": sha256_bytes((package_root / "manifests" / "package_manifest.json").read_bytes()),
        "package_file_checksums_sha256": sha256_bytes(
            (package_root / "manifests" / "file_checksums.csv").read_bytes()
        ),
        "package_run_provenance_sha256": sha256_bytes((package_root / "run_provenance.json").read_bytes()),
    }


def build_contract(package_root, basin_ids, dates, *, n_support=None, **overrides) -> dict:
    """A fixed-support contract bound to this package by real hashes.

    Support timestamps stop ``LEAD_HOURS`` before the package's last date so
    every lead-shifted lookup ``t + 6h`` exists -- which is the alignment the
    contract genuinely has, not a convenience.
    """
    hashes = package_identity_hashes(package_root)
    hashes.update(overrides)
    usable = len(dates) - LEAD_HOURS
    if n_support is not None:
        usable = min(usable, n_support)
    support = dates[:usable]
    return fixed.build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2,
        lead_hours=LEAD_HOURS,
        target_variable=TARGET_VARIABLE,
        period="validation",
        date_start=str(np.datetime64(dates[0], "s")),
        date_end=str(np.datetime64(dates[usable - 1], "s")),
        source_gap_policy_identity="fixture_gap_v001",
        screening_basin_ids_sha256="0" * 64,
        development_split_sha256="d" * 64,
        spatial_holdout_split_sha256="e" * 64,
        per_basin_date={basin_id: support for basin_id in basin_ids},
        per_basin_admitted={basin_id: np.ones(len(support), dtype=bool) for basin_id in basin_ids},
        **hashes,
    )


def write_validation_pickle(run_dir, epoch, *, basin_ids, contract, qobs_by_basin, area_km2=AREA_KM2,
                            obs_override=None, omit_basins=(), corrupt_basins=()):
    """Write a real ``validation_results.p`` whose observations are the
    package's own values passed through the same mm/h storage NH uses.

    The round trip ``m^3/s (float32) -> mm/h (float32) -> m^3/s (float64)``
    is deliberately lossy in exactly the way the real pipeline is, so the
    default fixture produces genuine, small, non-bitwise-equal differences
    rather than an artificial contrivance.

    ``obs_override(basin_id, obs_mm_per_h) -> obs_mm_per_h`` lets a test
    inject a controlled disagreement; ``omit_basins`` and ``corrupt_basins``
    exercise the ``basin_missing_from_trial`` and ``pickle_load_error``
    statuses.
    """
    run_dir = Path(run_dir)
    results = {}
    for basin_id in basin_ids:
        if basin_id in omit_basins:
            continue
        if basin_id in corrupt_basins:
            results[basin_id] = {"1h": {"xr": None}}
            continue
        support = fixed._deserialize_date_array(
            contract["per_basin_support"][basin_id], contract["date_dtype"]
        )
        qobs = qobs_by_basin[basin_id]
        # The package value aligned to support timestamp t lives at t + lead.
        aligned = qobs[LEAD_HOURS : LEAD_HOURS + len(support)].astype(np.float64)
        obs_mm_per_h = (3.6 * aligned / area_km2).astype(np.float32)
        if obs_override is not None:
            obs_mm_per_h = obs_override(basin_id, obs_mm_per_h)
        sim_mm_per_h = (obs_mm_per_h.astype(np.float64) * 0.9).astype(np.float32)
        results[basin_id] = {
            "1h": {
                "xr": xr.Dataset(
                    {
                        f"{TARGET_VARIABLE}_obs": ("date", obs_mm_per_h.astype(np.float64)),
                        f"{TARGET_VARIABLE}_sim": ("date", sim_mm_per_h.astype(np.float64)),
                    },
                    coords={"date": support},
                )
            }
        }
    period_dir = run_dir / "validation" / weight_stem(epoch)
    period_dir.mkdir(parents=True, exist_ok=True)
    with open(period_dir / "validation_results.p", "wb") as handle:
        pickle.dump(results, handle)
    return run_dir


def write_synthetic_execution_receipt(
    root,
    label,
    *,
    contract,
    run_dir,
    search_arm="bayesian",
    proposal_order=1,
    execution_generation=1,
    hyperparameters=None,
    trajectory=None,
):
    """Write a small self-consistent v2 execution receipt for auth tests.

    This does not emulate execution.  It writes the exact authoritative
    receipt interface consumed by ``build_v2_best_epoch_source`` with every
    identity derived by the production campaign helpers.
    """
    hp = dict(
        hyperparameters
        or {
            "learning_rate": 3e-4,
            "hidden_size": 128,
            "embedding_dropout": 0.10,
            "output_dropout": 0.25,
            "batch_size": 256,
            "seq_length": 96,
        }
    )
    scores = dict(trajectory or {epoch: 0.50 - abs(epoch - 3) * 0.01 for epoch in range(1, 13)})
    best_epoch = min(epoch for epoch, value in scores.items() if value == max(scores.values()))
    objective = float(scores[best_epoch])
    configuration_id = configuration_id_v2(
        hp,
        support_contract_version=contract["contract_id"],
        support_contract_sha256=contract["checksum_sha256"],
    )
    proposal_id = proposal_id_v2(search_arm, proposal_order)
    trial_id = trial_id_v2(configuration_id, proposal_id, execution_generation=execution_generation)
    preparation_record = {
        "campaign_id": CAMPAIGN_ID_V2,
        "domain_version": DOMAIN_VERSION_V2,
        "fidelity_id": FIDELITY_ID_V2,
        "evaluation_scope": "development_validation_2024_only",
        "sealed_scope": False,
        "hyperparameters": hp,
        "support_contract_version": contract["contract_id"],
        "support_contract_sha256": contract["checksum_sha256"],
        "configuration_id": configuration_id,
        "proposal_id": proposal_id,
        "trial_id": trial_id,
        "search_arm": search_arm,
        "proposal_order": proposal_order,
        "execution_generation": execution_generation,
        "model_seed": MODEL_SEED_A,
        "wandb_sweep_id": "fixture-sweep" if search_arm == "bayesian" else None,
        "wandb_run_id": f"fixture-{label}" if search_arm == "bayesian" else None,
    }
    record = {
        "campaign_id": CAMPAIGN_ID_V2,
        "search_arm": search_arm,
        "proposal_id": proposal_id,
        "configuration_id": configuration_id,
        "trial_id": trial_id,
        "execution_generation": execution_generation,
        "best_epoch": best_epoch,
        "objective_score": objective,
        "objective_eligible": True,
        "execution_status": "VALID",
        "fixed_support_metric_name": V2_METRIC_NAME,
        "fixed_support_epoch_trajectory": scores,
        "support_contract_version": contract["contract_id"],
        "support_contract_sha256": contract["checksum_sha256"],
        "preparation_record": preparation_record,
        "result": {"nh_run_dir": str(Path(run_dir))},
        "executor_mode": "monolithic",
        "retry_of_trial_id": None,
        "git_commit": "1" * 40,
    }
    path = Path(root) / f"{label}_execution_provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return path, record


def write_trial_roster(path, *, contract, receipt_entries, campaign_id=CAMPAIGN_ID_V2):
    """Write deterministic roster bytes and return ``(path, sha256)``."""
    payload = {
        "schema_name": "flashnh_rd1_c4_d1_trial_roster",
        "schema_version": 1,
        "campaign_id": campaign_id,
        "support_contract_version": contract["contract_id"],
        "support_contract_sha256": contract["checksum_sha256"],
        "trials": [dict(entry) for entry in receipt_entries],
    }
    path = Path(path)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return path, sha256_bytes(raw)


def fixture_authenticated_trial(
    trial_id,
    run_dir,
    *,
    contract,
    search_arm="bayesian",
    epoch=9,
    validation_pickle_sha256=None,
    receipt_path=None,
    receipt_sha256=None,
):
    """A fixture ``AuthenticatedTrialTarget`` for tests that are about D1's
    own comparison/publication behaviour rather than about receipt authority.

    It is built through the dataclass directly, on purpose: the receipt-
    authority proof lives in ``tests/test_rd1_c4_trial_authentication.py``,
    which drives the real
    ``authenticate_trial_roster`` -> ``build_v2_best_epoch_source`` ->
    ``_revalidate_source_against_authoritative_receipt`` chain against
    genuine ``execution_provenance.json`` receipts. Repeating that whole
    spine in every shard/publication test would make those tests about
    something else. The one thing this helper cannot do is make an
    unauthenticated object acceptable to D1 -- ``run_trial_observation_diagnostic``
    requires this exact type, and a plain namespace is refused.
    """
    from src.baseline.nh_seed_evaluation import period_results_path
    from src.baseline.rd1_c4_trial_authentication import AuthenticatedTrialTarget

    return AuthenticatedTrialTarget(
        trial_id=trial_id,
        search_arm=search_arm,
        campaign_id="fixture_campaign",
        domain_version="fixture_domain_v1",
        fidelity_id="fixture_fidelity",
        model_seed=1,
        proposal_id=f"proposal_{trial_id}",
        proposal_order=1,
        configuration_id=f"cfg_{trial_id}",
        execution_generation=1,
        retry_of_trial_id=None,
        best_epoch=epoch,
        official_objective=0.4388098707961096,
        fixed_support_metric_name="nse",
        evaluation_scope="development_validation",
        sealed_scope=False,
        support_contract_version=contract["contract_id"],
        support_contract_sha256=contract["checksum_sha256"],
        run_dir=str(run_dir),
        source_receipt_path=receipt_path or f"/fixture/{trial_id}/execution_provenance.json",
        source_receipt_sha256=receipt_sha256 or "f" * 64,
        validation_pickle_path=str(
            period_results_path(run_dir, contract["period"], epoch)
        ),
        validation_pickle_sha256=validation_pickle_sha256,
        wandb_sweep_id=None,
        wandb_run_id=None,
        git_commit=None,
        best_epoch_source=None,
    )


# Backwards-compatible alias for the existing D1 tests' import name.
trial_target = fixture_authenticated_trial
