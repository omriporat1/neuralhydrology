"""Run configs and score-blind per-epoch evaluation for the water-year rotation
baseline (campaign :data:`CAMPAIGN_ID`).

Five runs, one per cycle of the frozen split contract
``flashnh_wy_rotation_split_v001`` (decision log 2026-11-11 / 2026-11-12 /
2026-11-15). Each run is derived from the frozen P19 120 h NH config by
changing ONLY the keys in :data:`ALLOWED_CHANGED_KEYS`:

* identity: ``experiment_name``, ``run_dir``;
* the six NH date keys, taken from the contract (:func:`cycle_nh_dates`):
  training water years (a list of contiguous slices), the checkpoint year as
  the NH ``validation`` period and the assessment year as the NH ``test``
  period;
* the two sealing keys ``flashnh_split_contract`` (absolute path of a verified
  copy of the frozen contract) and ``flashnh_rotation_cycle``;
* all three basin files -> the contract's 2,307-basin development population
  (training and test membership equal P19's; the ``validation`` list grows
  from P19's 400-basin screening subset to the 2,307 basins the checkpoint
  choice uses, decision 2026-11-11);
* ``validate_n_random_basins`` 1000 -> 2307, so that NH's validation-period
  evaluation keeps every basin (NH silently shuffles and truncates to this
  number);
* ``validate_every`` 3 -> ``None``: no in-loop validation. Checkpoint-year
  evaluation is done after training, for every epoch, by
  :func:`evaluate_all_epochs` with no metrics (score-blind), so nothing is
  printed about checkpoint-year skill before the metric spec is approved.

Everything else (seed A, 12 epochs x 50k updates, package v002 ``data_dir``,
inputs, target/lead, model and optimizer settings) is copied verbatim and
checked. The WY2026 year never appears: the contract's dates end at
2025-09-30 and its observation upper bound is enforced by the sealed loader.

The per-epoch evaluation here writes NH's raw ``validation_results.p`` only.
It computes no metric. The checkpoint rule (median-NSE and flood-aware picks)
is applied later by a separate, approved step. :func:`verify_score_blind_run`
checks receipts, basin keys, shapes and finite counts only; it never compares
simulations with observations.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pickle
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from .nh_raw_space_evaluation import (
    FROZEN_SPLIT_CONTRACT_CHECKSUMS,
    OBSERVATION_CUTOFF_CONFIG_KEY,
    ROTATION_CYCLE_CONFIG_KEY,
    SPLIT_CONTRACT_CONFIG_KEY,
    read_run_rotation_sealing,
    split_sealing_receipt_path,
    verify_run_sealing_receipts,
)
from .wy_rotation_split import (
    BASIN_POPULATION_COUNT,
    BASIN_POPULATION_PATH,
    CONTRACT_ID,
    ROTATION_TABLE_V001,
    WyRotationSplitError,
    build_split_contract,
    cycle_nh_dates,
    load_split_contract,
    verify_basin_population,
)

__all__ = [
    "CAMPAIGN_ID", "SEED_A", "FIDELITY_TAG", "P19_CONFIG_SHA256", "ALLOWED_CHANGED_KEYS", "CANARY_CHANGED_KEYS",
    "DATE_KEYS", "BASIN_FILE_KEYS", "CAMPAIGN_MANIFEST_SCHEMA_NAME", "CAMPAIGN_MANIFEST_SCHEMA_VERSION",
    "CONTRACT_COPY_NAME", "WyRotationRunsError", "run_identity", "campaign_rows", "config_differences",
    "derive_cycle_config", "derive_canary_config", "prepare_campaign", "prepare_canary", "verify_campaign_row",
    "verify_canary_manifest", "results_pickle_path",
    "sha256_file", "evaluate_all_epochs", "results_pickle_problem", "delete_results_pickles",
    "verify_score_blind_run", "describe_results_pickle",
    "log_epoch_timestamps", "SCORE_LINE_PATTERN",
]

CAMPAIGN_ID = "flashnh_wy_rotation_baseline_v001"
SEED_A = 967139
FIDELITY_TAG = "mf12x50000"
# The frozen P19 config bytes (sha also recorded in its generation manifest).
P19_CONFIG_SHA256 = "f0d7fab7fa8a2ae45f3a0418d39adcef9cc90614e077cb15da7c809fdad426fc"
CONTRACT_COPY_NAME = "split_contract_v001.json"
CAMPAIGN_MANIFEST_SCHEMA_NAME = "flashnh_wy_rotation_campaign_manifest"
CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1

DATE_KEYS = ("train_start_date", "train_end_date", "validation_start_date", "validation_end_date",
             "test_start_date", "test_end_date")
BASIN_FILE_KEYS = ("train_basin_file", "validation_basin_file", "test_basin_file")
ALLOWED_CHANGED_KEYS = frozenset({"experiment_name", "run_dir", *DATE_KEYS, *BASIN_FILE_KEYS,
                                  "validate_n_random_basins", "validate_every",
                                  SPLIT_CONTRACT_CONFIG_KEY, ROTATION_CYCLE_CONFIG_KEY})
# A canary differs from its campaign config only in budget and identity.
CANARY_CHANGED_KEYS = frozenset({"epochs", "max_updates_per_epoch", "experiment_name", "run_dir"})

# Frozen P19 facts the base config must reproduce before anything is derived.
_BASE_EXPECTED = {"seq_length": 120, "seed": SEED_A, "epochs": 12, "max_updates_per_epoch": 50000,
                  "predict_last_n": 1, "dataset": "flashnh", "target_variables": ["qobs_mm_per_h_lead06"],
                  "dynamic_inputs": ["mrms_qpe_1h_mm", "rtma_2t_K"], "loss": "NSE", "validate_every": 3,
                  "validate_n_random_basins": 1000, "save_weights_every": 1,
                  "data_dir": "/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_scientific_package_v002"}
# Keys that must stay absent: any of them would compute or print scores, or
# bring data in around the sealed loader.
_FORBIDDEN_KEYS = ("metrics", "experiment_logger", "use_wandb", OBSERVATION_CUTOFF_CONFIG_KEY,
                   "flashnh_common_support_file", "train_data_file", "additional_feature_files",
                   "per_basin_train_periods_file", "per_basin_validation_periods_file", "per_basin_test_periods_file")


class WyRotationRunsError(ValueError):
    """Raised for any rotation-run identity/config/evidence violation."""


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_identity(cycle: int) -> str:
    if cycle not in {c.cycle for c in ROTATION_TABLE_V001}:
        raise WyRotationRunsError(f"cycle {cycle!r} is not in the frozen rotation table")
    return f"{CAMPAIGN_ID}__cycle{cycle}__L120__seedA{SEED_A}__{FIDELITY_TAG}__attempt001"


def campaign_rows() -> list[dict[str, Any]]:
    """One row per contract cycle, in table order (array index = cycle - 1)."""
    rows = []
    for c in ROTATION_TABLE_V001:
        rows.append({"index": len(rows), "cycle": c.cycle, "train_wys": list(c.train_wys),
                     "checkpoint_wy": c.checkpoint_wy, "assessment_wy": c.assessment_wy,
                     "run_id": run_identity(c.cycle)})
    return rows


def config_differences(base: Mapping[str, Any], derived: Mapping[str, Any]) -> dict[str, list]:
    keys = sorted(set(base) | set(derived))
    return {k: [base.get(k), derived.get(k)] for k in keys
            if base.get(k) != derived.get(k) or (k in base) != (k in derived)}


def _check_forbidden(config: Mapping[str, Any], what: str) -> None:
    present = [k for k in _FORBIDDEN_KEYS if config.get(k) not in (None, [], {}, "None")]
    if present:
        raise WyRotationRunsError(f"{what} sets forbidden key(s) {present}")


def derive_cycle_config(base: Mapping[str, Any], *, cycle: int, contract_path: str, basin_file: str,
                        experiment_name: str, run_dir: str) -> dict[str, Any]:
    """The NH config of one rotation cycle (see the module docstring)."""
    if SPLIT_CONTRACT_CONFIG_KEY in base or ROTATION_CYCLE_CONFIG_KEY in base:
        raise WyRotationRunsError("base config already carries split-contract keys")
    _check_forbidden(base, "base config")
    rotation = {c.cycle: c for c in ROTATION_TABLE_V001}
    if cycle not in rotation:
        raise WyRotationRunsError(f"cycle {cycle!r} is not in the frozen rotation table")
    derived = copy.deepcopy(dict(base))
    derived.update(cycle_nh_dates(rotation[cycle]))
    derived.update({key: str(basin_file) for key in BASIN_FILE_KEYS})
    derived.update({"experiment_name": experiment_name, "run_dir": str(run_dir),
                    "validate_n_random_basins": BASIN_POPULATION_COUNT, "validate_every": None,
                    SPLIT_CONTRACT_CONFIG_KEY: str(contract_path), ROTATION_CYCLE_CONFIG_KEY: int(cycle)})
    changed = set(config_differences(base, derived))
    if not changed <= ALLOWED_CHANGED_KEYS:
        raise WyRotationRunsError(f"derived config changes non-whitelisted keys: {sorted(changed - ALLOWED_CHANGED_KEYS)}")
    _check_forbidden(derived, "derived config")
    return derived


def derive_canary_config(cycle_config: Mapping[str, Any], *, epochs: int, max_updates_per_epoch: int,
                         experiment_name: str, run_dir: str) -> dict[str, Any]:
    """A canary of one campaign config: smaller budget and its own identity,
    nothing else. Never a campaign run."""
    if not (1 <= int(epochs) <= cycle_config["epochs"] and 1 <= int(max_updates_per_epoch)
            <= cycle_config["max_updates_per_epoch"]):
        raise WyRotationRunsError("a canary budget must not exceed the campaign budget")
    derived = copy.deepcopy(dict(cycle_config))
    derived.update({"epochs": int(epochs), "max_updates_per_epoch": int(max_updates_per_epoch),
                    "experiment_name": experiment_name, "run_dir": str(run_dir)})
    changed = set(config_differences(cycle_config, derived))
    if not changed <= CANARY_CHANGED_KEYS:
        raise WyRotationRunsError(f"canary changes non-budget keys: {sorted(changed - CANARY_CHANGED_KEYS)}")
    return derived


def _write_exact(path: Path, text: str, what: str) -> None:
    """Write ``text`` bytes (no newline translation); refuse different existing bytes."""
    if path.exists():
        if path.read_bytes() != text.encode("utf-8"):
            raise WyRotationRunsError(f"refusing to overwrite a different existing {what}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _load_verified_base(base_config_path, base_manifest_path, population: list) -> tuple[dict, dict]:
    base_config_path = Path(base_config_path)
    manifest = json.loads(Path(base_manifest_path).read_text(encoding="utf-8"))
    actual = sha256_file(base_config_path)
    if actual != P19_CONFIG_SHA256 or manifest["artifact_sha256"]["config.yaml"] != P19_CONFIG_SHA256:
        raise WyRotationRunsError(f"base config sha256 {actual} is not the frozen P19 config {P19_CONFIG_SHA256} "
                                  "(or its generation manifest disagrees)")
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    wrong = {k: (base.get(k), v) for k, v in _BASE_EXPECTED.items() if base.get(k) != v}
    if wrong:
        raise WyRotationRunsError(f"base config is not the frozen P19 config: {wrong}")
    # P19 trained on exactly the contract population; the rotation keeps that membership.
    p19_basins = manifest.get("basin_ids")
    if p19_basins is None or sorted(map(str, p19_basins)) != sorted(population):
        raise WyRotationRunsError("P19 training basins (generation manifest basin_ids) differ from the contract "
                                  "population")
    return base, manifest


def _contract_copy(output_root: Path, write: bool) -> tuple[Path, dict, str]:
    contract = build_split_contract()
    pinned = FROZEN_SPLIT_CONTRACT_CHECKSUMS.get(CONTRACT_ID)
    if contract["checksum_sha256"] != pinned:
        raise WyRotationRunsError(f"rebuilt contract checksum {contract['checksum_sha256']} != pinned {pinned}")
    text = json.dumps(contract, indent=1, sort_keys=True) + "\n"  # = write_split_contract's bytes
    path = (output_root / CONTRACT_COPY_NAME).resolve()
    if write:
        _write_exact(path, text, "split contract copy")
        if load_split_contract(path)["checksum_sha256"] != pinned:
            raise WyRotationRunsError(f"written contract copy {path} does not verify")
    return path, contract, _sha256_text(text)


def prepare_campaign(*, base_config_path, base_manifest_path, output_root, runtime_project_root,
                     write: bool = True) -> dict[str, Any]:
    """Derive the five cycle configs from the frozen P19 config.

    ``runtime_project_root`` is the project root of the machine that will
    run them (the basin file path in the configs is absolute under it, and
    the population is verified there). Writes the verified contract copy,
    ``<output_root>/<run_id>/config.yaml`` and
    ``<output_root>/campaign_manifest.json``; refuses to overwrite any
    different existing bytes. P19's own basin files are never opened."""
    output_root = Path(output_root).resolve()
    runtime_project_root = Path(runtime_project_root).resolve()
    contract_path, contract, contract_file_sha = _contract_copy(output_root, write)
    try:
        population = verify_basin_population(contract, runtime_project_root)
    except WyRotationSplitError as exc:
        raise WyRotationRunsError(f"basin population refused: {exc}") from exc
    base, manifest = _load_verified_base(base_config_path, base_manifest_path, population)
    basin_file = (runtime_project_root / BASIN_POPULATION_PATH).resolve()

    runs = []
    for row in campaign_rows():
        config_dir = output_root / row["run_id"]
        derived = derive_cycle_config(base, cycle=row["cycle"], contract_path=contract_path.as_posix(),
                                      basin_file=basin_file.as_posix(), experiment_name=row["run_id"],
                                      run_dir=(config_dir / "runs").as_posix())
        text = yaml.safe_dump(derived, sort_keys=False, default_flow_style=False)
        if write:
            _write_exact(config_dir / "config.yaml", text, "config")
        runs.append({**row, "config_dir": config_dir.as_posix(), "config_sha256": _sha256_text(text),
                     "nh_dates": {k: derived[k] for k in DATE_KEYS},
                     "config_differences_vs_p19": config_differences(base, derived)})
    result = {
        "schema_name": CAMPAIGN_MANIFEST_SCHEMA_NAME, "schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID, "seed": SEED_A, "fidelity_tag": FIDELITY_TAG,
        "allowed_changed_keys": sorted(ALLOWED_CHANGED_KEYS),
        "base_config": {"path": Path(base_config_path).resolve().as_posix(), "sha256": P19_CONFIG_SHA256,
                        "experiment_name": base["experiment_name"]},
        "base_generation_manifest": {"path": Path(base_manifest_path).resolve().as_posix(),
                                     "sha256": sha256_file(base_manifest_path),
                                     "git_commit": manifest.get("git_commit")},
        "split_contract": {"path": contract_path.as_posix(), "file_sha256": contract_file_sha,
                           "contract_id": contract["contract_id"], "checksum_sha256": contract["checksum_sha256"]},
        "basin_population": {"path": basin_file.as_posix(), "file_sha256": sha256_file(basin_file),
                             "count": len(population),
                             "membership_sha256": contract["basin_population"]["membership_sha256"]},
        "p19_basin_files_opened": False,
        "runtime_project_root": runtime_project_root.as_posix(),
        "runs": runs,
    }
    if write:
        _write_exact(output_root / "campaign_manifest.json", json.dumps(result, indent=2, sort_keys=True) + "\n",
                     "campaign manifest")
    return result


def verify_campaign_row(manifest: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Re-verify, at launch time, every frozen byte one run depends on;
    returns the row and its parsed config."""
    if (manifest.get("schema_name") != CAMPAIGN_MANIFEST_SCHEMA_NAME or manifest.get("campaign_id") != CAMPAIGN_ID
            or manifest.get("schema_version") != CAMPAIGN_MANIFEST_SCHEMA_VERSION):
        raise WyRotationRunsError("not a wy_rotation campaign manifest (v1)")
    runs = manifest["runs"]
    if len(runs) != len(ROTATION_TABLE_V001) or not 0 <= index < len(runs):
        raise WyRotationRunsError(f"index {index} out of range for {len(runs)} runs")
    row = dict(runs[index])
    if row["index"] != index or row["run_id"] != run_identity(row["cycle"]):
        raise WyRotationRunsError("campaign manifest row identity is inconsistent")
    config_path = Path(row["config_dir"]) / "config.yaml"
    checks = {"config": (config_path, row["config_sha256"]),
              "split contract": (Path(manifest["split_contract"]["path"]), manifest["split_contract"]["file_sha256"]),
              "basin population": (Path(manifest["basin_population"]["path"]),
                                   manifest["basin_population"]["file_sha256"])}
    bad = [name for name, (path, sha) in checks.items() if not path.is_file() or sha256_file(path) != sha]
    if bad:
        raise WyRotationRunsError(f"frozen launch inputs changed or missing: {bad}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {"experiment_name": row["run_id"], ROTATION_CYCLE_CONFIG_KEY: row["cycle"],
                SPLIT_CONTRACT_CONFIG_KEY: manifest["split_contract"]["path"],
                "validate_n_random_basins": BASIN_POPULATION_COUNT, "validate_every": None,
                **{k: manifest["basin_population"]["path"] for k in BASIN_FILE_KEYS},
                **{k: _BASE_EXPECTED[k] for k in ("seed", "epochs", "max_updates_per_epoch", "data_dir")}}
    wrong = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if wrong:
        raise WyRotationRunsError(f"config {config_path} does not match its manifest row: {wrong}")
    return {"row": row, "config": config, "config_path": config_path}


def prepare_canary(*, campaign_manifest_path, cycle: int, epochs: int, max_updates_per_epoch: int,
                   output_root, write: bool = True) -> dict[str, Any]:
    """Derive one canary config from a verified campaign row (budget and
    identity changes only), plus ``canary_manifest.json``."""
    manifest = json.loads(Path(campaign_manifest_path).read_text(encoding="utf-8"))
    index = next((r["index"] for r in manifest["runs"] if r["cycle"] == cycle), None)
    if index is None:
        raise WyRotationRunsError(f"no cycle {cycle} in {campaign_manifest_path}")
    checked = verify_campaign_row(manifest, index)
    output_root = Path(output_root).resolve()
    run_id = f"{CAMPAIGN_ID}__canary__cycle{cycle}__e{int(epochs)}x{int(max_updates_per_epoch)}"
    config_dir = output_root / run_id
    derived = derive_canary_config(checked["config"], epochs=epochs, max_updates_per_epoch=max_updates_per_epoch,
                                   experiment_name=run_id, run_dir=(config_dir / "runs").as_posix())
    text = yaml.safe_dump(derived, sort_keys=False, default_flow_style=False)
    result = {
        "schema_name": "flashnh_wy_rotation_canary_manifest", "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "is_canary": True, "run_id": run_id, "cycle": cycle, "epochs": int(epochs),
        "max_updates_per_epoch": int(max_updates_per_epoch),
        "campaign_manifest": {"path": Path(campaign_manifest_path).resolve().as_posix(),
                              "sha256": sha256_file(campaign_manifest_path)},
        "campaign_row": checked["row"], "config_dir": config_dir.as_posix(), "config_sha256": _sha256_text(text),
        "config_differences_vs_campaign_row": config_differences(checked["config"], derived),
        "split_contract": manifest["split_contract"], "basin_population": manifest["basin_population"],
    }
    if write:
        _write_exact(config_dir / "config.yaml", text, "canary config")
        _write_exact(output_root / "canary_manifest.json", json.dumps(result, indent=2, sort_keys=True) + "\n",
                     "canary manifest")
    return result


def verify_canary_manifest(canary_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Re-verify a canary at launch time: its campaign row (all frozen bytes),
    its own config bytes, and that it differs from the row only in budget and
    identity. Returns the same shape as :func:`verify_campaign_row`."""
    if (canary_manifest.get("schema_name") != "flashnh_wy_rotation_canary_manifest"
            or canary_manifest.get("campaign_id") != CAMPAIGN_ID or canary_manifest.get("is_canary") is not True):
        raise WyRotationRunsError("not a wy_rotation canary manifest")
    campaign_path = Path(canary_manifest["campaign_manifest"]["path"])
    if not campaign_path.is_file() or sha256_file(campaign_path) != canary_manifest["campaign_manifest"]["sha256"]:
        raise WyRotationRunsError(f"campaign manifest {campaign_path} changed or missing")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    checked = verify_campaign_row(campaign, canary_manifest["campaign_row"]["index"])
    config_path = Path(canary_manifest["config_dir"]) / "config.yaml"
    if not config_path.is_file() or sha256_file(config_path) != canary_manifest["config_sha256"]:
        raise WyRotationRunsError(f"canary config {config_path} changed or missing")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    changed = set(config_differences(checked["config"], config))
    if not changed <= CANARY_CHANGED_KEYS or config.get("experiment_name") != canary_manifest["run_id"]:
        raise WyRotationRunsError(f"canary config changes non-budget keys: {sorted(changed - CANARY_CHANGED_KEYS)}")
    row = {"run_id": canary_manifest["run_id"], "cycle": canary_manifest["cycle"],
           "config_dir": canary_manifest["config_dir"], "is_canary": True}
    return {"row": row, "config": config, "config_path": config_path, "campaign": campaign}


# --------------------------------------------------------------------------- #
# Score-blind per-epoch evaluation of the checkpoint year (NH "validation")
# --------------------------------------------------------------------------- #

def results_pickle_path(nh_run_dir, epoch: int) -> Path:
    from .nh_seed_evaluation import period_results_path

    return period_results_path(nh_run_dir, "validation", epoch)


def _default_tester_factory(nh_run_dir: Path):
    """One NH tester for the run's validation (= checkpoint-year) period, with
    the model initialised; datasets are cached across epochs by NH
    (``cache_validation_data`` default True)."""
    from .nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    from neuralhydrology.evaluation import get_tester
    from neuralhydrology.utils.config import Config

    cfg = Config(nh_run_dir / "config.yml")
    return get_tester(cfg=cfg, run_dir=nh_run_dir, period="validation", init_model=True)


def results_pickle_problem(path) -> Optional[str]:
    """``None`` when ``path`` unpickles to a dict (an NH results pickle), else
    why not. Only loadability is checked, no content: NH writes the pickle in
    place (``open(..., "wb")``), so a job killed mid-write leaves a truncated
    file that is present but does not load.

    ``MemoryError`` and ``ImportError`` (incl. ``ModuleNotFoundError``) are raised,
    not reported: they say the job or environment is broken, not the file, and a
    valid (~0.5 GB) pickle must not be deleted and re-evaluated because of them."""
    try:
        with Path(path).open("rb") as fh:
            results = pickle.load(fh)
    except (MemoryError, ImportError):
        raise
    except Exception as exc:  # noqa: BLE001 -- EOFError, UnpicklingError, ... all mean "redo this epoch"
        return f"does not load: {exc!r}"
    if not isinstance(results, dict):
        return f"loads as {type(results).__name__}, not an NH results dict"
    return None


def delete_results_pickles(nh_run_dir, epochs) -> list[int]:
    """Delete the ``validation_results.p`` of each epoch in ``epochs``; returns
    the epochs that had one. Nothing else in the run dir is touched."""
    deleted = []
    for epoch in sorted(int(e) for e in epochs):
        path = results_pickle_path(nh_run_dir, epoch)
        if path.is_file():
            path.unlink()
            deleted.append(epoch)
    return deleted


def evaluate_all_epochs(nh_run_dir, epochs, *, tester_factory: Optional[Callable] = None,
                        clock: Callable[[], float] = time.monotonic) -> list[dict[str, Any]]:
    """Write NH's ``validation_results.p`` for every epoch in ``epochs`` that
    lacks a loadable one, using one tester (datasets loaded once). An existing
    pickle is skipped only if it loads (:func:`results_pickle_problem`); one
    that does not is deleted and re-evaluated, and its record carries
    ``redone_existing`` (why). ``metrics=[]``: no metric is computed, printed
    or stored. Returns per-epoch timing records (seconds) only; the results
    dicts are dropped."""
    nh_run_dir = Path(nh_run_dir)
    config = yaml.safe_load((nh_run_dir / "config.yml").read_text(encoding="utf-8")) or {}
    _check_forbidden(config, f"run config {nh_run_dir / 'config.yml'}")
    if config.get("validate_n_random_basins") != BASIN_POPULATION_COUNT:
        raise WyRotationRunsError(f"validate_n_random_basins must be {BASIN_POPULATION_COUNT} (NH truncates the "
                                  f"validation basins to it), got {config.get('validate_n_random_basins')!r}")
    tester = None
    records = []
    for epoch in sorted(int(e) for e in epochs):
        path = results_pickle_path(nh_run_dir, epoch)
        redone = {}
        if path.is_file():
            problem = results_pickle_problem(path)
            if problem is None:
                records.append({"epoch": epoch, "skipped_existing": True})
                continue
            path.unlink()  # a later kill then leaves a missing pickle, never the bad one
            redone = {"redone_existing": problem}
        t0 = clock()
        if tester is None:
            tester = (tester_factory or _default_tester_factory)(nh_run_dir)
        t1 = clock()
        tester.evaluate(epoch=epoch, save_results=True, metrics=[])
        t2 = clock()
        if not path.is_file():
            raise WyRotationRunsError(f"NH evaluation of epoch {epoch} wrote no {path}")
        records.append({"epoch": epoch, "skipped_existing": False, "tester_setup_s": round(t1 - t0, 3),
                        "evaluate_s": round(t2 - t1, 3), **redone})
    return records


def describe_results_pickle(path, *, target: str) -> dict[str, Any]:
    """Shapes and counts of one NH results pickle -- never a comparison of
    simulation with observation. Per basin: frequencies, number of dates,
    first/last date, finite sim and finite obs counts."""
    import numpy as np

    with Path(path).open("rb") as fh:
        results = pickle.load(fh)
    if not isinstance(results, dict):
        raise WyRotationRunsError(f"{path} is not an NH results dict")
    per_basin = {}
    for basin, by_freq in results.items():
        entry = {}
        for freq, payload in by_freq.items():
            xr = payload["xr"]
            sim = np.asarray(xr[f"{target}_sim"].values, dtype=float)
            obs = np.asarray(xr[f"{target}_obs"].values, dtype=float)
            dates = xr["date"].values
            entry[str(freq)] = {"n_dates": int(len(dates)), "first_date": str(dates.min()) if len(dates) else None,
                                "last_date": str(dates.max()) if len(dates) else None,
                                "dates_increasing": bool(len(dates) < 2 or (np.diff(
                                    np.asarray(dates, dtype="datetime64[ns]").astype("int64")) > 0).all()),
                                "sim_shape": list(sim.shape), "n_sim_finite": int(np.isfinite(sim).sum()),
                                "n_obs_finite": int(np.isfinite(obs).sum()),
                                "extra_keys": sorted(k for k in payload if k != "xr")}
        per_basin[str(basin)] = entry
    return per_basin


# NH lines that report a validation score (in-loop validation or metrics).
SCORE_LINE_PATTERN = re.compile(r"validation loss|validation metrics|Stored metrics at", re.IGNORECASE)
_EPOCH_END_PATTERN = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+: Epoch (\d+) average loss")


def log_epoch_timestamps(output_log) -> dict[int, str]:
    """Wall-clock end time of each training epoch from NH's ``output.log``
    (timestamps only; the loss value on the line is not returned)."""
    out = {}
    for line in Path(output_log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _EPOCH_END_PATTERN.match(line)
        if m:
            out[int(m.group(2))] = m.group(1)
    return out


def count_admissible_checkpoint_targets(nh_run_dir, basins, *, sealing, target: str,
                                        project_root=None) -> dict[str, int]:
    """Independent count, per basin, of checkpoint-year issue times ``t`` with
    ``t`` and ``t + lead`` inside the checkpoint-year window and a finite
    ``target`` at ``t``. Reads the run's package NetCDF through the sealed
    reader (validation windows, contract upper bound, no forcing), so only
    checkpoint-year observations are touched; only finiteness is counted.

    Deliberately ignores the forcing-gap history rule: a basin that has
    finite targets but loses every sample to forcing gaps counts > 0 and is
    therefore NOT explained (fail closed)."""
    import numpy as np
    import pandas as pd
    import xarray

    from .nh_raw_space_evaluation import select_readable_observations

    nh_run_dir = Path(nh_run_dir)
    config = yaml.safe_load((nh_run_dir / "config.yml").read_text(encoding="utf-8")) or {}
    data_dir = Path(str(config.get("data_dir")))
    if not data_dir.is_absolute() and project_root is not None:
        data_dir = Path(project_root) / data_dir
    windows = sealing.readable_windows("validation")
    lead = pd.Timedelta(hours=int(sealing.lead_hours))
    first = pd.to_datetime(sealing.nh_dates["validation_start_date"], format="%d/%m/%Y")
    last = pd.to_datetime(sealing.nh_dates["validation_end_date"], format="%d/%m/%Y") + pd.Timedelta(hours=23)
    counts = {}
    for basin in sorted(map(str, basins)):
        found = [p for p in (data_dir / "time_series" / f"{basin}{ext}" for ext in (".nc4", ".nc")) if p.is_file()]
        if len(found) != 1:
            raise WyRotationRunsError(f"basin {basin}: expected one NetCDF in {data_dir / 'time_series'}, "
                                      f"found {[str(p) for p in found]}")
        with xarray.open_dataset(found[0]) as ds:
            coordinate = "date" if "date" in ds.coords else "time"
            if target not in ds.data_vars:
                raise WyRotationRunsError(f"basin {basin}: target {target!r} not in {found[0].name}")
            sealed = select_readable_observations(ds[[target]], windows=windows, coordinate=coordinate,
                                                  forcing_variables=(), upper_bound=sealing.upper_bound)
            issue = sealed[target].sel({coordinate: slice(first, last - lead)})
            counts[basin] = int(np.isfinite(np.asarray(issue.values, dtype=float)).sum())
    return counts


def verify_score_blind_run(nh_run_dir, *, cycle: int, epochs, population, target: str = "qobs_mm_per_h_lead06",
                           project_root=None, missing_basin_counter: Optional[Callable] = None) -> dict[str, Any]:
    """Fail closed unless the run was loaded sealed for ``cycle`` (train and
    validation receipts matching the contract), trained on ``population`` and
    evaluated, for every epoch in ``epochs``, over ``population`` minus an
    explained-missing set, inside the checkpoint-year window, and left no
    metric file or score line. Returns a score-free evidence record.

    Explained-missing gate (Omri, 2026-11-15): NH's tester silently skips a
    basin whose Flash-NH filter leaves no checkpoint-year sample. A basin
    absent from ``validation_results.p`` is accepted only when it is absent
    from every epoch alike and an independent count
    (``missing_basin_counter``, default
    :func:`count_admissible_checkpoint_targets`) finds 0 admissible
    finite-target issue times for it. Extra keys always fail. The full
    missing list and the counts are recorded."""
    import pandas as pd

    nh_run_dir = Path(nh_run_dir)
    problems = []
    sealing = read_run_rotation_sealing(nh_run_dir, project_root=project_root)
    if sealing is None or sealing.cycle != cycle:
        raise WyRotationRunsError(f"run {nh_run_dir} is not sealed for cycle {cycle}")
    verify_run_sealing_receipts(nh_run_dir, sealing, ["train", "validation"])
    train_receipt = json.loads(split_sealing_receipt_path(nh_run_dir, "train").read_text(encoding="utf-8"))
    population = sorted(map(str, population))
    if train_receipt.get("n_basins") != len(population):
        problems.append(f"train receipt n_basins {train_receipt.get('n_basins')} != {len(population)}")
    if (nh_run_dir / "test").exists():
        problems.append("a test-period (assessment-year) output folder exists")

    window_first = pd.to_datetime(sealing.nh_dates["validation_start_date"], format="%d/%m/%Y")
    window_last = pd.to_datetime(sealing.nh_dates["validation_end_date"], format="%d/%m/%Y") + pd.Timedelta(hours=23)
    per_epoch = {}
    missing_by_epoch = {}
    for epoch in sorted(int(e) for e in epochs):
        path = results_pickle_path(nh_run_dir, epoch)
        if not path.is_file():
            problems.append(f"epoch {epoch}: no {path.name}")
            continue
        described = describe_results_pickle(path, target=target)
        keys = sorted(described)
        freqs = sorted({f for entry in described.values() for f in entry})
        n_dates = sorted({v["n_dates"] for entry in described.values() for v in entry.values()})
        firsts = [pd.Timestamp(v["first_date"]) for e in described.values() for v in e.values() if v["first_date"]]
        lasts = [pd.Timestamp(v["last_date"]) for e in described.values() for v in e.values() if v["last_date"]]
        record = {"n_basins": len(keys), "basins_equal_population": keys == population, "frequencies": freqs,
                  "n_dates_values": n_dates, "first_date_min": str(min(firsts)) if firsts else None,
                  "last_date_max": str(max(lasts)) if lasts else None,
                  "n_basins_without_finite_sim": sum(1 for e in described.values()
                                                     if all(v["n_sim_finite"] == 0 for v in e.values())),
                  "n_basins_without_finite_obs": sum(1 for e in described.values()
                                                     if all(v["n_obs_finite"] == 0 for v in e.values())),
                  "n_basins_dates_not_increasing": sum(1 for e in described.values()
                                                       if not all(v["dates_increasing"] for v in e.values())),
                  # Coverage is recorded, not gated: a basin whose predictions start after or end before the
                  # checkpoint-year edges (e.g. data gaps at a boundary) is counted here for human review.
                  "n_basins_partial_window": sum(1 for e in described.values() if any(
                      v["first_date"] is None or pd.Timestamp(v["first_date"]) > window_first
                      or pd.Timestamp(v["last_date"]) < window_last for v in e.values())),
                  "pickle_bytes": path.stat().st_size}
        missing, extra = sorted(set(population) - set(keys)), sorted(set(keys) - set(population))
        record["n_basins_missing"] = len(missing)
        missing_by_epoch[epoch] = missing
        if extra:
            problems.append(f"epoch {epoch}: {len(extra)} basin keys outside the population: {extra}")
        if record["n_basins_without_finite_obs"]:
            problems.append(f"epoch {epoch}: {record['n_basins_without_finite_obs']} evaluated basins have no "
                            "finite observation")
        if firsts and (min(firsts) < window_first or max(lasts) > window_last):
            problems.append(f"epoch {epoch}: dates {min(firsts)}..{max(lasts)} leave the checkpoint-year window "
                            f"{window_first}..{window_last}")
        if record["n_basins_dates_not_increasing"]:
            problems.append(f"epoch {epoch}: {record['n_basins_dates_not_increasing']} basins have dates that are "
                            "not strictly increasing (NH reindexes from the first to the last date)")
        if record["n_basins_without_finite_sim"]:
            problems.append(f"epoch {epoch}: {record['n_basins_without_finite_sim']} basins have no finite simulation")
        metric_files = sorted(p.name for p in path.parent.glob("*metrics*"))
        if metric_files:
            problems.append(f"epoch {epoch}: metric file(s) present {metric_files}")
        per_epoch[epoch] = record

    missing_sets = {tuple(m) for m in missing_by_epoch.values()}
    missing_basins = sorted(set().union(*missing_by_epoch.values())) if missing_by_epoch else []
    if len(missing_sets) > 1:
        problems.append("the missing basin set differs between epochs: "
                        + "; ".join(f"epoch {e}: {m}" for e, m in sorted(missing_by_epoch.items())))
    missing_counts = {}
    if missing_basins and len(missing_sets) == 1:  # differing sets already fail; nothing to explain
        from .nh_raw_space_evaluation import RawSpaceEvaluationError

        counter = missing_basin_counter or count_admissible_checkpoint_targets
        try:
            missing_counts = {str(b): int(n) for b, n in counter(nh_run_dir, missing_basins, sealing=sealing,
                                                                 target=target, project_root=project_root).items()}
        except (WyRotationRunsError, RawSpaceEvaluationError, OSError, KeyError, TypeError, ValueError) as exc:
            # Fail closed but keep the evidence (the missing list) instead of losing it to the exception.
            problems.append(f"the missing-basin count failed: {exc!r}")
            missing_counts = {}
        unexplained = sorted(b for b in missing_basins if missing_counts.get(b, -1) != 0)
        if unexplained:
            problems.append(f"{len(unexplained)} missing basin(s) have admissible checkpoint-year targets (or no "
                            f"count): " + ", ".join(f"{b}={missing_counts.get(b)}" for b in unexplained))

    log = nh_run_dir / "output.log"
    score_lines = 0
    if log.is_file():
        score_lines = sum(1 for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                          if SCORE_LINE_PATTERN.search(line))
    if score_lines:
        problems.append(f"output.log has {score_lines} score line(s) (validation loss/metrics)")
    evidence = {"nh_run_dir": nh_run_dir.as_posix(), "cycle": cycle, "contract_id": sealing.contract_id,
                "contract_checksum_sha256": sealing.contract_checksum_sha256,
                "checkpoint_window": [str(window_first), str(window_last)],
                "train_receipt_counts": {k: train_receipt.get(k) for k in
                                         ("n_basins", "n_train_targets_masked_outside_period")},
                "receipts_verified": ["train", "validation"], "per_epoch": per_epoch,
                "missing_basins": missing_basins,
                "missing_basin_admissible_target_counts": missing_counts,
                "n_basins_evaluated": len(population) - len(missing_basins),
                "score_lines_in_output_log": score_lines, "problems": problems, "passed": not problems}
    return evidence
