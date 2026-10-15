"""Run identities, seeds and NH configs for the common-support, three-seed
sequence-length experiment (campaign :data:`CAMPAIGN_ID`).

15 fresh runs = 5 sequence lengths x 3 predeclared seeds. Every run is derived
from the frozen P19 NH config by changing ONLY ``seq_length``, ``seed``,
``experiment_name`` and ``run_dir``, plus adding the common training-support
key (:data:`~src.baseline.seqlen_common336_support.COMMON_SUPPORT_CONFIG_KEY`)
and the observation cutoff key
(:data:`~src.baseline.nh_raw_space_evaluation.OBSERVATION_CUTOFF_CONFIG_KEY`
= the screening contract's last hour, 2024-12-31 23:00 UTC), which keeps every
later observation out of the loader and the scorer.
:func:`derive_run_config` enforces that whitelist; every other key (inputs,
target, lead, dates, basin files, model/optimizer hyperparameters, the
``max_updates_per_epoch`` 50k cap, epochs) is copied verbatim.

Seeds (fixed before any run; never selected by outcome):

* A = 967139 -- the existing model seed A (``sweep_v1_campaign.MODEL_SEED_A``).
* B = first 8 hex chars of sha256(:data:`SEED_B_NAMESPACE`) as an unsigned int
  (329911494). It replaces the earlier choice 1729, which the decision log
  reserves for promoted finalists.
* C = first 8 hex chars of sha256(:data:`SEED_C_NAMESPACE`) as an unsigned int
  (3023892610).

B and C use the same derivation rule as the v2 random-control manifest seed. A
collision with any seed value already used in the project is a hard stop with
no fallback.

The three fresh L=120 runs are the references; historical P19 is context only
and is never one of the 15 runs.

Prespecified result summary (:data:`RESULT_SUMMARY_RULE`, frozen before any
run): each run's checkpoint is its best epoch under the existing rule
(:func:`summarize_trial`). Within each seed, every length is compared with
that seed's fresh L120 run on the identical Common-336 screening support by
the difference in median per-basin raw-space NSE. The summary reports all
three within-seed differences per length and their descriptive spread (min,
max, mean, range), next to the existing KGE and flood metrics. It names no
best seed, forms no composite score and draws no promotion inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .nh_raw_space_evaluation import OBSERVATION_CUTOFF_CONFIG_KEY
from .seqlen_common336_support import (
    CAMPAIGN_ID,
    COMMON_SEQ_LENGTH_FLOOR,
    COMMON_SUPPORT_CONFIG_KEY,
    SCREENING_CONTRACT_ID,
    SEQ_LENGTHS,
    TRAINING_SUPPORT_ID,
    Common336SupportError,
    load_common336_screening_contract,
    load_training_support,
    screening_observation_cutoff,
)

__all__ = [
    "SEED_A", "SEED_B", "SEED_B_NAMESPACE", "SEED_B_HEX", "SEED_C", "SEED_C_NAMESPACE", "SEED_C_HEX", "SEEDS",
    "REFERENCE_SEQ_LENGTH", "FIDELITY_TAG", "ALLOWED_CHANGED_KEYS", "CAMPAIGN_MANIFEST_SCHEMA_NAME",
    "CAMPAIGN_MANIFEST_SCHEMA_VERSION", "RESULT_SUMMARY_RULE", "OBSERVATION_CUTOFF_CONFIG_KEY",
    "derive_seed", "derive_seed_c", "run_identity", "campaign_rows", "derive_run_config", "config_differences",
    "prepare_campaign", "sha256_file", "verify_campaign_row", "verify_applied_training_support",
    "summarize_trial", "TRIAL_RECEIPT_NAME", "PER_EPOCH_RESULTS_NAME",
]

SEED_A = 967139
SEED_B_NAMESPACE = "flashnh_seqlen_common336_model_seed_b_v001"
SEED_B_HEX = "13aa0cc6"
SEED_B = 329911494
SEED_C_NAMESPACE = "flashnh_seqlen_common336_model_seed_c_v001"
SEED_C_HEX = "b43cf082"
SEED_C = 3023892610
# Every seed value already used anywhere in the project: model seeds A and B
# (1729, reserved for promoted finalists), the v1/v2 random-control manifest
# RNG seeds and the Part-D screening-subset seed-sensitivity seeds.
_EXISTING_PROJECT_SEEDS = {SEED_A: "model seed A", 1729: "model seed B (reserved for promoted finalists)",
                           20260822: "v1 random-control manifest seed", 158987734: "v2 random-control manifest seed",
                           **{s: "Part-D screening-subset seed" for s in (400, 121, 111, 103, 105)}}

REFERENCE_SEQ_LENGTH = 120
FIDELITY_TAG = "mf12x50000"
ALLOWED_CHANGED_KEYS = frozenset({"seq_length", "seed", "experiment_name", "run_dir", COMMON_SUPPORT_CONFIG_KEY,
                                  OBSERVATION_CUTOFF_CONFIG_KEY})
CAMPAIGN_MANIFEST_SCHEMA_NAME = "flashnh_seqlen_common336_campaign_manifest"
# v2: seed B re-derived (1729 -> 329911494), observation cutoff key added,
# result-summary rule recorded.
CAMPAIGN_MANIFEST_SCHEMA_VERSION = 2

RESULT_SUMMARY_RULE = {
    "checkpoint_rule": "per run, best epoch by median per-basin raw-space NSE on the Common-336 screening "
                       "contract over epochs 1..12, earliest epoch on ties (summarize_trial)",
    "primary_quantity": "delta median per-basin raw-space NSE = run(L, seed) - run(L120, same seed), both at "
                        "their selected checkpoints on the identical Common-336 screening support",
    "comparison": "within seed only; every length against that seed's fresh L120 run",
    "report_per_length": ["the three within-seed differences (A, B, C)",
                          "their descriptive spread: min, max, mean, range"],
    "also_report": "the existing KGE and flood metrics from the same scorer output, per run",
    "excluded": ["best-seed selection", "composite score", "promotion inference",
                 "pooling across seeds before differencing", "historical P19 as a reference"],
}

# Frozen P19 facts the base config must reproduce before anything is derived.
_BASE_EXPECTED = {"seq_length": 120, "seed": SEED_A, "epochs": 12, "max_updates_per_epoch": 50000,
                  "predict_last_n": 1, "dataset": "flashnh", "target_variables": ["qobs_mm_per_h_lead06"],
                  "dynamic_inputs": ["mrms_qpe_1h_mm", "rtma_2t_K"], "loss": "NSE"}


class SeqlenCampaignError(ValueError):
    """Raised for any campaign identity/config violation."""


def derive_seed(namespace: str, label: str = "") -> dict[str, Any]:
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    if seed in _EXISTING_PROJECT_SEEDS:
        raise SeqlenCampaignError(f"derived seed {label} {seed} collides with {_EXISTING_PROJECT_SEEDS[seed]}; STOP")
    return {"namespace": namespace, "sha256": digest, "hex_prefix": digest[:8], "seed": seed}


def derive_seed_c(namespace: str = SEED_C_NAMESPACE) -> dict[str, Any]:
    return derive_seed(namespace, "C")


for _label, _namespace, _hex, _seed in (("B", SEED_B_NAMESPACE, SEED_B_HEX, SEED_B),
                                        ("C", SEED_C_NAMESPACE, SEED_C_HEX, SEED_C)):
    _derived = derive_seed(_namespace, _label)
    if (_derived["hex_prefix"], _derived["seed"]) != (_hex, _seed):
        raise SeqlenCampaignError(f"frozen seed {_label} constants do not match their derivation")
if len({SEED_A, SEED_B, SEED_C}) != 3:
    raise SeqlenCampaignError("campaign seeds are not distinct")
SEEDS = (("A", SEED_A), ("B", SEED_B), ("C", SEED_C))


def run_identity(seq_length: int, seed_label: str) -> str:
    seed = dict(SEEDS)[seed_label]
    if seq_length not in SEQ_LENGTHS:
        raise SeqlenCampaignError(f"seq_length {seq_length} is not one of {SEQ_LENGTHS}")
    return f"{CAMPAIGN_ID}__L{seq_length:03d}__seed{seed_label}{seed}__{FIDELITY_TAG}__attempt001"


def campaign_rows() -> list[dict[str, Any]]:
    """Seed-major order: array tasks 0-4 are seed A at every length, 5-9 seed
    B, 10-14 seed C. A ``%5`` array concurrency limit only caps simultaneous
    tasks; it does not guarantee that one seed's five lengths run together
    (a freed slot starts the next index while others are still running)."""
    rows = []
    for seed_label, seed in SEEDS:
        for seq_length in SEQ_LENGTHS:
            rows.append({"index": len(rows), "seq_length": seq_length, "seed_label": seed_label, "seed": seed,
                         "run_id": run_identity(seq_length, seed_label),
                         "is_reference": seq_length == REFERENCE_SEQ_LENGTH})
    return rows


def config_differences(base: Mapping[str, Any], derived: Mapping[str, Any]) -> dict[str, list]:
    keys = sorted(set(base) | set(derived))
    return {k: [base.get(k), derived.get(k)] for k in keys if base.get(k) != derived.get(k) or (k in base) != (k in derived)}


def derive_run_config(base: Mapping[str, Any], *, seq_length: int, seed: int, experiment_name: str,
                      run_dir: str, support_file: str, observation_cutoff: str) -> dict[str, Any]:
    if COMMON_SUPPORT_CONFIG_KEY in base or OBSERVATION_CUTOFF_CONFIG_KEY in base:
        raise SeqlenCampaignError("base config already carries a common-support or observation-cutoff key")
    if not 0 < seq_length <= COMMON_SEQ_LENGTH_FLOOR:
        raise SeqlenCampaignError(f"seq_length {seq_length} exceeds the common floor {COMMON_SEQ_LENGTH_FLOOR}")
    derived = copy.deepcopy(dict(base))
    derived.update({"seq_length": int(seq_length), "seed": int(seed), "experiment_name": experiment_name,
                    "run_dir": str(run_dir), COMMON_SUPPORT_CONFIG_KEY: str(support_file),
                    OBSERVATION_CUTOFF_CONFIG_KEY: str(observation_cutoff)})
    changed = set(config_differences(base, derived))
    if not changed <= ALLOWED_CHANGED_KEYS:
        raise SeqlenCampaignError(f"derived config changes non-whitelisted keys: {sorted(changed - ALLOWED_CHANGED_KEYS)}")
    return derived


def _config_date(value) -> pd.Timestamp:
    """NH config dates are DD/MM/YYYY strings."""
    return pd.Timestamp(value) if not isinstance(value, str) else pd.to_datetime(value, format="%d/%m/%Y")


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_campaign(*, base_config_path, base_manifest_path, training_support_path, screening_contract_path,
                     output_root, write: bool = True, expected_screening_count: int = 400) -> dict[str, Any]:
    """Derive the 15 NH configs from the frozen P19 config.

    Verifies, before deriving anything: the P19 config bytes against its
    generation manifest, the P19 frozen scalar facts, the train/validation
    basin files against the manifest (``test_basins.txt`` is never opened; its
    path is copied verbatim and NH never reads it during training or
    validation), and that the training support / screening contract are the
    Common-336 artifacts for exactly this target, lead, inputs and train
    population. Writes ``<output_root>/<run_id>/config.yaml`` and
    ``<output_root>/campaign_manifest.json`` (refusing to overwrite a config
    whose bytes differ)."""
    base_config_path, output_root = Path(base_config_path), Path(output_root)
    manifest = json.loads(Path(base_manifest_path).read_text(encoding="utf-8"))
    artifact_sha = manifest["artifact_sha256"]
    if sha256_file(base_config_path) != artifact_sha["config.yaml"]:
        raise SeqlenCampaignError("P19 config bytes do not match its generation manifest")
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    wrong = {k: (base.get(k), v) for k, v in _BASE_EXPECTED.items() if base.get(k) != v}
    if wrong:
        raise SeqlenCampaignError(f"base config is not the frozen P19 config: {wrong}")
    basin_files = {}
    for key, name in (("train_basin_file", "train_basins.txt"), ("validation_basin_file", "validation_basins.txt")):
        actual = sha256_file(base[key])
        if actual != artifact_sha[name]:
            raise SeqlenCampaignError(f"{key} {base[key]} sha256 {actual} != P19 manifest {artifact_sha[name]}")
        basin_files[key] = {"path": base[key], "sha256": actual}

    support = load_training_support(training_support_path)
    contract = load_common336_screening_contract(screening_contract_path)
    checks = {
        "support id": support["support_id"] == TRAINING_SUPPORT_ID,
        "support lead": support["lead_hours"] == manifest["lead_hours"] == 6,
        "support target": [support["target_variable"]] == base["target_variables"],
        "support inputs": support["dynamic_inputs"] == base["dynamic_inputs"],
        "support train basins": support["train_basin_ids_sha256"] == artifact_sha["train_basins.txt"],
        "contract id": contract["contract_id"] == SCREENING_CONTRACT_ID,
        "contract lead": contract["lead_hours"] == 6,
        "contract target": [contract["target_variable"]] == base["target_variables"],
        "contract basins": len(contract["basin_ids"]) == expected_screening_count,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SeqlenCampaignError(f"Common-336 artifacts do not match the P19 config: {failed}")
    cutoff = screening_observation_cutoff(contract)
    cutoff_text = cutoff.isoformat()
    period_last_hours = {key: _config_date(base[key]) + pd.Timedelta(hours=23)
                         for key in ("train_end_date", "validation_end_date")}
    if period_last_hours["validation_end_date"] != cutoff or period_last_hours["train_end_date"] > cutoff:
        raise SeqlenCampaignError(f"observation cutoff {cutoff} does not close the P19 train/validation periods: "
                                  f"{period_last_hours}")

    support_path = Path(training_support_path).resolve()
    runs = []
    for row in campaign_rows():
        config_dir = output_root / row["run_id"]
        derived = derive_run_config(base, seq_length=row["seq_length"], seed=row["seed"],
                                    experiment_name=row["run_id"], run_dir=(config_dir / "runs").as_posix(),
                                    support_file=support_path.as_posix(), observation_cutoff=cutoff_text)
        text = yaml.safe_dump(derived, sort_keys=False, default_flow_style=False)
        config_path = config_dir / "config.yaml"
        if write:
            config_dir.mkdir(parents=True, exist_ok=True)
            if config_path.exists() and config_path.read_text(encoding="utf-8") != text:
                raise SeqlenCampaignError(f"refusing to overwrite a different existing config: {config_path}")
            config_path.write_bytes(text.encode("utf-8"))  # checksummed bytes; no newline translation
        runs.append({**row, "config_dir": config_dir.as_posix(),
                     "config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                     "config_differences_vs_p19": config_differences(base, derived)})
    result = {
        "schema_name": CAMPAIGN_MANIFEST_SCHEMA_NAME, "schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID, "seeds": {label: seed for label, seed in SEEDS},
        "seed_b_derivation": derive_seed(SEED_B_NAMESPACE, "B"), "seed_c_derivation": derive_seed_c(),
        "seed_b_replaced": {"previous": 1729, "reason": "reserved for promoted finalists (decision log)"},
        "observation_cutoff": {"config_key": OBSERVATION_CUTOFF_CONFIG_KEY, "value": cutoff_text,
                               "rule": "screening contract date_end + 23 h; nothing later is read by the "
                                       "loader or the scorer; lead targets end at cutoff - lead"},
        "result_summary_rule": RESULT_SUMMARY_RULE,
        "reference_seq_length": REFERENCE_SEQ_LENGTH, "allowed_changed_keys": sorted(ALLOWED_CHANGED_KEYS),
        "base_config": {"path": base_config_path.as_posix(), "sha256": artifact_sha["config.yaml"],
                        "experiment_name": base["experiment_name"]},
        "base_generation_manifest": {"path": Path(base_manifest_path).as_posix(),
                                     "sha256": sha256_file(base_manifest_path), "git_commit": manifest["git_commit"]},
        "basin_files": basin_files,
        "test_basin_file": {"path": base["test_basin_file"], "opened": False},
        "training_support": {"path": support_path.as_posix(), "file_sha256": sha256_file(support_path),
                             "checksum_sha256": support["checksum_sha256"], "support_id": support["support_id"]},
        "screening_contract": {"path": Path(screening_contract_path).resolve().as_posix(),
                               "file_sha256": sha256_file(screening_contract_path),
                               "checksum_sha256": contract["checksum_sha256"], "contract_id": contract["contract_id"]},
        "runs": runs,
    }
    if write:
        path = output_root / "campaign_manifest.json"
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise SeqlenCampaignError(f"refusing to overwrite a different existing campaign manifest: {path}")
        path.write_bytes(text.encode("utf-8"))
    return result


# --------------------------------------------------------------------------- #
# One-trial execution (training + post-hoc Common-336 screening)
# --------------------------------------------------------------------------- #

TRIAL_RECEIPT_NAME = "seqlen_common336_trial_receipt.json"
PER_EPOCH_RESULTS_NAME = "common336_screening_by_epoch.json"


def verify_campaign_row(manifest: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Re-verify, at launch time, every frozen byte a trial depends on."""
    runs = manifest["runs"]
    if manifest.get("schema_name") != CAMPAIGN_MANIFEST_SCHEMA_NAME or manifest.get("campaign_id") != CAMPAIGN_ID:
        raise SeqlenCampaignError("not a seqlen_common336 campaign manifest")
    if manifest.get("schema_version") != CAMPAIGN_MANIFEST_SCHEMA_VERSION or manifest.get("seeds") != dict(SEEDS):
        raise SeqlenCampaignError("campaign manifest predates the frozen seeds/cutoff (schema v2); regenerate it")
    if not 0 <= index < len(runs) or len(runs) != 15:
        raise SeqlenCampaignError(f"index {index} out of range for {len(runs)} runs")
    row = dict(runs[index])
    if row["index"] != index or row["run_id"] != run_identity(row["seq_length"], row["seed_label"]):
        raise SeqlenCampaignError("campaign manifest row identity is inconsistent")
    config_path = Path(row["config_dir"]) / "config.yaml"
    checks = {"config": (config_path, row["config_sha256"])}
    for key, entry in manifest["basin_files"].items():
        checks[key] = (Path(entry["path"]), entry["sha256"])
    checks["training support"] = (Path(manifest["training_support"]["path"]), manifest["training_support"]["file_sha256"])
    checks["screening contract"] = (Path(manifest["screening_contract"]["path"]),
                                    manifest["screening_contract"]["file_sha256"])
    bad = [name for name, (path, sha) in checks.items() if not path.is_file() or sha256_file(path) != sha]
    if bad:
        raise SeqlenCampaignError(f"frozen launch inputs changed or missing: {bad}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {"seq_length": row["seq_length"], "seed": row["seed"], "experiment_name": row["run_id"],
                COMMON_SUPPORT_CONFIG_KEY: manifest["training_support"]["path"],
                OBSERVATION_CUTOFF_CONFIG_KEY: manifest["observation_cutoff"]["value"]}
    wrong = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if wrong:
        raise SeqlenCampaignError(f"config does not carry the row identity: {wrong}")
    return row


def verify_applied_training_support(nh_run_dir, *, seq_length: int, seed: int, support: Mapping[str, Any]) -> dict:
    """Prove from the run's own artifacts that the actual training loader
    applied the support: the dumped config carries the key/length/seed and the
    dataset's ``train_data/`` receipt names the support's checksum with
    exactly ``sum(eligible_counts)`` kept samples."""
    from .nh_dataset import COMMON_SUPPORT_RECEIPT_NAME

    nh_run_dir = Path(nh_run_dir)
    dumped = yaml.safe_load((nh_run_dir / "config.yml").read_text(encoding="utf-8"))
    if (dumped.get("seq_length"), dumped.get("seed")) != (seq_length, seed) or not dumped.get(COMMON_SUPPORT_CONFIG_KEY):
        raise SeqlenCampaignError("dumped NH config does not carry this trial's length/seed/support key")
    receipt_path = nh_run_dir / "train_data" / COMMON_SUPPORT_RECEIPT_NAME
    if not receipt_path.is_file():
        raise SeqlenCampaignError(f"training loader left no common-support receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {"support_id": support["support_id"], "checksum_sha256": support["checksum_sha256"],
                "seq_length": seq_length, "n_kept": sum(support["eligible_counts"].values()),
                "n_basins": len(support["basin_ids"])}
    wrong = {k: (receipt.get(k), v) for k, v in expected.items() if receipt.get(k) != v}
    if wrong:
        raise SeqlenCampaignError(f"training loader applied a different support: {wrong}")
    return receipt


def summarize_trial(result, *, row: Mapping[str, Any], screening_population: int = 400) -> dict[str, Any]:
    """Common-336 per-epoch objective trajectory and best epoch (median
    per-basin raw-space NSE; earliest epoch on ties -- the existing
    ``sweep_v1_campaign.derive_trajectory_diagnostics`` rule)."""
    from .sweep_v1_campaign import TARGET_EPOCH, derive_trajectory_diagnostics

    if result.blocked or result.final_status != "monolithic_training_and_screening_complete":
        raise SeqlenCampaignError(f"trial did not complete: {result.final_status}: {result.blocked_reason}")
    supplemental = result.supplemental_epoch_results
    if set(supplemental) != set(range(1, TARGET_EPOCH + 1)):
        raise SeqlenCampaignError("Common-336 results must cover exactly epochs 1..12")
    trajectory, per_epoch = {}, {}
    for epoch in sorted(supplemental):
        scored = supplemental[epoch]["common336"]
        if (scored["contract_id"] != SCREENING_CONTRACT_ID or scored["n_basins_evaluated"] != screening_population
                or scored["n_basins_excluded"] != 0):
            raise SeqlenCampaignError(f"epoch {epoch}: Common-336 screening is not complete over {screening_population} basins")
        value = scored["aggregate"]["metrics"]["nse"]["median"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SeqlenCampaignError(f"epoch {epoch}: Common-336 median NSE missing or non-finite")
        trajectory[epoch] = float(value)
        per_epoch[epoch] = {k: scored[k] for k in ("aggregate", "n_basins_evaluated", "evaluation_source")}
    diagnostics = derive_trajectory_diagnostics(trajectory)
    return {"run_id": row["run_id"], "seq_length": row["seq_length"], "seed_label": row["seed_label"],
            "seed": row["seed"], "selection_rule": "best epoch by median per-basin raw-space NSE on "
            f"{SCREENING_CONTRACT_ID} (earliest on ties)", "common336_median_nse_by_epoch": trajectory,
            "best_epoch": diagnostics["best_epoch"], "best_score": diagnostics["best_score"],
            "trajectory_diagnostics": diagnostics, "per_epoch_aggregate": per_epoch}
