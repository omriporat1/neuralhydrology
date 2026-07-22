"""Stage 1 NH-config generation (local implementation increment).

Generates a single, concrete NeuralHydrology 1.13 runtime config + matching
basin-list files for one approved (lead, seq_length) combination against the
certified Stage 1 Compact Scientific Package
(``stage1_compact_scientific_package_v001``, Gate 4 PASS). This module does
not build the 16-config matrix, does not write a Slurm script, does not
configure W&B, and never touches h2o/Moriah -- it is local-only and reads its
inputs (policy YAML, package manifest, split-list files, optional static
column-role manifest) as plain files via injectable paths.

Design mirrors :mod:`src.baseline.splits` (small, deterministic, testable
functions rather than a class hierarchy) and reuses rather than reimplements:
``policy.load_stage1_baseline_policy`` for the scientific policy,
``lead_targets.variable_name_for_lead`` for the target-variable name,
``splits.load_eligible_basins``/``sha256_of`` for split-file reading and
checksums, ``static_preparation.load_column_manifest`` /
``model_input_columns_from_manifest`` for the optional external static
column-role re-derivation, and ``staid.normalize_staid`` for STAID handling.

Known, accepted documentation debt (see ``docs/decision_log.md``): the
committed policy YAML's ``nh.dataset`` value is ``"generic"`` -- a distinct,
still-enforced invariant checked by ``policy.validate_stage1_baseline_policy``
for a different (Smoke-era) historical purpose -- while every rendered
config produced here hardcodes ``dataset: "flashnh"`` per this task's binding
requirement. The policy YAML itself is signed-off (policy_version 2) and is
intentionally not edited to resolve this; the discrepancy is documented here
and in the final report rather than papered over.

The rendered config also includes a frozen, explicitly-labeled **compact
smoke-run profile** (``model``, ``hidden_size``, ``optimizer``, ``loss``,
``epochs``, ...): the small set of runnable NH 1.13 training settings used
for the first real integration-validation training run against the
certified 32-basin compact package (lead06/seq24 only, ~2 epochs, CPU-safe
single-process data loading, no W&B). These values are NOT the scientific
baseline/tuning seed and are not sourced from the policy YAML -- they are
recorded as ``compact_smoke_run_profile: true`` in the generation manifest
and must not be read as a tuned model-architecture or hyperparameter choice.
See ``docs/decision_log.md``'s 2026-07-22 entry for the full rationale.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .lead_targets import DEFAULT_VARIABLE_NAME_TEMPLATE, variable_name_for_lead
from .policy import Stage1BaselinePolicyError, load_stage1_baseline_policy
from .splits import SplitGenerationError, load_eligible_basins, sha256_of
from .staid import normalize_staid
from .static_preparation import StaticPreparationError, load_column_manifest, model_input_columns_from_manifest

__all__ = [
    "NHConfigGenerationError",
    "StaticAttributeContractResult",
    "GeneratedConfigBundle",
    "read_package_manifest",
    "read_package_attribute_columns",
    "resolve_target_variable",
    "validate_seq_length",
    "validate_lead_hours",
    "validate_target_variables",
    "validate_dynamic_inputs",
    "validate_static_attribute_contract",
    "validate_basin_membership",
    "build_nh_config_mapping",
    "generate_stage1_nh_config",
    "write_generated_config",
]


class NHConfigGenerationError(ValueError):
    """Raised for an invalid config-generation input, package/policy mismatch,
    or contract violation (multi-target, raw-target-as-target, static/dynamic
    mismatch, non-canonical basin membership, forbidden config key, etc.)."""


# Frozen compact smoke-run profile: the runnable NH 1.13 training settings
# for the FIRST real integration-validation training run against the
# certified 32-basin compact package (lead06/seq24 only). This is an
# explicit, deliberately small technical profile -- NOT the scientific
# baseline/tuning seed -- chosen to prove end-to-end data loading, a
# forward/backward pass, an optimizer step, validation, and checkpoint
# writing, without a hyperparameter search or W&B. hidden_size/batch_size are
# larger than the old structural-placeholder values (8) because this profile
# is now actually executed rather than only parsed; validate_n_random_basins
# covers all 32 compact basins since that is cheap at this sample count and
# gives a real per-basin validation signal instead of a single-basin sample.
# device is "cuda:0" because the training smoke (Moriah GPU job) is the only
# consumer of this value -- the CPU-only construction-preflight job never
# instantiates an NH Trainer/device (see nh_structural_preflight.py, which
# never reads cfg.device), so it is unaffected by this setting. Values may be
# revisited if NH 1.13 mechanics or the compact sample count make a
# different small value more appropriate; see docs/decision_log.md's
# 2026-07-22 entry.
_COMPACT_SMOKE_RUN_PROFILE = {
    "model": "cudalstm",
    "hidden_size": 64,
    "batch_size": 64,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "MSE",
    "save_weights_every": 1,
    "validate_every": 1,
    "validate_n_random_basins": 32,
    "log_interval": 50,
    "num_workers": 0,
    "epochs": 2,
    "device": "cuda:0",
    "verbose": 0,
}

_FORBIDDEN_KEY_SUBSTRINGS = (
    "partition", "gres", "gpu", "hostname", "username", "password", "token", "secret", "credential",
)


@dataclass(frozen=True)
class StaticAttributeContractResult:
    """Result of the two-sided static-attribute contract check (task item 2)."""

    columns: list
    count: int
    columns_sha256: str
    package_manifest_columns_sha256: str
    package_attributes_csv_columns_sha256: str
    external_column_manifest_path: "str | None" = None
    external_column_manifest_sha256: "str | None" = None
    external_column_manifest_derived_columns_sha256: "str | None" = None


@dataclass(frozen=True)
class GeneratedConfigBundle:
    """Everything needed to write the generated config + basin lists + manifest.

    Pure data, produced with no filesystem writes by
    :func:`generate_stage1_nh_config`; :func:`write_generated_config` is the
    only function in this module that writes to disk.
    """

    config_mapping: dict
    basin_ids: list
    lead_hours: int
    seq_length: int
    target_variable: str
    dynamic_inputs: list
    static_attribute_result: StaticAttributeContractResult
    package_root: str
    package_manifest_identity: dict
    policy_path: str
    policy_sha256: str
    splits_dir: str
    generated_at_utc: str
    git_commit: "str | None"
    package_type: str = "compact_temporal_integration_validation"


# ---------------------------------------------------------------------------
# Package-side readers
# ---------------------------------------------------------------------------

def read_package_manifest(package_root) -> dict:
    p = Path(package_root) / "manifests" / "package_manifest.json"
    if not p.is_file():
        raise NHConfigGenerationError(f"package manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def read_package_attribute_columns(package_root) -> list:
    """Read ``attributes/attributes.csv``'s header, excluding the leading
    basin-ID/index field (``gauge_id``)."""
    p = Path(package_root) / "attributes" / "attributes.csv"
    if not p.is_file():
        raise NHConfigGenerationError(f"package attributes.csv not found: {p}")
    with open(p, "r", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), None)
    if not header:
        raise NHConfigGenerationError(f"{p}: empty or unreadable header row")
    if header[0] != "gauge_id":
        raise NHConfigGenerationError(f"{p}: expected first column 'gauge_id', got {header[0]!r}")
    return header[1:]


# ---------------------------------------------------------------------------
# Target / lead / sequence-length validation
# ---------------------------------------------------------------------------

def resolve_target_variable(lead_hours: int, variable_name_template: str = DEFAULT_VARIABLE_NAME_TEMPLATE) -> str:
    return variable_name_for_lead(lead_hours, variable_name_template)


def validate_seq_length(seq_length: int, policy: dict) -> None:
    allowed = policy["seq_lengths_hours"]
    if not isinstance(seq_length, int) or isinstance(seq_length, bool) or seq_length not in allowed:
        raise NHConfigGenerationError(
            f"seq_length {seq_length!r} is not one of the policy-allowed seq_lengths_hours {allowed}"
        )


def validate_lead_hours(lead_hours: int, policy: dict) -> None:
    allowed = policy["target"]["leads_hours"]
    if not isinstance(lead_hours, int) or isinstance(lead_hours, bool) or lead_hours not in allowed:
        raise NHConfigGenerationError(
            f"lead_hours {lead_hours!r} is not one of the policy-allowed target.leads_hours {allowed}"
        )


def validate_target_variables(target_variables: list, policy: dict) -> None:
    if not isinstance(target_variables, list) or len(target_variables) != 1:
        raise NHConfigGenerationError(
            f"exactly one target variable is required, got {target_variables!r}"
        )
    raw_target = policy["target"]["source_variable"]
    name = target_variables[0]
    if name == raw_target:
        raise NHConfigGenerationError(
            f"raw target variable {raw_target!r} must never be used directly as target_variables; "
            "select a lead-shifted variable instead"
        )


def validate_dynamic_inputs(dynamic_inputs: list, policy: dict) -> None:
    """Exact, order-sensitive equality against the policy's binding dynamic-input list."""
    expected = list(policy["dynamic_inputs"])
    if list(dynamic_inputs) != expected:
        raise NHConfigGenerationError(
            f"dynamic_inputs must exactly equal the policy order {expected}, got {list(dynamic_inputs)}"
        )


# ---------------------------------------------------------------------------
# Static-attribute contract (task item 2)
# ---------------------------------------------------------------------------

def validate_static_attribute_contract(
    policy: dict,
    package_manifest: dict,
    package_attribute_columns: list,
    *,
    static_column_manifest_path=None,
) -> StaticAttributeContractResult:
    """Two-sided static-attribute contract.

    Mandatory core check: ``package_manifest.json``'s recorded
    ``static_model_input_columns`` (already build-time-enforced by the
    package builder to equal the sorted, manifest-role-derived list) must
    exactly equal ``attributes.csv``'s header, excluding ``gauge_id``, in
    both name and order. Count, duplicate, and forbidden-field checks are
    then applied to that agreed-upon list.

    Optional additional check, only performed when
    ``static_column_manifest_path`` is given: independently re-derive the
    ``model_input``-role column list from that external column-role
    manifest and require it to equal the package's own list (not merely
    "all non-ID columns").
    """
    expected_count = policy["static_attributes"]["expected_model_input_columns"]
    forbidden = tuple(policy["static_attributes"]["forbidden_model_inputs"])
    allowed_role = policy["static_attributes"]["allowed_role"]

    manifest_columns = list(package_manifest.get("static_model_input_columns", []))
    if not manifest_columns:
        raise NHConfigGenerationError("package manifest has no static_model_input_columns recorded")
    if len(manifest_columns) != len(set(manifest_columns)):
        dupes = sorted({c for c in manifest_columns if manifest_columns.count(c) > 1})
        raise NHConfigGenerationError(f"package manifest static_model_input_columns contains duplicates: {dupes}")

    if list(package_attribute_columns) != manifest_columns:
        raise NHConfigGenerationError(
            "package manifest static_model_input_columns does not exactly equal "
            "attributes.csv's header (excluding gauge_id) in name and order; "
            f"manifest has {len(manifest_columns)} column(s), attributes.csv has "
            f"{len(package_attribute_columns)} column(s)"
        )

    if len(manifest_columns) != expected_count:
        raise NHConfigGenerationError(
            f"expected exactly {expected_count} static model_input columns, got {len(manifest_columns)}"
        )

    forbidden_present = [c for c in manifest_columns if c in forbidden]
    if forbidden_present:
        raise NHConfigGenerationError(f"forbidden static column(s) present in package: {forbidden_present}")

    columns_sha256 = hashlib.sha256("\n".join(manifest_columns).encode("utf-8")).hexdigest()
    package_manifest_columns_sha256 = package_manifest.get("static_model_input_columns_sha256", "") or ""
    if package_manifest_columns_sha256 and package_manifest_columns_sha256 != columns_sha256:
        raise NHConfigGenerationError(
            "recomputed static column-list checksum does not match package_manifest.json's "
            "own recorded static_model_input_columns_sha256"
        )
    package_attributes_csv_columns_sha256 = hashlib.sha256(
        "\n".join(package_attribute_columns).encode("utf-8")
    ).hexdigest()

    external_manifest_sha256 = None
    external_derived_sha256 = None
    if static_column_manifest_path is not None:
        external_manifest_sha256 = sha256_of(static_column_manifest_path)
        try:
            manifest_doc = load_column_manifest(static_column_manifest_path)
            derived_columns = model_input_columns_from_manifest(manifest_doc, role=allowed_role)
        except StaticPreparationError as exc:
            raise NHConfigGenerationError(f"external static column-role manifest invalid: {exc}") from exc
        external_derived_sha256 = hashlib.sha256("\n".join(derived_columns).encode("utf-8")).hexdigest()
        if derived_columns != manifest_columns:
            raise NHConfigGenerationError(
                "package static_model_input_columns does not match the model_input-role "
                "derivation from the externally supplied column-role manifest"
            )

    return StaticAttributeContractResult(
        columns=manifest_columns,
        count=len(manifest_columns),
        columns_sha256=columns_sha256,
        package_manifest_columns_sha256=package_manifest_columns_sha256,
        package_attributes_csv_columns_sha256=package_attributes_csv_columns_sha256,
        external_column_manifest_path=str(static_column_manifest_path) if static_column_manifest_path else None,
        external_column_manifest_sha256=external_manifest_sha256,
        external_column_manifest_derived_columns_sha256=external_derived_sha256,
    )


# ---------------------------------------------------------------------------
# Basin membership + leakage safeguards (task item 4)
# ---------------------------------------------------------------------------

def validate_basin_membership(package_manifest: dict, splits_dir) -> list:
    """Confirm every package basin is a canonical ``development_train`` member
    and that none belongs to the non-CA spatial holdout or the California
    universe. Returns the sorted, normalized basin-ID list (same 32 IDs used
    for train/validation/test -- separation here is temporal only)."""
    splits_dir = Path(splits_dir)
    raw_basin_ids = package_manifest.get("basin_ids", [])
    if not raw_basin_ids:
        raise NHConfigGenerationError("package manifest has no basin_ids recorded")

    package_basins = []
    for raw in raw_basin_ids:
        try:
            package_basins.append(normalize_staid(raw))
        except (TypeError, ValueError) as exc:
            raise NHConfigGenerationError(f"malformed basin id {raw!r} in package manifest: {exc}") from exc
    if len(package_basins) != len(set(package_basins)):
        raise NHConfigGenerationError("package manifest basin_ids contains duplicates after normalization")

    try:
        development_train = set(load_eligible_basins(splits_dir / "development_train.txt"))
        spatial_holdout = set(load_eligible_basins(splits_dir / "spatial_holdout_nonca.txt"))
        california_all = set(load_eligible_basins(splits_dir / "california_all.txt"))
    except SplitGenerationError as exc:
        raise NHConfigGenerationError(f"could not read canonical split file(s): {exc}") from exc

    not_dev_train = sorted(set(package_basins) - development_train)
    if not_dev_train:
        raise NHConfigGenerationError(
            f"package basin(s) are not members of the canonical development_train split: {not_dev_train}"
        )

    holdout_overlap = sorted(set(package_basins) & (spatial_holdout | california_all))
    if holdout_overlap:
        raise NHConfigGenerationError(
            "package basin(s) overlap the non-CA spatial holdout or California universe -- "
            f"forbidden for a compact temporal-integration-validation config: {holdout_overlap}"
        )

    return sorted(package_basins)


# ---------------------------------------------------------------------------
# Config mapping construction
# ---------------------------------------------------------------------------

def _format_ddmmyyyy(iso_date: str) -> str:
    y, m, d = iso_date.split("-")
    return f"{d}/{m}/{y}"


def build_nh_config_mapping(
    *,
    policy: dict,
    target_variable: str,
    seq_length: int,
    dynamic_inputs: list,
    static_attributes: list,
) -> dict:
    """Pure function: assemble the policy/target/structural fields of the
    rendered config. Does not include experiment_name, basin-file paths,
    data_dir, or run_dir -- those depend on the concrete output directory and
    are filled in by :func:`write_generated_config`."""
    temporal = policy["temporal_split"]
    nh_policy = policy["nh"]

    mapping = {
        # This is the actual NH 1.13 runtime dataset key (registered by
        # nh_register.register_flashnh_dataset()) and is intentionally NOT
        # sourced from policy["nh"]["dataset"] (== "generic"). The policy
        # field is a separate, signed-off NH-1.13-compat invariant describing
        # inherited storage-layout compatibility from an earlier (Smoke-era)
        # policy revision -- it is not a runtime dataset-key choice and must
        # not be conflated with this key. See module docstring's "Known,
        # accepted documentation debt" note and the generation manifest's
        # "nh_runtime_dataset_key_note" field for the same clarification
        # surfaced in the generated evidence artifact.
        "dataset": "flashnh",
        "train_start_date": _format_ddmmyyyy(temporal["training"]["start"]),
        "train_end_date": _format_ddmmyyyy(temporal["training"]["end"]),
        "validation_start_date": _format_ddmmyyyy(temporal["validation"]["start"]),
        "validation_end_date": _format_ddmmyyyy(temporal["validation"]["end"]),
        "test_start_date": _format_ddmmyyyy(temporal["test"]["start"]),
        "test_end_date": _format_ddmmyyyy(temporal["test"]["end"]),
        "target_variables": [target_variable],
        "head": nh_policy["head"],
        "output_activation": nh_policy["output_activation"],
        "predict_last_n": nh_policy["predict_last_n"],
        "seq_length": seq_length,
        "dynamic_inputs": list(dynamic_inputs),
        "static_attributes": list(static_attributes),
    }
    mapping.update(_COMPACT_SMOKE_RUN_PROFILE)
    # nan_handling_method deliberately absent: hard-exclusion baseline
    # (accepted finding #7); never set as a defensive backstop here.
    return mapping


def _get_git_commit(cwd=None) -> "str | None":
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd if cwd is not None else Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Top-level orchestration (no filesystem writes)
# ---------------------------------------------------------------------------

def generate_stage1_nh_config(
    *,
    policy_path,
    package_root,
    splits_dir,
    lead_hours: int,
    seq_length: int,
    static_column_manifest_path=None,
) -> GeneratedConfigBundle:
    policy_path = Path(policy_path)
    package_root = Path(package_root)
    splits_dir = Path(splits_dir)

    try:
        policy = load_stage1_baseline_policy(policy_path)
    except Stage1BaselinePolicyError as exc:
        raise NHConfigGenerationError(f"policy failed validation: {exc}") from exc

    validate_seq_length(seq_length, policy)
    validate_lead_hours(lead_hours, policy)

    target_variable = resolve_target_variable(lead_hours, policy["target"]["variable_name_template"])
    validate_target_variables([target_variable], policy)

    package_manifest = read_package_manifest(package_root)
    package_attribute_columns = read_package_attribute_columns(package_root)

    dynamic_inputs = list(policy["dynamic_inputs"])
    validate_dynamic_inputs(package_manifest.get("dynamic_variables", []), policy)

    static_result = validate_static_attribute_contract(
        policy,
        package_manifest,
        package_attribute_columns,
        static_column_manifest_path=static_column_manifest_path,
    )

    basin_ids = validate_basin_membership(package_manifest, splits_dir)

    config_mapping = build_nh_config_mapping(
        policy=policy,
        target_variable=target_variable,
        seq_length=seq_length,
        dynamic_inputs=dynamic_inputs,
        static_attributes=static_result.columns,
    )

    package_manifest_identity = {
        "schema_name": package_manifest.get("schema_name"),
        "schema_version": package_manifest.get("schema_version"),
        "package_role": package_manifest.get("package_role"),
        "basin_count": package_manifest.get("basin_count"),
        "static_model_input_columns_sha256": package_manifest.get("static_model_input_columns_sha256"),
    }

    return GeneratedConfigBundle(
        config_mapping=config_mapping,
        basin_ids=basin_ids,
        lead_hours=lead_hours,
        seq_length=seq_length,
        target_variable=target_variable,
        dynamic_inputs=dynamic_inputs,
        static_attribute_result=static_result,
        package_root=str(package_root),
        package_manifest_identity=package_manifest_identity,
        policy_path=str(policy_path),
        policy_sha256=sha256_of(policy_path),
        splits_dir=str(splits_dir),
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_commit=_get_git_commit(),
    )


# ---------------------------------------------------------------------------
# Writer (the only function in this module that touches disk for output)
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _check_no_forbidden_keys(mapping: dict) -> None:
    for key in mapping:
        lowered = key.lower()
        if any(bad in lowered for bad in _FORBIDDEN_KEY_SUBSTRINGS):
            raise NHConfigGenerationError(f"generated config contains a forbidden key: {key!r}")


def write_generated_config(
    bundle: GeneratedConfigBundle,
    out_dir,
    *,
    experiment_name: "str | None" = None,
    force: bool = False,
) -> dict:
    """Atomically write the generated basin-list files, ``config.yaml``, and
    ``generation_manifest.json`` under ``out_dir``.

    Fails if ``out_dir`` already exists and is non-empty, unless
    ``force=True`` (mirrors ``splits.write_split_artifacts``'s safety
    pattern). Never writes into a tracked source/config directory implicitly
    -- callers choose ``out_dir`` explicitly (CLI default is under
    ``tmp/``, which is gitignored).
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise NHConfigGenerationError(
            f"output directory already exists and is non-empty: {out_dir} (use force=True/--force)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_name = experiment_name or f"stage1_compact_lead{bundle.lead_hours:02d}_seq{bundle.seq_length}_v001"

    basin_list_text = "\n".join(bundle.basin_ids) + "\n"
    train_basin_file = out_dir / "train_basins.txt"
    validation_basin_file = out_dir / "validation_basins.txt"
    test_basin_file = out_dir / "test_basins.txt"
    for p in (train_basin_file, validation_basin_file, test_basin_file):
        _atomic_write_text(p, basin_list_text)

    run_dir = out_dir / "runs"

    full_mapping = dict(bundle.config_mapping)
    full_mapping["experiment_name"] = exp_name
    full_mapping["train_basin_file"] = str(train_basin_file)
    full_mapping["validation_basin_file"] = str(validation_basin_file)
    full_mapping["test_basin_file"] = str(test_basin_file)
    full_mapping["data_dir"] = bundle.package_root
    full_mapping["run_dir"] = str(run_dir)

    _check_no_forbidden_keys(full_mapping)

    config_path = out_dir / "config.yaml"
    _atomic_write_text(config_path, yaml.safe_dump(full_mapping, sort_keys=False))

    written_paths = {
        "train_basins.txt": train_basin_file,
        "validation_basins.txt": validation_basin_file,
        "test_basins.txt": test_basin_file,
        "config.yaml": config_path,
    }
    artifact_sha256 = {name: sha256_of(p) for name, p in sorted(written_paths.items())}

    generation_manifest = {
        "schema_name": "stage1_nh_config_generation_manifest",
        "schema_version": 1,
        "generated_at_utc": bundle.generated_at_utc,
        "git_commit": bundle.git_commit,
        "package_type": bundle.package_type,
        "lead_hours": bundle.lead_hours,
        "seq_length": bundle.seq_length,
        "target_variable": bundle.target_variable,
        "dates": {
            "train_start_date": full_mapping["train_start_date"],
            "train_end_date": full_mapping["train_end_date"],
            "validation_start_date": full_mapping["validation_start_date"],
            "validation_end_date": full_mapping["validation_end_date"],
            "test_start_date": full_mapping["test_start_date"],
            "test_end_date": full_mapping["test_end_date"],
        },
        "dynamic_inputs": bundle.dynamic_inputs,
        "static_attribute_count": bundle.static_attribute_result.count,
        "static_attribute_columns_sha256": bundle.static_attribute_result.columns_sha256,
        "static_attribute_contract": {
            "package_manifest_columns_sha256": bundle.static_attribute_result.package_manifest_columns_sha256,
            "package_attributes_csv_columns_sha256": bundle.static_attribute_result.package_attributes_csv_columns_sha256,
            "external_column_manifest_path": bundle.static_attribute_result.external_column_manifest_path,
            "external_column_manifest_sha256": bundle.static_attribute_result.external_column_manifest_sha256,
            "external_column_manifest_derived_columns_sha256":
                bundle.static_attribute_result.external_column_manifest_derived_columns_sha256,
        },
        "basin_count": len(bundle.basin_ids),
        "basin_ids": bundle.basin_ids,
        "package_root": bundle.package_root,
        "package_manifest_identity": bundle.package_manifest_identity,
        "policy_path": bundle.policy_path,
        "policy_sha256": bundle.policy_sha256,
        "splits_dir": bundle.splits_dir,
        "compact_smoke_run_profile": True,
        "compact_smoke_run_profile_note": (
            "model/hidden_size/optimizer/loss/epochs/etc. are the frozen "
            "compact-smoke-run technical settings for this first lead06/seq24 "
            "integration-validation training run; they are NOT the scientific "
            "baseline or a hyperparameter-tuning seed."
        ),
        "nh_runtime_dataset_key": full_mapping["dataset"],
        "nh_runtime_dataset_key_note": (
            "The NH 1.13 config 'dataset' key actually used at runtime "
            "(registered by nh_register.register_flashnh_dataset()). "
            "Intentionally distinct from policy['nh']['dataset'] == 'generic', "
            "which is a separate, signed-off NH-1.13-compat invariant from an "
            "earlier (Smoke-era) policy revision describing inherited "
            "storage-layout compatibility, not the runtime dataset key."
        ),
        "nan_handling_method": None,
        "artifact_sha256": artifact_sha256,
    }
    manifest_path = out_dir / "generation_manifest.json"
    _atomic_write_text(manifest_path, json.dumps(generation_manifest, indent=2, default=str))

    written_paths["generation_manifest.json"] = manifest_path
    return written_paths
