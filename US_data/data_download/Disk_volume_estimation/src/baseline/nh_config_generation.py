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
import math
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
    "FullPopulationBasinMembership",
    "FullPopulationConfigBundles",
    "EXPECTED_DEVELOPMENT_BASIN_COUNT",
    "EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT",
    "COMPACT_SMOKE_RUN_PROFILE_NAME",
    "INITIAL_SEED_RUN_PROFILE_NAME",
    "EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME",
    "PILOT_LEAD06_RAW_SEEDA_PROFILE_NAME",
    "PILOT_LEAD06_RAW_SEEDB_PROFILE_NAME",
    "PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME",
    "PILOT_LEAD06_EMB128X64_SEEDB_PROFILE_NAME",
    "PILOT_LEAD06_EMB64_SEEDA_PROFILE_NAME",
    "PILOT_LEAD06_EMB128_SEEDA_PROFILE_NAME",
    "PILOT_LEAD06_EMB64X32_SEEDA_PROFILE_NAME",
    "PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME",
    "PILOT_LEAD06_EMB256X64_SEEDA_PROFILE_NAME",
    "PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME",
    "KNOWN_RUN_PROFILE_NAMES",
    "get_run_profile_mapping",
    "validate_statics_embedding_spec",
    "HOLDOUT_MARKER_FILENAME",
    "HoldoutBundleTrainingRejected",
    "raise_if_holdout_bundle",
    "read_package_manifest",
    "read_package_attribute_columns",
    "resolve_target_variable",
    "validate_seq_length",
    "validate_lead_hours",
    "validate_target_variables",
    "validate_dynamic_inputs",
    "validate_static_attribute_contract",
    "validate_basin_membership",
    "validate_full_population_basin_membership",
    "build_nh_config_mapping",
    "validate_embedding_dropout_override",
    "generate_stage1_nh_config",
    "generate_stage1_full_population_nh_config_bundles",
    "write_generated_config",
]

# Pinned facts about the certified stage1_scientific_package_v002 full non-CA
# population (see docs/decision_log.md's Gate 4 PASS entry and
# scripts/prepare_stage1_full_static_attributes.py's identically pinned
# constants). Never exposed as a CLI argument: a caller cannot silently ask
# for a different split size, only the canonical one.
EXPECTED_DEVELOPMENT_BASIN_COUNT = 2307
EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT = 250


class NHConfigGenerationError(ValueError):
    """Raised for an invalid config-generation input, package/policy mismatch,
    or contract violation (multi-target, raw-target-as-target, static/dynamic
    mismatch, non-canonical basin membership, forbidden config key, etc.)."""


# Sibling-file safeguard name for a spatial-holdout (test-only) generated
# bundle -- see write_generated_config's rationale (NH's own
# Config._check_cfg_keys forbids marking this inside config.yaml itself, so
# it must be a file NH never reads).
HOLDOUT_MARKER_FILENAME = "TEST_ONLY_DO_NOT_TRAIN.txt"


class HoldoutBundleTrainingRejected(NHConfigGenerationError):
    """Raised by raise_if_holdout_bundle when a training entrypoint is
    pointed at a spatial-holdout (test-only) generated config directory."""


def raise_if_holdout_bundle(generated_dir) -> None:
    """Guard for any training entrypoint (the GPU training sbatch script,
    ``scripts/run_stage1_nh.py train``/``continue`` callers): refuses to
    proceed if ``generated_dir`` is a spatial-holdout bundle, identified by
    the presence of :data:`HOLDOUT_MARKER_FILENAME`. This is the only
    reliable holdout/development discriminator available to a training
    launcher, since the holdout bundle's config.yaml otherwise looks like an
    ordinary runnable NH config (its train/validation basin lists are the
    development population, by design -- see write_generated_config)."""
    marker_path = Path(generated_dir) / HOLDOUT_MARKER_FILENAME
    if marker_path.is_file():
        raise HoldoutBundleTrainingRejected(
            f"refusing to train: {generated_dir} is a SPATIAL-HOLDOUT TEST-ONLY bundle "
            f"(found {marker_path}). Point the training launcher at the development bundle instead."
        )


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

# Initial full-population seed training profile: the FIRST real scientific
# training run against the certified 2,307-basin development population
# (stage1_scientific_package_v002, lead06/seq24). This is an initial,
# untuned seed -- not the official Stage 1 benchmark, not a sweep result.
# Values are taken from docs/stage1_scientific_baseline_design.md Sec 9c
# ("initial seed / first-viable-config only") wherever that table gives an
# exact value; two documented ranges required a narrow, non-inventive
# resolution, recorded here rather than left implicit:
#   - dropout ("~0.2-0.3"): resolved to the range midpoint, 0.25. The config
#     key name is "output_dropout" (not "dropout") -- confirmed directly
#     against the installed NH 1.13 package on Moriah
#     (neuralhydrology/utils/config.py's Config.output_dropout property,
#     `self._cfg.get("output_dropout", 0.0)`). An earlier version of this
#     comment cited scripts/build_stage1_neuralhydrology_january_pilot.py as
#     precedent for the plain "dropout" key name; that script is an unrun
#     placeholder template (explicitly commented "LSTM placeholder -- update
#     before running"), not executed evidence, and the key name it used was
#     wrong -- caught only when job 45639408's real training run crashed
#     immediately with `ValueError: ['dropout'] are not recognized config
#     keys.` See docs/decision_log.md.
#   - epochs ("max 30-50, with early stopping"): resolved to the range
#     midpoint, 40. NH 1.13 has no confirmed native early-stopping/patience
#     config key anywhere in this repo's own prior source inspection; this
#     project's established convention (already required independently by
#     this task's own checkpoint-selection step) is instead to train a fixed
#     epoch count with per-epoch checkpointing (save_weights_every=1,
#     validate_every=1) and select the best epoch post-hoc from validation
#     metrics -- i.e. Sec 9d's "Validation raw-space NSE" selection rule IS
#     the early-stopping mechanism for this project, not a live callback.
#   - loss: Sec 7 leaves the training loss "likely an NSE-family loss"
#     (target-scaling-dependent, not yet pinned to one formula); resolved to
#     NH's built-in "NSE" loss, consistent with that steer and with this
#     project's own prior Smoke 1 run (job 45370873, loss: NSE).
#   - validate_n_random_basins: set to the full development population size
#     (not a subsample), mirroring the compact profile's own convention of
#     covering every available basin every epoch -- required so that
#     per-epoch validation coverage is identical across epochs, which the
#     post-hoc checkpoint-selection step (Part E) depends on for a fair
#     cross-epoch comparison.
#   - num_workers / verbose: operational-only settings (GPU data-loading
#     parallelism, log verbosity), not scientific hyperparameters; chosen for
#     a real multi-hour GPU run rather than the compact profile's CPU-safe
#     single-process/quiet smoke-run defaults.
# See docs/decision_log.md for the recorded decision entry.
_INITIAL_SEED_TRAINING_PROFILE = {
    "model": "cudalstm",
    "hidden_size": 128,
    "output_dropout": 0.25,
    "batch_size": 256,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "NSE",
    "save_weights_every": 1,
    "validate_every": 1,
    "validate_n_random_basins": EXPECTED_DEVELOPMENT_BASIN_COUNT,
    "log_interval": 50,
    "num_workers": 4,
    "epochs": 40,
    "device": "cuda:0",
    "verbose": 1,
}

# Embedded-static CudaLSTM PILOT profile (Part I, section 13). Per Part B's
# static-pathway audit (reports/stage1_validation_optimization_foundation_v001/
# part_b_static_pathway_audit/static_pathway_audit.md), the seed run's
# `statics_embedding` config key was absent, which NH 1.13 resolves to
# `nn.Identity()` -- raw concatenation, no learned static representation.
# Section 2.5 permits a first embedded-static candidate only because that
# audit confirmed the seed did NOT already use an equivalent learned
# representation; this profile is that candidate.
#
# This is a DESIGN/CONFIG + STRUCTURAL-SMOKE-ONLY profile, not a scientific
# candidate to be trained in this phase (section 13's explicit scope). It is
# therefore built on top of the small, CPU-safe COMPACT_SMOKE_RUN_PROFILE
# (32-basin compact package, 2 epochs) rather than the full-population seed
# profile -- the only change relative to that compact-smoke baseline is the
# addition of `statics_embedding`, isolating exactly the one architectural
# axis this candidate exists to exercise. hiddens=[128, 64] and dropout=0.1
# are a reasonable, unremarkable first FC-embedding shape (final width 64
# comparable in order of magnitude to the compact profile's own hidden_size),
# not a tuned choice -- no training is run against this profile in this
# phase, so there is nothing to tune yet.
_EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE = {
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
    "statics_embedding": {
        "type": "fc",
        "hiddens": [128, 64],
        "activation": "tanh",
        "dropout": 0.1,
    },
}

# Stage 1 lead-6 optimization pilot (config/stage1_lead06_pilot_v001.yaml):
# exactly 6 named profiles, one per agreed run_id. All 6 share the initial
# full-population seed profile's scientific hyperparameters (hidden_size 128,
# output_dropout 0.25, batch_size 256, Adam lr 0.001, NSE loss, num_workers
# 4) -- none of those are varied by this pilot. Each profile differs only in:
#   - seed (967139 = "seed A", the historical seed run's own NH-assigned
#     value, recovered read-only from its frozen run_dir/config.yml; 1729 =
#     "seed B")
#   - static pathway: raw_* profiles have no statics_embedding key (NH 1.13
#     resolves that to nn.Identity(), i.e. raw concatenation); emb* profiles
#     add a learned FC statics_embedding of the agreed shape, with
#     activation/dropout held fixed (tanh/0.1) across all three shapes, per
#     section 2.5.
#   - epochs=6: the FIRST bounded training chunk only (this pilot's
#     screening cadence validates every 3 epochs, diagnostic-only at epoch 3,
#     stopping-eligible from epoch 6 -- see
#     config/stage1_lead06_pilot_v001.yaml's early_stopping block). The pilot
#     orchestrator (src/baseline/pilot_orchestration.py) extends training
#     past epoch 6 via NH's own `continue_run` + a raised `epochs` overlay,
#     one bounded chunk at a time, up to the pilot's 36-epoch sub-cap -- it
#     never edits these frozen profiles to "restart" at a larger epoch count.
#   - validate_n_random_basins=1000: larger than the screening-subset
#     policy's max allowed count (500, config/stage1_screening_subset_v001.yaml),
#     so NH validates every basin in the (already deliberately-scoped)
#     validation basin file each screening epoch rather than subsampling it
#     further.
# See docs/stage1_lead06_pilot_v001.md.
_PILOT_LEAD06_BASE_PROFILE = {
    "model": "cudalstm",
    "hidden_size": 128,
    "output_dropout": 0.25,
    "batch_size": 256,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "NSE",
    "save_weights_every": 1,
    "validate_every": 3,
    "validate_n_random_basins": 1000,
    "log_interval": 50,
    "num_workers": 4,
    "epochs": 6,
    "device": "cuda:0",
    "verbose": 1,
}

_PILOT_LEAD06_EMBEDDING_SHAPES = {
    "emb128x64": [128, 64],
    "emb64": [64],
    "emb128": [128],
}


def _pilot_lead06_profile(*, seed: int, embedding_hiddens: "list | None") -> dict:
    profile = dict(_PILOT_LEAD06_BASE_PROFILE)
    profile["seed"] = seed
    if embedding_hiddens is not None:
        profile["statics_embedding"] = {
            "type": "fc",
            "hiddens": list(embedding_hiddens),
            "activation": "tanh",
            "dropout": 0.1,
        }
    return profile


PILOT_LEAD06_RAW_SEEDA_PROFILE_NAME = "pilot_lead06_raw_seedA_v001"
PILOT_LEAD06_RAW_SEEDB_PROFILE_NAME = "pilot_lead06_raw_seedB_v001"
PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME = "pilot_lead06_emb128x64_seedA_v001"
PILOT_LEAD06_EMB128X64_SEEDB_PROFILE_NAME = "pilot_lead06_emb128x64_seedB_v001"
PILOT_LEAD06_EMB64_SEEDA_PROFILE_NAME = "pilot_lead06_emb64_seedA_v001"
PILOT_LEAD06_EMB128_SEEDA_PROFILE_NAME = "pilot_lead06_emb128_seedA_v001"

_SEED_A = 967139
_SEED_B = 1729

_PILOT_LEAD06_RAW_SEEDA_PROFILE = _pilot_lead06_profile(seed=_SEED_A, embedding_hiddens=None)
_PILOT_LEAD06_RAW_SEEDB_PROFILE = _pilot_lead06_profile(seed=_SEED_B, embedding_hiddens=None)
_PILOT_LEAD06_EMB128X64_SEEDA_PROFILE = _pilot_lead06_profile(seed=_SEED_A, embedding_hiddens=[128, 64])
_PILOT_LEAD06_EMB128X64_SEEDB_PROFILE = _pilot_lead06_profile(seed=_SEED_B, embedding_hiddens=[128, 64])
_PILOT_LEAD06_EMB64_SEEDA_PROFILE = _pilot_lead06_profile(seed=_SEED_A, embedding_hiddens=[64])
_PILOT_LEAD06_EMB128_SEEDA_PROFILE = _pilot_lead06_profile(seed=_SEED_A, embedding_hiddens=[128])

PILOT_LEAD06_EMB64X32_SEEDA_PROFILE_NAME = "pilot_lead06_emb64x32_seedA_v001"
PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME = "pilot_lead06_emb128x32_seedA_v001"
PILOT_LEAD06_EMB256X64_SEEDA_PROFILE_NAME = "pilot_lead06_emb256x64_seedA_v001"

_PILOT_LEAD06_EMB64X32_SEEDA_PROFILE = _pilot_lead06_profile(seed=_SEED_A, embedding_hiddens=[64, 32])
_PILOT_LEAD06_EMB128X32_SEEDA_PROFILE = _pilot_lead06_profile(seed=_SEED_A, embedding_hiddens=[128, 32])
_PILOT_LEAD06_EMB256X64_SEEDA_PROFILE = _pilot_lead06_profile(seed=_SEED_A, embedding_hiddens=[256, 64])

# Canonical run-profile registry: name -> (hyperparameter mapping, manifest
# note). New profiles must be added here, never spliced in ad hoc, so every
# generated manifest can unambiguously record which named profile it used.
COMPACT_SMOKE_RUN_PROFILE_NAME = "compact_smoke_v1"
INITIAL_SEED_RUN_PROFILE_NAME = "initial_full_population_seed_v001"
EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME = "embedded_static_cudalstm_pilot"

_RUN_PROFILES = {
    COMPACT_SMOKE_RUN_PROFILE_NAME: _COMPACT_SMOKE_RUN_PROFILE,
    INITIAL_SEED_RUN_PROFILE_NAME: _INITIAL_SEED_TRAINING_PROFILE,
    EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME: _EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE,
    PILOT_LEAD06_RAW_SEEDA_PROFILE_NAME: _PILOT_LEAD06_RAW_SEEDA_PROFILE,
    PILOT_LEAD06_RAW_SEEDB_PROFILE_NAME: _PILOT_LEAD06_RAW_SEEDB_PROFILE,
    PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME: _PILOT_LEAD06_EMB128X64_SEEDA_PROFILE,
    PILOT_LEAD06_EMB128X64_SEEDB_PROFILE_NAME: _PILOT_LEAD06_EMB128X64_SEEDB_PROFILE,
    PILOT_LEAD06_EMB64_SEEDA_PROFILE_NAME: _PILOT_LEAD06_EMB64_SEEDA_PROFILE,
    PILOT_LEAD06_EMB128_SEEDA_PROFILE_NAME: _PILOT_LEAD06_EMB128_SEEDA_PROFILE,
    PILOT_LEAD06_EMB64X32_SEEDA_PROFILE_NAME: _PILOT_LEAD06_EMB64X32_SEEDA_PROFILE,
    PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME: _PILOT_LEAD06_EMB128X32_SEEDA_PROFILE,
    PILOT_LEAD06_EMB256X64_SEEDA_PROFILE_NAME: _PILOT_LEAD06_EMB256X64_SEEDA_PROFILE,
}

# run_id (config/stage1_lead06_pilot_v001.yaml) -> run_profile_name, so
# src/baseline/pilot_lead06_config.py never hardcodes this mapping itself.
PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME = {
    "raw_seedA": PILOT_LEAD06_RAW_SEEDA_PROFILE_NAME,
    "raw_seedB": PILOT_LEAD06_RAW_SEEDB_PROFILE_NAME,
    "emb128x64_seedA": PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME,
    "emb128x64_seedB": PILOT_LEAD06_EMB128X64_SEEDB_PROFILE_NAME,
    "emb64_seedA": PILOT_LEAD06_EMB64_SEEDA_PROFILE_NAME,
    "emb128_seedA": PILOT_LEAD06_EMB128_SEEDA_PROFILE_NAME,
}

# Sorted, public list of known run-profile names, for CLI --help/choices use.
KNOWN_RUN_PROFILE_NAMES = tuple(sorted(_RUN_PROFILES))


def get_run_profile_mapping(run_profile_name: str) -> dict:
    """Public read-only accessor for one named entry of the internal
    ``_RUN_PROFILES`` registry, so other modules (e.g.
    ``pilot_lead06_config.py``'s policy/profile semantic cross-check) never
    need to import the private ``_RUN_PROFILES`` name directly."""
    if run_profile_name not in _RUN_PROFILES:
        raise NHConfigGenerationError(f"unknown run_profile_name: {run_profile_name!r}")
    return dict(_RUN_PROFILES[run_profile_name])

_RUN_PROFILE_NOTES = {
    COMPACT_SMOKE_RUN_PROFILE_NAME: (
        "model/hidden_size/optimizer/loss/epochs/etc. are the frozen "
        "compact-smoke-run technical settings for the first lead06/seq24 "
        "integration-validation training run; they are NOT the scientific "
        "baseline or a hyperparameter-tuning seed."
    ),
    INITIAL_SEED_RUN_PROFILE_NAME: (
        "model/hidden_size/output_dropout/optimizer/loss/epochs/etc. are the initial "
        "full-population seed run / not tuned / not the official Stage 1 "
        "benchmark. Sourced from docs/stage1_scientific_baseline_design.md "
        "Sec 9c; this is the first real full-population training run, not the "
        "result of a hyperparameter sweep."
    ),
    EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME: (
        "First embedded-static CudaLSTM candidate (section 13, Part I): adds a "
        "learned `statics_embedding` FC network on top of the compact-smoke "
        "technical settings. DESIGN/CONFIG + STRUCTURAL-SMOKE-ONLY -- not "
        "trained in this phase, not a tuned candidate, not a claim that this "
        "embedding shape is optimal."
    ),
    PILOT_LEAD06_RAW_SEEDA_PROFILE_NAME: (
        "Stage 1 lead-6 optimization pilot, config/stage1_lead06_pilot_v001.yaml "
        "run_id=raw_seedA: raw (nn.Identity()) static concatenation, "
        "seed=967139 ('seed A', the historical full-population seed run's own "
        "NH-assigned value). One of 2 raw-pathway controls, not itself a "
        "learned-embedding candidate."
    ),
    PILOT_LEAD06_RAW_SEEDB_PROFILE_NAME: (
        "Stage 1 lead-6 optimization pilot run_id=raw_seedB: raw static "
        "concatenation, seed=1729 ('seed B'). Second raw-pathway control, for "
        "seed-sensitivity comparison against raw_seedA."
    ),
    PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME: (
        "Stage 1 lead-6 optimization pilot run_id=emb128x64_seedA: learned FC "
        "statics_embedding hiddens=[128, 64], seed=967139 ('seed A'). This is "
        "the pilot's workflow-qualification run (config/"
        "stage1_lead06_pilot_v001.yaml's workflow_qualification_run_id) -- "
        "prepared, not launched, by this implementation increment."
    ),
    PILOT_LEAD06_EMB128X64_SEEDB_PROFILE_NAME: (
        "Stage 1 lead-6 optimization pilot run_id=emb128x64_seedB: learned FC "
        "statics_embedding hiddens=[128, 64], seed=1729 ('seed B'). Matched-"
        "seed pair with emb128x64_seedA for embedding-vs-seed variance "
        "comparison."
    ),
    PILOT_LEAD06_EMB64_SEEDA_PROFILE_NAME: (
        "Stage 1 lead-6 optimization pilot run_id=emb64_seedA: learned FC "
        "statics_embedding hiddens=[64] (single narrower layer), seed=967139 "
        "('seed A'). Embedding-width ablation against emb128x64_seedA."
    ),
    PILOT_LEAD06_EMB128_SEEDA_PROFILE_NAME: (
        "Stage 1 lead-6 optimization pilot run_id=emb128_seedA: learned FC "
        "statics_embedding hiddens=[128] (single wider layer), seed=967139 "
        "('seed A'). Embedding-depth ablation against emb128x64_seedA."
    ),
    PILOT_LEAD06_EMB64X32_SEEDA_PROFILE_NAME: (
        "Stage 1 lead-6 embedding-shape neighborhood: learned FC "
        "statics_embedding hiddens=[64, 32] (two-layer neighbor of "
        "emb128x64_seedA), seed=967139 ('seed A'). Canonical profile for the "
        "next approved 25k embedding-shape batch; not a config/"
        "stage1_lead06_pilot_v001.yaml run_id and not yet trained."
    ),
    PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME: (
        "Stage 1 lead-6 embedding-shape neighborhood: learned FC "
        "statics_embedding hiddens=[128, 32] (two-layer neighbor of "
        "emb128x64_seedA), seed=967139 ('seed A'). Canonical profile for the "
        "next approved 25k embedding-shape batch; not a config/"
        "stage1_lead06_pilot_v001.yaml run_id and not yet trained."
    ),
    PILOT_LEAD06_EMB256X64_SEEDA_PROFILE_NAME: (
        "Stage 1 lead-6 embedding-shape neighborhood: learned FC "
        "statics_embedding hiddens=[256, 64] (two-layer neighbor of "
        "emb128x64_seedA), seed=967139 ('seed A'). Canonical profile for the "
        "next approved 25k embedding-shape batch; not a config/"
        "stage1_lead06_pilot_v001.yaml run_id and not yet trained."
    ),
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
    # Population-role fields (task: full-population dev/spatial-holdout
    # separation). All default to None/"identical_all_periods", which
    # preserves the original single-population behavior byte-for-byte:
    # write_generated_config() falls back to basin_ids for every period when
    # these are unset. Only generate_stage1_full_population_nh_config_bundles
    # sets them explicitly.
    population_role: str = "identical_all_periods"
    train_basin_ids: "list | None" = None
    validation_basin_ids: "list | None" = None
    test_basin_ids: "list | None" = None
    # Defaults to the pre-existing compact-smoke profile, which preserves
    # every previous caller's behavior byte-for-byte. Only a caller that
    # explicitly asks for a different named profile (see _RUN_PROFILES)
    # gets something else.
    run_profile_name: str = COMPACT_SMOKE_RUN_PROFILE_NAME
    # Optional per-epoch NH training-batch cap (see NHConfigGenerationError
    # validate_max_updates_per_epoch and docs/stage1_validation_optimization_
    # foundation.md Part L.5/L.6): None (the default) means uncapped/full
    # fidelity, byte-identical to every pre-existing caller. A positive int
    # is this candidate's frozen fidelity-screening cap -- never mutated
    # after a bundle is built, and never adopted/compared loosely (capped vs
    # uncapped, or two different int caps, are distinct identities; see
    # pilot_orchestration.enforce_pilot_cap_identity).
    max_updates_per_epoch: "int | None" = None
    # Optional per-candidate learning-rate override (LR-A range-
    # characterization campaign; see docs/decision_log.md and
    # pilot_lead06_config.PilotRunSpec.learning_rate). None (the default)
    # means "use whatever the named run_profile already specifies" --
    # byte-identical to every pre-existing caller. A float override is this
    # candidate's frozen learning-rate identity for its entire lifetime;
    # never mutated after a bundle is built, and never adopted/compared
    # loosely against a differently-resolved value for the same run identity
    # (see pilot_orchestration's LR resume-contradiction guard).
    learning_rate: "float | None" = None
    # Optional per-candidate LSTM hidden-size override (Hidden-size-A range-
    # characterization campaign; see docs/decision_log.md and
    # pilot_lead06_config.PilotRunSpec.hidden_size). None (the default) means
    # "use whatever the named run_profile already specifies" -- byte-
    # identical to every pre-existing caller. An int override is this
    # candidate's frozen hidden-size identity for its entire lifetime; never
    # mutated after a bundle is built, and never adopted/compared loosely
    # against a differently-resolved value for the same run identity (see
    # pilot_orchestration.enforce_pilot_hidden_size_identity).
    hidden_size: "int | None" = None
    # Optional per-candidate statics_embedding dropout override (Embedding-
    # Dropout-A range-characterization campaign; see docs/decision_log.md
    # and pilot_lead06_config.PilotRunSpec.embedding_dropout). None (the
    # default) means "use whatever the named run_profile already
    # specifies" -- byte-identical to every pre-existing caller. A float
    # override (including 0.0, always checked with "is not None", never
    # truthiness) is this candidate's frozen embedding-dropout identity for
    # its entire lifetime; never mutated after a bundle is built, and never
    # adopted/compared loosely against a differently-resolved value for the
    # same run identity (see pilot_orchestration.
    # enforce_pilot_embedding_dropout_identity).
    embedding_dropout: "float | None" = None


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


@dataclass(frozen=True)
class FullPopulationBasinMembership:
    """Partition of a full-population (dev + spatial-holdout) package's basin
    IDs into their two strictly separated scientific roles. Both lists are
    sorted, normalized STAIDs; neither ever contains a California basin."""

    development_basins: list
    spatial_holdout_basins: list


def validate_full_population_basin_membership(package_manifest: dict, splits_dir) -> FullPopulationBasinMembership:
    """Partition a full non-CA population package's basins into the
    ``development_train`` (2,307) and ``spatial_holdout_nonca`` (250) roles,
    failing loudly on any deviation from the certified Stage 1 contract:
    duplicate/malformed IDs, dev/holdout overlap (should be structurally
    impossible from the canonical split files, but re-checked here rather
    than assumed), any California basin present, a package basin that is
    neither a dev nor a holdout member, a dev/holdout member missing from the
    package, or either partition not being exactly the pinned expected size.

    Unlike :func:`validate_basin_membership` (which requires every package
    basin to be a ``development_train`` member -- the compact 32-basin
    package's contract), this function is for the certified full
    non-California population package
    (``stage1_scientific_package_v002``, 2,557 basins == 2,307 + 250) and
    returns both roles instead of a single combined list, so callers can keep
    them in strictly separate config bundles.
    """
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
    package_basin_set = set(package_basins)

    try:
        development_train = set(load_eligible_basins(splits_dir / "development_train.txt"))
        spatial_holdout = set(load_eligible_basins(splits_dir / "spatial_holdout_nonca.txt"))
        california_all = set(load_eligible_basins(splits_dir / "california_all.txt"))
    except SplitGenerationError as exc:
        raise NHConfigGenerationError(f"could not read canonical split file(s): {exc}") from exc

    dev_holdout_overlap = sorted(development_train & spatial_holdout)
    if dev_holdout_overlap:
        raise NHConfigGenerationError(
            f"canonical development_train and spatial_holdout_nonca splits overlap: {dev_holdout_overlap}"
        )

    california_overlap = sorted(package_basin_set & california_all)
    if california_overlap:
        raise NHConfigGenerationError(
            f"package basin(s) include California basin(s), forbidden for the non-CA full population: "
            f"{california_overlap}"
        )

    expected_union = development_train | spatial_holdout
    missing_from_package = sorted(expected_union - package_basin_set)
    extra_in_package = sorted(package_basin_set - expected_union)
    if missing_from_package or extra_in_package:
        raise NHConfigGenerationError(
            "package basin_ids do not exactly equal the union of the canonical development_train and "
            f"spatial_holdout_nonca splits: missing={missing_from_package} extra={extra_in_package}"
        )

    development_basins = sorted(package_basin_set & development_train)
    spatial_holdout_basins = sorted(package_basin_set & spatial_holdout)

    if len(development_basins) != EXPECTED_DEVELOPMENT_BASIN_COUNT:
        raise NHConfigGenerationError(
            f"expected exactly {EXPECTED_DEVELOPMENT_BASIN_COUNT} development basins, "
            f"got {len(development_basins)}"
        )
    if len(spatial_holdout_basins) != EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT:
        raise NHConfigGenerationError(
            f"expected exactly {EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT} spatial-holdout basins, "
            f"got {len(spatial_holdout_basins)}"
        )

    return FullPopulationBasinMembership(
        development_basins=development_basins,
        spatial_holdout_basins=spatial_holdout_basins,
    )


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
    run_profile_name: str = COMPACT_SMOKE_RUN_PROFILE_NAME,
    max_updates_per_epoch: "int | None" = None,
    learning_rate: "float | None" = None,
    hidden_size: "int | None" = None,
    embedding_dropout: "float | None" = None,
) -> dict:
    """Pure function: assemble the policy/target/structural fields of the
    rendered config. Does not include experiment_name, basin-file paths,
    data_dir, or run_dir -- those depend on the concrete output directory and
    are filled in by :func:`write_generated_config`.

    ``run_profile_name`` selects which named training-hyperparameter profile
    (see ``_RUN_PROFILES``) is merged in; it defaults to the compact-smoke
    profile so every pre-existing caller is unaffected.

    ``max_updates_per_epoch`` (default ``None``, uncapped) is this candidate's
    optional per-epoch NH training-batch cap (see
    :func:`validate_max_updates_per_epoch` and docs/
    stage1_validation_optimization_foundation.md Part L.5/L.6). When ``None``
    the key is omitted from the returned mapping entirely -- not written as
    ``null`` -- so every pre-existing caller's generated ``config.yaml`` is
    byte-for-byte unchanged. This is never part of a named ``_RUN_PROFILES``
    entry: profiles are shared scientific-architecture identities reused
    across candidates, while a fidelity cap is a per-candidate screening
    concern layered on top.

    ``learning_rate`` (default ``None``) is this candidate's optional
    per-candidate learning-rate override (LR-A range-characterization
    campaign; see :func:`validate_learning_rate_override`). When ``None`` the
    profile's own ``learning_rate`` entry (merged in below) is left
    untouched, so every pre-existing caller is byte-for-byte unaffected. When
    given, it is applied AFTER the profile merge so it always wins -- a
    learning-rate override is a per-candidate identity, never part of a
    shared named profile.

    ``hidden_size`` (default ``None``) is this candidate's optional
    per-candidate LSTM hidden-size override (Hidden-size-A range-
    characterization campaign; see :func:`validate_hidden_size_override`).
    When ``None`` the profile's own ``hidden_size`` entry (merged in below)
    is left untouched, so every pre-existing caller is byte-for-byte
    unaffected. When given, it is applied AFTER the profile merge so it
    always wins -- the same always-wins pattern as ``learning_rate``.

    ``embedding_dropout`` (default ``None``) is this candidate's optional
    per-candidate ``statics_embedding.dropout`` override (Embedding-Dropout-A
    range-characterization campaign; see
    :func:`validate_embedding_dropout_override`). When ``None`` the profile's
    own ``statics_embedding.dropout`` entry (merged in below) is left
    untouched, so every pre-existing caller is byte-for-byte unaffected. When
    given (including ``0.0``, always checked with ``is not None``, never
    truthiness), it is applied AFTER the profile merge so it always wins --
    the same always-wins pattern as ``learning_rate``/``hidden_size``. A
    profile with no ``statics_embedding`` section (the raw-static pathway)
    has nothing to override, so an explicit override against such a profile
    is rejected rather than silently ignored or silently creating a new
    ``statics_embedding`` section."""
    if run_profile_name not in _RUN_PROFILES:
        raise NHConfigGenerationError(
            f"unknown run_profile_name {run_profile_name!r}; known profiles: {sorted(_RUN_PROFILES)}"
        )
    if max_updates_per_epoch is not None:
        validate_max_updates_per_epoch(max_updates_per_epoch)
    if learning_rate is not None:
        validate_learning_rate_override(learning_rate)
    if hidden_size is not None:
        validate_hidden_size_override(hidden_size)
    if embedding_dropout is not None:
        validate_embedding_dropout_override(embedding_dropout)
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
    mapping.update(_RUN_PROFILES[run_profile_name])
    # nan_handling_method deliberately absent: hard-exclusion baseline
    # (accepted finding #7); never set as a defensive backstop here.
    if "statics_embedding" in mapping:
        if embedding_dropout is not None:
            # Copy-before-mutate: mapping["statics_embedding"] is still the
            # exact same dict object as _RUN_PROFILES[run_profile_name]
            # ["statics_embedding"] after dict.update() above (a shallow
            # copy) -- mutating it in place would permanently corrupt the
            # shared module-level profile registry for every future caller
            # that reuses this run_profile_name. Copy first so this
            # candidate's override is local to this one mapping.
            mapping["statics_embedding"] = dict(mapping["statics_embedding"])
            mapping["statics_embedding"]["dropout"] = embedding_dropout
        validate_statics_embedding_spec(mapping["statics_embedding"])
    elif embedding_dropout is not None:
        raise NHConfigGenerationError(
            f"embedding_dropout override given ({embedding_dropout!r}) but run_profile_name "
            f"{run_profile_name!r} has no statics_embedding section (raw-static pathway) to override"
        )
    if max_updates_per_epoch is not None:
        mapping["max_updates_per_epoch"] = max_updates_per_epoch
    if learning_rate is not None:
        mapping["learning_rate"] = learning_rate
    if hidden_size is not None:
        mapping["hidden_size"] = hidden_size
    return mapping


def validate_max_updates_per_epoch(value) -> None:
    """Reject anything but a positive Python int (bools rejected even though
    ``bool`` is an ``int`` subclass, per this codebase's established
    positive-int idiom -- see :func:`validate_statics_embedding_spec`'s
    ``hiddens`` check). ``None`` (uncapped) is never passed to this
    function -- callers only call it once they already know a cap was
    requested; see :func:`build_nh_config_mapping`."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise NHConfigGenerationError(
            f"max_updates_per_epoch must be a positive int or None (uncapped), got {value!r}"
        )


def validate_learning_rate_override(value) -> None:
    """Reject anything but a positive, finite real number (bools rejected
    even though ``bool`` is an ``int`` subclass, per this codebase's
    established positive-value idiom -- see
    :func:`validate_max_updates_per_epoch`). ``None`` (no override, use the
    profile's own value) is never passed to this function -- callers only
    call it once they already know an override was requested; see
    :func:`build_nh_config_mapping`. This intentionally does not enforce any
    broad optimizer-specific bounds -- the frozen LR-A closed matrix (see
    docs/decision_log.md) defines the currently approved candidate values;
    this validator only rejects structurally-invalid input (non-numeric,
    boolean, zero, negative, NaN/inf)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise NHConfigGenerationError(
            f"learning_rate override must be a positive finite number or None, got {value!r}"
        )
    if not math.isfinite(value) or value <= 0:
        raise NHConfigGenerationError(
            f"learning_rate override must be a positive finite number or None, got {value!r}"
        )


def validate_hidden_size_override(value) -> None:
    """Reject anything but a positive Python int (bools rejected even though
    ``bool`` is an ``int`` subclass, per this codebase's established
    positive-int idiom -- see :func:`validate_max_updates_per_epoch`).
    ``None`` (no override, use the profile's own value) is never passed to
    this function -- callers only call it once they already know an override
    was requested; see :func:`build_nh_config_mapping`. This intentionally
    does not enforce any campaign-specific allowlist (e.g. Hidden-size-A's
    ``{64,128,256,512}``) -- that closed set is enforced by the campaign
    launcher, not this general-purpose structural validator."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise NHConfigGenerationError(
            f"hidden_size override must be a positive int or None, got {value!r}"
        )


def validate_embedding_dropout_override(value) -> None:
    """Reject anything but a finite real number in ``[0, 1)`` (bools rejected
    even though ``bool`` is an ``int`` subclass, per this codebase's
    established numeric-override idiom -- see
    :func:`validate_learning_rate_override`). ``0.0`` is a valid, distinct-
    from-``None`` value (the Embedding-Dropout-A ``drop00`` candidate) --
    this validator does not treat it as falsy/absent. ``None`` (no override,
    use the profile's own value) is never passed to this function -- callers
    only call it once they already know an override was requested; see
    :func:`build_nh_config_mapping`. The bound matches
    :func:`validate_statics_embedding_spec`'s own ``dropout`` range check, so
    an override that passes here is guaranteed to also pass that structural
    check once merged in. This intentionally does not enforce any campaign-
    specific allowlist (e.g. Embedding-Dropout-A's ``{0.0, 0.05, 0.1, 0.2,
    0.4}``) -- that closed set is enforced by the campaign launcher, not this
    general-purpose structural validator."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise NHConfigGenerationError(
            f"embedding_dropout override must be a real number in [0, 1) or None, got {value!r}"
        )
    if not math.isfinite(value) or not (0 <= value < 1):
        raise NHConfigGenerationError(
            f"embedding_dropout override must be in [0, 1), got {value!r}"
        )


_ALLOWED_STATICS_EMBEDDING_ACTIVATIONS = ("tanh", "sigmoid", "linear")


def validate_statics_embedding_spec(spec: dict) -> None:
    """Structurally validate a ``statics_embedding`` config spec (section 13,
    Part I). Pure Python only -- no torch/NeuralHydrology import, so this
    check can run at config-generation time, before any training dependency
    is even needed.

    Per Part B's (section 6) audit of ``neuralhydrology/modelzoo/inputlayer.py``,
    NH 1.13's FC embedding spec requires exactly: ``type: "fc"``, a non-empty
    list of positive-int ``hiddens``, an ``activation`` from NH's supported
    set, and a ``dropout`` in ``[0, 1)``."""
    if not isinstance(spec, dict):
        raise NHConfigGenerationError(f"statics_embedding must be a mapping, got {type(spec).__name__}")

    if spec.get("type") != "fc":
        raise NHConfigGenerationError(f"statics_embedding.type must be 'fc', got {spec.get('type')!r}")

    hiddens = spec.get("hiddens")
    if not isinstance(hiddens, list) or not hiddens:
        raise NHConfigGenerationError(f"statics_embedding.hiddens must be a non-empty list, got {hiddens!r}")
    if not all(isinstance(h, int) and not isinstance(h, bool) and h > 0 for h in hiddens):
        raise NHConfigGenerationError(f"statics_embedding.hiddens must all be positive ints, got {hiddens!r}")

    activation = spec.get("activation")
    if activation not in _ALLOWED_STATICS_EMBEDDING_ACTIVATIONS:
        raise NHConfigGenerationError(
            f"statics_embedding.activation must be one of {_ALLOWED_STATICS_EMBEDDING_ACTIVATIONS}, got {activation!r}"
        )

    dropout = spec.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not (0 <= dropout < 1):
        raise NHConfigGenerationError(f"statics_embedding.dropout must be in [0, 1), got {dropout!r}")


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
    run_profile_name: str = COMPACT_SMOKE_RUN_PROFILE_NAME,
    max_updates_per_epoch: "int | None" = None,
    learning_rate: "float | None" = None,
    hidden_size: "int | None" = None,
    embedding_dropout: "float | None" = None,
) -> GeneratedConfigBundle:
    policy_path = Path(policy_path)
    package_root = Path(package_root)
    splits_dir = Path(splits_dir)

    if run_profile_name not in _RUN_PROFILES:
        raise NHConfigGenerationError(
            f"unknown run_profile_name {run_profile_name!r}; known profiles: {sorted(_RUN_PROFILES)}"
        )

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
        run_profile_name=run_profile_name,
        max_updates_per_epoch=max_updates_per_epoch,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        embedding_dropout=embedding_dropout,
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
        run_profile_name=run_profile_name,
        max_updates_per_epoch=max_updates_per_epoch,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        embedding_dropout=embedding_dropout,
    )


@dataclass(frozen=True)
class FullPopulationConfigBundles:
    """The two strictly-separated config bundles for one full-population
    (lead, seq_length) configuration: a development bundle (train/validation/
    temporal-test all drawn from the 2,307 development basins, differing only
    by date period) and a spatial-holdout bundle (test-only, drawn from the
    250 spatial-holdout basins, whose train/validation basin lists are the
    *development* bundle's -- so a holdout basin can never appear in a
    training or validation basin list even if this bundle's config is
    misused directly)."""

    development: GeneratedConfigBundle
    spatial_holdout: GeneratedConfigBundle
    basin_membership: FullPopulationBasinMembership


def generate_stage1_full_population_nh_config_bundles(
    *,
    policy_path,
    package_root,
    splits_dir,
    lead_hours: int,
    seq_length: int,
    static_column_manifest_path=None,
    run_profile_name: str = COMPACT_SMOKE_RUN_PROFILE_NAME,
    max_updates_per_epoch: "int | None" = None,
) -> FullPopulationConfigBundles:
    """Generate the development + spatial-holdout config bundle pair for one
    approved (lead, seq_length) combination against the certified full
    non-CA population package (``stage1_scientific_package_v002``, 2,557
    basins).

    Reuses every validation/mapping function from
    :func:`generate_stage1_nh_config` unchanged (policy loading, seq_length/
    lead_hours/target-variable/dynamic-input/static-attribute-contract
    validation, :func:`build_nh_config_mapping`) -- the only new logic is
    :func:`validate_full_population_basin_membership`'s dev/spatial-holdout
    partition (replacing :func:`validate_basin_membership`'s single-population
    check) and assembling two bundles instead of one from that partition.

    The spatial-holdout bundle's ``basin_ids`` is the 250-basin holdout
    population (its own scientific contract); its ``test_basin_ids`` is the
    same 250 basins, while ``train_basin_ids``/``validation_basin_ids`` are
    the *development* bundle's 2,307 basins -- so this bundle's own generated
    ``train_basins.txt``/``validation_basins.txt`` never contain a
    spatial-holdout basin. Evaluating the spatial-holdout bundle must reuse
    the development bundle's fitted training scaler (never fit a new one);
    see ``src.baseline.nh_structural_preflight.check_flashnh_external_scaler_test_construction``
    for the corresponding dataset-construction-time safeguard.
    """
    policy_path = Path(policy_path)
    package_root = Path(package_root)
    splits_dir = Path(splits_dir)

    if run_profile_name not in _RUN_PROFILES:
        raise NHConfigGenerationError(
            f"unknown run_profile_name {run_profile_name!r}; known profiles: {sorted(_RUN_PROFILES)}"
        )

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

    basin_membership = validate_full_population_basin_membership(package_manifest, splits_dir)

    config_mapping = build_nh_config_mapping(
        policy=policy,
        target_variable=target_variable,
        seq_length=seq_length,
        dynamic_inputs=dynamic_inputs,
        static_attributes=static_result.columns,
        run_profile_name=run_profile_name,
        max_updates_per_epoch=max_updates_per_epoch,
    )

    package_manifest_identity = {
        "schema_name": package_manifest.get("schema_name"),
        "schema_version": package_manifest.get("schema_version"),
        "package_role": package_manifest.get("package_role"),
        "basin_count": package_manifest.get("basin_count"),
        "static_model_input_columns_sha256": package_manifest.get("static_model_input_columns_sha256"),
    }

    generated_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_commit = _get_git_commit()

    common_kwargs = dict(
        config_mapping=config_mapping,
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
        generated_at_utc=generated_at_utc,
        git_commit=git_commit,
        run_profile_name=run_profile_name,
        max_updates_per_epoch=max_updates_per_epoch,
    )

    development_bundle = GeneratedConfigBundle(
        basin_ids=basin_membership.development_basins,
        package_type="full_population_development",
        population_role="development_identical_all_periods",
        **common_kwargs,
    )
    spatial_holdout_bundle = GeneratedConfigBundle(
        basin_ids=basin_membership.spatial_holdout_basins,
        package_type="full_population_spatial_holdout_test_only",
        population_role="spatial_holdout_test_only",
        train_basin_ids=basin_membership.development_basins,
        validation_basin_ids=basin_membership.development_basins,
        test_basin_ids=basin_membership.spatial_holdout_basins,
        **common_kwargs,
    )

    return FullPopulationConfigBundles(
        development=development_bundle,
        spatial_holdout=spatial_holdout_bundle,
        basin_membership=basin_membership,
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
    if experiment_name is None and bundle.population_role == "spatial_holdout_test_only":
        # Deliberately distinct from the development bundle's default name so the
        # two bundles never collide and a holdout config is not mistaken for an
        # independently trainable experiment (review: holdout bundle safety).
        exp_name = f"{exp_name}_spatial_holdout_test_only_eval"

    # Per-period basin lists default to bundle.basin_ids (preserves the
    # original single-population train==validation==test behavior exactly);
    # a full-population spatial-holdout bundle overrides train/validation to
    # the development population instead (see
    # generate_stage1_full_population_nh_config_bundles), so a holdout basin
    # can never appear in a training or validation basin-list file.
    train_ids = bundle.train_basin_ids if bundle.train_basin_ids is not None else bundle.basin_ids
    validation_ids = bundle.validation_basin_ids if bundle.validation_basin_ids is not None else bundle.basin_ids
    test_ids = bundle.test_basin_ids if bundle.test_basin_ids is not None else bundle.basin_ids

    train_basin_file = out_dir / "train_basins.txt"
    validation_basin_file = out_dir / "validation_basins.txt"
    test_basin_file = out_dir / "test_basins.txt"
    for p, ids in (
        (train_basin_file, train_ids),
        (validation_basin_file, validation_ids),
        (test_basin_file, test_ids),
    ):
        _atomic_write_text(p, "\n".join(ids) + "\n")

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

    if bundle.population_role == "spatial_holdout_test_only":
        # NH 1.13's Config._check_cfg_keys rejects any unrecognized top-level
        # key, so this safeguard cannot live inside config.yaml itself -- it
        # must be a sibling file NH never reads.
        marker_path = out_dir / HOLDOUT_MARKER_FILENAME
        _atomic_write_text(
            marker_path,
            "This config bundle is SPATIAL-HOLDOUT TEST-ONLY EVALUATION MACHINERY.\n"
            "Do NOT run a trainer against this config.yaml.\n"
            "\n"
            "Its train_basin_file / validation_basin_file list the DEVELOPMENT\n"
            "population, never a holdout basin -- they are present only because\n"
            "NeuralHydrology's config schema requires train/validation basin files\n"
            "and date ranges to exist. Fitting against this bundle would silently\n"
            "reuse development data and produce a misleading, meaningless run.\n"
            "\n"
            "Construct only the 'test' period, supplying the already-fitted\n"
            "development-training scaler as an explicit external scaler; never fit\n"
            "a new scaler from this bundle. See\n"
            "src/baseline/nh_structural_preflight.py::"
            "check_flashnh_external_scaler_test_construction and the full-population\n"
            "config-generation entry in docs/decision_log.md.\n",
        )
        written_paths[HOLDOUT_MARKER_FILENAME] = marker_path
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
        "population_role": bundle.population_role,
        "train_basin_count": len(train_ids),
        "validation_basin_count": len(validation_ids),
        "test_basin_count": len(test_ids),
        "package_root": bundle.package_root,
        "package_manifest_identity": bundle.package_manifest_identity,
        "policy_path": bundle.policy_path,
        "policy_sha256": bundle.policy_sha256,
        "splits_dir": bundle.splits_dir,
        "run_profile_name": bundle.run_profile_name,
        "run_profile_values": dict(_RUN_PROFILES[bundle.run_profile_name]),
        "run_profile_note": _RUN_PROFILE_NOTES[bundle.run_profile_name],
        **(
            # Legacy field, preserved byte-for-byte ONLY for the compact-smoke
            # profile so pre-existing consumers/tests of this exact key are
            # unaffected. Deliberately absent (never written as False) for
            # every other profile, per this task's explicit requirement that
            # a seed-run manifest must not contain `compact_smoke_run_profile:
            # true` -- omission, not a False value, is the unambiguous signal.
            {
                "compact_smoke_run_profile": True,
                "compact_smoke_run_profile_note": _RUN_PROFILE_NOTES[COMPACT_SMOKE_RUN_PROFILE_NAME],
            }
            if bundle.run_profile_name == COMPACT_SMOKE_RUN_PROFILE_NAME
            else {}
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
        "max_updates_per_epoch": bundle.max_updates_per_epoch,
        "learning_rate_override": bundle.learning_rate,
        "resolved_learning_rate": full_mapping.get("learning_rate"),
        "hidden_size_override": bundle.hidden_size,
        "resolved_hidden_size": full_mapping.get("hidden_size"),
        "embedding_dropout_override": bundle.embedding_dropout,
        "resolved_embedding_dropout": (
            full_mapping["statics_embedding"]["dropout"]
            if isinstance(full_mapping.get("statics_embedding"), dict)
            else None
        ),
        "artifact_sha256": artifact_sha256,
    }
    manifest_path = out_dir / "generation_manifest.json"
    _atomic_write_text(manifest_path, json.dumps(generation_manifest, indent=2, default=str))

    written_paths["generation_manifest.json"] = manifest_path
    return written_paths
