"""Stage 1 lead-6 optimization pilot: run-spec generation (task item 1).

Reads ``config/stage1_lead06_pilot_v001.yaml`` (the closed, 6-entry pilot run
matrix) and composes the existing, unmodified config-generation machinery in
:mod:`src.baseline.nh_config_generation` -- ``validate_full_population_basin_membership``,
``build_nh_config_mapping``, the new named ``_RUN_PROFILES`` entries added
for this pilot, and ``write_generated_config`` -- to produce one
``GeneratedConfigBundle`` per pilot run_id.

The one genuinely new shape this pilot needs (train = full 2,307-basin
development population, validation = the ~400-basin screening subset, a
PROPER SUBSET of train, not equal to it) is not natively covered by either
existing bundle constructor
(``generate_stage1_nh_config``: single population, train==validation==test;
``generate_stage1_full_population_nh_config_bundles``: dev/spatial-holdout
pair, dev bundle has train==validation). It is built here directly from
``GeneratedConfigBundle``'s pre-existing independent
``train_basin_ids``/``validation_basin_ids``/``test_basin_ids`` fields --
the same mechanism ``generate_stage1_full_population_nh_config_bundles``
already uses for its spatial-holdout bundle -- with ``test_basin_ids`` set
to the development population (never the spatial holdout or temporal-test
basin list). NOTE: this scopes the *basin membership* of ``test_basin_ids``
only -- the generated config's ``test_start_date``/``test_end_date`` are
still the real, sealed temporal-test date window (``build_nh_config_mapping``
sets them unconditionally from the baseline policy). The bundle's basin-list
scoping alone does not make sealed-data access impossible; what actually
prevents it is that this pilot's own orchestration and screening code
(``src.baseline.pilot_orchestration``, ``src.baseline.pilot_screening_eval``)
never invoke a temporal-test, spatial-holdout, or California evaluation --
see ``evaluate_screening_checkpoint``'s ``period`` guard.

No filesystem writes happen in this module except via the unmodified
``write_generated_config``, which every caller invokes explicitly and
which refuses to write into an existing non-empty directory without
``force=True``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .nh_config_generation import (
    GeneratedConfigBundle,
    NHConfigGenerationError,
    PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME,
    Stage1BaselinePolicyError,
    build_nh_config_mapping,
    get_run_profile_mapping,
    load_stage1_baseline_policy,
    read_package_attribute_columns,
    read_package_manifest,
    resolve_target_variable,
    validate_dynamic_inputs,
    validate_dynamic_inputs_override,
    validate_full_population_basin_membership,
    validate_lead_hours,
    validate_seq_length,
    validate_static_attribute_contract,
    validate_target_variables,
    validate_max_updates_per_epoch,
    validate_learning_rate_override,
    validate_hidden_size_override,
    validate_embedding_dropout_override,
    _get_git_commit,
)
from datetime import datetime, timezone

from .splits import SplitGenerationError, load_eligible_basins, sha256_of

__all__ = [
    "PilotConfigError",
    "PilotPolicy",
    "PilotRunSpec",
    "load_pilot_policy",
    "pilot_run_ids",
    "resolve_pilot_run_spec",
    "load_screening_basin_ids",
    "SCREENING_VALIDATION_POPULATION_ROLE",
    "build_pilot_bundle",
    "build_pilot_bundle_with_validation_scope",
    "build_all_pilot_bundles",
]


class PilotConfigError(NHConfigGenerationError):
    """Raised for an invalid pilot policy, unknown run_id, or screening-subset
    contract violation."""


# Population role used by write_generated_config's generation_manifest.json
# for every pilot bundle whose validation scope is the screening subset (not
# the eventual, not-yet-run full-population validation -- see task item 9 /
# src/baseline/pilot_full_validation.py, which reuses this module's shared
# builder with a different validation scope and its own, distinct role).
SCREENING_VALIDATION_POPULATION_ROLE = "development_pilot_screening_validation"

# The pilot's closed 6-run scientific-semantics contract (task item 5): for
# each run_id, the exact declared static_pathway/embedding_hiddens/seed_name
# this pilot agreed to. A small explicit mapping, not a generalized schema
# engine -- deliberately not derived from the YAML itself, so a typo'd or
# swapped policy entry is caught rather than silently trusted.
_RAW_IDENTITY_PATHWAY = "raw_identity_concatenation"
_LEARNED_EMBEDDING_PATHWAY = "learned_fc_embedding"

_EXPECTED_RUN_SEMANTICS = {
    "raw_seedA": {"static_pathway": _RAW_IDENTITY_PATHWAY, "embedding_hiddens": None, "seed_name": "seed_a"},
    "raw_seedB": {"static_pathway": _RAW_IDENTITY_PATHWAY, "embedding_hiddens": None, "seed_name": "seed_b"},
    "emb128x64_seedA": {
        "static_pathway": _LEARNED_EMBEDDING_PATHWAY, "embedding_hiddens": [128, 64], "seed_name": "seed_a",
    },
    "emb128x64_seedB": {
        "static_pathway": _LEARNED_EMBEDDING_PATHWAY, "embedding_hiddens": [128, 64], "seed_name": "seed_b",
    },
    "emb64_seedA": {"static_pathway": _LEARNED_EMBEDDING_PATHWAY, "embedding_hiddens": [64], "seed_name": "seed_a"},
    "emb128_seedA": {"static_pathway": _LEARNED_EMBEDDING_PATHWAY, "embedding_hiddens": [128], "seed_name": "seed_a"},
}

# Fixed learned-embedding settings (untuned, held constant across all 3
# embedding shapes -- see config/stage1_lead06_pilot_v001.yaml comment).
_EXPECTED_EMBEDDING_ACTIVATION = "tanh"
_EXPECTED_EMBEDDING_DROPOUT = 0.1


@dataclass(frozen=True)
class PilotRunSpec:
    run_id: str
    static_pathway: str
    embedding_hiddens: "list | None"
    seed_name: str
    seed: int
    run_profile_name: str
    # Optional per-epoch NH training-batch cap for cheap early-fidelity
    # screening (see nh_config_generation.validate_max_updates_per_epoch and
    # docs/stage1_validation_optimization_foundation.md Part L.5/L.6). None
    # (the default) is uncapped/full-fidelity -- byte-identical to every
    # pre-existing pilot run. Frozen for this run's entire lifetime; never
    # mutated or overridden at launch time (see pilot_orchestration's
    # enforce_pilot_cap_identity, which rejects any later change).
    max_updates_per_epoch: "int | None" = None
    # Optional per-candidate learning-rate override (LR-A range-
    # characterization campaign; see nh_config_generation.
    # validate_learning_rate_override and docs/decision_log.md). None (the
    # default) means "use whatever the named run_profile_name already
    # specifies" -- byte-identical to every pre-existing pilot run, including
    # the closed six-run matrix and the cap25k/cap50k closure candidates.
    # Frozen for this run's entire lifetime; never mutated or overridden at
    # launch time (see pilot_orchestration's LR resume-contradiction guard,
    # which mirrors enforce_pilot_cap_identity's always-active design).
    learning_rate: "float | None" = None
    # Optional per-candidate LSTM hidden-size override (Hidden-size-A
    # range-characterization campaign; see nh_config_generation.
    # validate_hidden_size_override and docs/decision_log.md). None (the
    # default) means "use whatever the named run_profile_name already
    # specifies" -- byte-identical to every pre-existing pilot run. Frozen
    # for this run's entire lifetime; never mutated or overridden at launch
    # time (see pilot_orchestration's enforce_pilot_hidden_size_identity,
    # which mirrors enforce_pilot_cap_identity's always-active design).
    hidden_size: "int | None" = None
    # Optional per-candidate statics_embedding dropout override (Embedding-
    # Dropout-A range-characterization campaign; see nh_config_generation.
    # validate_embedding_dropout_override and docs/decision_log.md). None
    # (the default) means "use whatever the named run_profile_name already
    # specifies" -- byte-identical to every pre-existing pilot run, including
    # the closed six-run matrix and the LR-A/Hidden-size-A/cap25k/cap50k
    # closure candidates. Frozen for this run's entire lifetime; never
    # mutated or overridden at launch time (see pilot_orchestration's
    # enforce_pilot_embedding_dropout_identity, which mirrors
    # enforce_pilot_cap_identity's always-active design). 0.0 is a valid,
    # distinct-from-None override (the drop00 candidate) -- always checked
    # with "is not None", never truthiness.
    embedding_dropout: "float | None" = None
    # Optional per-candidate seq_length override (Sequence-Length-A range-
    # characterization campaign; see nh_config_generation.validate_seq_length
    # and docs/decision_log.md). None (the default) means "use the
    # campaign-wide PilotPolicy.seq_length" -- byte-identical to every
    # pre-existing pilot run, including the closed six-run matrix and the
    # LR-A/Hidden-size-A/Embedding-Dropout-A/cap25k/cap50k closure
    # candidates. When set, validated against the same closed
    # {12, 24, 48, 72} set as PilotPolicy.seq_length (see
    # nh_config_generation.validate_seq_length, invoked unconditionally on
    # whichever seq_length is actually resolved -- no separate structural
    # validator is needed for this field). Frozen for this run's entire
    # lifetime; never mutated or overridden at launch time (see
    # pilot_orchestration's enforce_pilot_seq_length_identity, which mirrors
    # enforce_pilot_cap_identity's always-active design).
    seq_length: "int | None" = None
    # Optional per-candidate dynamic-input-variable-set override (Dynamic-
    # Input-Family-A range-characterization campaign; see
    # nh_config_generation.validate_dynamic_inputs_override and
    # docs/decision_log.md). None (the default) means "use the campaign-wide
    # PilotPolicy's/baseline policy's own binding dynamic_inputs list" --
    # byte-identical to every pre-existing pilot run, including the closed
    # six-run matrix and the LR-A/Hidden-size-A/Embedding-Dropout-A/
    # Sequence-Length-A/cap25k/cap50k closure candidates. When set, must be a
    # non-empty, duplicate-free tuple of package-advertised dynamic-variable
    # names, order preserved exactly (validated structurally by
    # validate_dynamic_inputs_override against the package's actually-
    # advertised schema -- never re-sorted, never deduplicated-by-set, and
    # never checked against the baseline policy's exact 8-variable list,
    # unlike the unconditional package-integrity check every config
    # generation still performs via validate_dynamic_inputs). A tuple (not a
    # list) so this frozen dataclass's own instances stay genuinely
    # immutable/hashable. Frozen for this run's entire lifetime; never
    # mutated or overridden at launch time (see pilot_orchestration's
    # enforce_pilot_dynamic_inputs_identity, which mirrors
    # enforce_pilot_cap_identity's always-active design).
    dynamic_inputs: "tuple[str, ...] | None" = None


@dataclass(frozen=True)
class PilotPolicy:
    raw: dict
    path: str
    sha256: str
    lead_hours: int
    seq_length: int
    seeds: dict
    runs: "dict[str, PilotRunSpec]"
    workflow_qualification_run_id: str
    pilot_max_epoch_budget: int
    screening_validation_every_n_epochs: int
    diagnostic_only_epoch: int
    stopping_eligible_from_epoch: int
    screening_basin_ids_path: str
    screening_expected_count: int
    screening_expected_sha256: str
    base_early_stopping_policy_path: str
    wandb_policy_path: str


def load_pilot_policy(path) -> PilotPolicy:
    p = Path(path)
    if not p.is_file():
        raise PilotConfigError(f"pilot policy file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise PilotConfigError(f"pilot policy {p} did not parse to a mapping")

    required_top_keys = (
        "policy_name", "lead_hours", "seq_length", "seeds", "runs",
        "workflow_qualification_run_id", "early_stopping", "screening",
        "embedding_activation", "embedding_dropout",
    )
    missing = [k for k in required_top_keys if k not in raw]
    if missing:
        raise PilotConfigError(f"pilot policy {p} missing required key(s): {missing}")

    if raw["embedding_activation"] != _EXPECTED_EMBEDDING_ACTIVATION:
        raise PilotConfigError(
            f"pilot policy {p}: embedding_activation={raw['embedding_activation']!r}, "
            f"expected the frozen pilot value {_EXPECTED_EMBEDDING_ACTIVATION!r}"
        )
    if raw["embedding_dropout"] != _EXPECTED_EMBEDDING_DROPOUT:
        raise PilotConfigError(
            f"pilot policy {p}: embedding_dropout={raw['embedding_dropout']!r}, "
            f"expected the frozen pilot value {_EXPECTED_EMBEDDING_DROPOUT!r}"
        )

    seeds = raw["seeds"]
    if "seed_a" not in seeds or "seed_b" not in seeds:
        raise PilotConfigError(f"pilot policy {p}: seeds must define seed_a and seed_b")
    if seeds["seed_a"] == seeds["seed_b"]:
        raise PilotConfigError(f"pilot policy {p}: seed_a and seed_b must differ")

    runs: "dict[str, PilotRunSpec]" = {}
    for entry in raw["runs"]:
        run_id = entry["run_id"]
        if run_id in runs:
            raise PilotConfigError(f"pilot policy {p}: duplicate run_id {run_id!r}")
        if run_id not in PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME:
            raise PilotConfigError(
                f"pilot policy {p}: run_id {run_id!r} has no corresponding "
                "nh_config_generation._RUN_PROFILES entry -- this pilot's run "
                "matrix is closed to exactly the 6 agreed runs"
            )
        expected_profile_name = PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME[run_id]
        if entry["run_profile_name"] != expected_profile_name:
            raise PilotConfigError(
                f"pilot policy {p}: run_id {run_id!r} declares run_profile_name "
                f"{entry['run_profile_name']!r}, expected {expected_profile_name!r}"
            )
        seed_name = entry["seed_name"]
        if seed_name not in seeds:
            raise PilotConfigError(f"pilot policy {p}: run_id {run_id!r} references unknown seed_name {seed_name!r}")

        static_pathway = entry["static_pathway"]
        embedding_hiddens = entry.get("embedding_hiddens")

        # Optional per-run embedding-dropout override (Embedding-Dropout-A
        # range-characterization campaign): read before the semantics checks
        # below so the profile-resolved dropout check can reconcile against
        # it instead of the frozen 0.1 default when present. None of the
        # committed policy's six real run entries declare this key today, so
        # this is purely additive -- every existing entry resolves
        # embedding_dropout=None here exactly as before. Explicit "is not
        # None" (never truthiness) so a hypothetical explicit 0.0 entry is
        # not silently treated as "no override".
        run_embedding_dropout = entry.get("embedding_dropout")
        if run_embedding_dropout is not None:
            validate_embedding_dropout_override(run_embedding_dropout)

        expected_semantics = _EXPECTED_RUN_SEMANTICS[run_id]
        if static_pathway != expected_semantics["static_pathway"]:
            raise PilotConfigError(
                f"pilot policy {p}: run_id {run_id!r} declares static_pathway {static_pathway!r}, "
                f"expected {expected_semantics['static_pathway']!r}"
            )
        if embedding_hiddens != expected_semantics["embedding_hiddens"]:
            raise PilotConfigError(
                f"pilot policy {p}: run_id {run_id!r} declares embedding_hiddens {embedding_hiddens!r}, "
                f"expected {expected_semantics['embedding_hiddens']!r}"
            )
        if seed_name != expected_semantics["seed_name"]:
            raise PilotConfigError(
                f"pilot policy {p}: run_id {run_id!r} declares seed_name {seed_name!r}, "
                f"expected {expected_semantics['seed_name']!r}"
            )

        # Cross-check the declared semantics against the ACTUAL generated
        # profile this run_profile_name resolves to -- a YAML entry must not
        # describe one architecture while selecting a profile for another.
        profile = get_run_profile_mapping(entry["run_profile_name"])
        has_embedding = "statics_embedding" in profile
        if static_pathway == _RAW_IDENTITY_PATHWAY and has_embedding:
            raise PilotConfigError(
                f"pilot policy {p}: run_id {run_id!r} declares static_pathway="
                f"{_RAW_IDENTITY_PATHWAY!r} (no embedding) but its run_profile_name "
                f"{entry['run_profile_name']!r} defines statics_embedding"
            )
        if static_pathway == _LEARNED_EMBEDDING_PATHWAY:
            if not has_embedding:
                raise PilotConfigError(
                    f"pilot policy {p}: run_id {run_id!r} declares static_pathway="
                    f"{_LEARNED_EMBEDDING_PATHWAY!r} but its run_profile_name "
                    f"{entry['run_profile_name']!r} defines no statics_embedding"
                )
            embedding_spec = profile["statics_embedding"]
            if embedding_spec.get("hiddens") != embedding_hiddens:
                raise PilotConfigError(
                    f"pilot policy {p}: run_id {run_id!r} declares embedding_hiddens "
                    f"{embedding_hiddens!r}, but profile {entry['run_profile_name']!r} defines "
                    f"statics_embedding hiddens {embedding_spec.get('hiddens')!r}"
                )
            if embedding_spec.get("activation") != _EXPECTED_EMBEDDING_ACTIVATION:
                raise PilotConfigError(
                    f"pilot policy {p}: run_id {run_id!r} profile {entry['run_profile_name']!r} "
                    f"defines statics_embedding activation {embedding_spec.get('activation')!r}, "
                    f"expected {_EXPECTED_EMBEDDING_ACTIVATION!r}"
                )
            # No explicit per-run embedding_dropout override (every real
            # committed run today): preserve the old strict check exactly --
            # the named profile itself must still declare the frozen 0.1.
            # Explicit override present: the profile's own declared dropout
            # is no longer required to equal 0.1 -- build_nh_config_mapping
            # replaces it with the override value after profile merge (see
            # nh_config_generation.build_nh_config_mapping's embedding_dropout
            # parameter), so this profile-identity check is not meaningful
            # for an overridden run and is skipped; validate_embedding_
            # dropout_override above already checked the override's own
            # numeric validity.
            if run_embedding_dropout is None and embedding_spec.get("dropout") != _EXPECTED_EMBEDDING_DROPOUT:
                raise PilotConfigError(
                    f"pilot policy {p}: run_id {run_id!r} profile {entry['run_profile_name']!r} "
                    f"defines statics_embedding dropout {embedding_spec.get('dropout')!r}, "
                    f"expected {_EXPECTED_EMBEDDING_DROPOUT!r}"
                )
        if profile.get("seed") != seeds[seed_name]:
            raise PilotConfigError(
                f"pilot policy {p}: run_id {run_id!r} declares seed_name {seed_name!r} "
                f"(seed={seeds[seed_name]!r}), but profile {entry['run_profile_name']!r} defines "
                f"seed {profile.get('seed')!r}"
            )

        max_updates_per_epoch = entry.get("max_updates_per_epoch")
        if max_updates_per_epoch is not None:
            validate_max_updates_per_epoch(max_updates_per_epoch)

        learning_rate = entry.get("learning_rate")
        if learning_rate is not None:
            validate_learning_rate_override(learning_rate)

        hidden_size = entry.get("hidden_size")
        if hidden_size is not None:
            validate_hidden_size_override(hidden_size)

        runs[run_id] = PilotRunSpec(
            run_id=run_id,
            static_pathway=static_pathway,
            embedding_hiddens=embedding_hiddens,
            seed_name=seed_name,
            seed=seeds[seed_name],
            run_profile_name=entry["run_profile_name"],
            max_updates_per_epoch=max_updates_per_epoch,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            embedding_dropout=run_embedding_dropout,
        )

    known_run_ids = set(PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME)
    declared_run_ids = set(runs)
    if declared_run_ids != known_run_ids:
        raise PilotConfigError(
            f"pilot policy {p}: declared run_ids {sorted(declared_run_ids)} do not exactly "
            f"match the closed 6-run matrix {sorted(known_run_ids)}"
        )

    if raw["workflow_qualification_run_id"] not in runs:
        raise PilotConfigError(
            f"pilot policy {p}: workflow_qualification_run_id "
            f"{raw['workflow_qualification_run_id']!r} is not a declared run_id"
        )

    early_stopping = raw["early_stopping"]
    screening = raw["screening"]

    return PilotPolicy(
        raw=raw,
        path=str(p),
        sha256=sha256_of(p),
        lead_hours=raw["lead_hours"],
        seq_length=raw["seq_length"],
        seeds=dict(seeds),
        runs=runs,
        workflow_qualification_run_id=raw["workflow_qualification_run_id"],
        pilot_max_epoch_budget=early_stopping["pilot_max_epoch_budget"],
        screening_validation_every_n_epochs=early_stopping["screening_validation_every_n_epochs"],
        diagnostic_only_epoch=early_stopping["diagnostic_only_epoch"],
        stopping_eligible_from_epoch=early_stopping["stopping_eligible_from_epoch"],
        screening_basin_ids_path=screening["basin_ids_path"],
        screening_expected_count=screening["expected_count"],
        screening_expected_sha256=screening["expected_sha256"],
        base_early_stopping_policy_path=early_stopping["base_policy_path"],
        wandb_policy_path=raw["wandb"]["policy_path"],
    )


def pilot_run_ids(pilot_policy: PilotPolicy) -> list:
    return sorted(pilot_policy.runs)


def resolve_pilot_run_spec(pilot_policy: PilotPolicy, run_id: str) -> PilotRunSpec:
    if run_id not in pilot_policy.runs:
        raise PilotConfigError(
            f"unknown pilot run_id {run_id!r}; known run_ids: {sorted(pilot_policy.runs)}"
        )
    return pilot_policy.runs[run_id]


def load_screening_basin_ids(path, *, development_basins, expected_count: int, expected_sha256: str) -> list:
    """Read the screening-subset basin-ID artifact and check its contract:
    the file must be byte-identical (exact SHA-256) to the accepted
    provisional 400-basin screening realization recorded in
    ``reports/stage1_validation_optimization_foundation_v001/part_d_screening_subset/``
    -- not merely "some subset in a plausible size range" -- every screening
    basin must be a member of the (already-validated) development
    population, and (redundantly, since development_basins is itself
    already guaranteed CA/spatial-holdout free) no duplicate/malformed IDs.
    """
    try:
        screening_ids = load_eligible_basins(path)
    except SplitGenerationError as exc:
        raise PilotConfigError(f"could not read screening-subset basin-ids file {path}: {exc}") from exc

    if len(screening_ids) != expected_count:
        raise PilotConfigError(
            f"screening-subset basin-ids file {path} has {len(screening_ids)} basins, "
            f"expected exactly the accepted screening realization's {expected_count}"
        )
    actual_sha256 = sha256_of(path)
    if actual_sha256 != expected_sha256:
        raise PilotConfigError(
            f"screening-subset basin-ids file {path} has sha256={actual_sha256}, "
            f"expected the accepted screening realization's {expected_sha256} -- this pilot "
            "must use the exact accepted 400-basin screening subset, not a re-selected or "
            "substituted one"
        )

    development_set = set(development_basins)
    not_in_development = sorted(set(screening_ids) - development_set)
    if not_in_development:
        raise PilotConfigError(
            f"screening-subset basin-ids file {path} contains basin(s) not in the "
            f"development population (sealed-data-access risk): {not_in_development}"
        )
    if len(screening_ids) == len(development_basins):
        raise PilotConfigError(
            f"screening-subset basin-ids file {path} equals the full development population "
            "-- expected a proper subset"
        )

    return screening_ids


def build_pilot_bundle_with_validation_scope(
    *,
    baseline_policy_path,
    package_root,
    splits_dir,
    lead_hours: int,
    seq_length: int,
    run_profile_name: str,
    validation_basin_ids: list,
    population_role: str,
    package_type: str,
    static_column_manifest_path=None,
    max_updates_per_epoch: "int | None" = None,
    learning_rate: "float | None" = None,
    hidden_size: "int | None" = None,
    embedding_dropout: "float | None" = None,
    dynamic_inputs: "list | None" = None,
) -> GeneratedConfigBundle:
    """Shared builder underlying both this module's screening-validation
    bundle (task item 1) and ``pilot_full_validation.py``'s full-population
    validation-readiness bundle (task item 9) -- the only difference between
    the two is which basin list is passed as ``validation_basin_ids`` and
    what ``population_role``/``package_type`` label the resulting bundle
    carries. ``train_basin_ids`` and ``test_basin_ids`` are always the full
    development population (never a spatial-holdout basin). This scopes
    basin membership only -- the generated config still carries the real
    sealed temporal-test date window (see module docstring); callers must
    not treat this bundle as making sealed-period access impossible on its
    own. The pilot's orchestration/screening code is what never invokes
    that period -- see ``pilot_screening_eval.evaluate_screening_checkpoint``.

    ``dynamic_inputs`` (Dynamic-Input-Family-A range-characterization
    campaign; see ``PilotRunSpec.dynamic_inputs``): None (the default) means
    "use the baseline policy's own binding dynamic_inputs list", resolved and
    validated exactly as every pre-existing caller already does (the
    unconditional package-integrity check below always still runs
    regardless). A non-None list is validated structurally via
    ``validate_dynamic_inputs_override`` against the package's actually-
    advertised schema and used as this bundle's dynamic_inputs verbatim
    (order preserved), instead of the policy's full list.
    """
    baseline_policy_path = Path(baseline_policy_path)
    package_root = Path(package_root)
    splits_dir = Path(splits_dir)

    try:
        policy = load_stage1_baseline_policy(baseline_policy_path)
    except Stage1BaselinePolicyError as exc:
        raise PilotConfigError(f"scientific baseline policy failed validation: {exc}") from exc

    validate_seq_length(seq_length, policy)
    validate_lead_hours(lead_hours, policy)

    target_variable = resolve_target_variable(lead_hours, policy["target"]["variable_name_template"])
    validate_target_variables([target_variable], policy)

    package_manifest = read_package_manifest(package_root)
    package_attribute_columns = read_package_attribute_columns(package_root)

    package_dynamic_variables = list(package_manifest.get("dynamic_variables", []))
    validate_dynamic_inputs(package_dynamic_variables, policy)
    if dynamic_inputs is not None:
        resolved_dynamic_inputs = validate_dynamic_inputs_override(dynamic_inputs, package_dynamic_variables)
    else:
        resolved_dynamic_inputs = list(policy["dynamic_inputs"])

    static_result = validate_static_attribute_contract(
        policy,
        package_manifest,
        package_attribute_columns,
        static_column_manifest_path=static_column_manifest_path,
    )

    basin_membership = validate_full_population_basin_membership(package_manifest, splits_dir)
    development_basins = basin_membership.development_basins

    not_in_development = sorted(set(validation_basin_ids) - set(development_basins))
    if not_in_development:
        raise PilotConfigError(
            f"validation_basin_ids contains basin(s) outside the development population: {not_in_development}"
        )

    config_mapping = build_nh_config_mapping(
        policy=policy,
        target_variable=target_variable,
        seq_length=seq_length,
        dynamic_inputs=resolved_dynamic_inputs,
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
        basin_ids=development_basins,
        lead_hours=lead_hours,
        seq_length=seq_length,
        target_variable=target_variable,
        dynamic_inputs=resolved_dynamic_inputs,
        static_attribute_result=static_result,
        package_root=str(package_root),
        package_manifest_identity=package_manifest_identity,
        policy_path=str(baseline_policy_path),
        policy_sha256=sha256_of(baseline_policy_path),
        splits_dir=str(splits_dir),
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_commit=_get_git_commit(),
        package_type=package_type,
        population_role=population_role,
        train_basin_ids=development_basins,
        validation_basin_ids=list(validation_basin_ids),
        test_basin_ids=development_basins,
        run_profile_name=run_profile_name,
        max_updates_per_epoch=max_updates_per_epoch,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        embedding_dropout=embedding_dropout,
    )


def build_pilot_bundle(
    *,
    pilot_policy: PilotPolicy,
    run_id: str,
    baseline_policy_path,
    package_root,
    splits_dir,
    static_column_manifest_path=None,
) -> GeneratedConfigBundle:
    """Build the screening-validation ``GeneratedConfigBundle`` for one pilot
    run_id: train on the full development population, screen (NH's own
    ``validate_every``-driven validation) on the ~400-basin screening
    subset. Never touches the spatial-holdout or temporal-test period."""
    run_spec = resolve_pilot_run_spec(pilot_policy, run_id)

    package_manifest = read_package_manifest(package_root)
    basin_membership = validate_full_population_basin_membership(package_manifest, splits_dir)
    screening_ids = load_screening_basin_ids(
        pilot_policy.screening_basin_ids_path,
        development_basins=basin_membership.development_basins,
        expected_count=pilot_policy.screening_expected_count,
        expected_sha256=pilot_policy.screening_expected_sha256,
    )

    return build_pilot_bundle_with_validation_scope(
        baseline_policy_path=baseline_policy_path,
        package_root=package_root,
        splits_dir=splits_dir,
        lead_hours=pilot_policy.lead_hours,
        # Sequence-Length-A range-characterization campaign: a run_spec's
        # explicit seq_length override always wins over the campaign-wide
        # PilotPolicy default, mirroring max_updates_per_epoch/learning_rate/
        # hidden_size/embedding_dropout's "run_spec override wins" pattern
        # below -- byte-identical to every pre-existing pilot run, whose
        # run_spec.seq_length is always None and therefore always falls
        # through to pilot_policy.seq_length exactly as before.
        seq_length=(run_spec.seq_length if run_spec.seq_length is not None else pilot_policy.seq_length),
        run_profile_name=run_spec.run_profile_name,
        validation_basin_ids=screening_ids,
        population_role=SCREENING_VALIDATION_POPULATION_ROLE,
        package_type=f"stage1_lead06_pilot_{run_id}",
        static_column_manifest_path=static_column_manifest_path,
        max_updates_per_epoch=run_spec.max_updates_per_epoch,
        learning_rate=run_spec.learning_rate,
        hidden_size=run_spec.hidden_size,
        embedding_dropout=run_spec.embedding_dropout,
        # Dynamic-Input-Family-A range-characterization campaign: a run_spec's
        # explicit dynamic_inputs override always wins over the baseline
        # policy's own binding list, mirroring seq_length/max_updates_per_epoch/
        # learning_rate/hidden_size/embedding_dropout's "run_spec override
        # wins" pattern above -- byte-identical to every pre-existing pilot
        # run, whose run_spec.dynamic_inputs is always None and therefore
        # always falls through to the baseline policy's list exactly as
        # before. Converted list->tuple->list only insofar as
        # build_pilot_bundle_with_validation_scope accepts any list/tuple.
        dynamic_inputs=(list(run_spec.dynamic_inputs) if run_spec.dynamic_inputs is not None else None),
    )


def build_all_pilot_bundles(
    *,
    pilot_policy: PilotPolicy,
    baseline_policy_path,
    package_root,
    splits_dir,
    static_column_manifest_path=None,
) -> "dict[str, GeneratedConfigBundle]":
    """Build all 6 pilot bundles. No Cartesian expansion -- iterates exactly
    the run_ids declared in the pilot policy (which load_pilot_policy already
    validated equal the closed 6-run matrix)."""
    return {
        run_id: build_pilot_bundle(
            pilot_policy=pilot_policy,
            run_id=run_id,
            baseline_policy_path=baseline_policy_path,
            package_root=package_root,
            splits_dir=splits_dir,
            static_column_manifest_path=static_column_manifest_path,
        )
        for run_id in pilot_run_ids(pilot_policy)
    }
