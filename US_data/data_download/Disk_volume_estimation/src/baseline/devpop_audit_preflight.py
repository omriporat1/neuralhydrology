"""SHARED-A5 Workstream A: manifest-driven, configuration-generic per-entry
audit preflight.

The scratch script ``.scratch_local/devpop_audit_a5_rerun_*/prep_and_preflight_p1_rerun.py``
hard-codes P1's ``hidden_size`` (256), ``seq_length`` (60) and
``checkpoint_epoch`` (1) as literal pass conditions -- so it can only ever
validate the P1 canary. This module derives every configuration-specific fact
from

* the VALIDATED frozen selection-manifest entry
  (:mod:`devpop_audit_selection_manifest`), for the per-configuration axes
  (``hidden_size``, ``seq_length``, the four other six-axis values,
  ``screening_best_epoch``, ``checkpoint_filename``, ``checkpoint_sha256``);
* the authoritative frozen v2 fixed configuration
  (``sweep_v2_six_axis_campaign.FROZEN_FIXED_CONFIGURATION_V2``) plus the
  SHARED-A1 audit-contract constants, for the campaign-frozen facts shared by
  all seven configurations (dynamic-input family, static-embedding shape and
  implied LSTM input dimension, lead hours, model seed, ``save_weights_every``,
  the 120h ``seq_length`` floor, target variable, evaluation period/window, the
  canonical 2,307-basin development population identity).

so the same code validates P1/P2/P3/R1/R2/R3/R4 -- and any future frozen
entry -- without a single P1 literal.

If the repository does not unambiguously freeze a value this preflight needs,
it is NOT invented here: the derivation raises
:class:`DevpopAuditPreflightError` naming the ambiguity.

Nothing here runs NeuralHydrology, contacts a remote host, or reads a remote
checkpoint. :func:`derive_expected_preflight_facts` is pure (manifest entry +
frozen constants only); :func:`preflight_devpop_audit_entry` additionally reads
local staged-run artifacts (the generated ``config.yml`` / eval-run manifest /
optionally the byte-verified staged checkpoint).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .devpop_audit_eval_run_producer import (
    AUDIT_EVAL_RUN_MANIFEST_FILENAME,
    AUDIT_EVAL_RUN_MARKER_FILENAME,
    AUDIT_EVAL_RUN_POPULATION_ROLE,
)
from .devpop_audit_selection_manifest import (
    DevpopAuditSelectionManifestError,
    validate_devpop_audit_selection_manifest,
    validate_devpop_audit_selection_manifest_entry,
)
from .devpop_common120_audit_contract import (
    AUDIT_DATE_MAX,
    AUDIT_DATE_MIN,
    AUDIT_PERIOD_NAME,
    CANONICAL_LEAD_HOURS,
    CANONICAL_SEQ_LENGTH_FLOOR,
    CANONICAL_TARGET_VARIABLE,
    DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256,
    DEVELOPMENT_TRAIN_ROLE,
    EXPECTED_DEVELOPMENT_POPULATION_SIZE,
)
from .nh_config_generation import EXPECTED_DEVELOPMENT_BASIN_COUNT
from .nh_seed_evaluation import weight_stem
from .package_audit import sha256_file
from .sweep_v1_campaign import MODEL_SEED_A
from .sweep_v2_six_axis_campaign import (
    FROZEN_FIXED_CONFIGURATION_V2,
    OBJECTIVE_ID_V2,
    SweepV2CampaignError,
    normalize_seq_length_axis,
)

__all__ = [
    "DevpopAuditPreflightError",
    "CampaignFrozenAuditFacts",
    "ExpectedPreflightFacts",
    "campaign_frozen_audit_facts",
    "derive_expected_preflight_facts",
    "preflight_devpop_audit_entry",
    "prepare_and_preflight_devpop_audit_entry",
]


class DevpopAuditPreflightError(ValueError):
    """Raised for a fail-closed preflight contradiction: a manifest entry that
    is internally inconsistent, a repository value the audit needs but does not
    unambiguously freeze, or a staged eval-run whose generated config /
    eval-run manifest / staged checkpoint does not match the facts derived from
    the frozen entry + frozen v2 configuration.  Never raised for an ordinary
    poor-skill outcome.

    On a staged-run check failure the partially-built report is attached as
    ``.report`` so a caller can log every failed check, not just the first.
    """

    def __init__(self, message: str, *, report: "Optional[dict]" = None) -> None:
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------- #
# campaign-frozen facts (identical for all seven configurations)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CampaignFrozenAuditFacts:
    """Facts every one of the seven audit configurations shares, derived from
    ``FROZEN_FIXED_CONFIGURATION_V2`` + the SHARED-A1 audit-contract constants.
    Never copied from any single configuration (P1 included)."""

    dynamic_inputs: tuple
    dynamic_input_count: int
    static_embedding_hiddens: tuple
    static_embedding_output_dim: int
    static_embedding_activation: str
    implied_lstm_input_dim: int
    lead_hours: int
    model_seed: int
    save_weights_every: int
    seq_length_floor: int
    target_variable: str
    period: str
    date_window: tuple
    authoritative_screening_epochs: tuple
    population_role: str
    expected_population_size: int
    membership_ids_sha256: str

    def as_dict(self) -> dict:
        return {
            "dynamic_inputs": list(self.dynamic_inputs),
            "dynamic_input_count": self.dynamic_input_count,
            "static_embedding_hiddens": list(self.static_embedding_hiddens),
            "static_embedding_output_dim": self.static_embedding_output_dim,
            "static_embedding_activation": self.static_embedding_activation,
            "implied_lstm_input_dim": self.implied_lstm_input_dim,
            "lead_hours": self.lead_hours,
            "model_seed": self.model_seed,
            "save_weights_every": self.save_weights_every,
            "seq_length_floor": self.seq_length_floor,
            "target_variable": self.target_variable,
            "period": self.period,
            "date_window": list(self.date_window),
            "authoritative_screening_epochs": list(self.authoritative_screening_epochs),
            "population_role": self.population_role,
            "expected_population_size": self.expected_population_size,
            "membership_ids_sha256": self.membership_ids_sha256,
        }


def _require_frozen(mapping: Mapping, key: str) -> Any:
    if key not in mapping:
        raise DevpopAuditPreflightError(
            f"FROZEN_FIXED_CONFIGURATION_V2 does not freeze {key!r} -- the audit needs this value "
            "and it must not be invented here"
        )
    return mapping[key]


def campaign_frozen_audit_facts() -> CampaignFrozenAuditFacts:
    """Derive the campaign-frozen audit facts from the authoritative frozen v2
    configuration + audit-contract constants.  Fails closed (naming the value)
    if the frozen configuration does not carry something the audit needs."""
    frozen = FROZEN_FIXED_CONFIGURATION_V2

    dynamic_inputs = tuple(_require_frozen(frozen, "dynamic_inputs"))
    if not dynamic_inputs or not all(isinstance(x, str) and x for x in dynamic_inputs):
        raise DevpopAuditPreflightError(
            f"frozen dynamic_inputs is not a non-empty list of strings: {dynamic_inputs!r}"
        )

    static_embedding = _require_frozen(frozen, "static_embedding")
    if not isinstance(static_embedding, Mapping) or "hiddens" not in static_embedding:
        raise DevpopAuditPreflightError(
            f"frozen static_embedding is not a mapping with 'hiddens': {static_embedding!r}"
        )
    hiddens = tuple(int(h) for h in static_embedding["hiddens"])
    if not hiddens:
        raise DevpopAuditPreflightError("frozen static_embedding.hiddens is empty")
    activation = static_embedding.get("activation")
    if not isinstance(activation, str) or not activation:
        raise DevpopAuditPreflightError(
            f"frozen static_embedding.activation is not a non-empty string: {activation!r}"
        )
    embedding_output_dim = hiddens[-1]
    implied_input_dim = len(dynamic_inputs) + embedding_output_dim

    lead_hours = int(_require_frozen(frozen, "lead_hours"))
    if lead_hours != CANONICAL_LEAD_HOURS:
        raise DevpopAuditPreflightError(
            f"frozen v2 lead_hours ({lead_hours}) does not equal the SHARED-A1 canonical audit "
            f"lead_hours ({CANONICAL_LEAD_HOURS}) -- the repository is internally ambiguous here"
        )
    save_weights_every = int(_require_frozen(frozen, "save_weights_every"))
    screening_epochs = tuple(int(e) for e in _require_frozen(frozen, "authoritative_screening_epochs"))
    if not screening_epochs or sorted(screening_epochs) != list(screening_epochs):
        raise DevpopAuditPreflightError(
            f"frozen authoritative_screening_epochs is empty or not ascending: {screening_epochs!r}"
        )

    return CampaignFrozenAuditFacts(
        dynamic_inputs=dynamic_inputs,
        dynamic_input_count=len(dynamic_inputs),
        static_embedding_hiddens=hiddens,
        static_embedding_output_dim=embedding_output_dim,
        static_embedding_activation=activation,
        implied_lstm_input_dim=implied_input_dim,
        lead_hours=lead_hours,
        model_seed=int(MODEL_SEED_A),
        save_weights_every=save_weights_every,
        seq_length_floor=int(CANONICAL_SEQ_LENGTH_FLOOR),
        target_variable=CANONICAL_TARGET_VARIABLE,
        period=AUDIT_PERIOD_NAME,
        date_window=(AUDIT_DATE_MIN, AUDIT_DATE_MAX),
        authoritative_screening_epochs=screening_epochs,
        population_role=DEVELOPMENT_TRAIN_ROLE,
        expected_population_size=int(EXPECTED_DEVELOPMENT_POPULATION_SIZE),
        membership_ids_sha256=DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256,
    )


# --------------------------------------------------------------------------- #
# per-entry derived facts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExpectedPreflightFacts:
    """Everything a per-entry audit preflight expects to see, derived from one
    validated manifest entry + the campaign-frozen facts.  The per-configuration
    architecture values (``hidden_size``, ``seq_length``, ...) come ONLY from the
    manifest entry; the shared values come ONLY from
    :func:`campaign_frozen_audit_facts`."""

    trial_id: str
    configuration_id: str
    proposal_id: str
    search_arm: str
    proposal_order: int
    execution_generation: int
    checkpoint_epoch: int
    checkpoint_filename: str
    checkpoint_sha256: str
    screening_score: float
    # per-configuration architecture (from the validated entry only)
    hidden_size: int
    seq_length: int
    learning_rate: str
    embedding_dropout: str
    output_dropout: str
    batch_size: int
    # shared campaign-frozen facts
    campaign_frozen: CampaignFrozenAuditFacts = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "configuration_id": self.configuration_id,
            "proposal_id": self.proposal_id,
            "search_arm": self.search_arm,
            "proposal_order": self.proposal_order,
            "execution_generation": self.execution_generation,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_filename": self.checkpoint_filename,
            "checkpoint_sha256": self.checkpoint_sha256,
            "screening_score": self.screening_score,
            "hidden_size": self.hidden_size,
            "seq_length": self.seq_length,
            "learning_rate": self.learning_rate,
            "embedding_dropout": self.embedding_dropout,
            "output_dropout": self.output_dropout,
            "batch_size": self.batch_size,
            "campaign_frozen": self.campaign_frozen.as_dict(),
        }


def _strict_axis_int(axes: Mapping, key: str) -> int:
    value = axes.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DevpopAuditPreflightError(
            f"manifest entry hyperparameters[{key!r}] is not a plain int: {value!r}"
        )
    return value


def _axis_str(axes: Mapping, key: str) -> str:
    value = axes.get(key)
    if not isinstance(value, str) or not value:
        raise DevpopAuditPreflightError(
            f"manifest entry hyperparameters[{key!r}] is not a canonical string decimal: {value!r}"
        )
    return value


def derive_expected_preflight_facts(
    entry: Mapping[str, Any],
    *,
    support_contract_version: "str | None" = None,
    support_contract_sha256: "str | None" = None,
) -> ExpectedPreflightFacts:
    """Validate one selection-manifest entry and derive the facts a per-entry
    audit preflight expects.  Pure: no filesystem, no network, no remote bytes.

    Raises :class:`DevpopAuditPreflightError` on a corrupted / internally
    inconsistent entry or on a value the repository does not unambiguously
    freeze.
    """
    try:
        validate_devpop_audit_selection_manifest_entry(
            entry,
            support_contract_version=support_contract_version,
            support_contract_sha256=support_contract_sha256,
        )
    except DevpopAuditSelectionManifestError as exc:
        raise DevpopAuditPreflightError(f"selection-manifest entry failed validation: {exc}") from exc

    frozen = campaign_frozen_audit_facts()

    if entry["objective_id"] != OBJECTIVE_ID_V2:
        raise DevpopAuditPreflightError(
            f"entry objective_id {entry['objective_id']!r} is not the frozen v2 screening objective "
            f"{OBJECTIVE_ID_V2!r}"
        )

    axes = entry["hyperparameters"]
    hidden_size = _strict_axis_int(axes, "hidden_size")
    batch_size = _strict_axis_int(axes, "batch_size")
    raw_seq_length = _strict_axis_int(axes, "seq_length")
    try:
        seq_length = normalize_seq_length_axis(raw_seq_length)
    except SweepV2CampaignError as exc:
        raise DevpopAuditPreflightError(f"entry seq_length is not a legal v2 axis value: {exc}") from exc
    if seq_length > frozen.seq_length_floor:
        raise DevpopAuditPreflightError(
            f"entry seq_length {seq_length} exceeds the fixed {frozen.seq_length_floor}h Common-120 "
            "audit floor -- the frozen support could not cover this configuration's warm-up"
        )

    epoch = entry["screening_best_epoch"]
    if epoch not in frozen.authoritative_screening_epochs:
        raise DevpopAuditPreflightError(
            f"entry screening_best_epoch {epoch} is not one of the frozen authoritative screening "
            f"epochs {list(frozen.authoritative_screening_epochs)}"
        )
    expected_ckpt_name = f"{weight_stem(epoch)}.pt"
    if entry["checkpoint_filename"] != expected_ckpt_name:
        raise DevpopAuditPreflightError(
            f"entry checkpoint_filename {entry['checkpoint_filename']!r} is not bound to "
            f"screening_best_epoch {epoch} (expected {expected_ckpt_name!r})"
        )

    return ExpectedPreflightFacts(
        trial_id=entry["trial_id"],
        configuration_id=entry["configuration_id"],
        proposal_id=entry["proposal_id"],
        search_arm=entry["search_arm"],
        proposal_order=entry["proposal_order"],
        execution_generation=entry["execution_generation"],
        checkpoint_epoch=epoch,
        checkpoint_filename=entry["checkpoint_filename"],
        checkpoint_sha256=entry["checkpoint_sha256"],
        screening_score=entry["screening_score"],
        hidden_size=hidden_size,
        seq_length=seq_length,
        learning_rate=_axis_str(axes, "learning_rate"),
        embedding_dropout=_axis_str(axes, "embedding_dropout"),
        output_dropout=_axis_str(axes, "output_dropout"),
        batch_size=batch_size,
        campaign_frozen=frozen,
    )


# --------------------------------------------------------------------------- #
# staged-run preflight
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # PyYAML

        loaded = yaml.safe_load(text)
    except ImportError:  # pragma: no cover - repo ships PyYAML
        from ruamel.yaml import YAML

        loaded = YAML(typ="safe").load(text)
    if not isinstance(loaded, Mapping):
        raise DevpopAuditPreflightError(f"generated config at {path} did not parse to a mapping")
    return dict(loaded)


def _select_manifest_entry(selection_manifest: Mapping, entry_trial_id: str) -> tuple[dict, dict]:
    if not isinstance(selection_manifest, Mapping):
        raise DevpopAuditPreflightError(
            "selection_manifest must be the validated seven-entry selection-manifest mapping"
        )
    try:
        validated = validate_devpop_audit_selection_manifest(selection_manifest.get("entries"))
    except DevpopAuditSelectionManifestError as exc:
        raise DevpopAuditPreflightError(f"selection manifest failed validation: {exc}") from exc
    recorded = selection_manifest.get("manifest_sha256")
    if recorded is not None and recorded != validated["manifest_sha256"]:
        raise DevpopAuditPreflightError(
            f"selection_manifest manifest_sha256 {recorded!r} does not match the recomputed identity "
            f"of its own entries ({validated['manifest_sha256']!r}) -- possible tampering"
        )
    selected = [e for e in validated["entries"] if e["trial_id"] == entry_trial_id]
    if len(selected) != 1:
        raise DevpopAuditPreflightError(
            f"entry_trial_id {entry_trial_id!r} does not identify exactly one entry of the validated "
            f"seven-entry selection manifest (matched {len(selected)})"
        )
    return validated, selected[0]


def preflight_devpop_audit_entry(
    *,
    selection_manifest: Mapping,
    entry_trial_id: str,
    eval_run_manifest: "Mapping | str | Path | None" = None,
    generated_config: "Mapping | str | Path | None" = None,
    run_dir: "str | Path | None" = None,
    verify_staged_checkpoint_bytes: bool = True,
    strict: bool = True,
) -> dict:
    """Verify one staged development-population audit eval-run against the facts
    derived from its frozen selection-manifest entry + the frozen v2
    configuration.

    ``eval_run_manifest`` / ``generated_config`` may be a mapping or a path; if
    omitted they are read from ``run_dir`` (``DEVPOP_AUDIT_EVAL_RUN_MANIFEST.json``
    and ``config.yml``).  When ``run_dir`` is given, the staged marker /
    manifest / scaler presence and (unless disabled) the staged checkpoint's
    bytes are checked too.

    Returns a report ``{"ok": bool, "entry_trial_id", "derived_facts",
    "checks": {name: {"ok", "expected", "observed"}}}``.  With ``strict=True``
    (the default, fail-closed) any failed check raises
    :class:`DevpopAuditPreflightError` with the full report attached as
    ``.report``.
    """
    validated_manifest, entry = _select_manifest_entry(selection_manifest, entry_trial_id)
    facts = derive_expected_preflight_facts(entry)
    frozen = facts.campaign_frozen

    run_dir_path = Path(run_dir) if run_dir is not None else None

    if eval_run_manifest is None:
        if run_dir_path is None:
            raise DevpopAuditPreflightError("need eval_run_manifest or run_dir")
        eval_run_manifest = run_dir_path / AUDIT_EVAL_RUN_MANIFEST_FILENAME
    if isinstance(eval_run_manifest, (str, Path)):
        p = Path(eval_run_manifest)
        if not p.is_file():
            raise DevpopAuditPreflightError(f"eval-run manifest not found: {p}")
        eval_run_manifest = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(eval_run_manifest, Mapping):
        raise DevpopAuditPreflightError("eval_run_manifest must be a mapping or a path to its JSON")

    if generated_config is None:
        if run_dir_path is None:
            raise DevpopAuditPreflightError("need generated_config or run_dir")
        generated_config = run_dir_path / "config.yml"
    if isinstance(generated_config, (str, Path)):
        generated_config = _load_yaml(Path(generated_config))
    if not isinstance(generated_config, Mapping):
        raise DevpopAuditPreflightError("generated_config must be a mapping or a path to config yaml")

    m = dict(eval_run_manifest)
    cfg = dict(generated_config)
    checks: dict[str, dict] = {}

    def _check(name: str, expected: Any, observed: Any) -> None:
        checks[name] = {"ok": expected == observed, "expected": expected, "observed": observed}

    # -- eval-run manifest identity ---------------------------------------- #
    _check("selection_manifest_sha256", validated_manifest["manifest_sha256"], m.get("selection_manifest_sha256"))
    _check("trial_id", facts.trial_id, m.get("trial_id"))
    _check("configuration_id", facts.configuration_id, m.get("configuration_id"))
    _check("search_arm", facts.search_arm, m.get("search_arm"))
    _check("proposal_order", facts.proposal_order, m.get("proposal_order"))
    _check("checkpoint_epoch", facts.checkpoint_epoch, m.get("checkpoint_epoch"))
    _check("checkpoint_sha256", facts.checkpoint_sha256, m.get("checkpoint_sha256"))
    _check("period", frozen.period, m.get("period"))
    _check("target_variable", frozen.target_variable, m.get("target_variable"))
    _check("lead_hours", frozen.lead_hours, m.get("lead_hours"))
    _check("eval_run_population_role", AUDIT_EVAL_RUN_POPULATION_ROLE, m.get("population_role"))
    _check("validation_basin_count", frozen.expected_population_size, m.get("validation_basin_count"))
    _check("nh_evaluation_population_count", frozen.expected_population_size, m.get("nh_evaluation_population_count"))
    _check("nh_evaluation_population_count_matches_config_generation",
           EXPECTED_DEVELOPMENT_BASIN_COUNT, m.get("nh_evaluation_population_count"))
    _check("date_window", list(frozen.date_window), m.get("date_window"))
    ckpt_path_name = Path(str(m.get("checkpoint_path", ""))).name
    _check("checkpoint_path_filename", facts.checkpoint_filename, ckpt_path_name)

    # -- generated config: campaign-frozen architecture (NOT P1 constants) - #
    cfg_dyn = list(cfg.get("dynamic_inputs", []))
    _check("dynamic_inputs", list(frozen.dynamic_inputs), cfg_dyn)
    _check("dynamic_input_count", frozen.dynamic_input_count, len(cfg_dyn))
    emb = cfg.get("statics_embedding") or {}
    emb_hiddens = [int(h) for h in (emb.get("hiddens") or [])]
    _check("static_embedding_hiddens", list(frozen.static_embedding_hiddens), emb_hiddens)
    _check("static_embedding_output_dim", frozen.static_embedding_output_dim,
           emb_hiddens[-1] if emb_hiddens else None)
    _check("static_embedding_activation", frozen.static_embedding_activation, emb.get("activation"))
    implied = (len(cfg_dyn) + emb_hiddens[-1]) if emb_hiddens else None
    _check("implied_lstm_input_dim", frozen.implied_lstm_input_dim, implied)
    _check("save_weights_every", frozen.save_weights_every, cfg.get("save_weights_every"))
    _check("seed", frozen.model_seed, cfg.get("seed"))

    # -- generated config: per-configuration architecture (from the entry) - #
    _check("hidden_size", facts.hidden_size, cfg.get("hidden_size"))
    _check("seq_length", facts.seq_length, cfg.get("seq_length"))
    _check("validate_n_random_basins", frozen.expected_population_size, cfg.get("validate_n_random_basins"))
    _check("validation_start_date", "01/01/2024", cfg.get("validation_start_date"))
    _check("validation_end_date", "31/12/2024", cfg.get("validation_end_date"))

    # -- staged run directory --------------------------------------------- #
    if run_dir_path is not None:
        _check("eval_only_marker_present", True, (run_dir_path / AUDIT_EVAL_RUN_MARKER_FILENAME).is_file())
        _check("eval_run_manifest_present", True, (run_dir_path / AUDIT_EVAL_RUN_MANIFEST_FILENAME).is_file())
        _check("staged_scaler_present", True, (run_dir_path / "train_data" / "train_data_scaler.yml").is_file())
        _check("staged_config_present", True, (run_dir_path / "config.yml").is_file())
        staged_ckpt = run_dir_path / facts.checkpoint_filename
        _check("staged_checkpoint_present", True, staged_ckpt.is_file())
        if verify_staged_checkpoint_bytes and staged_ckpt.is_file():
            _check("staged_checkpoint_sha256", facts.checkpoint_sha256, sha256_file(staged_ckpt))

    failed = sorted(k for k, v in checks.items() if not v["ok"])
    report = {
        "schema": "flashnh_devpop_audit_preflight_report_v001",
        "ok": not failed,
        "entry_trial_id": entry_trial_id,
        "failed_checks": failed,
        "derived_facts": facts.as_dict(),
        "checks": checks,
    }
    if failed and strict:
        raise DevpopAuditPreflightError(
            f"devpop audit preflight failed for {entry_trial_id!r}: {failed}", report=report
        )
    return report


def prepare_and_preflight_devpop_audit_entry(
    *,
    selection_manifest: Mapping,
    entry_trial_id: str,
    baseline_policy_path,
    policy_overlay_path,
    package_root,
    splits_dir,
    fixed_support_contract_path,
    checkpoint_src_path,
    scaler_src_path,
    run_profile_name: str,
    out_generated_dir,
    out_run_dir,
    expected_scaler_sha256: "str | None" = None,
    force: bool = False,
    strict: bool = True,
) -> dict:
    """Tracked replacement for the P1-only scratch ``prep_and_preflight`` script:
    call the SHARED-A4 producer for one frozen entry, then run the generic
    per-entry preflight against the staged run.  Returns
    ``{"eval_run_manifest": <producer manifest>, "preflight_report": <report>}``.
    """
    from .devpop_audit_eval_run_producer import prepare_devpop_audit_eval_run_dir

    eval_run_manifest = prepare_devpop_audit_eval_run_dir(
        selection_manifest=selection_manifest,
        entry_trial_id=entry_trial_id,
        baseline_policy_path=baseline_policy_path,
        policy_overlay_path=policy_overlay_path,
        package_root=package_root,
        splits_dir=splits_dir,
        fixed_support_contract_path=fixed_support_contract_path,
        checkpoint_src_path=checkpoint_src_path,
        scaler_src_path=scaler_src_path,
        run_profile_name=run_profile_name,
        out_generated_dir=out_generated_dir,
        out_run_dir=out_run_dir,
        expected_scaler_sha256=expected_scaler_sha256,
        force=force,
    )
    report = preflight_devpop_audit_entry(
        selection_manifest=selection_manifest,
        entry_trial_id=entry_trial_id,
        eval_run_manifest=eval_run_manifest,
        generated_config=Path(out_generated_dir) / "config.yaml",
        run_dir=Path(out_run_dir),
        strict=strict,
    )
    return {"eval_run_manifest": eval_run_manifest, "preflight_report": report}
