"""Development-population Common-120 audit consumer/evaluator seam (SHARED-A2).

This module is the *smallest useful one-configuration vertical seam* that turns

    an explicit trial/checkpoint identity
      + an already-produced NH evaluation result (period-results pickle)
      + a frozen development-population Common-120 audit contract (SHARED-A1)
      + an explicit expected :class:`ExpectedPopulationSpec`

into **one audit row**, by:

1. validating the audit contract through the SHARED-A1 contract functions
   (:func:`validate_devpop_audit_contract` /
   :func:`validate_canonical_devpop_audit_contract`);
2. obtaining each basin's already-produced obs/sim series through the
   established result helper
   :func:`src.baseline.common_support_audit.basin_date_and_admitted`;
3. obtaining each basin's frozen admitted-timestamp set through the SHARED-A1
   helper :func:`src.baseline.devpop_common120_audit_contract.deserialize_contract_support`;
4. requiring *exact* timestamp membership -- no silent realignment, no
   truncation -- then masking observations outside Common-120 support to NaN so
   the qualified raw-space path's own ``admitted_mask = isfinite(obs)``
   naturally reduces to exactly the frozen support subset (no new metric math);
5. reusing :func:`derive_basin_area_km2_from_netcdf`,
   :func:`evaluate_basin_raw_space` and :func:`aggregate_raw_space_metrics`
   **verbatim**;
6. passing the assembled result through the SHARED-A1 completeness gate
   (:func:`require_complete_synthetic_devpop_audit_population` for a fixture,
   :func:`require_complete_devpop_audit_population` for the canonical
   population) before emitting a successful audit row.

**Scope of this milestone.**  This is a *local synthetic integration* seam.  It
does **not** run or launch checkpoint inference, prepare an evaluation run,
select a checkpoint / "best epoch", build the real 2,307-basin Common-120
support artifact, run the seven-configuration audit, or contact
W&B / Moriah / Slurm / any remote service.  The caller supplies the exact
trial/checkpoint identity; this module never infers it.

**Diagnostic-only.**  Every row this module emits carries
``objective_scope="devpop_audit"`` and the diagnostic
:data:`DEVPOP_AUDIT_CONTRACT_ID`; this module exposes no bare optimizer-objective
float and no ``flashnh/``-prefixed metric name, and imports nothing from the
W&B bridge / sweep-execution layer.  A fixture completeness receipt can never be
relabelled ``canonical_completeness`` -- that label is produced only by
SHARED-A1's :func:`require_complete_devpop_audit_population`; the *only* way this
module recognises canonical completeness is to route through that gate with a
canonical :class:`ExpectedPopulationSpec` and canonical contract
(``require_canonical=True``).  This module deliberately exposes **no**
row-level "assert canonical" helper: a post-hoc check that trusts caller-set
boolean labels on a row is forgeable and would not establish that canonical
completeness actually came through A1's gate.

**Result provenance.**  A caller-supplied checkpoint SHA and a result path are
*not* proof that the loaded ``{period}_results.p`` was produced from that
checkpoint.  :func:`evaluate_devpop_common120_audit_row` therefore requires a
:class:`DevpopAuditProvenanceReceipt` -- an authoritative producer artifact that
binds the exact checkpoint bytes and the exact consumed period-results bytes to
one declared trial / configuration / period / epoch identity -- and verifies it
against the actual files the row is built from before emitting a valid row.  The
future real evaluation runner emits the real receipt (see
:func:`build_devpop_audit_provenance_receipt`); this module never fabricates one
from caller identity alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .common_support_audit import CommonSupportAuditError, basin_date_and_admitted
from .devpop_common120_audit_contract import (
    AUDIT_OBJECTIVE_SCOPE,
    DEVPOP_AUDIT_CONTRACT_ID,
    DevpopAuditContractError,
    ExpectedPopulationSpec,
    deserialize_contract_support,
    require_complete_devpop_audit_population,
    require_complete_synthetic_devpop_audit_population,
    validate_canonical_devpop_audit_contract,
    validate_devpop_audit_contract,
)
from .nh_raw_space_evaluation import (
    DEFAULT_MAX_RELATIVE_MAD,
    DEFAULT_MIN_AREA_SAMPLES,
    RawSpaceEvaluationError,
    aggregate_raw_space_metrics,
    derive_basin_area_km2_from_netcdf,
    evaluate_basin_raw_space,
)
from .nh_seed_evaluation import (
    NHSeedEvaluationError,
    basin_netcdf_path,
    load_period_results,
    period_results_path,
)
from .package_audit import sha256_file

__all__ = [
    "DevpopAuditEvaluatorError",
    "DevpopAuditCheckpointIdentity",
    "DevpopAuditProvenanceReceipt",
    "PROVENANCE_RECEIPT_SCHEMA",
    "build_devpop_audit_provenance_receipt",
    "write_devpop_audit_provenance_receipt",
    "load_devpop_audit_provenance_receipt",
    "evaluate_devpop_common120_audit_row",
]

_HEX = frozenset("0123456789abcdef")

PROVENANCE_RECEIPT_SCHEMA = "flashnh_stage1_devpop_common120_audit_provenance_receipt_v001"


class DevpopAuditEvaluatorError(Exception):
    """Raised for a consumer-boundary setup / identity contradiction: a
    contract that does not describe the expected population, an expected basin
    missing from the supplied NH result, a frozen support timestamp absent from
    (or unmatched in) the run's own ``date`` coordinate, a frozen admitted
    observation that is not naturally finite, a non-finite simulation at an
    admitted Common-120 timestamp, an area derivation that fails or is
    inconsistent, or a supplied checkpoint file whose hash contradicts the
    declared identity.  Never raised for an ordinary poor-skill outcome (that
    surfaces as the assembled row's metrics), but a non-finite per-basin NSE
    still blocks completeness in the SHARED-A1 gate.
    """


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(c in _HEX for c in value)
    )


@dataclass(frozen=True)
class DevpopAuditCheckpointIdentity:
    """Explicit, caller-supplied identity of the *one* already-produced
    checkpoint whose NH evaluation result this row is built from.

    The four fields are the minimum needed to make the row self-describing and
    reproducible; this module does **not** invent checkpoint-selection
    semantics (no "best epoch", no ranking) -- the caller is authoritative for
    which trial / configuration / epoch this is.

    * ``trial_id`` / ``configuration_id`` -- opaque upstream identity strings,
      preserved verbatim onto the row.
    * ``checkpoint_epoch`` -- the epoch whose ``{period}_results`` pickle is
      read (``run_dir/{period}/model_epoch{epoch:03d}/{period}_results.p``).
    * ``checkpoint_sha256`` -- the checkpoint weight file's SHA-256.  This is a
      *declared* value; :func:`evaluate_devpop_common120_audit_row` only accepts
      it once it agrees with an authoritative
      :class:`DevpopAuditProvenanceReceipt` **and** the actual checkpoint file
      on disk hashes to it (a declared hash on its own is never sufficient).
    """

    trial_id: str
    configuration_id: str
    checkpoint_epoch: int
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        for name, value in (("trial_id", self.trial_id), ("configuration_id", self.configuration_id)):
            if not isinstance(value, str) or not value or value.isspace():
                raise DevpopAuditEvaluatorError(f"{name} must be a non-empty string")
        if not isinstance(self.checkpoint_epoch, int) or isinstance(self.checkpoint_epoch, bool):
            raise DevpopAuditEvaluatorError("checkpoint_epoch must be a strict int (no bool, no float)")
        if self.checkpoint_epoch < 0:
            raise DevpopAuditEvaluatorError(f"checkpoint_epoch must be >= 0, got {self.checkpoint_epoch!r}")
        if not _is_sha256(self.checkpoint_sha256):
            raise DevpopAuditEvaluatorError("checkpoint_sha256 must be a lowercase 64-char SHA-256")

    @classmethod
    def for_synthetic_fixture(
        cls,
        *,
        trial_id: str = "synthetic-trial-0000",
        configuration_id: str = "synthetic-config-0000",
        checkpoint_epoch: int = 1,
        checkpoint_sha256: Optional[str] = None,
    ) -> "DevpopAuditCheckpointIdentity":
        """Test-only convenience constructor with harmless defaults.  Carries no
        production meaning -- there is no canonical checkpoint identity here."""
        return cls(
            trial_id=trial_id,
            configuration_id=configuration_id,
            checkpoint_epoch=checkpoint_epoch,
            checkpoint_sha256=checkpoint_sha256 or ("0" * 64),
        )

    def identity_payload(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "configuration_id": self.configuration_id,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


_RECEIPT_STR_FIELDS = (
    "trial_id",
    "configuration_id",
    "period",
    "checkpoint_filename",
    "period_results_relpath",
)
_RECEIPT_FIELDS = (
    "schema",
    *_RECEIPT_STR_FIELDS,
    "checkpoint_epoch",
    "checkpoint_sha256",
    "period_results_sha256",
)


@dataclass(frozen=True)
class DevpopAuditProvenanceReceipt:
    """Authoritative producer -> consumer binding for one audit row.

    Separately hashing a caller-supplied checkpoint and separately selecting a
    result pickle by ``(run_dir, period, epoch)`` does **not** establish that
    the consumed result artifact was produced from that checkpoint.  This
    receipt closes that gap: it records, in one structured artifact,

    * the trial / configuration identity;
    * the period and checkpoint epoch;
    * the exact checkpoint weight file name and its SHA-256;
    * the canonical ``run_dir``-relative path of the consumed
      ``{period}_results.p`` and its exact SHA-256.

    The future real evaluation runner emits this receipt immediately after
    NeuralHydrology writes ``{period}_results.p`` (via
    :func:`build_devpop_audit_provenance_receipt`, which reads the real bytes on
    disk).  :func:`evaluate_devpop_common120_audit_row` then re-hashes the
    actual checkpoint and the actual result pickle it consumes and fails closed
    on any contradiction with the receipt or with the declared checkpoint
    identity.  A receipt whose fields are merely caller-asserted, with no
    matching bytes on disk, cannot produce a valid audit row.
    """

    trial_id: str
    configuration_id: str
    period: str
    checkpoint_epoch: int
    checkpoint_filename: str
    checkpoint_sha256: str
    period_results_relpath: str
    period_results_sha256: str
    schema: str = PROVENANCE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROVENANCE_RECEIPT_SCHEMA:
            raise DevpopAuditEvaluatorError(
                f"provenance receipt schema must be {PROVENANCE_RECEIPT_SCHEMA!r}, got {self.schema!r}"
            )
        for name in _RECEIPT_STR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.isspace():
                raise DevpopAuditEvaluatorError(f"provenance receipt {name} must be a non-empty string")
        if not isinstance(self.checkpoint_epoch, int) or isinstance(self.checkpoint_epoch, bool):
            raise DevpopAuditEvaluatorError("provenance receipt checkpoint_epoch must be a strict int")
        if self.checkpoint_epoch < 0:
            raise DevpopAuditEvaluatorError("provenance receipt checkpoint_epoch must be >= 0")
        for name in ("checkpoint_sha256", "period_results_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise DevpopAuditEvaluatorError(
                    f"provenance receipt {name} must be a lowercase 64-char SHA-256"
                )
        if Path(self.checkpoint_filename).name != self.checkpoint_filename:
            raise DevpopAuditEvaluatorError(
                "provenance receipt checkpoint_filename must be a bare file name, not a path"
            )
        rel = Path(self.period_results_relpath)
        if rel.is_absolute() or self.period_results_relpath != rel.as_posix() or ".." in rel.parts:
            raise DevpopAuditEvaluatorError(
                "provenance receipt period_results_relpath must be a normalized relative POSIX path"
            )

    def to_mapping(self) -> dict:
        return {name: getattr(self, name) for name in _RECEIPT_FIELDS}

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "DevpopAuditProvenanceReceipt":
        if not isinstance(mapping, Mapping):
            raise DevpopAuditEvaluatorError("provenance receipt must be a mapping")
        keys = set(mapping)
        missing = sorted(set(_RECEIPT_FIELDS) - keys)
        extra = sorted(keys - set(_RECEIPT_FIELDS))
        if missing or extra:
            raise DevpopAuditEvaluatorError(
                f"provenance receipt mapping has missing={missing} extra={extra} keys"
            )
        return cls(**{name: mapping[name] for name in _RECEIPT_FIELDS})


def build_devpop_audit_provenance_receipt(
    *,
    trial_id: str,
    configuration_id: str,
    run_dir,
    period: str,
    checkpoint_epoch: int,
    checkpoint_path,
) -> DevpopAuditProvenanceReceipt:
    """Producer-side receipt builder.

    Reads the **actual** checkpoint file and the **actual** canonical
    ``{period}_results.p`` under ``run_dir`` (:func:`period_results_path`) and
    binds their real SHA-256s to the declared identity.  This runs no inference
    -- the real evaluation runner calls it once NeuralHydrology has already
    written the result pickle; the synthetic A2 fixture calls it after writing
    fabricated checkpoint / NH-style result bytes.
    """
    run_dir = Path(run_dir)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise DevpopAuditEvaluatorError(f"checkpoint artifact does not exist: {checkpoint_path}")
    results_path = period_results_path(run_dir, period, checkpoint_epoch)
    if not results_path.is_file():
        raise DevpopAuditEvaluatorError(f"period-results artifact does not exist: {results_path}")
    return DevpopAuditProvenanceReceipt(
        trial_id=trial_id,
        configuration_id=configuration_id,
        period=period,
        checkpoint_epoch=int(checkpoint_epoch),
        checkpoint_filename=checkpoint_path.name,
        checkpoint_sha256=sha256_file(checkpoint_path),
        period_results_relpath=results_path.relative_to(run_dir).as_posix(),
        period_results_sha256=sha256_file(results_path),
    )


def write_devpop_audit_provenance_receipt(receipt: DevpopAuditProvenanceReceipt, path) -> Path:
    if not isinstance(receipt, DevpopAuditProvenanceReceipt):
        raise DevpopAuditEvaluatorError("receipt must be a DevpopAuditProvenanceReceipt")
    path = Path(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(receipt.to_mapping(), fh, indent=2, sort_keys=True)
    return path


def load_devpop_audit_provenance_receipt(path) -> DevpopAuditProvenanceReceipt:
    path = Path(path)
    if not path.is_file():
        raise DevpopAuditEvaluatorError(f"provenance receipt file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return DevpopAuditProvenanceReceipt.from_mapping(raw)


def _coerce_provenance_receipt(obj) -> DevpopAuditProvenanceReceipt:
    if isinstance(obj, DevpopAuditProvenanceReceipt):
        return obj
    if isinstance(obj, Mapping):
        return DevpopAuditProvenanceReceipt.from_mapping(obj)
    if isinstance(obj, (str, Path)):
        return load_devpop_audit_provenance_receipt(obj)
    raise DevpopAuditEvaluatorError(
        "provenance_receipt must be a DevpopAuditProvenanceReceipt, a mapping, or a path to a receipt JSON"
    )


def _verify_provenance_receipt(
    receipt: DevpopAuditProvenanceReceipt,
    *,
    checkpoint_identity: "DevpopAuditCheckpointIdentity",
    run_dir: Path,
    period: str,
    checkpoint_path: Path,
) -> dict:
    """Fail-closed verification of ``receipt`` against the actual files this row
    is built from.  Returns a compact verified-provenance block on success;
    raises :class:`DevpopAuditEvaluatorError` on the first contradiction."""
    # (1) declared checkpoint / trial identity must agree with the receipt
    for label, declared, in_receipt in (
        ("trial_id", checkpoint_identity.trial_id, receipt.trial_id),
        ("configuration_id", checkpoint_identity.configuration_id, receipt.configuration_id),
        ("checkpoint_epoch", checkpoint_identity.checkpoint_epoch, receipt.checkpoint_epoch),
        ("checkpoint_sha256", checkpoint_identity.checkpoint_sha256, receipt.checkpoint_sha256),
    ):
        if declared != in_receipt:
            raise DevpopAuditEvaluatorError(
                f"provenance receipt {label}={in_receipt!r} contradicts the declared checkpoint identity "
                f"({declared!r})"
            )

    # (4) period / epoch / canonical result-path identity must agree with the request
    if receipt.period != period:
        raise DevpopAuditEvaluatorError(
            f"provenance receipt period={receipt.period!r} contradicts the audit contract period ({period!r})"
        )
    canonical_relpath = (
        period_results_path(run_dir, period, checkpoint_identity.checkpoint_epoch)
        .relative_to(run_dir)
        .as_posix()
    )
    if receipt.period_results_relpath != canonical_relpath:
        raise DevpopAuditEvaluatorError(
            f"provenance receipt period_results_relpath={receipt.period_results_relpath!r} is not the canonical "
            f"period/epoch result path ({canonical_relpath!r})"
        )
    if receipt.checkpoint_filename != checkpoint_path.name:
        raise DevpopAuditEvaluatorError(
            f"provenance receipt checkpoint_filename={receipt.checkpoint_filename!r} does not match the supplied "
            f"checkpoint file ({checkpoint_path.name!r})"
        )

    # (2) the ACTUAL checkpoint artifact must hash to the declared / receipt SHA-256
    if not checkpoint_path.is_file():
        raise DevpopAuditEvaluatorError(f"checkpoint artifact does not exist: {checkpoint_path}")
    observed_ckpt = sha256_file(checkpoint_path)
    if observed_ckpt != receipt.checkpoint_sha256:
        raise DevpopAuditEvaluatorError(
            "the checkpoint artifact consumed does not match the provenance receipt "
            f"(file {observed_ckpt}, receipt {receipt.checkpoint_sha256})"
        )

    # (3) the ACTUAL result pickle at the canonical path must hash to the receipt SHA-256
    results_path = period_results_path(run_dir, period, checkpoint_identity.checkpoint_epoch)
    if not results_path.is_file():
        raise DevpopAuditEvaluatorError(f"period-results artifact does not exist: {results_path}")
    observed_results = sha256_file(results_path)
    if observed_results != receipt.period_results_sha256:
        raise DevpopAuditEvaluatorError(
            "the period-results artifact consumed does not match the provenance receipt "
            f"(file {observed_results}, receipt {receipt.period_results_sha256})"
        )

    return {
        "schema": PROVENANCE_RECEIPT_SCHEMA,
        "trial_id": receipt.trial_id,
        "configuration_id": receipt.configuration_id,
        "period": receipt.period,
        "checkpoint_epoch": receipt.checkpoint_epoch,
        "checkpoint_filename": receipt.checkpoint_filename,
        "checkpoint_sha256": observed_ckpt,
        "period_results_relpath": canonical_relpath,
        "period_results_sha256": observed_results,
        "provenance_verified": True,
    }


def _basin_support_metrics(
    *,
    basin_id: str,
    period_results: Mapping,
    contract: Mapping,
    package_root,
    target_variable: str,
    lead_hours: int,
    min_area_samples: int,
    max_relative_mad: float,
) -> dict:
    """Evaluate one basin restricted to its frozen Common-120 support.

    Every failure here is a consumer-boundary identity/integrity contradiction
    (:class:`DevpopAuditEvaluatorError`) -- never a soft exclusion, because the
    SHARED-A1 completeness gate forbids any excluded basin.
    """
    try:
        date_values, obs_mm_per_h, sim_mm_per_h = basin_date_and_admitted(
            period_results, basin_id, target_variable
        )
    except CommonSupportAuditError as exc:
        raise DevpopAuditEvaluatorError(
            f"basin {basin_id!r}: expected NH result not usable -- {exc}"
        ) from exc

    run_date_values = np.asarray(date_values)
    support_dates = deserialize_contract_support(contract, basin_id)
    if len(np.unique(run_date_values)) != len(run_date_values):
        raise DevpopAuditEvaluatorError(f"basin {basin_id!r}: run date coordinate contains duplicates")
    if len(np.unique(support_dates)) != len(support_dates):
        raise DevpopAuditEvaluatorError(f"basin {basin_id!r}: frozen Common-120 support contains duplicate timestamps")

    support_mask = np.isin(run_date_values, support_dates)
    n_matched = int(support_mask.sum())
    if n_matched != len(support_dates):
        raise DevpopAuditEvaluatorError(
            f"basin {basin_id!r}: {len(support_dates)} frozen Common-120 support timestamp(s) but only "
            f"{n_matched} present in this run's own date coordinate -- date/period contradiction "
            "(refusing to silently realign or truncate)"
        )

    obs_arr = np.asarray(obs_mm_per_h, dtype=np.float64)
    sim_arr = np.asarray(sim_mm_per_h, dtype=np.float64)
    if obs_arr.shape != run_date_values.shape or sim_arr.shape != run_date_values.shape:
        raise DevpopAuditEvaluatorError(
            f"basin {basin_id!r}: obs/sim length does not match the run date coordinate"
        )

    obs_support = np.where(support_mask, obs_arr, np.nan)
    if not np.isfinite(obs_support[support_mask]).all():
        raise DevpopAuditEvaluatorError(
            f"basin {basin_id!r}: a frozen Common-120 admitted timestamp is not a naturally finite observation"
        )
    if not np.isfinite(sim_arr[support_mask]).all():
        raise DevpopAuditEvaluatorError(
            f"basin {basin_id!r}: non-finite simulation at an admitted Common-120 timestamp"
        )

    nc_path = basin_netcdf_path(package_root, basin_id)
    try:
        area_result = derive_basin_area_km2_from_netcdf(
            nc_path,
            basin_id=basin_id,
            target_variable=target_variable,
            lead_hours=lead_hours,
            min_samples=min_area_samples,
            max_relative_mad=max_relative_mad,
        )
    except RawSpaceEvaluationError as exc:
        raise DevpopAuditEvaluatorError(
            f"basin {basin_id!r}: basin-area derivation failed -- {exc} "
            "(a full-development-population audit forbids excluding a basin)"
        ) from exc
    if not area_result.consistent:
        raise DevpopAuditEvaluatorError(
            f"basin {basin_id!r}: basin-area derivation inconsistent "
            f"(relative_mad={area_result.relative_mad:.6g}) -- audit forbids excluding a basin"
        )

    basin_metrics = evaluate_basin_raw_space(
        basin_id=basin_id,
        obs_mm_per_h=obs_support,
        sim_mm_per_h=sim_arr,
        area_km2=area_result.area_km2,
    )
    if basin_metrics.get("basin_id") != basin_id:
        raise DevpopAuditEvaluatorError(
            f"basin {basin_id!r}: raw-space evaluation returned a different basin identity "
            f"({basin_metrics.get('basin_id')!r})"
        )
    basin_metrics["freq"] = "1h"
    basin_metrics["n_common120_support_eligible"] = int(len(support_dates))
    basin_metrics["area_n_samples"] = area_result.n_samples
    basin_metrics["area_relative_mad"] = area_result.relative_mad
    return basin_metrics


def evaluate_devpop_common120_audit_row(
    *,
    checkpoint_identity: DevpopAuditCheckpointIdentity,
    run_dir,
    package_root,
    population: ExpectedPopulationSpec,
    contract: Mapping,
    provenance_receipt,
    checkpoint_path,
    require_canonical: bool = False,
    min_area_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
) -> dict:
    """Build one development-population Common-120 audit row for the *single*
    checkpoint identified by ``checkpoint_identity`` from an already-produced NH
    evaluation result under ``run_dir``.

    ``population`` and ``contract`` are the SHARED-A1 expected-population spec
    and (already-built) audit contract.  With ``require_canonical=False`` (the
    default, and the only path exercised in this synthetic milestone) the
    generic contract validator and the *synthetic* completeness gate are used,
    and the row's ``completeness`` receipt is labelled ``fixture_completeness``.
    ``require_canonical=True`` routes through
    :func:`validate_canonical_devpop_audit_contract` and
    :func:`require_complete_devpop_audit_population`; a synthetic fixture
    population or a generic contract cannot satisfy that path (it fails closed
    in SHARED-A1).  Canonical completeness is recognised **only** by routing
    through that gate here -- there is no forgeable row-level "assert canonical"
    helper.

    ``provenance_receipt`` (a :class:`DevpopAuditProvenanceReceipt`, an
    equivalent mapping, or a path to its JSON) and ``checkpoint_path`` (the
    actual checkpoint weight file) are **required**.  The receipt is verified by
    :func:`_verify_provenance_receipt` against the declared
    ``checkpoint_identity`` and against the real bytes of both the checkpoint
    file and the canonical ``{period}_results.p`` this row consumes; any
    contradiction fails closed before a valid row is emitted.  A declared
    checkpoint hash plus a caller-asserted receipt without matching bytes on
    disk is never sufficient.

    Raises :class:`DevpopAuditEvaluatorError` (or a SHARED-A1
    ``DevpopAuditContractError`` / ``DevpopAuditCompletenessError``) on any
    identity / provenance / completeness contradiction; never silently drops or
    realigns.
    """
    if not isinstance(checkpoint_identity, DevpopAuditCheckpointIdentity):
        raise DevpopAuditEvaluatorError("checkpoint_identity must be a DevpopAuditCheckpointIdentity")
    if not isinstance(population, ExpectedPopulationSpec):
        raise DevpopAuditEvaluatorError("population must be an ExpectedPopulationSpec")
    if not isinstance(contract, Mapping):
        raise DevpopAuditEvaluatorError("contract must be a mapping")

    # -- contract identity, through the SHARED-A1 validator(s) ---------------- #
    try:
        if require_canonical:
            validate_canonical_devpop_audit_contract(dict(contract))
        else:
            validate_devpop_audit_contract(dict(contract))
    except DevpopAuditContractError as exc:
        raise DevpopAuditEvaluatorError(f"audit contract failed SHARED-A1 validation: {exc}") from exc

    if contract["contract_id"] != DEVPOP_AUDIT_CONTRACT_ID:
        raise DevpopAuditEvaluatorError("audit contract is not the diagnostic development-population audit contract")
    if list(contract["basin_ids"]) != list(population.basin_ids):
        raise DevpopAuditEvaluatorError(
            "audit contract basin_ids do not equal the expected population basin_ids "
            "(the consumer never reconciles a population/contract mismatch)"
        )
    if contract["membership_ids_sha256"] != population.membership_ids_sha256:
        raise DevpopAuditEvaluatorError("audit contract membership hash does not equal the expected population hash")
    if contract["period"] != population.period:
        raise DevpopAuditEvaluatorError("audit contract period does not equal the expected population period")

    target_variable = contract["target_variable"]
    lead_hours = contract["lead_hours"]
    period = contract["period"]

    # -- authoritative producer -> consumer provenance receipt -------------- #
    # A declared checkpoint hash plus a result pickle selected only by
    # (run_dir, period, epoch) does NOT establish that the loaded result was
    # produced from that checkpoint.  The receipt binds the actual checkpoint
    # bytes and the actual consumed {period}_results.p bytes to one declared
    # trial/configuration/period/epoch identity; it is verified here against the
    # files this row is built from, fail-closed, before anything else.
    run_dir = Path(run_dir)
    checkpoint_path = Path(checkpoint_path)
    provenance = _verify_provenance_receipt(
        _coerce_provenance_receipt(provenance_receipt),
        checkpoint_identity=checkpoint_identity,
        run_dir=run_dir,
        period=period,
        checkpoint_path=checkpoint_path,
    )

    # -- the already-produced NH evaluation result (never (re)computed here) -- #
    try:
        period_results = load_period_results(run_dir, period, checkpoint_identity.checkpoint_epoch)
    except NHSeedEvaluationError as exc:
        raise DevpopAuditEvaluatorError(
            f"could not load the already-produced NH {period} result for "
            f"epoch {checkpoint_identity.checkpoint_epoch}: {exc}"
        ) from exc

    per_basin: list = []
    for basin_id in population.basin_ids:
        per_basin.append(
            _basin_support_metrics(
                basin_id=basin_id,
                period_results=period_results,
                contract=contract,
                package_root=package_root,
                target_variable=target_variable,
                lead_hours=lead_hours,
                min_area_samples=min_area_samples,
                max_relative_mad=max_relative_mad,
            )
        )

    aggregate = aggregate_raw_space_metrics(per_basin)
    evaluated_basin_ids = sorted(row["basin_id"] for row in per_basin)

    result = {
        "objective_scope": AUDIT_OBJECTIVE_SCOPE,
        "contract_id": contract["contract_id"],
        "contract_checksum_sha256": contract["checksum_sha256"],
        "seq_length_floor": contract["seq_length_floor"],
        "target_variable": target_variable,
        "lead_hours": lead_hours,
        "period": period,
        "checkpoint_epoch": checkpoint_identity.checkpoint_epoch,
        "requested_basin_ids": list(population.basin_ids),
        "evaluated_basin_ids": evaluated_basin_ids,
        "n_basins_requested": len(population.basin_ids),
        "n_basins_evaluated": len(per_basin),
        "n_basins_excluded": 0,
        "basins_excluded": [],
        "per_basin": per_basin,
        "aggregate": aggregate,
    }

    # -- SHARED-A1 completeness gate (the only producer of the receipt) ------ #
    if require_canonical:
        receipt = require_complete_devpop_audit_population(
            result, population=population, contract=contract
        )
    else:
        receipt = require_complete_synthetic_devpop_audit_population(
            result, population=population, contract=contract
        )

    canonical_completeness = receipt.get("canonical_completeness") is True
    canonical_population_verified = receipt.get("canonical_population_verified") is True

    return {
        "schema": "flashnh_stage1_devpop_common120_audit_row_v001",
        "objective_scope": AUDIT_OBJECTIVE_SCOPE,
        "contract_id": contract["contract_id"],
        "contract_checksum_sha256": contract["checksum_sha256"],
        "population_role": population.role,
        "expected_population_size": population.expected_size,
        "membership_ids_sha256": population.membership_ids_sha256,
        "checkpoint_identity": checkpoint_identity.identity_payload(),
        "trial_id": checkpoint_identity.trial_id,
        "configuration_id": checkpoint_identity.configuration_id,
        "checkpoint_epoch": checkpoint_identity.checkpoint_epoch,
        "checkpoint_sha256": checkpoint_identity.checkpoint_sha256,
        "provenance": provenance,
        "provenance_verified": True,
        "canonical_completeness": canonical_completeness,
        "canonical_population_verified": canonical_population_verified,
        "fixture_completeness": receipt.get("fixture_completeness") is True,
        "completeness": receipt,
        "result": result,
        "aggregate": aggregate,
        "nse_median": aggregate["metrics"]["nse"]["median"],
    }


# NOTE: there is deliberately no ``assert_canonical_devpop_audit_row(row)`` here.
# An independent review demonstrated that such a helper -- reading
# ``canonical_completeness`` / ``canonical_population_verified`` booleans off a
# caller-controlled row -- accepts a fabricated minimal mapping and therefore
# establishes nothing.  Canonical authority is established in exactly one place:
# :func:`evaluate_devpop_common120_audit_row` with ``require_canonical=True``,
# which routes a canonical :class:`ExpectedPopulationSpec` and canonical contract
# through SHARED-A1's :func:`require_complete_devpop_audit_population` gate (which
# re-derives the pinned 2,307-basin identity and every package/split/provenance
# identity).  A post-hoc row check cannot re-establish that without re-running
# that gate on the authoritative inputs.
