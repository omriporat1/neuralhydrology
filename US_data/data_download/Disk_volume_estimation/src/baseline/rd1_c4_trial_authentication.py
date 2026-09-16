"""Authoritative, receipt-bound authentication of the frozen RD1-C4-D1 trial roster.

RD1-C4-D1 Correction Pass A, BLOCKER A1: the D1 CLI used to build its trial
targets straight out of an arbitrary JSON file. Every scientifically load-
bearing fact -- run directory, official best epoch, official objective,
search arm, configuration and proposal identity -- was whatever that file
said, and the ``source_receipt_path``/``source_receipt_sha256`` it carried
were recorded into the diagnostic's provenance without the receipt ever
being opened. A hand-written trial list could therefore aim the diagnostic
at any run directory on disk, including one belonging to a sealed
evaluation scope, and the resulting shard would carry a receipt reference
that nothing had checked.

This module removes the caller's authority entirely. A trial list may name
only two things per trial: which receipt to open, and that receipt's
expected SHA-256. Everything else is *derived from the receipt* through the
existing qualified consumer interface
(:func:`stage1_v2_12plus12_hydrological_consumer.build_v2_best_epoch_source`),
which is the repository's authority for what a v2 execution receipt means.
Nothing is reinterpreted here.

What the caller may assert, and what it may not
-----------------------------------------------

Accepted from the trial list::

    {
      "schema_name": "flashnh_rd1_c4_d1_trial_roster",
      "schema_version": 1,
      "campaign_id": "<v2 campaign id>",
      "support_contract_version": "<fixed-support contract_id>",
      "support_contract_sha256": "<fixed-support checksum_sha256>",
      "trials": [
        {"trial_id": ..., "search_arm": ...,
         "source_receipt_path": ..., "source_receipt_sha256": "<64 hex>"},
        ...
      ]
    }

Rejected outright as caller-asserted authority: ``run_dir``,
``best_epoch``, ``official_objective``, ``configuration_id``,
``proposal_id``, ``validation_pickle_sha256``, ``evaluation_scope``,
``sealed_scope`` and any other receipt-owned field. Supplying one is an
error, not a silently-ignored key -- a trial list that *looks* like it is
setting the run directory must never appear to have worked.

``trial_id`` and ``search_arm`` are accepted only as the caller's *claim*
about which trial this entry is, and both are then required to equal what
the receipt itself says. They exist so a roster is human-readable and so a
substituted receipt (a real, valid receipt for a different trial) is caught
by name rather than silently accepted.

How each fact is proven
-----------------------

For every entry, in order:

1. ``source_receipt_path`` is read once as raw bytes, and those bytes are
   SHA-256'd. The digest must equal the entry's declared
   ``source_receipt_sha256``. A missing file, an unreadable file, a
   non-JSON file, or a tampered file fails here.
2. The bytes are parsed and handed to
   :func:`~.stage1_v2_12plus12_hydrological_consumer.build_v2_best_epoch_source`,
   which re-reads and re-hashes the same path, requires the parsed record to
   match it exactly, requires every top-level and ``preparation_record``
   field the v2 receipt schema defines, derives the official best epoch and
   objective from the fixed-support epoch trajectory rather than trusting a
   stated integer, binds ``run_dir`` from the receipt's own
   ``result.nh_run_dir``, and finishes by re-verifying campaign, domain
   version, fidelity, model seed, evaluation scope, sealed-scope state,
   search-arm membership and the recomputed configuration/proposal/trial
   identities. None of that logic is reimplemented here.
3. The freshly built source is independently re-proven against its own
   receipt with
   :func:`~.stage1_v2_12plus12_hydrological_consumer._revalidate_source_against_authoritative_receipt`,
   the same gate the formal C4 batch entry point applies.
4. The receipt's identity is cross-checked against the roster header and
   against the fixed-support contract actually being used: campaign id,
   support-contract id and support-contract checksum must agree, so a
   receipt produced against a different contract cannot enter.
5. The official validation product is resolved -- never asserted -- as
   ``period_results_path(run_dir, contract["period"], best_epoch)`` from the
   receipt-bound run directory and receipt-derived best epoch, and must
   exist on disk.

Then, across the whole roster: exactly :data:`EXPECTED_TRIAL_COUNT` trials,
exactly :data:`EXPECTED_ARM_COUNTS` per arm, and no duplicated trial id,
receipt path, run directory or validation product. A roster that is missing
a trial, repeats one, substitutes one, or adds an unexpected one fails
closed before any comparison is performed.

Cost of the validation-pickle proof
-----------------------------------

``verify_validation_pickles`` defaults to ``False``. Each trial's
``validation_results.p`` is ~84.5 MB, so hashing all 24 costs ~2 GB of
reads; doing that inside every one of 24 Slurm array tasks would cost ~48 GB
for a fact each task needs only for its own trial. The default therefore
binds the *path* (receipt-derived, existence-checked) and leaves the
cryptographic proof to the point of use:
:func:`~.authenticated_period_results.load_authenticated_period_results`
hashes the pickle exactly once when the trial is actually evaluated, and the
diagnostic compares that digest against
:attr:`AuthenticatedTrialTarget.validation_pickle_sha256` when the roster was
authenticated with ``verify_validation_pickles=True``. Set it to ``True``
for a single whole-roster qualification pass; leave it ``False`` per array
task.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from .fixed_support_contract_v2 import validate_fixed_support_contract
from .nh_seed_evaluation import period_results_path
from .stage1_v2_12plus12_hydrological_consumer import (
    HydrologicalConsumerError,
    V2BestEpochSource,
    _revalidate_source_against_authoritative_receipt,
    build_v2_best_epoch_source,
)

__all__ = [
    "TrialAuthenticationError",
    "TRIAL_ROSTER_SCHEMA_NAME",
    "TRIAL_ROSTER_SCHEMA_VERSION",
    "EXPECTED_TRIAL_COUNT",
    "EXPECTED_ARM_COUNTS",
    "EXPECTED_PROPOSAL_ORDERS_BY_ARM",
    "RECEIPT_OWNED_FIELDS",
    "AuthenticatedTrialTarget",
    "AuthenticatedTrialRoster",
    "fixture_only_expected_roster",
    "authenticate_trial_roster",
]


class TrialAuthenticationError(ValueError):
    """Raised when a trial roster, or any trial in it, cannot be proven to
    derive from authoritative, unmodified v2 execution receipts for the
    expected campaign, contract and frozen population."""


#: Identity of the trial-roster file format this module accepts.
TRIAL_ROSTER_SCHEMA_NAME = "flashnh_rd1_c4_d1_trial_roster"
TRIAL_ROSTER_SCHEMA_VERSION = 1

#: The frozen RD1-C4 population is exactly P1--P12 and R1--R12.  Proposal
#: identity, rather than a caller-supplied count, is the authority: this
#: rejects a same-sized roster that silently replaces P1 with P13.  The
#: reducer imports the derived counts below, so these facts have one meaning.
EXPECTED_PROPOSAL_ORDERS_BY_ARM: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {"bayesian": tuple(range(1, 13)), "random_control": tuple(range(1, 13))}
)
EXPECTED_ARM_COUNTS: Mapping[str, int] = MappingProxyType(
    {arm: len(orders) for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_ARM.items()}
)
EXPECTED_TRIAL_COUNT = sum(EXPECTED_ARM_COUNTS.values())


@dataclass(frozen=True)
class _FixtureOnlyExpectedRoster:
    """Explicit test-only replacement for the frozen P1--P12/R1--R12 shape."""

    proposal_orders_by_arm: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposal_orders_by_arm",
            MappingProxyType({arm: tuple(orders) for arm, orders in self.proposal_orders_by_arm.items()}),
        )


def fixture_only_expected_roster(
    proposal_orders_by_arm: Mapping[str, Sequence[int]],
) -> _FixtureOnlyExpectedRoster:
    """Build an unmistakably test-only reduced-roster expectation.

    Production callers omit this object and therefore always receive the
    immutable P1--P12/R1--R12 requirement.  No count-only override exists.
    """
    normalized: dict[str, tuple[int, ...]] = {}
    for arm, orders in proposal_orders_by_arm.items():
        _require(arm in EXPECTED_PROPOSAL_ORDERS_BY_ARM, f"unknown test-only search arm {arm!r}")
        values = tuple(orders)
        _require(values != (), f"test-only expected roster arm {arm!r} is empty")
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in values),
            f"test-only expected roster arm {arm!r} must contain positive strict integers",
        )
        _require(len(set(values)) == len(values), f"test-only expected roster arm {arm!r} contains duplicates")
        normalized[arm] = values
    _require(normalized != {}, "test-only expected roster must contain at least one arm")
    return _FixtureOnlyExpectedRoster(normalized)

#: Facts a trial-list entry may NOT assert: each is owned by the execution
#: receipt (or derived from it) and accepting a caller's value for any of
#: them is exactly the authority inversion BLOCKER A1 reports.
RECEIPT_OWNED_FIELDS: tuple[str, ...] = (
    "run_dir",
    "best_epoch",
    "official_objective",
    "objective_score",
    "configuration_id",
    "proposal_id",
    "proposal_order",
    "campaign_id",
    "domain_version",
    "fidelity_id",
    "model_seed",
    "execution_generation",
    "evaluation_scope",
    "sealed_scope",
    "support_contract_version",
    "support_contract_sha256",
    "validation_pickle_path",
    "validation_pickle_sha256",
    "wandb_sweep_id",
    "wandb_run_id",
    "git_commit",
)

_ALLOWED_ENTRY_KEYS = frozenset(
    {"trial_id", "search_arm", "source_receipt_path", "source_receipt_sha256"}
)
_REQUIRED_HEADER_KEYS = (
    "schema_name",
    "schema_version",
    "campaign_id",
    "support_contract_version",
    "support_contract_sha256",
    "trials",
)
_ALLOWED_HEADER_KEYS = frozenset(_REQUIRED_HEADER_KEYS)

_SHA256_LENGTH = 64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrialAuthenticationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class AuthenticatedTrialTarget:
    """One trial whose every scientifically load-bearing fact was derived
    from its own authoritative, hash-verified execution receipt.

    This is the object the D1 diagnostic consumes in place of the former
    caller-asserted ``TrialTarget``. It intentionally exposes the complete
    set of source-identity facts a later atomic receipt must pin (RD1-C4-D1
    "closely related identity preparation"): receipt bytes/digest, run
    directory, official best epoch and objective, trial/configuration/
    proposal/arm identity, campaign/domain/fidelity, evaluation scope and
    sealed-scope state, the support-contract identity the trial was executed
    against, and the resolved official validation product.

    ``validation_pickle_sha256`` is ``None`` unless the roster was
    authenticated with ``verify_validation_pickles=True`` -- it is never a
    caller's claim, and its absence is reported rather than filled in.
    """

    trial_id: str
    search_arm: str
    campaign_id: str
    domain_version: str
    fidelity_id: str
    model_seed: int
    proposal_id: str
    proposal_order: int
    configuration_id: str
    execution_generation: int
    retry_of_trial_id: Optional[str]
    best_epoch: int
    official_objective: float
    fixed_support_metric_name: str
    evaluation_scope: str
    sealed_scope: bool
    support_contract_version: str
    support_contract_sha256: str
    run_dir: str
    source_receipt_path: str
    source_receipt_sha256: str
    validation_pickle_path: str
    validation_pickle_sha256: Optional[str]
    wandb_sweep_id: Optional[str]
    wandb_run_id: Optional[str]
    git_commit: Optional[str]
    best_epoch_source: V2BestEpochSource = field(repr=False, compare=False)

    def identity_fields(self) -> dict:
        """Flat, JSON-serializable identity record for shard receipts and
        diagnostic manifests. Every field is receipt-derived."""
        return {
            "trial_id": self.trial_id,
            "search_arm": self.search_arm,
            "campaign_id": self.campaign_id,
            "domain_version": self.domain_version,
            "fidelity_id": self.fidelity_id,
            "model_seed": self.model_seed,
            "proposal_id": self.proposal_id,
            "proposal_order": self.proposal_order,
            "configuration_id": self.configuration_id,
            "execution_generation": self.execution_generation,
            "retry_of_trial_id": self.retry_of_trial_id,
            "best_epoch": self.best_epoch,
            "official_objective": self.official_objective,
            "fixed_support_metric_name": self.fixed_support_metric_name,
            "evaluation_scope": self.evaluation_scope,
            "sealed_scope": self.sealed_scope,
            "support_contract_version": self.support_contract_version,
            "support_contract_sha256": self.support_contract_sha256,
            "run_dir": self.run_dir,
            "source_receipt_path": self.source_receipt_path,
            "source_receipt_sha256": self.source_receipt_sha256,
            "validation_pickle_path": self.validation_pickle_path,
            "validation_pickle_sha256": self.validation_pickle_sha256,
            "wandb_sweep_id": self.wandb_sweep_id,
            "wandb_run_id": self.wandb_run_id,
            "git_commit": self.git_commit,
        }


@dataclass(frozen=True)
class AuthenticatedTrialRoster:
    """The complete frozen 24-trial roster, every trial receipt-authenticated.

    ``trial_list_sha256`` is the SHA-256 of the roster file's own raw bytes,
    so a downstream receipt can pin not only each trial but the exact roster
    document that selected them.
    """

    schema_name: str
    schema_version: int
    campaign_id: str
    support_contract_version: str
    support_contract_sha256: str
    trial_list_path: str
    trial_list_sha256: str
    period: str
    validation_pickles_verified: bool
    targets: tuple = ()

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self):
        return iter(self.targets)

    def trial_ids(self) -> tuple:
        return tuple(t.trial_id for t in self.targets)

    def by_trial_id(self, trial_id: str) -> AuthenticatedTrialTarget:
        for target in self.targets:
            if target.trial_id == trial_id:
                return target
        raise TrialAuthenticationError(
            f"trial {trial_id!r} is not part of the authenticated roster {self.trial_ids()}"
        )

    def identity_fields(self) -> dict:
        """Roster-level identity for receipts: the roster document itself
        plus the ordered trial identities it authenticated."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "support_contract_version": self.support_contract_version,
            "support_contract_sha256": self.support_contract_sha256,
            "trial_list_path": self.trial_list_path,
            "trial_list_sha256": self.trial_list_sha256,
            "period": self.period,
            "validation_pickles_verified": self.validation_pickles_verified,
            "n_trials": len(self.targets),
            "trial_ids": list(self.trial_ids()),
            "source_receipt_sha256_by_trial_id": {
                t.trial_id: t.source_receipt_sha256 for t in self.targets
            },
        }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticate_one_trial(
    entry: Mapping[str, Any],
    *,
    index: int,
    contract: Mapping[str, Any],
    roster_campaign_id: str,
    verify_validation_pickles: bool,
) -> AuthenticatedTrialTarget:
    _require(
        isinstance(entry, Mapping),
        f"trial entry {index}: expected a JSON object, got {type(entry).__name__}",
    )
    supplied = set(entry)
    forbidden = sorted(supplied & set(RECEIPT_OWNED_FIELDS))
    _require(
        not forbidden,
        f"trial entry {index}: {forbidden} may not be asserted by a trial list -- "
        "every one of those facts is owned by the trial's execution receipt and is derived from it",
    )
    unexpected = sorted(supplied - _ALLOWED_ENTRY_KEYS)
    _require(
        not unexpected,
        f"trial entry {index}: unexpected key(s) {unexpected}; a trial entry may contain exactly "
        f"{sorted(_ALLOWED_ENTRY_KEYS)}",
    )
    missing = sorted(_ALLOWED_ENTRY_KEYS - supplied)
    _require(not missing, f"trial entry {index}: missing required key(s) {missing}")

    claimed_trial_id = entry["trial_id"]
    claimed_arm = entry["search_arm"]
    _require(
        isinstance(claimed_trial_id, str) and claimed_trial_id != "",
        f"trial entry {index}: trial_id must be a non-empty string",
    )
    _require(
        isinstance(claimed_arm, str) and claimed_arm != "",
        f"trial entry {index}: search_arm must be a non-empty string",
    )
    declared_digest = entry["source_receipt_sha256"]
    _require(
        _is_sha256(declared_digest),
        f"trial {claimed_trial_id!r}: source_receipt_sha256 {declared_digest!r} is not 64 lowercase hex "
        "characters",
    )
    receipt_path_value = entry["source_receipt_path"]
    _require(
        isinstance(receipt_path_value, str) and receipt_path_value != "",
        f"trial {claimed_trial_id!r}: source_receipt_path must be a non-empty string",
    )
    receipt_path = Path(receipt_path_value)
    _require(
        receipt_path.is_file(),
        f"trial {claimed_trial_id!r}: execution receipt is absent: {receipt_path} -- a trial cannot enter "
        "the diagnostic without its authoritative receipt",
    )

    raw = receipt_path.read_bytes()
    observed_digest = _sha256_bytes(raw)
    _require(
        observed_digest == declared_digest,
        f"trial {claimed_trial_id!r}: {receipt_path} has sha256 {observed_digest} but the trial list "
        f"declares {declared_digest} -- the receipt is not the document this roster was built from",
    )
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TrialAuthenticationError(
            f"trial {claimed_trial_id!r}: {receipt_path} is not valid UTF-8 JSON: {exc}"
        ) from exc
    _require(
        isinstance(record, dict),
        f"trial {claimed_trial_id!r}: {receipt_path} is not a JSON object",
    )

    # The authoritative interpretation of a v2 execution receipt. Every
    # schema requirement, the derived official best epoch/objective, the
    # receipt-bound run directory, and the campaign/domain/fidelity/seed/
    # scope/sealed-scope/arm/identity verification all live here -- none of
    # it is reimplemented in this module.
    try:
        source = build_v2_best_epoch_source(
            execution_provenance=record, source_receipt_path=receipt_path
        )
        source = _revalidate_source_against_authoritative_receipt(source)
    except HydrologicalConsumerError as exc:
        raise TrialAuthenticationError(
            f"trial {claimed_trial_id!r}: {receipt_path} is not a qualified v2 execution receipt: {exc}"
        ) from exc

    _require(
        source.trial_id == claimed_trial_id,
        f"trial entry {index}: the trial list claims trial_id {claimed_trial_id!r} but {receipt_path} is the "
        f"receipt for {source.trial_id!r} -- substituted receipt",
    )
    _require(
        source.search_arm == claimed_arm,
        f"trial {claimed_trial_id!r}: the trial list claims search_arm {claimed_arm!r} but the receipt "
        f"records {source.search_arm!r}",
    )
    _require(
        source.campaign_id == roster_campaign_id,
        f"trial {claimed_trial_id!r}: receipt campaign_id {source.campaign_id!r} != roster campaign_id "
        f"{roster_campaign_id!r}",
    )
    _require(
        source.support_contract_version == contract["contract_id"],
        f"trial {claimed_trial_id!r}: receipt support_contract_version {source.support_contract_version!r} "
        f"!= the fixed-support contract in use {contract['contract_id']!r} -- this trial was executed "
        "against a different support contract",
    )
    _require(
        source.support_contract_sha256 == contract["checksum_sha256"],
        f"trial {claimed_trial_id!r}: receipt support_contract_sha256 does not match the fixed-support "
        "contract's checksum_sha256 -- contract identity contradiction",
    )

    # The official validation product is RESOLVED from receipt-bound facts,
    # never asserted: run directory from the receipt, epoch from the
    # receipt-derived official best epoch, period from the contract.
    validation_pickle = period_results_path(source.run_dir, contract["period"], source.best_epoch)
    _require(
        validation_pickle.is_file(),
        f"trial {claimed_trial_id!r}: the receipt-bound official validation product is absent at "
        f"{validation_pickle} (run_dir={source.run_dir!r}, period={contract['period']!r}, "
        f"best_epoch={source.best_epoch!r})",
    )
    validation_digest = _sha256_path(validation_pickle) if verify_validation_pickles else None

    return AuthenticatedTrialTarget(
        trial_id=source.trial_id,
        search_arm=source.search_arm,
        campaign_id=source.campaign_id,
        domain_version=source.domain_version,
        fidelity_id=source.fidelity_id,
        model_seed=source.model_seed,
        proposal_id=source.proposal_id,
        proposal_order=source.proposal_order,
        configuration_id=source.configuration_id,
        execution_generation=source.execution_generation,
        retry_of_trial_id=source.retry_of_trial_id,
        best_epoch=source.best_epoch,
        official_objective=source.official_objective,
        fixed_support_metric_name=source.fixed_support_metric_name,
        evaluation_scope=source.evaluation_scope,
        sealed_scope=source.sealed_scope,
        support_contract_version=source.support_contract_version,
        support_contract_sha256=source.support_contract_sha256,
        run_dir=source.run_dir,
        source_receipt_path=str(receipt_path),
        source_receipt_sha256=observed_digest,
        validation_pickle_path=str(validation_pickle),
        validation_pickle_sha256=validation_digest,
        wandb_sweep_id=source.wandb_sweep_id,
        wandb_run_id=source.wandb_run_id,
        git_commit=source.git_commit,
        best_epoch_source=source,
    )


def authenticate_trial_roster(
    *,
    trial_list_path,
    contract: Mapping[str, Any],
    verify_validation_pickles: bool = False,
    test_only_expected_roster: Optional[_FixtureOnlyExpectedRoster] = None,
) -> AuthenticatedTrialRoster:
    """Authenticate a complete frozen trial roster against its receipts.

    See the module docstring for the exact accepted file shape, the per-trial
    proof chain, and the cost model for ``verify_validation_pickles``.

    Raises :class:`TrialAuthenticationError` for: a malformed or
    wrong-schema roster; a trial entry that asserts a receipt-owned fact; a
    missing, unreadable, non-JSON, tampered or substituted receipt; a receipt
    that is not a qualified v2 execution receipt (wrong campaign, domain,
    fidelity, seed, arm, scope, sealed scope, or recomputed configuration/
    proposal/trial identity); a receipt bound to a different support
    contract; a missing official validation product; and any roster-level
    population contradiction (wrong trial count, wrong per-arm counts,
    duplicate trial/receipt/run directory/validation product).
    """
    contract = validate_fixed_support_contract(dict(contract))
    if test_only_expected_roster is None:
        expected_proposals = EXPECTED_PROPOSAL_ORDERS_BY_ARM
    elif isinstance(test_only_expected_roster, _FixtureOnlyExpectedRoster):
        expected_proposals = test_only_expected_roster.proposal_orders_by_arm
    else:
        raise TrialAuthenticationError(
            "test_only_expected_roster must come from fixture_only_expected_roster(); production callers "
            "must omit it and therefore cannot replace the frozen P1--P12/R1--R12 population with counts"
        )
    expected_arm_counts = {arm: len(orders) for arm, orders in expected_proposals.items()}
    expected_trial_count = sum(expected_arm_counts.values())
    path = Path(trial_list_path)
    _require(path.is_file(), f"trial list is absent: {path}")
    raw = path.read_bytes()
    trial_list_sha256 = _sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TrialAuthenticationError(f"{path}: trial list is not valid UTF-8 JSON: {exc}") from exc
    _require(isinstance(payload, dict), f"{path}: trial list is not a JSON object")

    missing_header = [key for key in _REQUIRED_HEADER_KEYS if key not in payload]
    _require(not missing_header, f"{path}: trial list is missing required key(s) {missing_header}")
    unexpected_header = sorted(set(payload) - _ALLOWED_HEADER_KEYS)
    _require(
        not unexpected_header,
        f"{path}: trial list has unexpected header key(s) {unexpected_header}; scope, sealed-scope, run, "
        "epoch and objective facts belong only to the authenticated receipts",
    )
    _require(
        payload["schema_name"] == TRIAL_ROSTER_SCHEMA_NAME,
        f"{path}: trial list schema_name is {payload['schema_name']!r}; expected "
        f"{TRIAL_ROSTER_SCHEMA_NAME!r}",
    )
    _require(
        payload["schema_version"] == TRIAL_ROSTER_SCHEMA_VERSION,
        f"{path}: trial list schema_version is {payload['schema_version']!r}; expected "
        f"{TRIAL_ROSTER_SCHEMA_VERSION!r}",
    )
    campaign_id = payload["campaign_id"]
    _require(
        isinstance(campaign_id, str) and campaign_id != "",
        f"{path}: campaign_id must be a non-empty string",
    )
    _require(
        payload["support_contract_version"] == contract["contract_id"],
        f"{path}: trial list support_contract_version {payload['support_contract_version']!r} != the "
        f"fixed-support contract in use {contract['contract_id']!r}",
    )
    _require(
        payload["support_contract_sha256"] == contract["checksum_sha256"],
        f"{path}: trial list support_contract_sha256 does not match the fixed-support contract's "
        "checksum_sha256",
    )
    entries = payload["trials"]
    _require(isinstance(entries, list), f"{path}: 'trials' must be a list")
    _require(
        len(entries) == expected_trial_count,
        f"{path}: the frozen RD1-C4 roster contains exactly {expected_trial_count} trials, this list "
        f"contains {len(entries)} -- refusing to diagnose a partial or extended roster",
    )

    targets = [
        _authenticate_one_trial(
            entry,
            index=index,
            contract=contract,
            roster_campaign_id=campaign_id,
            verify_validation_pickles=verify_validation_pickles,
        )
        for index, entry in enumerate(entries)
    ]

    for label, values in (
        ("trial_id", [t.trial_id for t in targets]),
        ("source_receipt_path", [t.source_receipt_path for t in targets]),
        ("run_dir", [t.run_dir for t in targets]),
        ("validation_pickle_path", [t.validation_pickle_path for t in targets]),
    ):
        duplicates = sorted({v for v in values if values.count(v) > 1})
        _require(
            not duplicates,
            f"{path}: duplicate {label} in the authenticated roster: {duplicates} -- two roster entries "
            "resolve to the same trial",
        )

    observed_arm_counts: dict = {}
    for target in targets:
        observed_arm_counts[target.search_arm] = observed_arm_counts.get(target.search_arm, 0) + 1
    _require(
        observed_arm_counts == dict(expected_arm_counts),
        f"{path}: the frozen RD1-C4 roster requires {dict(expected_arm_counts)} but the authenticated "
        f"receipts give {observed_arm_counts} -- population contradiction",
    )

    observed_proposals = {(target.search_arm, target.proposal_order) for target in targets}
    expected_proposal_set = {
        (arm, proposal_order)
        for arm, proposal_orders in expected_proposals.items()
        for proposal_order in proposal_orders
    }
    missing_proposals = sorted(expected_proposal_set - observed_proposals)
    unexpected_proposals = sorted(observed_proposals - expected_proposal_set)
    _require(
        not missing_proposals and not unexpected_proposals and len(observed_proposals) == len(targets),
        f"{path}: authenticated receipts do not form the exact frozen proposal roster "
        f"(missing={missing_proposals}, unexpected={unexpected_proposals}, "
        f"distinct={len(observed_proposals)}, trials={len(targets)})",
    )

    return AuthenticatedTrialRoster(
        schema_name=TRIAL_ROSTER_SCHEMA_NAME,
        schema_version=TRIAL_ROSTER_SCHEMA_VERSION,
        campaign_id=campaign_id,
        support_contract_version=contract["contract_id"],
        support_contract_sha256=contract["checksum_sha256"],
        trial_list_path=str(path),
        trial_list_sha256=trial_list_sha256,
        period=contract["period"],
        validation_pickles_verified=bool(verify_validation_pickles),
        targets=tuple(targets),
    )
