"""RD1 continuation evidence contract -- receipt-bound authentication of the
staged Bayesian continuation rosters approved beyond the frozen RD1-C4
24-run controlled comparison
(``docs/stage1_v2_12plus12_review_design_v001.md``).

Scientific framing (frozen, user-approved roadmap): the purpose of the
Bayesian continuation blocks is candidate discovery before later full-model
training -- never a claim that Bayesian optimization outperforms random
search. The staged path is:

* ``initial_controlled_24``  -- bayesian P1-P12 + random_control R1-R12
  (the existing, immutable :mod:`.rd1_c4_trial_authentication` population).
  This is the ONLY roster shape that supports the original Bayesian-versus-
  random descriptive comparison.
* ``bayesian_continuation_36`` -- the above 24 plus bayesian P13-P24 (36
  total). Bayesian search-progress / candidate-discovery evidence only.
* ``final_bayesian_48`` -- the above 36 plus bayesian P25-P36 (48 total,
  then stop). Bayesian search-progress / candidate-discovery evidence only.

No random-control runs are planned, or admitted here, beyond R12.

This module does not replace, extend, or loosen
:mod:`.rd1_c4_trial_authentication`: that module's ``authenticate_trial_roster``
remains the sole authority for the frozen 24-run production contract and is
untouched by this file. This module is a separate, versioned continuation
layer that reuses the *same* receipt-derived-identity primitive
(:func:`~.rd1_c4_trial_authentication._authenticate_one_trial`) rather than
reimplementing any part of the proof chain, and adds only what the
continuation phases need: a phase-aware expected-proposal-order shape, an
explicit ``phase`` roster header, and phase/slice-membership bookkeeping
(``initial_controlled_trial_ids`` / ``continuation_only_trial_ids``) so a
downstream consumer never has to re-derive which trials belong to the
immutable original comparison slice.

A continuation trial list may assert exactly the same two facts the frozen
roster allows (``trial_id``, ``search_arm`` as a claim; ``source_receipt_path``
/ ``source_receipt_sha256`` as the actual authority) -- every other
scientifically load-bearing fact remains receipt-owned and is proven, never
trusted, by :func:`~.rd1_c4_trial_authentication._authenticate_one_trial`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Optional

from .fixed_support_contract_v2 import validate_fixed_support_contract
from .rd1_c4_trial_authentication import (
    AuthenticatedTrialTarget,
    TrialAuthenticationError,
    _authenticate_one_trial,
    authenticate_trial_roster,
)

__all__ = [
    "ContinuationAuthenticationError",
    "PHASE_INITIAL_CONTROLLED_24",
    "PHASE_BAYESIAN_CONTINUATION_36",
    "PHASE_FINAL_BAYESIAN_48",
    "CONTINUATION_PHASES",
    "EXPECTED_PROPOSAL_ORDERS_BY_PHASE",
    "CONTINUATION_ROSTER_SCHEMA_NAME",
    "CONTINUATION_ROSTER_SCHEMA_VERSION",
    "AuthenticatedContinuationRoster",
    "authenticate_continuation_roster",
    "require_two_arm_comparison_support",
]


class ContinuationAuthenticationError(ValueError):
    """Raised for a continuation-roster-level contract violation: an unknown
    or mismatched phase, a malformed/wrong-schema roster, or a roster-level
    population contradiction (wrong trial count, wrong per-arm counts,
    duplicate trial/receipt/run directory/validation product, or an
    admitted proposal order the phase does not authorize). Per-trial receipt
    proof failures continue to raise
    :class:`~.rd1_c4_trial_authentication.TrialAuthenticationError` from the
    reused per-trial authority."""


PHASE_INITIAL_CONTROLLED_24 = "initial_controlled_24"
PHASE_BAYESIAN_CONTINUATION_36 = "bayesian_continuation_36"
PHASE_FINAL_BAYESIAN_48 = "final_bayesian_48"

#: The frozen random-control population. No random-control run is planned,
#: or admitted by this module, beyond R12 -- this tuple is identical across
#: every legal phase.
_RANDOM_CONTROL_ORDERS: tuple = tuple(range(1, 13))

#: The three legal staged roster shapes (docs/stage1_v2_12plus12_review_design_v001.md
#: Section 19; task-authorized roadmap). Each phase is the ordered union of
#: the frozen initial 24 plus the next approved Bayesian block -- never a
#: replacement of it, and never an additional random-control block.
CONTINUATION_PHASES: tuple = (
    PHASE_INITIAL_CONTROLLED_24,
    PHASE_BAYESIAN_CONTINUATION_36,
    PHASE_FINAL_BAYESIAN_48,
)

EXPECTED_PROPOSAL_ORDERS_BY_PHASE: dict = MappingProxyType(
    {
        PHASE_INITIAL_CONTROLLED_24: MappingProxyType(
            {"bayesian": tuple(range(1, 13)), "random_control": _RANDOM_CONTROL_ORDERS}
        ),
        PHASE_BAYESIAN_CONTINUATION_36: MappingProxyType(
            {"bayesian": tuple(range(1, 25)), "random_control": _RANDOM_CONTROL_ORDERS}
        ),
        PHASE_FINAL_BAYESIAN_48: MappingProxyType(
            {"bayesian": tuple(range(1, 37)), "random_control": _RANDOM_CONTROL_ORDERS}
        ),
    }
)

#: The (search_arm, proposal_order) set identifying the immutable initial
#: 24-run slice, independent of which phase a roster is being authenticated
#: as -- used to split every phase's targets into
#: ``initial_controlled_trial_ids`` / ``continuation_only_trial_ids``.
_INITIAL_CONTROLLED_PROPOSAL_SET: frozenset = frozenset(
    (arm, order)
    for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[PHASE_INITIAL_CONTROLLED_24].items()
    for order in orders
)

CONTINUATION_ROSTER_SCHEMA_NAME = "flashnh_rd1_continuation_trial_roster"
CONTINUATION_ROSTER_SCHEMA_VERSION = 1

#: The immutable RD1-C4 original initial 24-run roster's authoritative
#: raw-byte SHA-256 (the roster *file*, i.e. ``trial_list_sha256`` as
#: computed by :func:`~.rd1_c4_trial_authentication.authenticate_trial_roster`).
#: A 36-/48-run continuation roster's ``initial_roster_path`` must
#: authenticate to exactly this file, not merely to *some* internally
#: receipt-valid 24-run roster -- otherwise a caller could bind a
#: continuation phase to an alternate, never-frozen initial slice that
#: happens to be individually valid. There is deliberately no production
#: parameter on :func:`authenticate_continuation_roster` that can override
#: this value; the only sanctioned way to change it in a test is to
#: monkeypatch this exact module attribute (see
#: ``tests/test_rd1_continuation_trial_authentication.py``), which is not
#: reachable through any production call path.
_AUTHORITATIVE_INITIAL_ROSTER_SHA256 = "01569743a7fe69f180f2ca4a2be2f518e45ed70b2b2822642894b8e22cc52e4e"

_ALLOWED_ENTRY_KEYS = frozenset({"trial_id", "search_arm", "source_receipt_path", "source_receipt_sha256"})
_REQUIRED_HEADER_KEYS = (
    "schema_name",
    "schema_version",
    "campaign_id",
    "phase",
    "support_contract_version",
    "support_contract_sha256",
    "trials",
)
_ALLOWED_HEADER_KEYS = frozenset(_REQUIRED_HEADER_KEYS)

_SHA256_LENGTH = 64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuationAuthenticationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AuthenticatedContinuationRoster:
    """A receipt-authenticated staged continuation roster.

    Carries the same per-trial authority as
    :class:`~.rd1_c4_trial_authentication.AuthenticatedTrialRoster`
    (``targets`` are :class:`~.rd1_c4_trial_authentication.AuthenticatedTrialTarget`,
    proven exactly the same way), plus explicit phase identity and slice
    membership so a consumer never has to re-derive which trials belong to
    the immutable original 12+12 comparison slice.
    """

    schema_name: str
    schema_version: int
    campaign_id: str
    phase: str
    support_contract_version: str
    support_contract_sha256: str
    trial_list_path: str
    trial_list_sha256: str
    period: str
    validation_pickles_verified: bool
    targets: tuple = ()
    initial_controlled_trial_ids: tuple = ()
    continuation_only_trial_ids: tuple = ()
    initial_slice_identity: "MappingProxyType" = MappingProxyType({})

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
        raise ContinuationAuthenticationError(
            f"trial {trial_id!r} is not part of the authenticated continuation roster {self.trial_ids()}"
        )

    @property
    def two_arm_comparison_supported(self) -> bool:
        """Only ``initial_controlled_24`` supports the original Bayesian-
        versus-random-control descriptive comparison; see
        :func:`require_two_arm_comparison_support`."""
        return self.phase == PHASE_INITIAL_CONTROLLED_24

    def identity_fields(self) -> dict:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "phase": self.phase,
            "support_contract_version": self.support_contract_version,
            "support_contract_sha256": self.support_contract_sha256,
            "trial_list_path": self.trial_list_path,
            "trial_list_sha256": self.trial_list_sha256,
            "period": self.period,
            "validation_pickles_verified": self.validation_pickles_verified,
            "n_trials": len(self.targets),
            "trial_ids": list(self.trial_ids()),
            "initial_controlled_trial_ids": list(self.initial_controlled_trial_ids),
            "continuation_only_trial_ids": list(self.continuation_only_trial_ids),
            "two_arm_comparison_supported": self.two_arm_comparison_supported,
            "source_receipt_sha256_by_trial_id": {
                t.trial_id: t.source_receipt_sha256 for t in self.targets
            },
            "initial_slice_identity": dict(self.initial_slice_identity),
        }


def require_two_arm_comparison_support(obj) -> None:
    """Code-level enforcement of the interpretation boundary (task design
    constraint 5): raise unless ``obj`` (an :class:`AuthenticatedContinuationRoster`
    or a continuation review result exposing a ``.phase`` attribute) is the
    immutable ``initial_controlled_24`` slice. The 36/48 continuation blocks
    support Bayesian search-progress / candidate-discovery evidence only --
    never a winner, promotion, classifier, tolerance, range-change, or
    sealed-scope conclusion, and never a re-presentation of the original
    Bayesian-versus-random two-arm comparison over a larger population."""
    phase = getattr(obj, "phase", None)
    if phase != PHASE_INITIAL_CONTROLLED_24:
        raise ContinuationAuthenticationError(
            f"phase {phase!r} does not support the original Bayesian-versus-random-control two-arm "
            f"comparison -- only {PHASE_INITIAL_CONTROLLED_24!r} (the frozen P1-12 + R1-12 slice) supports "
            "that descriptive comparison; continuation blocks provide Bayesian search-progress / "
            "candidate-discovery evidence only"
        )


def authenticate_continuation_roster(
    *,
    trial_list_path,
    contract,
    phase: str,
    initial_roster_path=None,
    verify_validation_pickles: bool = False,
) -> AuthenticatedContinuationRoster:
    """Authenticate a staged continuation trial roster against its receipts.

    ``phase`` selects the expected proposal-order shape from
    :data:`EXPECTED_PROPOSAL_ORDERS_BY_PHASE`; the roster file's own
    ``phase`` header must equal it. Every per-trial proof is delegated to
    :func:`~.rd1_c4_trial_authentication._authenticate_one_trial`, the exact
    receipt-derived-identity primitive the frozen 24-run contract uses --
    nothing about how a trial is proven is reimplemented here.

    Immutable initial-slice binding (continuation phases only): a 36- or
    48-run continuation roster is not accepted merely because it has the
    right shape -- its claimed initial-24 subset must be bound, trial-id
    for trial-id and receipt-sha256 for receipt-sha256, to the exact
    already-frozen initial 24-run roster. ``initial_roster_path`` is
    required for ``bayesian_continuation_36``/``final_bayesian_48`` and is
    authenticated by reusing the frozen, unmodified
    :func:`~.rd1_c4_trial_authentication.authenticate_trial_roster` (the
    sole authority for the frozen 24-run contract). ``initial_roster_path``
    must be omitted for ``initial_controlled_24``, which *is* the initial
    slice and is trivially self-bound.

    Raises :class:`ContinuationAuthenticationError` for an unknown phase, a
    malformed/wrong-schema/wrong-phase roster, a roster-level population
    contradiction (wrong trial count, wrong per-arm counts, an admitted
    proposal order the phase does not authorize, or a duplicate trial/
    receipt/run directory/validation product), a missing/wrongly-supplied
    ``initial_roster_path``, or an initial-slice binding mismatch. Raises
    :class:`~.rd1_c4_trial_authentication.TrialAuthenticationError` for any
    per-trial receipt-proof failure (missing/tampered/substituted receipt,
    caller-asserted receipt-owned field, wrong contract binding, etc.).
    """
    _require(phase in EXPECTED_PROPOSAL_ORDERS_BY_PHASE, f"unknown continuation phase {phase!r}")
    _require(
        (phase == PHASE_INITIAL_CONTROLLED_24) == (initial_roster_path is None),
        f"initial_roster_path must be omitted for {PHASE_INITIAL_CONTROLLED_24!r} (it is the initial slice) "
        f"and must be supplied for {PHASE_BAYESIAN_CONTINUATION_36!r}/{PHASE_FINAL_BAYESIAN_48!r} "
        "(the continuation roster's initial-24 subset must be bound to it)",
    )
    expected_proposals = EXPECTED_PROPOSAL_ORDERS_BY_PHASE[phase]
    expected_arm_counts = {arm: len(orders) for arm, orders in expected_proposals.items()}
    expected_trial_count = sum(expected_arm_counts.values())

    contract = validate_fixed_support_contract(dict(contract))
    path = Path(trial_list_path)
    _require(path.is_file(), f"continuation trial list is absent: {path}")
    raw = path.read_bytes()
    trial_list_sha256 = _sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContinuationAuthenticationError(
            f"{path}: continuation trial list is not valid UTF-8 JSON: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path}: continuation trial list is not a JSON object")

    missing_header = [key for key in _REQUIRED_HEADER_KEYS if key not in payload]
    _require(not missing_header, f"{path}: continuation trial list is missing required key(s) {missing_header}")
    unexpected_header = sorted(set(payload) - _ALLOWED_HEADER_KEYS)
    _require(
        not unexpected_header,
        f"{path}: continuation trial list has unexpected header key(s) {unexpected_header}",
    )
    _require(
        payload["schema_name"] == CONTINUATION_ROSTER_SCHEMA_NAME,
        f"{path}: continuation trial list schema_name is {payload['schema_name']!r}; expected "
        f"{CONTINUATION_ROSTER_SCHEMA_NAME!r}",
    )
    _require(
        payload["schema_version"] == CONTINUATION_ROSTER_SCHEMA_VERSION,
        f"{path}: continuation trial list schema_version is {payload['schema_version']!r}; expected "
        f"{CONTINUATION_ROSTER_SCHEMA_VERSION!r}",
    )
    _require(
        payload["phase"] == phase,
        f"{path}: continuation trial list declares phase {payload['phase']!r} but was requested to be "
        f"authenticated as {phase!r} -- a roster's phase header cannot be overridden by the caller",
    )
    campaign_id = payload["campaign_id"]
    _require(isinstance(campaign_id, str) and campaign_id != "", f"{path}: campaign_id must be a non-empty string")
    _require(
        payload["support_contract_version"] == contract["contract_id"],
        f"{path}: continuation trial list support_contract_version {payload['support_contract_version']!r} "
        f"!= the fixed-support contract in use {contract['contract_id']!r}",
    )
    _require(
        payload["support_contract_sha256"] == contract["checksum_sha256"],
        f"{path}: continuation trial list support_contract_sha256 does not match the fixed-support "
        "contract's checksum_sha256",
    )
    entries = payload["trials"]
    _require(isinstance(entries, list), f"{path}: 'trials' must be a list")
    _require(
        len(entries) == expected_trial_count,
        f"{path}: continuation phase {phase!r} requires exactly {expected_trial_count} trials, this list "
        f"contains {len(entries)} -- refusing to authenticate a partial or extended roster",
    )

    targets = tuple(
        _authenticate_one_trial(
            entry,
            index=index,
            contract=contract,
            roster_campaign_id=campaign_id,
            verify_validation_pickles=verify_validation_pickles,
        )
        for index, entry in enumerate(entries)
    )

    for label, values in (
        ("trial_id", [t.trial_id for t in targets]),
        ("source_receipt_path", [t.source_receipt_path for t in targets]),
        ("run_dir", [t.run_dir for t in targets]),
        ("validation_pickle_path", [t.validation_pickle_path for t in targets]),
    ):
        duplicates = sorted({v for v in values if values.count(v) > 1})
        _require(
            not duplicates,
            f"{path}: duplicate {label} in the authenticated continuation roster: {duplicates} -- two "
            "roster entries resolve to the same trial",
        )

    observed_arm_counts: dict = {}
    for target in targets:
        observed_arm_counts[target.search_arm] = observed_arm_counts.get(target.search_arm, 0) + 1
    _require(
        observed_arm_counts == dict(expected_arm_counts),
        f"{path}: continuation phase {phase!r} requires {dict(expected_arm_counts)} but the authenticated "
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
        f"{path}: continuation phase {phase!r} authenticated receipts do not form the exact expected "
        f"proposal roster (missing={missing_proposals}, unexpected={unexpected_proposals}, "
        f"distinct={len(observed_proposals)}, trials={len(targets)})",
    )

    initial_controlled_trial_ids = tuple(
        t.trial_id for t in targets if (t.search_arm, t.proposal_order) in _INITIAL_CONTROLLED_PROPOSAL_SET
    )
    continuation_only_trial_ids = tuple(
        t.trial_id for t in targets if (t.search_arm, t.proposal_order) not in _INITIAL_CONTROLLED_PROPOSAL_SET
    )

    this_roster_initial_map = {
        t.trial_id: t.source_receipt_sha256
        for t in targets
        if (t.search_arm, t.proposal_order) in _INITIAL_CONTROLLED_PROPOSAL_SET
    }

    if phase == PHASE_INITIAL_CONTROLLED_24:
        # This roster *is* the initial slice: it is trivially bound to itself.
        initial_slice_identity = MappingProxyType(
            {
                "initial_roster_path": str(path),
                "initial_roster_sha256": trial_list_sha256,
                "source_receipt_sha256_by_trial_id": MappingProxyType(dict(this_roster_initial_map)),
            }
        )
    else:
        initial_path = Path(initial_roster_path)
        try:
            authenticated_initial = authenticate_trial_roster(
                trial_list_path=initial_path,
                contract=contract,
                verify_validation_pickles=verify_validation_pickles,
            )
        except TrialAuthenticationError as exc:
            raise ContinuationAuthenticationError(
                f"{initial_path}: initial-24 roster supplied to bind continuation phase {phase!r} against "
                f"is not receipt-qualified: {exc}"
            ) from exc

        _require(
            authenticated_initial.trial_list_sha256 == _AUTHORITATIVE_INITIAL_ROSTER_SHA256,
            f"{initial_path}: initial-24 roster raw-byte SHA-256 {authenticated_initial.trial_list_sha256!r} "
            f"does not match the immutable authoritative RD1-C4 initial roster SHA-256 "
            f"{_AUTHORITATIVE_INITIAL_ROSTER_SHA256!r} -- refusing to bind continuation phase {phase!r} to an "
            "alternate initial roster, even one that is itself fully receipt-valid",
        )

        frozen_initial_map = dict(authenticated_initial.identity_fields()["source_receipt_sha256_by_trial_id"])

        missing_trial_ids = sorted(set(frozen_initial_map) - set(this_roster_initial_map))
        unexpected_trial_ids = sorted(set(this_roster_initial_map) - set(frozen_initial_map))
        _require(
            not missing_trial_ids and not unexpected_trial_ids,
            f"{path}: continuation roster's initial-24 subset does not match the authenticated frozen "
            f"initial roster {initial_path} by trial_id (missing={missing_trial_ids}, "
            f"unexpected={unexpected_trial_ids}) -- refusing to bind an altered initial slice",
        )
        mismatched_receipt_sha256 = sorted(
            trial_id
            for trial_id in frozen_initial_map
            if frozen_initial_map[trial_id] != this_roster_initial_map[trial_id]
        )
        _require(
            not mismatched_receipt_sha256,
            f"{path}: continuation roster's initial-24 subset claims a different source_receipt_sha256 than "
            f"the authenticated frozen initial roster {initial_path} for trial_id(s) "
            f"{mismatched_receipt_sha256} -- refusing to bind an altered initial slice",
        )

        initial_slice_identity = MappingProxyType(
            {
                "initial_roster_path": str(initial_path),
                "initial_roster_sha256": authenticated_initial.trial_list_sha256,
                "source_receipt_sha256_by_trial_id": MappingProxyType(dict(frozen_initial_map)),
            }
        )

    return AuthenticatedContinuationRoster(
        schema_name=CONTINUATION_ROSTER_SCHEMA_NAME,
        schema_version=CONTINUATION_ROSTER_SCHEMA_VERSION,
        campaign_id=campaign_id,
        phase=phase,
        support_contract_version=contract["contract_id"],
        support_contract_sha256=contract["checksum_sha256"],
        trial_list_path=str(path),
        trial_list_sha256=trial_list_sha256,
        period=contract["period"],
        validation_pickles_verified=bool(verify_validation_pickles),
        targets=targets,
        initial_controlled_trial_ids=initial_controlled_trial_ids,
        continuation_only_trial_ids=continuation_only_trial_ids,
        initial_slice_identity=initial_slice_identity,
    )
