#!/usr/bin/env python
"""Per-proposal driver for the RD1 P20-P24 fixed Slurm chain.

Replaces the retired self-propagating controller
(``.scratch_local/rd1_continuation_chain_20260925T123812Z/controller/
rd1_continuation_controller.py`` on Moriah). The structural fix is "no job
submits another job": this driver runs once, inside one already-submitted
Slurm job, and never calls ``sbatch``. The fixed ``afterok`` dependency
chain across P20-P24 is created once, up front, by
``scripts/submit_rd1_p20_p24_chain.sh`` from a clean login shell.

This module is split into pure, host-independent logic (importable and
tested without Slurm, W&B, or a POSIX filesystem) and a thin ``main()`` CLI
that performs the actual subprocess/W&B/registry I/O. The pure functions
carry every behavior this replacement was specifically required to get
right (contiguity/hard-stop gating, socket-failure retry eligibility
wiring); the CLI wiring is intentionally thin.

Correction (Phase A.1 review, 2026-09-27): ``run_agent_with_retry`` below was
present and fully tested but never wired to any CLI subcommand -- the actual
sbatch script ran its own, separate, weaker bash retry loop instead (it
never captured a "before" run-ids snapshot, so it could not detect that a
retried attempt might have already created a run). This file now adds the
one ``run-agent-with-retry`` subcommand that is the actual execution path's
only retry-eligibility implementation, delegating entirely to the
already-tested ``run_agent_with_retry``/``is_socket_failure_retry_eligible``
functions below. ``rd1_p20_p24_job.sbatch`` calls this subcommand instead of
running its own loop.

Reuses rather than reimplements:

* ``scripts/run_sweep_v2_six_axis_wandb_agent_moriah.sbatch`` for manifest
  validation, WANDB_PROJECT/ENTITY resolution, ``.netrc``-based
  ``WANDB_API_KEY`` resolution, and the single ``wandb agent --count 1``
  invocation (invoked here as a plain subprocess, never via ``sbatch``);
* ``scripts/create_sweep_v2_six_axis_wandb_bridge_production_sweep.py
  build-manifest`` for building each of P21-P24's own manifest, after its
  predecessor has already passed (enforced by the Slurm ``afterok``
  dependency itself -- this driver never runs early);
* :mod:`src.baseline.rd1_p20_p24_registry` for the lock/append/verify
  contract;
* :mod:`src.baseline.rd1_p20_p24_env` for environment sanitization;
* :mod:`src.baseline.rd1_p20_p24_retry` for bounded backoff and the
  socket-failure retry-eligibility gate.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.rd1_p20_p24_env import sanitize_environment  # noqa: E402
from src.baseline.rd1_p20_p24_registry import (  # noqa: E402
    RegistryError,
    RegistryLock,
    append_row_verified,
    read_registry,
    registry_orders,
)
from src.baseline.rd1_p20_p24_retry import (  # noqa: E402
    RetryAuditError,
    fetch_with_backoff,
    is_socket_failure_retry_eligible,
    no_new_run_appeared,
)
from src.baseline.sweep_v2_six_axis_campaign import (  # noqa: E402
    CAMPAIGN_ID_V2,
    DOMAIN_VERSION_V2,
    OBJECTIVE_ID_V2,
    SweepV2CampaignError,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
)
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import (  # noqa: E402
    load_v2_wandb_bridge_manifest,
)

__all__ = [
    "FIRST_PROPOSAL_ORDER",
    "HARD_STOP_ORDER",
    "V2_METRIC_NAME",
    "PreflightError",
    "ManifestIdentityError",
    "RunAcceptanceError",
    "AgentAttemptResult",
    "compute_expected_prior_orders",
    "check_preflight",
    "verify_pinned_manifest_checksum",
    "run_agent_with_retry",
    "redact_secrets",
    "find_local_execution_provenance",
    "find_local_execution_provenance_by_order",
    "validate_local_provenance_record",
    "validate_terminal_wandb_run",
    "validate_run_for_registry_acceptance",
    "resolve_run_id_from_local_provenance",
    "main",
]

FIRST_PROPOSAL_ORDER = 1
HARD_STOP_ORDER = 24

# Derived locally, never imported from sweep_v2_six_axis_config: that module
# pulls in sweep_v2_six_axis_execution's full NH/torch training dependency
# chain (pilot_orchestration, nh_config_generation, fixed_support_contract_v2,
# ...), which this deliberately narrow, orchestration-only deployed harness
# bundle must not need. The formula is identical to
# sweep_v2_six_axis_config.V2_METRIC_NAME (f"flashnh/{OBJECTIVE_ID_V2}"),
# and OBJECTIVE_ID_V2 itself is already an existing deployed dependency.
V2_METRIC_NAME = f"flashnh/{OBJECTIVE_ID_V2}"


class PreflightError(Exception):
    """Any reason a proposal's job must refuse to proceed before touching
    W&B or the registry. Carries a machine-readable reason code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class ManifestIdentityError(Exception):
    """P20's manifest failed the byte-for-byte checksum-pinned reuse check."""


class RunAcceptanceError(Exception):
    """Any reason a completed W&B run must be refused RD1 P20-P24 registry
    acceptance. Mirrors :class:`PreflightError`'s reason/detail pattern.
    This is the sole gate ``_cmd_append_registry_row`` trusts -- a run may
    be appended to the registry only after every check this exception type
    can raise has passed."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


_SECRET_KEYVALUE_PATTERN = re.compile(
    r"(?i)\b(wandb_api_key|api[_-]?key|password|token|secret)\b(\s*[:=]\s*)(\S+)"
)
_SECRET_BEARER_PATTERN = re.compile(r"(?i)(authorization:\s*bearer\s+)(\S+)")
_SECRET_NETRC_PATTERN = re.compile(
    r"(?i)(machine\s+\S+\s+login\s+\S+\s+password\s+)(\S+)"
)
_SECRET_BARE_TOKEN_PATTERN = re.compile(r"\b[0-9a-fA-F]{32,64}\b")


def redact_secrets(text: str) -> str:
    """Redact every credential-shaped substring from ``text`` before it is
    ever persisted as attempt evidence. Deliberately fails toward
    over-redaction, never under-redaction: covers ``key=value``/``key:
    value`` forms for common credential-bearing names (api key, password,
    token, secret), ``Authorization: Bearer`` headers, ``.netrc``-style
    ``machine ... login ... password ...`` lines, and -- independent of any
    surrounding keyword -- any bare 32-64 hex-character run (the shape of a
    W&B API key), so a credential is still caught even if it appears next
    to an unanticipated label."""
    redacted = _SECRET_KEYVALUE_PATTERN.sub(r"\1\2***REDACTED***", text)
    redacted = _SECRET_BEARER_PATTERN.sub(r"\1***REDACTED***", redacted)
    redacted = _SECRET_NETRC_PATTERN.sub(r"\1***REDACTED***", redacted)
    redacted = _SECRET_BARE_TOKEN_PATTERN.sub("***REDACTED***", redacted)
    return redacted


def compute_expected_prior_orders(order: int) -> list[int]:
    """The full contiguous order set that must already be registered
    before ``order`` may run: every order from :data:`FIRST_PROPOSAL_ORDER`
    up to (not including) ``order``."""
    return list(range(FIRST_PROPOSAL_ORDER, order))


def check_preflight(
    order: int,
    current_orders: Iterable[int],
    *,
    hard_stop_order: int = HARD_STOP_ORDER,
) -> None:
    """Raise :class:`PreflightError` unless ``order`` is exactly the next
    proposal the registry is ready for. Refuses (in order): any order past
    the hard stop, a non-positive order, an order already registered, and
    an order whose full prior contiguous history is not exactly present
    (catches both "too early" and any drifted/ambiguous registry state).
    Never mutates anything -- callers hold a shared/exclusive
    :class:`~src.baseline.rd1_p20_p24_registry.RegistryLock` around the
    read that produced ``current_orders``.
    """
    if order > hard_stop_order:
        raise PreflightError("hard_stop_exceeded", f"order {order} > hard stop {hard_stop_order}")
    if order < FIRST_PROPOSAL_ORDER:
        raise PreflightError("invalid_order", str(order))

    current = sorted(int(o) for o in current_orders)
    if order in current:
        raise PreflightError("duplicate_order_already_registered", str(order))

    expected_prior = compute_expected_prior_orders(order)
    if current != expected_prior:
        raise PreflightError(
            "registry_not_contiguous_for_order",
            f"current={current} expected_prior={expected_prior}",
        )


def verify_pinned_manifest_checksum(manifest_path: "str | Path", expected_sha256: str) -> None:
    """P20 reuses its already-validated manifest byte-for-byte (design
    requirement: no rebuild). Raise :class:`ManifestIdentityError` if the
    file's sha256 does not exactly match ``expected_sha256``; never
    silently proceeds with a manifest that merely looks similar."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ManifestIdentityError(f"pinned manifest not found: {manifest_path}")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ManifestIdentityError(
            f"pinned manifest checksum mismatch: got {digest} expected {expected_sha256}"
        )


@dataclasses.dataclass(frozen=True)
class AgentAttemptResult:
    success: bool
    returncode: int
    output_text: str
    run_id: "str | None"


def run_agent_with_retry(
    *,
    attempt_fn: "Callable[[], AgentAttemptResult]",
    audit_run_ids_fn: "Callable[[], Iterable[str]]",
    max_attempts: int = 2,
    on_attempt: "Callable[[int, AgentAttemptResult], None] | None" = None,
) -> AgentAttemptResult:
    """Run ``attempt_fn`` (one ``wandb agent --count 1`` invocation) and, on
    failure, retry at most once -- and only for the exact pre-run W&B
    service-socket failure signature, only after a bounded read-only W&B
    audit independently confirms the failed attempt created no run. Any
    other failure is returned as-is (fail-closed: no retry).

    ``audit_run_ids_fn`` is a read-only sweep-runs listing call, wrapped by
    the caller-supplied function itself in nothing extra -- backoff is
    applied here via :func:`~src.baseline.rd1_p20_p24_retry.fetch_with_backoff`.
    If every backoff attempt raises, :class:`~src.baseline.rd1_p20_p24_retry.
    RetryAuditError` propagates to the caller uncaught: an unavailable or
    ambiguous audit must fail closed (no retry decision made at all), never
    be silently treated as "no new run".

    ``on_attempt``, when given, is called with ``(attempt_index, result)``
    immediately after every single ``attempt_fn()`` call -- success or
    failure, retried or not, and before any subsequent audit call that
    could itself raise. This is purely additive evidence persistence (the
    root-cause fix for silently discarding launcher output on an exit-0
    subprocess whose dispatched W&B run had already crashed): it never
    influences the retry decision itself, which remains exactly as before
    when ``on_attempt`` is ``None``.
    """
    run_ids_before = list(fetch_with_backoff(audit_run_ids_fn))
    attempt_index = 0
    result: "AgentAttemptResult | None" = None
    while True:
        attempt_index += 1
        result = attempt_fn()
        if on_attempt is not None:
            on_attempt(attempt_index, result)
        if result.success or attempt_index >= max_attempts:
            return result

        run_ids_after = list(fetch_with_backoff(audit_run_ids_fn))
        no_new = no_new_run_appeared(run_ids_before, run_ids_after)
        eligible, _reason = is_socket_failure_retry_eligible(
            attempt_index=attempt_index,
            max_attempts=max_attempts,
            attempt_output_text=result.output_text,
            run_id_resolved=result.run_id,
            no_new_run_confirmed=no_new,
        )
        if not eligible:
            return result
        run_ids_before = run_ids_after


def _subprocess_attempt_fn(agent_launcher_path: Path, env: dict) -> AgentAttemptResult:
    """One ``bash <agent_launcher_path>`` subprocess attempt, run with
    exactly the caller-supplied ``env`` (no inheritance from this process'
    own environment -- callers pass a narrow, explicit allowlist, mirroring
    the ``env -i`` pattern previously used directly in the sbatch script).

    Never parses a run id out of launcher stdout/stderr: run-id resolution
    is a separate, explicit step performed by the CLI wiring (see
    ``resolve_run_id_from_local_provenance``), not text-scraped from the
    launcher's own output.
    """
    proc = subprocess.run(
        ["bash", str(agent_launcher_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output_text = (proc.stdout or "") + (proc.stderr or "")
    return AgentAttemptResult(
        success=(proc.returncode == 0),
        returncode=proc.returncode,
        output_text=output_text,
        run_id=None,
    )


def _resolve_canonical_sweep_path(manifest_path: "str | Path", expected_sweep_id: str) -> str:
    """Build the ``entity/project/sweep_id`` path ``wandb.Api().sweep``
    actually requires. A bare sweep id resolves against whatever
    entity/project happen to be ambient defaults for the calling account,
    which this chain must never rely on -- this is exactly how a prior
    production attempt failed with ``Invalid path: 'wta85z3b' (missing
    project)``. Reuses the same loader-validated manifest that is already
    the sole launch identity everywhere else in this chain (never a second,
    parallel source for entity/project). Fails closed -- raises instead of
    ever returning a bare or partial path -- if the manifest's own sweep id
    disagrees with ``expected_sweep_id``, or if it carries no non-empty
    project/entity."""
    manifest = load_v2_wandb_bridge_manifest(manifest_path)
    manifest_sweep_id = manifest["wandb_sweep_id"]
    if manifest_sweep_id != expected_sweep_id:
        raise RuntimeError(
            f"manifest wandb_sweep_id {manifest_sweep_id!r} disagrees with expected "
            f"{expected_sweep_id!r}; refusing to guess a sweep path"
        )
    project = manifest.get("wandb_project")
    entity = manifest.get("wandb_entity")
    if not isinstance(project, str) or not project or not isinstance(entity, str) or not entity:
        raise RuntimeError(
            f"manifest carries no usable non-empty wandb_project/wandb_entity "
            f"(project={project!r} entity={entity!r}); refusing to contact W&B with "
            "an ambiguous sweep path"
        )
    return f"{entity}/{project}/{expected_sweep_id}"


def _resolve_canonical_run_path(manifest_path: "str | Path", run_id: str) -> str:
    """Build the ``entity/project/run_id`` path ``wandb.Api().run`` actually
    requires -- the run-path sibling of ``_resolve_canonical_sweep_path``
    (a W&B run path never includes the sweep id, unlike a sweep path).
    Reuses the same loader-validated manifest as the sole source of
    entity/project everywhere else in this chain."""
    manifest = load_v2_wandb_bridge_manifest(manifest_path)
    project = manifest.get("wandb_project")
    entity = manifest.get("wandb_entity")
    if not isinstance(project, str) or not project or not isinstance(entity, str) or not entity:
        raise RuntimeError(
            f"manifest carries no usable non-empty wandb_project/wandb_entity "
            f"(project={project!r} entity={entity!r}); refusing to contact W&B with "
            "an ambiguous run path"
        )
    return f"{entity}/{project}/{run_id}"


def _wandb_sweep_run_ids(manifest_path: "str | Path", sweep_id: str) -> list[str]:
    """Read-only listing of every run id currently in ``sweep_id``. Imports
    ``wandb`` lazily so importing this module for the pure-function unit
    tests never requires the ``wandb`` package to be installed."""
    sweep_path = _resolve_canonical_sweep_path(manifest_path, sweep_id)
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    return sorted(run.id for run in sweep.runs)


def find_local_execution_provenance(
    output_root: "str | Path",
    *,
    expected_order: int,
    expected_run_id: str,
) -> Path:
    """Locate this proposal's ``execution_provenance.json`` beneath
    ``output_root`` (the manifest's own ``output_root``, the documented
    one-directory-per-trial layout: ``<output_root>/<trial_id>/
    execution_provenance.json``) without assuming or reconstructing any
    particular trial-id naming scheme: search the immediate subdirectories
    once, and require exactly one candidate whose own ``proposal_order``/
    ``wandb_run_id`` fields match what the caller independently expects.

    Fails closed -- raises :class:`RunAcceptanceError` -- on a missing
    output root, zero matches, more than one match, or a candidate that is
    unreadable/not valid JSON; never silently guesses "the newest one"."""
    root = Path(output_root)
    if not root.is_dir():
        raise RunAcceptanceError("output_root_missing", str(root))

    matches: list[Path] = []
    for candidate in sorted(root.glob("*/execution_provenance.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("proposal_order") == expected_order
            and payload.get("wandb_run_id") == expected_run_id
        ):
            matches.append(candidate)

    if not matches:
        raise RunAcceptanceError(
            "local_execution_provenance_not_found",
            f"no execution_provenance.json beneath {root} matches "
            f"order={expected_order} run_id={expected_run_id}",
        )
    if len(matches) > 1:
        raise RunAcceptanceError(
            "local_execution_provenance_ambiguous",
            f"{len(matches)} candidates matched order={expected_order} "
            f"run_id={expected_run_id}: {[str(m) for m in matches]}",
        )
    return matches[0]


def find_local_execution_provenance_by_order(
    output_root: "str | Path",
    *,
    expected_order: int,
) -> Path:
    """Order-only sibling of :func:`find_local_execution_provenance`, used
    solely to resolve which run id this proposal's own attempt actually
    created -- before any run id is known, so it cannot yet be part of the
    match criteria. This is the fix for the root cause of a mis-registered
    run (2026-09-30): the prior resolution path
    (``_wandb_resolve_latest_run_id``, now removed) picked the sweep-wide
    "most recently created" run, which silently selects a *different*,
    later-created, possibly-crashed proposal's run whenever one exists in
    the same sweep -- there was no requirement that the "latest" run be
    the one this attempt itself produced.

    A run id resolved via this order-only match is never trusted on its
    own: the caller (:func:`resolve_run_id_from_local_provenance`) runs it
    through the full, order+run_id-matching acceptance gate
    (:func:`_run_registry_acceptance_core`) before returning it.

    Fails closed -- raises :class:`RunAcceptanceError` -- on a missing
    output root, zero matches, or more than one candidate for
    ``expected_order``; never guesses "the newest one"."""
    root = Path(output_root)
    if not root.is_dir():
        raise RunAcceptanceError("output_root_missing", str(root))

    matches: list[Path] = []
    for candidate in sorted(root.glob("*/execution_provenance.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("proposal_order") == expected_order:
            matches.append(candidate)

    if not matches:
        raise RunAcceptanceError(
            "local_execution_provenance_not_found",
            f"no execution_provenance.json beneath {root} matches order={expected_order}",
        )
    if len(matches) > 1:
        raise RunAcceptanceError(
            "local_execution_provenance_ambiguous",
            f"{len(matches)} candidates matched order={expected_order}: {[str(m) for m in matches]}",
        )
    return matches[0]


_LOCAL_PROVENANCE_REQUIRED_FIELDS = (
    "hyperparameters",
    "search_arm",
    "proposal_order",
    "execution_generation",
    "configuration_id",
    "proposal_id",
    "trial_id",
    "campaign_id",
    "domain_version",
    "wandb_sweep_id",
    "wandb_run_id",
    "support_contract_version",
    "support_contract_sha256",
    "execution_status",
    "objective_eligible",
    "fixed_support_metric_name",
    "objective_score",
)


def validate_local_provenance_record(
    provenance_path: "str | Path",
    *,
    expected_order: int,
    expected_sweep_id: str,
    expected_run_id: str,
) -> float:
    """Load and validate one local ``execution_provenance.json`` record.

    Mirrors -- rather than imports -- the identity re-derivation contract of
    ``src.baseline.sweep_v2_six_axis_retry.load_frozen_proposal_record_v2``/
    ``assert_matches_pinned_identity_v2``. Deliberately not imported:
    ``sweep_v2_six_axis_retry.py`` and its dependency ``sweep_v1_retry.py``
    were both added to this repository by commits landing AFTER
    ``FROZEN_SCIENTIFIC_COMMIT`` (verified: neither file exists in that
    commit's tree) that are not on the reviewed
    ``APPROVED_OPERATIONAL_BASE_COMMITS`` allowlist in
    ``scripts/deploy_rd1_p20_p24_harness.py`` -- deploying either module
    into this harness would silently require widening that provenance
    allowlist, which this recovery task's scope does not authorize (CLAUDE.md
    S9: a provenance-safeguard change is an escalation, not an autonomous
    repair). Every identity value re-derived below instead uses only
    ``src.baseline.sweep_v2_six_axis_campaign`` (and its own dependency
    ``sweep_v1_campaign``), both confirmed byte-for-byte unchanged since
    ``FROZEN_SCIENTIFIC_COMMIT`` and already part of this harness's deployed
    bundle -- so this validation needs no new deployment surface at all.

    Never trusts any persisted identity field at face value: every
    identity-bearing field is either independently re-derived from the
    record's own raw persisted fields via the canonical v2 helpers
    (``configuration_id_v2``/``proposal_id_v2``/``trial_id_v2``), or
    cross-checked against what this CLI invocation independently already
    knows (the proposal order it was launched for, and the W&B sweep/run id
    the agent actually resolved).

    Returns the locally recorded objective value, for the caller to
    cross-check against W&B's own published summary. Fails closed -- raises
    :class:`RunAcceptanceError` -- on any read, schema, re-derivation,
    identity, terminal-state, or finiteness problem."""
    path = Path(provenance_path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunAcceptanceError("local_provenance_record_unreadable", f"{path}: {exc}") from exc
    if not isinstance(record, dict):
        raise RunAcceptanceError(
            "local_provenance_record_malformed", f"{path}: top level is not a JSON object"
        )

    missing = [field for field in _LOCAL_PROVENANCE_REQUIRED_FIELDS if field not in record]
    if missing:
        raise RunAcceptanceError("local_provenance_record_missing_fields", f"{path}: missing {missing}")

    if record["campaign_id"] != CAMPAIGN_ID_V2 or record["domain_version"] != DOMAIN_VERSION_V2:
        raise RunAcceptanceError(
            "local_provenance_campaign_identity_mismatch",
            f"campaign_id={record['campaign_id']!r} domain_version={record['domain_version']!r} "
            f"expected campaign_id={CAMPAIGN_ID_V2!r} domain_version={DOMAIN_VERSION_V2!r}",
        )

    try:
        re_derived_configuration_id = configuration_id_v2(
            record["hyperparameters"],
            support_contract_version=record["support_contract_version"],
            support_contract_sha256=record["support_contract_sha256"],
        )
        re_derived_proposal_id = proposal_id_v2(record["search_arm"], record["proposal_order"])
        re_derived_trial_id = trial_id_v2(
            re_derived_configuration_id,
            re_derived_proposal_id,
            execution_generation=record["execution_generation"],
        )
    except SweepV2CampaignError as exc:
        raise RunAcceptanceError("local_provenance_re_derivation_failed", str(exc)) from exc

    mismatches: dict[str, tuple[object, object]] = {}
    if re_derived_configuration_id != record["configuration_id"]:
        mismatches["configuration_id"] = (re_derived_configuration_id, record["configuration_id"])
    if re_derived_proposal_id != record["proposal_id"]:
        mismatches["proposal_id"] = (re_derived_proposal_id, record["proposal_id"])
    if re_derived_trial_id != record["trial_id"]:
        mismatches["trial_id"] = (re_derived_trial_id, record["trial_id"])
    if record["proposal_order"] != expected_order:
        mismatches["proposal_order"] = (expected_order, record["proposal_order"])
    if record["wandb_sweep_id"] != expected_sweep_id:
        mismatches["wandb_sweep_id"] = (expected_sweep_id, record["wandb_sweep_id"])
    if record["wandb_run_id"] != expected_run_id:
        mismatches["wandb_run_id"] = (expected_run_id, record["wandb_run_id"])
    if mismatches:
        raise RunAcceptanceError("local_provenance_identity_mismatch", f"{path}: {mismatches}")

    if record["execution_status"] != "VALID":
        raise RunAcceptanceError(
            "local_provenance_not_valid", f"execution_status={record['execution_status']!r}"
        )
    if record["objective_eligible"] is not True:
        raise RunAcceptanceError(
            "local_provenance_objective_ineligible",
            f"objective_eligible={record['objective_eligible']!r}",
        )
    if record["fixed_support_metric_name"] != V2_METRIC_NAME:
        raise RunAcceptanceError(
            "local_provenance_metric_name_mismatch",
            f"fixed_support_metric_name={record['fixed_support_metric_name']!r} "
            f"expected={V2_METRIC_NAME!r}",
        )

    objective = record["objective_score"]
    try:
        objective_value = float(objective)
    except (TypeError, ValueError):
        raise RunAcceptanceError(
            "local_provenance_objective_not_numeric", f"objective_score={objective!r}"
        )
    if not math.isfinite(objective_value):
        raise RunAcceptanceError(
            "local_provenance_objective_not_finite", f"objective_score={objective_value!r}"
        )
    return objective_value


def validate_terminal_wandb_run(
    manifest_path: "str | Path",
    run_id: str,
    *,
    sleep_fn: "Callable[[float], None]" = time.sleep,
) -> float:
    """Independently confirm, via a fresh read-only W&B API call, that
    ``run_id`` reached a genuinely terminal, scientifically valid state --
    never trusted from local files or subprocess exit codes alone (the
    root-cause-#2 defect this validation gate exists to close). Returns the
    published objective value for the caller to cross-check against the
    local ``execution_provenance.json`` record. Fails closed -- raises
    :class:`RunAcceptanceError` -- on any non-``finished`` state, a
    missing/false ``flashnh/valid`` flag, a missing or non-finite objective
    metric, or a W&B API error that persists across
    :func:`~src.baseline.rd1_p20_p24_retry.fetch_with_backoff`'s bounded
    retries (root-cause-#3 fix, 2026-09-30: this lookup was a single
    transient-W&B-API-error hazard, now protected by ``fetch_with_backoff``
    like every other W&B read in this module). Imports ``wandb`` lazily so
    importing this module for the pure-function unit tests never requires
    the ``wandb`` package to be installed. ``sleep_fn`` defaults to real
    ``time.sleep`` in production; tests inject a no-op to stay fast."""
    run_path = _resolve_canonical_run_path(manifest_path, run_id)
    import wandb  # noqa: PLC0415

    def _fetch_run():
        return wandb.Api().run(run_path)

    try:
        run = fetch_with_backoff(_fetch_run, sleep_fn=sleep_fn)
    except Exception as exc:  # noqa: BLE001 -- any lookup failure is fatal, never silently skipped
        raise RunAcceptanceError("wandb_run_lookup_failed", f"{run_path}: {exc}") from exc

    if run.state != "finished":
        raise RunAcceptanceError("wandb_run_not_finished", f"{run_path} state={run.state!r}")

    summary = dict(run.summary)
    if summary.get("flashnh/valid") is not True:
        raise RunAcceptanceError(
            "wandb_run_not_flagged_valid",
            f"{run_path} flashnh/valid={summary.get('flashnh/valid')!r}",
        )

    objective = summary.get(V2_METRIC_NAME)
    try:
        objective_value = float(objective)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RunAcceptanceError(
            "wandb_objective_missing_or_not_numeric",
            f"{run_path} {V2_METRIC_NAME}={objective!r}",
        )
    if not math.isfinite(objective_value):
        raise RunAcceptanceError(
            "wandb_objective_not_finite", f"{run_path} {V2_METRIC_NAME}={objective_value!r}"
        )
    return objective_value


_OBJECTIVE_CROSS_CHECK_ABS_TOL = 1e-9


def _run_registry_acceptance_core(
    *,
    manifest_path: Path,
    manifest: dict,
    wandb_sweep_id: str,
    run_id: str,
    order: int,
) -> dict[str, Any]:
    """Shared acceptance-check body behind both
    :func:`validate_run_for_registry_acceptance` and
    :func:`resolve_run_id_from_local_provenance`: exactly one local
    ``execution_provenance.json`` matching ``order``/``run_id`` that
    independently re-derives to a matching, ``VALID``, objective-eligible,
    finite-objective identity; a fresh W&B API lookup of the same run id
    independently confirming ``state == "finished"``, ``flashnh/valid is
    True``, and a finite published objective; and agreement between the
    two independently-obtained objective values. Assumes the caller has
    already checked ``manifest["wandb_sweep_id"] == wandb_sweep_id``.

    Returns ``{"objective": <float>, "manifest_sha256": <hex str>}``.
    Raises :class:`RunAcceptanceError` (fail-closed) on any disagreement."""
    output_root = manifest.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise RunAcceptanceError("manifest_output_root_missing", repr(output_root))

    provenance_path = find_local_execution_provenance(
        output_root, expected_order=order, expected_run_id=run_id
    )
    local_objective = validate_local_provenance_record(
        provenance_path,
        expected_order=order,
        expected_sweep_id=wandb_sweep_id,
        expected_run_id=run_id,
    )

    wandb_objective = validate_terminal_wandb_run(manifest_path, run_id)

    if not math.isclose(
        local_objective, wandb_objective, rel_tol=0.0, abs_tol=_OBJECTIVE_CROSS_CHECK_ABS_TOL
    ):
        raise RunAcceptanceError(
            "objective_cross_check_mismatch",
            f"local execution_provenance.json objective={local_objective!r} disagrees "
            f"with W&B summary objective={wandb_objective!r}",
        )

    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {"objective": local_objective, "manifest_sha256": manifest_sha256}


def validate_run_for_registry_acceptance(
    *,
    manifest_path: "str | Path",
    wandb_sweep_id: str,
    run_id: str,
    order: int,
) -> dict[str, Any]:
    """The full RD1 P20-P24 registry-acceptance gate. A production run may
    be accepted into the registry only if ALL of the following
    independently hold:

    * the manifest's own pinned sweep id agrees with ``wandb_sweep_id``;
    * exactly one local ``execution_provenance.json`` exists beneath the
      manifest's ``output_root`` for this exact order/run_id, and it
      independently re-derives to a matching, ``VALID``,
      objective-eligible, finite-objective identity
      (:func:`validate_local_provenance_record`);
    * a fresh W&B API lookup of the same run id independently confirms
      ``state == "finished"``, ``flashnh/valid is True``, and a finite
      published objective (:func:`validate_terminal_wandb_run`);
    * the two independently-obtained objective values agree.

    Returns ``{"objective": <float>, "manifest_sha256": <hex str>}`` for
    the caller to persist into the registry row. Raises
    :class:`RunAcceptanceError` (fail-closed) on any disagreement --
    ``_cmd_append_registry_row`` must never append a registry row on
    partial, ambiguous, or single-source evidence.
    """
    manifest_path = Path(manifest_path)
    manifest = load_v2_wandb_bridge_manifest(manifest_path)
    manifest_sweep_id = manifest["wandb_sweep_id"]
    if manifest_sweep_id != wandb_sweep_id:
        raise RunAcceptanceError(
            "manifest_sweep_id_mismatch",
            f"manifest wandb_sweep_id {manifest_sweep_id!r} disagrees with "
            f"expected {wandb_sweep_id!r}",
        )
    return _run_registry_acceptance_core(
        manifest_path=manifest_path,
        manifest=manifest,
        wandb_sweep_id=wandb_sweep_id,
        run_id=run_id,
        order=order,
    )


def resolve_run_id_from_local_provenance(
    *,
    manifest_path: "str | Path",
    wandb_sweep_id: str,
    order: int,
) -> str:
    """Resolve the run id this exact proposal (``order``) actually created,
    from this attempt's own local ``execution_provenance.json`` -- never by
    asking W&B for "whatever was most recently created in the sweep" (the
    removed ``_wandb_resolve_latest_run_id``, root cause of a mis-registered
    run on 2026-09-30: a sweep can contain another proposal's later-created,
    crashed run, which a sweep-wide "latest" lookup has no way to exclude).

    The resolved run id is never returned on local evidence alone: it is
    run through the full registry-acceptance gate
    (:func:`_run_registry_acceptance_core` -- identity re-derivation, a
    fresh W&B terminal-state/``flashnh/valid``/finite-objective check, and
    the local/W&B objective cross-check) before being returned, so a run id
    this function hands back has already cleared full acceptance --
    ``AGENT_OK run_id=...`` is only ever printed for a run that would also
    pass ``append-registry-row``.

    Fails closed -- raises :class:`RunAcceptanceError` -- if the manifest's
    pinned sweep id disagrees with ``wandb_sweep_id``, if local provenance
    for this order is missing or ambiguous, if that record carries no
    usable ``wandb_run_id``, or if any acceptance check subsequently fails.
    """
    manifest_path = Path(manifest_path)
    manifest = load_v2_wandb_bridge_manifest(manifest_path)
    manifest_sweep_id = manifest["wandb_sweep_id"]
    if manifest_sweep_id != wandb_sweep_id:
        raise RunAcceptanceError(
            "manifest_sweep_id_mismatch",
            f"manifest wandb_sweep_id {manifest_sweep_id!r} disagrees with "
            f"expected {wandb_sweep_id!r}",
        )
    output_root = manifest.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise RunAcceptanceError("manifest_output_root_missing", repr(output_root))

    provenance_path = find_local_execution_provenance_by_order(output_root, expected_order=order)
    try:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunAcceptanceError("local_provenance_record_unreadable", f"{provenance_path}: {exc}") from exc
    run_id = record.get("wandb_run_id") if isinstance(record, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise RunAcceptanceError(
            "local_provenance_run_id_missing", f"{provenance_path}: wandb_run_id={run_id!r}"
        )

    _run_registry_acceptance_core(
        manifest_path=manifest_path,
        manifest=manifest,
        wandb_sweep_id=wandb_sweep_id,
        run_id=run_id,
        order=order,
    )
    return run_id


def _cmd_preflight(args: argparse.Namespace) -> int:
    with RegistryLock(args.lock_path, exclusive=False):
        registry = read_registry(args.registry_path)
    current_orders = registry_orders(registry)
    try:
        check_preflight(args.order, current_orders, hard_stop_order=args.hard_stop_order)
    except PreflightError as exc:
        print(f"PREFLIGHT_REFUSED reason={exc.reason} detail={exc.detail}", file=sys.stderr)
        return 1
    print(f"PREFLIGHT_OK order={args.order} expected_prior_orders={current_orders}")
    return 0


def _cmd_append_registry_row(args: argparse.Namespace) -> int:
    try:
        acceptance = validate_run_for_registry_acceptance(
            manifest_path=args.manifest_path,
            wandb_sweep_id=args.wandb_sweep_id,
            run_id=args.run_id,
            order=args.order,
        )
    except RunAcceptanceError as exc:
        print(f"APPEND_REFUSED reason={exc.reason} detail={exc.detail}", file=sys.stderr)
        return 1

    row = {
        "order": args.order,
        "wandb_run_id": args.run_id,
        "recorded_at_utc": args.recorded_at_utc,
        "objective": acceptance["objective"],
        "manifest_sha256": acceptance["manifest_sha256"],
    }
    with RegistryLock(args.lock_path, exclusive=True):
        try:
            result = append_row_verified(
                args.registry_path,
                row,
                expected_prior_orders=compute_expected_prior_orders(args.order),
                backup_dir=args.backup_dir,
            )
        except RegistryError as exc:
            print(f"APPEND_REFUSED reason={exc.reason} detail={exc.detail}", file=sys.stderr)
            return 1
    print(f"APPEND_OK order={args.order} backup={result['registry_backup_path']}")
    return 0


def _cmd_verify_pinned_manifest(args: argparse.Namespace) -> int:
    try:
        verify_pinned_manifest_checksum(args.manifest_path, args.expected_sha256)
    except ManifestIdentityError as exc:
        print(f"MANIFEST_IDENTITY_REFUSED {exc}", file=sys.stderr)
        return 1
    print(f"MANIFEST_IDENTITY_OK {args.manifest_path}")
    return 0


def _cmd_sanitize_env_probe(args: argparse.Namespace) -> int:
    import os

    cleaned = sanitize_environment(os.environ)
    leaked = sorted(name for name in cleaned if name == "WANDB" or name.startswith("WANDB_"))
    if leaked:
        print(f"SANITIZE_FAILED leaked={leaked}", file=sys.stderr)
        return 1
    print("SANITIZE_OK no WANDB_* variable present after sanitization")
    return 0


def _persist_attempt_evidence(attempts_dir: Path, attempt_index: int, result: AgentAttemptResult) -> None:
    """Persist one agent-launcher attempt's redacted evidence beneath
    ``attempts_dir``, unconditionally -- this is the root-cause-#1 fix:
    called for every attempt regardless of exit code, so evidence survives
    even when the launcher subprocess exited 0 but the W&B run it dispatched
    had already crashed. Never writes raw, unredacted output -- ``run_id``
    is always ``None`` at this point (see ``_subprocess_attempt_fn``), so
    only ``output_text`` needs redaction."""
    attempts_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "attempt_index": attempt_index,
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "success": result.success,
        "returncode": result.returncode,
        "run_id": result.run_id,
        "output_text_redacted": redact_secrets(result.output_text),
    }
    evidence_path = attempts_dir / f"attempt_{attempt_index:02d}.json"
    evidence_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def _cmd_run_agent_with_retry(args: argparse.Namespace) -> int:
    """The one live retry-eligibility execution path. Builds the exact
    narrow env allowlist the agent launcher runs under, then delegates the
    entire attempt/retry decision to the tested ``run_agent_with_retry`` /
    ``is_socket_failure_retry_eligible`` functions above -- this function
    adds no additional retry logic of its own.

    ``REPO_WORKDIR`` is part of the allowlist (not just ``PATH``/``HOME``)
    because ``run_sweep_v2_six_axis_wandb_agent_moriah.sbatch`` falls back to
    its own hardcoded production default whenever ``REPO_WORKDIR`` is unset
    in its environment, and the bridge script then refuses on a
    ``repository_root`` mismatch if the caller actually meant somewhere else
    (e.g. a dual-provenance payload checkout).

    When ``--attempts-dir`` is given, every attempt's redacted evidence is
    persisted beneath it via ``on_attempt``, regardless of outcome -- purely
    additive; the ``AGENT_OK``/``AGENT_FAILED`` contract below is unchanged.

    On overall success, resolves the created run id from this attempt's own
    local execution-provenance record (:func:`resolve_run_id_from_local_
    provenance` -- order-scoped, never a sweep-wide "latest run" guess) and
    prints it for the sbatch wrapper to pass to ``append-registry-row``.
    Never appends to the registry itself (kept as a separate, existing
    subcommand/step).
    """
    attempt_env = {
        "PATH": args.path,
        "HOME": args.home,
        "FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST": args.manifest_path,
        "WANDB_SWEEP_ID": args.wandb_sweep_id,
        "REPO_WORKDIR": args.repo_workdir,
    }
    agent_launcher_path = Path(args.agent_launcher_path)

    def attempt_fn() -> AgentAttemptResult:
        return _subprocess_attempt_fn(agent_launcher_path, dict(attempt_env))

    def audit_run_ids_fn() -> list[str]:
        return _wandb_sweep_run_ids(args.manifest_path, args.wandb_sweep_id)

    on_attempt = None
    if args.attempts_dir:
        attempts_dir = Path(args.attempts_dir)

        def on_attempt(attempt_index: int, result: AgentAttemptResult) -> None:
            _persist_attempt_evidence(attempts_dir, attempt_index, result)

    try:
        result = run_agent_with_retry(
            attempt_fn=attempt_fn,
            audit_run_ids_fn=audit_run_ids_fn,
            max_attempts=args.max_attempts,
            on_attempt=on_attempt,
        )
    except RetryAuditError as exc:
        print(f"AGENT_FAILED reason=retry_audit_unavailable detail={exc}", file=sys.stderr)
        return 1

    if not result.success:
        print(
            f"AGENT_FAILED reason=agent_attempts_exhausted returncode={result.returncode}",
            file=sys.stderr,
        )
        return 1

    try:
        run_id = resolve_run_id_from_local_provenance(
            manifest_path=args.manifest_path,
            wandb_sweep_id=args.wandb_sweep_id,
            order=args.order,
        )
    except RunAcceptanceError as exc:
        print(
            f"AGENT_FAILED reason=run_id_resolution_failed detail={exc.reason}: {exc.detail}",
            file=sys.stderr,
        )
        return 1

    print(f"AGENT_OK run_id={run_id}")
    return 0


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_preflight = sub.add_parser("preflight", help="Refuse unless the registry is exactly ready for --order.")
    p_preflight.add_argument("--order", type=int, required=True)
    p_preflight.add_argument("--registry-path", required=True)
    p_preflight.add_argument("--lock-path", required=True)
    p_preflight.add_argument("--hard-stop-order", type=int, default=HARD_STOP_ORDER)
    p_preflight.set_defaults(func=_cmd_preflight)

    p_append = sub.add_parser("append-registry-row", help="Verified append of one completed proposal's row.")
    p_append.add_argument("--order", type=int, required=True)
    p_append.add_argument("--run-id", required=True)
    p_append.add_argument("--recorded-at-utc", required=True)
    p_append.add_argument("--registry-path", required=True)
    p_append.add_argument("--lock-path", required=True)
    p_append.add_argument("--backup-dir", required=True)
    p_append.add_argument(
        "--manifest-path", required=True, help="Manifest whose pinned sweep id/output_root gate acceptance."
    )
    p_append.add_argument(
        "--wandb-sweep-id", required=True, help="Expected sweep id, cross-checked against the manifest."
    )
    p_append.set_defaults(func=_cmd_append_registry_row)

    p_manifest = sub.add_parser("verify-pinned-manifest", help="P20-only: checksum-pin the reused manifest.")
    p_manifest.add_argument("--manifest-path", required=True)
    p_manifest.add_argument("--expected-sha256", required=True)
    p_manifest.set_defaults(func=_cmd_verify_pinned_manifest)

    p_sanitize = sub.add_parser("sanitize-env-probe", help="Fail if any WANDB_* var survives sanitization.")
    p_sanitize.set_defaults(func=_cmd_sanitize_env_probe)

    p_run_agent = sub.add_parser(
        "run-agent-with-retry",
        help="The one live retry-eligibility execution path: run the agent launcher, "
        "retrying at most once under the tested socket-failure eligibility gate.",
    )
    p_run_agent.add_argument("--agent-launcher-path", required=True)
    p_run_agent.add_argument("--manifest-path", required=True)
    p_run_agent.add_argument("--wandb-sweep-id", required=True)
    p_run_agent.add_argument(
        "--order",
        type=int,
        required=True,
        help="This attempt's own proposal order, used to resolve its run id from its own "
        "local execution-provenance record rather than a sweep-wide 'latest run' guess.",
    )
    p_run_agent.add_argument("--path", required=True, help="PATH value for the narrow attempt-env allowlist.")
    p_run_agent.add_argument("--home", required=True, help="HOME value for the narrow attempt-env allowlist.")
    p_run_agent.add_argument(
        "--repo-workdir",
        required=True,
        help="REPO_WORKDIR value for the narrow attempt-env allowlist (the agent-launcher "
        "sbatch script falls back to its own production default when unset, which is wrong "
        "whenever the caller's own REPO_WORKDIR points somewhere else, e.g. a dual-provenance "
        "payload checkout).",
    )
    p_run_agent.add_argument("--max-attempts", type=int, default=2)
    p_run_agent.add_argument(
        "--attempts-dir",
        default=None,
        help="Project-local dir to persist redacted per-attempt evidence into (optional).",
    )
    p_run_agent.set_defaults(func=_cmd_run_agent_with_retry)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
