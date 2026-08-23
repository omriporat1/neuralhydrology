"""QUALIFICATION ONLY -- NON-SCIENTIFIC.

Pure, wandb-free logic shared by the Phase-B online W&B sweep/agent
qualification scripts (``wandb_online_sweep_qualification_preflight.py``,
``_run.py``, ``_lifecycle.py``). Kept import-light (no real ``wandb``) so it
is trivially unit-testable and so the qualification harness's scientific
mapping can be reviewed independently of any network/backend behavior.

This module never trains anything, never touches scientific data, and never
mutates the frozen Sweep-v1 campaign inputs in
``src/baseline/sweep_v1_campaign.py`` -- it only *reuses* that module's
domain-legality/configuration-ID helpers (docs/
stage1_phase_b_sweep_v1_launch_contract.md section 12) to prove a toy W&B
proposal can be validated against the real frozen Sweep-v1 domain shape.
Toy proposals produced here are NEVER valid Bayesian or random-control
Sweep-v1 trials: ``QUALIFICATION_CAMPAIGN_ID`` is deliberately distinct from
``src.baseline.sweep_v1_campaign.CAMPAIGN_ID``, and nothing here writes to
any file/counter the real campaign reads.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from src.baseline.sweep_v1_campaign import (
    SEARCH_DOMAIN,
    canonical_hyperparameters,
    configuration_id as flashnh_configuration_id,
)

__all__ = [
    "QUALIFICATION_CAMPAIGN_ID",
    "TOY_METRIC_NAME",
    "SWEEP_NAME",
    "DEFAULT_PROJECT",
    "QUALIFICATION_TAGS",
    "HYPERPARAMETER_FIELDS",
    "compute_toy_objective",
    "build_sweep_config",
    "check_flashnh_legality",
    "build_run_identity",
]

# Deliberately NOT src.baseline.sweep_v1_campaign.CAMPAIGN_ID -- this
# identity must never be mistaken for the real frozen Sweep-v1 wave.
QUALIFICATION_CAMPAIGN_ID = "phase_b_wandb_online_sweep_qualification_v001"

# Deliberately outside the "screening/*" scientific metric namespace (see
# docs/stage1_wandb_user_guide.md section 4-5) so it can never be confused
# with "screening/primary_metric_median" or another real objective key.
TOY_METRIC_NAME = "qualification/toy_objective"

SWEEP_NAME = "phase_b_wandb_online_sweep_qualification_v001_toy_bayes"
DEFAULT_PROJECT = "flashnh-stage1"
QUALIFICATION_TAGS = ("qualification", "non_scientific", QUALIFICATION_CAMPAIGN_ID)

HYPERPARAMETER_FIELDS = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")


def compute_toy_objective(hyperparameters: Mapping[str, Any]) -> float:
    """Deterministic, cheap algebraic function of the five Sweep-v1 axes.

    No scientific data, no NH/torch import, runs in microseconds. Purely a
    numeric probe so a real W&B run has a real finite metric to receive --
    it carries no scientific meaning whatsoever.
    """
    missing = [k for k in HYPERPARAMETER_FIELDS if k not in hyperparameters]
    if missing:
        raise ValueError(f"toy objective requires all of {HYPERPARAMETER_FIELDS}, missing {missing}")
    lr = float(hyperparameters["learning_rate"])
    hidden = float(hyperparameters["hidden_size"])
    embedding_dropout = float(hyperparameters["embedding_dropout"])
    output_dropout = float(hyperparameters["output_dropout"])
    batch = float(hyperparameters["batch_size"])
    if lr <= 0:
        raise ValueError(f"learning_rate must be positive, got {lr!r}")

    log_lr_term = -abs(math.log10(lr) - math.log10(3e-4))
    hidden_term = 0.1 * (hidden / 256.0)
    dropout_term = -(embedding_dropout + output_dropout)
    batch_term = 0.05 * (batch / 512.0)
    objective = log_lr_term + hidden_term + dropout_term + batch_term
    if not math.isfinite(objective):
        raise ValueError(f"toy objective produced a non-finite value: {objective!r}")
    return objective


def build_sweep_config(sweep_name: str = SWEEP_NAME) -> dict:
    """W&B Bayesian sweep config exercising the frozen Sweep-v1 parameter
    TYPES/NAMES/DOMAINS (log-uniform continuous, ordinary-uniform continuous,
    categorical), read directly from ``sweep_v1_campaign.SEARCH_DOMAIN`` so
    this qualification's shape can never silently drift from the real
    frozen domain. The sweep's own name/metric never overlap real Sweep-v1
    identity/metric names.
    """
    lr_domain = SEARCH_DOMAIN["learning_rate"]
    hidden_domain = SEARCH_DOMAIN["hidden_size"]
    embedding_dropout_domain = SEARCH_DOMAIN["embedding_dropout"]
    output_dropout_domain = SEARCH_DOMAIN["output_dropout"]
    batch_domain = SEARCH_DOMAIN["batch_size"]
    return {
        "method": "bayes",
        "name": sweep_name,
        "metric": {"name": TOY_METRIC_NAME, "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": lr_domain["lower"],
                "max": lr_domain["upper"],
            },
            "hidden_size": {"values": list(hidden_domain["values"])},
            "embedding_dropout": {
                "distribution": "uniform",
                "min": embedding_dropout_domain["lower"],
                "max": embedding_dropout_domain["upper"],
            },
            "output_dropout": {
                "distribution": "uniform",
                "min": output_dropout_domain["lower"],
                "max": output_dropout_domain["upper"],
            },
            "batch_size": {"values": list(batch_domain["values"])},
        },
    }


def check_flashnh_legality(hyperparameters: Mapping[str, Any]) -> dict:
    """Pass a toy W&B proposal through the EXISTING frozen Sweep-v1
    domain-legality/configuration-ID helper (never a reimplementation). Never
    raises -- an illegal/malformed proposal is reported as
    ``legality_pass=False`` with an error string, not an exception, so the
    qualification harness can record and continue rather than crash on an
    edge-of-domain Bayesian proposal.
    """
    try:
        canonical = canonical_hyperparameters(dict(hyperparameters))
        config_id = flashnh_configuration_id(canonical)
        return {"legality_pass": True, "configuration_id": config_id, "canonical_hyperparameters": canonical, "error": None}
    except (ValueError, TypeError) as exc:
        return {"legality_pass": False, "configuration_id": None, "canonical_hyperparameters": None, "error": str(exc)}


def build_run_identity(proposal_label: str, *, extra: Mapping[str, Any] | None = None) -> dict:
    """Non-scientific run-identity metadata common to every toy run.

    Deliberately shaped so it can never be mistaken for a real Flash-NH
    pilot/campaign run identity: no ``pilot_policy_name``, ``run_id`` (in the
    real campaign sense), ``package_manifest_identity``, or similar fields
    exist here.
    """
    identity = {
        "qualification_kind": "wandb_online_sweep_qualification",
        "online_sweep_qualification": True,
        "qualification_campaign_id": QUALIFICATION_CAMPAIGN_ID,
        "proposal_label": proposal_label,
        "scientific_trial": False,
    }
    if extra:
        identity.update(extra)
    return identity
