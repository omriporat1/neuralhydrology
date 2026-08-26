"""v2 six-axis seq_length policy overlay loader/validator (additive, Section I).

Never broadens the v1 baseline policy's own meaning: it does not touch
``config/stage1_scientific_baseline_v001.yaml`` or ``policy.py``'s
validator, and it does not retroactively claim that 84h/96h/108h/120h were
approved under the v1 policy's 2K-G-H/2K-G-I sign-off. It only produces an
in-memory, v2-scoped copy of the fully-validated v1 policy with
``seq_lengths_hours`` replaced by the explicitly versioned six-axis overlay
domain, for callers that opt into the v2 campaign contract via
``build_pilot_bundle_with_validation_scope(policy_override=...)``.

Public API:
    load_stage1_baseline_policy_v2_six_axis(base_policy_path, overlay_path) -> dict
    validate_v2_six_axis_policy_overlay(overlay_data) -> dict
    PolicyOverlayError (subclass of ValueError)
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .policy import Stage1BaselinePolicyError, load_stage1_baseline_policy
from .sweep_v2_six_axis_campaign import SEQ_LENGTH_DOMAIN_V2

__all__ = [
    "PolicyOverlayError",
    "load_stage1_baseline_policy_v2_six_axis",
    "validate_v2_six_axis_policy_overlay",
]

_OVERLAY_NAME = "stage1_scientific_baseline_v2_six_axis_overlay_v001"
_OVERLAY_VERSION = 1
_BASE_POLICY_NAME = "stage1_scientific_baseline_v001"
_BASE_POLICY_VERSION = 2


class PolicyOverlayError(ValueError):
    """Raised when the v2 six-axis policy overlay is missing, invalid, or
    contradicts either its declared base policy or the v2 sweep's own
    seq_length domain."""


def validate_v2_six_axis_policy_overlay(overlay_data) -> dict:
    """Validate an in-memory overlay mapping. Returns it unmodified on
    success; never touches the filesystem."""
    if not isinstance(overlay_data, dict):
        raise PolicyOverlayError(f"overlay top level must be a mapping, got {type(overlay_data).__name__}")

    def _expect(key, expected):
        if key not in overlay_data:
            raise PolicyOverlayError(f"{key}: missing required key")
        value = overlay_data[key]
        if value != expected:
            raise PolicyOverlayError(f"{key}: must equal {expected!r}, got {value!r}")

    _expect("overlay_name", _OVERLAY_NAME)
    _expect("overlay_version", _OVERLAY_VERSION)
    _expect("base_policy_name", _BASE_POLICY_NAME)
    _expect("base_policy_version", _BASE_POLICY_VERSION)

    seq_lengths = overlay_data.get("seq_lengths_hours")
    if not isinstance(seq_lengths, list) or any(isinstance(v, bool) for v in seq_lengths):
        raise PolicyOverlayError(f"seq_lengths_hours: must be a plain list of ints, got {seq_lengths!r}")
    if tuple(seq_lengths) != SEQ_LENGTH_DOMAIN_V2:
        raise PolicyOverlayError(
            "seq_lengths_hours: must exactly equal sweep_v2_six_axis_campaign.SEQ_LENGTH_DOMAIN_V2 "
            f"{SEQ_LENGTH_DOMAIN_V2!r}, got {tuple(seq_lengths)!r} -- the policy overlay and the v2 "
            "sweep search domain must never diverge"
        )

    return overlay_data


def load_stage1_baseline_policy_v2_six_axis(base_policy_path, overlay_path) -> dict:
    """Load + fully validate the unmodified v1 baseline policy, load +
    validate the v2 six-axis overlay, cross-check overlay/base identity,
    and return a new in-memory dict: a shallow copy of the v1-validated
    policy with ``seq_lengths_hours`` replaced by the overlay's domain and
    an added ``policy_overlay`` provenance block. The returned dict is
    intended only for
    ``build_pilot_bundle_with_validation_scope(policy_override=...)`` --
    it is never written back to ``config/stage1_scientific_baseline_v001.yaml``
    and the v1 validator/file are never touched.
    """
    try:
        base_policy = load_stage1_baseline_policy(base_policy_path)
    except Stage1BaselinePolicyError as exc:
        raise PolicyOverlayError(f"base v1 policy failed validation: {exc}") from exc

    overlay_path = Path(overlay_path)
    if not overlay_path.is_file():
        raise PolicyOverlayError(f"v2 overlay file not found: {overlay_path}")
    text = overlay_path.read_text(encoding="utf-8")
    if not text.strip():
        raise PolicyOverlayError(f"v2 overlay file is empty: {overlay_path}")
    overlay_data = yaml.safe_load(text)
    validate_v2_six_axis_policy_overlay(overlay_data)

    if overlay_data["base_policy_name"] != base_policy["policy_name"]:
        raise PolicyOverlayError(
            f"overlay base_policy_name {overlay_data['base_policy_name']!r} does not match the "
            f"loaded base policy's policy_name {base_policy['policy_name']!r}"
        )
    if overlay_data["base_policy_version"] != base_policy["policy_version"]:
        raise PolicyOverlayError(
            f"overlay base_policy_version {overlay_data['base_policy_version']!r} does not match the "
            f"loaded base policy's policy_version {base_policy['policy_version']!r}"
        )

    merged = dict(base_policy)
    merged["seq_lengths_hours"] = list(SEQ_LENGTH_DOMAIN_V2)
    merged["policy_overlay"] = {
        "overlay_name": overlay_data["overlay_name"],
        "overlay_version": overlay_data["overlay_version"],
        "base_policy_name": overlay_data["base_policy_name"],
        "base_policy_version": overlay_data["base_policy_version"],
    }
    return merged
