"""Generate or audit the frozen offline Sweep-v2 six-axis IID random-control manifest.

Intentionally an offline preparation utility: it imports neither
NeuralHydrology nor W&B, never contacts the live Bayesian controller, and
never starts a run. The manifest was frozen AFTER Bayesian observation 1 and
is drawn only from the exact committed v2 six-axis priors; it is
scientifically independent of Proposal 1 but not a fully pre-outcome-frozen
prospective control (see docs/decision_log.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.baseline.sweep_v2_six_axis_random_control import (
    GENERATOR_DRAW_ORDER_V2,
    GENERATOR_RNG_IMPLEMENTATION_V2,
    GENERATOR_VERSION_V2,
    MANIFEST_RNG_NAMESPACE_V2,
    MANIFEST_RNG_SEED_V2,
    RANDOM_CONTROL_MANIFEST_SHA256_V2,
    SEQ_LENGTH_SAMPLER_V2,
    SUPPORT_CONTRACT_SHA256_V2,
    SUPPORT_CONTRACT_VERSION_V2,
    derive_manifest_rng_seed,
    generate_random_control_rows_v2,
    render_manifest_v2,
    sha256_bytes,
    validate_manifest_rows_v2,
)


def build_audit() -> dict:
    rows = generate_random_control_rows_v2()
    validate_manifest_rows_v2(rows)
    manifest_bytes = render_manifest_v2(rows)
    seed_derivation = derive_manifest_rng_seed()
    return {
        "audit_scope": "LOCAL_OFFLINE_STATIC_CAMPAIGN_FOUNDATION_ONLY",
        "seed_derivation": seed_derivation,
        "manifest_rng_namespace": MANIFEST_RNG_NAMESPACE_V2,
        "manifest_rng_seed": MANIFEST_RNG_SEED_V2,
        "generator_version": GENERATOR_VERSION_V2,
        "generator_rng_implementation": GENERATOR_RNG_IMPLEMENTATION_V2,
        "per_row_draw_order": list(GENERATOR_DRAW_ORDER_V2),
        "seq_length_sampler": SEQ_LENGTH_SAMPLER_V2,
        "support_contract_version": SUPPORT_CONTRACT_VERSION_V2,
        "support_contract_sha256": SUPPORT_CONTRACT_SHA256_V2,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "committed_manifest_sha256_expected": RANDOM_CONTROL_MANIFEST_SHA256_V2,
        "row_count": len(rows),
        "rows": rows,
        "hidden_size_counts": dict(sorted(Counter(row["hidden_size"] for row in rows).items())),
        "batch_size_counts": dict(sorted(Counter(row["batch_size"] for row in rows).items())),
        "seq_length_counts": dict(sorted(Counter(row["seq_length"] for row in rows).items())),
        "learning_rate_min": min(float(row["learning_rate"]) for row in rows),
        "learning_rate_max": max(float(row["learning_rate"]) for row in rows),
        "embedding_dropout_min": min(float(row["embedding_dropout"]) for row in rows),
        "embedding_dropout_max": max(float(row["embedding_dropout"]) for row in rows),
        "output_dropout_min": min(float(row["output_dropout"]) for row in rows),
        "output_dropout_max": max(float(row["output_dropout"]) for row in rows),
        "distinct_configuration_ids": sorted({row["configuration_id"] for row in rows}),
        "no_wandb": True,
        "no_training": True,
        "no_sealed_scope": True,
        "no_bayesian_controller_contact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", type=Path, help="Write deterministic canonical manifest bytes.")
    parser.add_argument("--write-sha256", type=Path, help="Write the '<sha>  <name>' sidecar for --write-manifest.")
    parser.add_argument("--audit-output", type=Path, required=True, help="Write compact local-only audit JSON.")
    parser.add_argument("--verify-manifest", type=Path, help="Require an existing manifest to equal deterministic bytes.")
    args = parser.parse_args()

    manifest_bytes = render_manifest_v2()
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_bytes(manifest_bytes)
        if args.write_sha256:
            args.write_sha256.write_text(
                f"{sha256_bytes(manifest_bytes)}  {args.write_manifest.name}\n", encoding="utf-8"
            )
    if args.verify_manifest and args.verify_manifest.read_bytes() != manifest_bytes:
        raise SystemExit("manifest bytes differ from deterministic frozen generation")

    audit = build_audit()
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("row_count", "manifest_rng_seed", "manifest_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
