"""Generate or audit the frozen offline Sweep-v1 IID random-control manifest.

This is intentionally an offline preparation utility: it imports neither
NeuralHydrology nor W&B and never starts a run.
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

from src.baseline.sweep_v1_campaign import (
    MANIFEST_RNG_SEED,
    RANDOM_CONTROL_MANIFEST_SHA256,
    GENERATOR_VERSION,
    GENERATOR_RNG_IMPLEMENTATION,
    GENERATOR_DRAW_ORDER,
    generate_random_control_rows,
    render_manifest,
    sha256_bytes,
    validate_manifest_rows,
)


def build_audit() -> dict:
    rows = generate_random_control_rows()
    validate_manifest_rows(rows)
    manifest_bytes = render_manifest(rows)
    return {
        "audit_scope": "LOCAL_OFFLINE_STATIC_CAMPAIGN_FOUNDATION_ONLY",
        "manifest_rng_seed": MANIFEST_RNG_SEED,
        "generator_version": GENERATOR_VERSION,
        "generator_rng_implementation": GENERATOR_RNG_IMPLEMENTATION,
        "per_row_draw_order": list(GENERATOR_DRAW_ORDER),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "committed_manifest_sha256_expected": RANDOM_CONTROL_MANIFEST_SHA256,
        "row_count": len(rows),
        "rows": rows,
        "hidden_size_counts": dict(sorted(Counter(row["hidden_size"] for row in rows).items())),
        "batch_size_counts": dict(sorted(Counter(row["batch_size"] for row in rows).items())),
        "learning_rate_min": min(float(row["learning_rate"]) for row in rows),
        "learning_rate_max": max(float(row["learning_rate"]) for row in rows),
        "embedding_dropout_min": min(float(row["embedding_dropout"]) for row in rows),
        "embedding_dropout_max": max(float(row["embedding_dropout"]) for row in rows),
        "output_dropout_min": min(float(row["output_dropout"]) for row in rows),
        "output_dropout_max": max(float(row["output_dropout"]) for row in rows),
        "no_wandb": True,
        "no_training": True,
        "no_sealed_scope": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", type=Path, help="Write deterministic canonical manifest bytes.")
    parser.add_argument("--audit-output", type=Path, required=True, help="Write compact local-only audit JSON.")
    parser.add_argument("--verify-manifest", type=Path, help="Require an existing manifest to equal deterministic bytes.")
    args = parser.parse_args()

    manifest_bytes = render_manifest()
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_bytes(manifest_bytes)
    if args.verify_manifest and args.verify_manifest.read_bytes() != manifest_bytes:
        raise SystemExit("manifest bytes differ from deterministic frozen generation")

    audit = build_audit()
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("row_count", "manifest_rng_seed", "manifest_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
