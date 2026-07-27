#!/usr/bin/env python
"""Stage 1 validation and optimization foundation, Part A: percentile-closure
plots.

Local-only (no NeuralHydrology/torch/xarray dependency): reads the small
``percentile_diagnostics.json`` artifact produced by
``scripts/compute_seed_percentile_diagnostics.py`` (already pulled back from
Moriah -- this script never reads prediction pickles or package NetCDFs
itself) and renders the two required diagnostic plots:

  1. Epoch-colored quantile function: percentile (x) vs. raw-space NSE (y),
     one line per epoch.
  2. Percentile-through-epoch: epoch (x) vs. raw-space NSE (y), one line per
     percentile track.

Raw-space NSE is unbounded below (the existing seed evidence records mean
NSE as negative as roughly -40 at some epochs), so both plots use a
``symlog`` y-axis (linear near zero, logarithmic in the tails) rather than a
linear axis that would compress the informative near-zero region -- this is
a display choice only, it does not transform or reinterpret any value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.percentile_diagnostics import PERCENTILES

_SYMLOG_LINTHRESH = 1.0


def plot_quantile_functions(epoch_tables: dict, out_path: Path) -> None:
    epochs = sorted(epoch_tables.keys())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, epoch in enumerate(epochs):
        t = epoch_tables[epoch]
        ys = [t[f"p{p}"] for p in PERCENTILES]
        color = cmap(i / max(1, len(epochs) - 1))
        ax.plot(PERCENTILES, ys, marker="o", color=color, label=f"epoch {epoch}")
    ax.set_yscale("symlog", linthresh=_SYMLOG_LINTHRESH)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("percentile")
    ax.set_ylabel("raw-space NSE (symlog scale)")
    ax.set_title("Development-validation per-basin NSE: quantile function by epoch")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_percentile_through_epoch(epoch_tables: dict, out_path: Path) -> None:
    epochs = sorted(epoch_tables.keys())
    cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, p in enumerate(PERCENTILES):
        ys = [epoch_tables[e][f"p{p}"] for e in epochs]
        color = cmap(i / max(1, len(PERCENTILES) - 1))
        ax.plot(epochs, ys, marker="o", color=color, label=f"p{p}")
    ax.set_yscale("symlog", linthresh=_SYMLOG_LINTHRESH)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("raw-space NSE (symlog scale)")
    ax.set_title("Development-validation per-basin NSE percentiles through training")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--percentile-json", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = json.loads(Path(args.percentile_json).read_text(encoding="utf-8"))
    epoch_tables = {int(k): v for k, v in artifact["per_epoch_percentile_table"].items()}
    if not epoch_tables:
        raise SystemExit(f"no per-epoch percentile tables found in {args.percentile_json}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_quantile_functions(epoch_tables, out_dir / "quantile_function_by_epoch.png")
    plot_percentile_through_epoch(epoch_tables, out_dir / "percentile_through_epoch.png")
    print(f"wrote 2 plot(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
