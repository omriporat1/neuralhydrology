#!/usr/bin/env python3
"""Read-only inspection of the installed NeuralHydrology package (Moriah).

Phase A scaffold for Milestone 2K-G-G (Target Scaling + Gap Policy +
Lead-Time Feasibility Report, see
`docs/stage1_target_scaling_gap_leadtime_feasibility.md`).

This tool only IMPORTS and INTROSPECTS the installed `neuralhydrology`
package. It does not:
  - run training or call any training entry point (`nh-run`, etc.),
  - build or touch any NH dataset package,
  - submit a Slurm job,
  - require or use a GPU.
Importing library modules executes their module-level definitions (as any
Python import does) but does not invoke NH's training/data pipeline, since
no NH `Config`, dataset, or model object is constructed here.

It is safe to run on the Moriah login node or inside a light CPU
allocation, after activating the `flashnh-moriah` environment.

Usage (Moriah):
  python scripts/inspect_neuralhydrology_stage1_mechanics.py \\
      --out-dir /sci/labs/efratmorin/omripo/Flash-NH/tmp/nh13_inspection_<TS>

Usage (local, NeuralHydrology likely not installed -- syntax/smoke check only):
  python scripts/inspect_neuralhydrology_stage1_mechanics.py \\
      --out-dir tmp/nh13_inspection_smoke

If `neuralhydrology` cannot be imported, the script does not crash: it
prints a clear warning, still writes all output files (noting NH is
unavailable), and exits 0. A non-zero exit only happens on an unexpected
internal error while writing the report.

Output files written under --out-dir:
  nh13_inspection_summary.md    human-readable report
  nh13_inspection_summary.json  same content, machine-readable
  module_paths.txt              resolved module/class/function file:line list
  source_hits.txt               keyword text-search hits across installed NH source
  run_command.txt               exact argv used
  git_commit.txt                this repo's HEAD commit (best-effort)
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Search targets
# ---------------------------------------------------------------------------

# Class names we most want to find and fully describe.
TARGET_CLASS_NAMES = ["GenericDataset", "BaseDataset"]

# Substrings used to flag member functions/methods of interest during the
# generic package walk (matched against lowercase name).
TARGET_MEMBER_KEYWORDS = [
    "seq_length", "predict_last_n", "nan_handling", "target_variable",
    "static_attribute", "qobs", "scaler", "normalize", "denormalize",
    "inverse_transform", "lead_time", "horizon", "forecast", "mask",
    "sample", "window", "lookup_table",
]

# Methods always worth recording for any Dataset-like class, regardless of
# whether their name matches a keyword above (these are the classic
# GenericDataset/BaseDataset windowing/loading hooks in upstream NH, but we
# do not assume they exist -- we just always check for them).
ALWAYS_RECORD_METHODS = {
    "__getitem__", "__len__", "_load_data", "_load_attributes",
    "_create_lookup_table", "_get_start_and_end_dates", "_load_basin_data",
    "_load_or_create_xarray_dataset", "_initialize_frequency_configuration",
}

# Plain-text keyword search across every installed .py file (independent of
# whether the module imports cleanly or the symbol is a top-level name).
SOURCE_SEARCH_KEYWORDS = [
    "GenericDataset", "BaseDataset", "seq_length", "predict_last_n",
    "nan_handling_method", "target_variables", "static_attributes",
    "qobs", "forecast", "lead_time", "horizon", "scaler", "normalize",
    "inverse_transform", "mask",
]

MAX_HITS_PER_KEYWORD = 40


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", required=True,
                   help="Directory to write the inspection report to (expected under tmp/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment facts
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    repo_dir = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - best effort only
        return f"unknown ({type(exc).__name__}: {exc})"


def _try_import_nh():
    """Returns (available, module_or_None, version_str_or_None, module_path_or_None, error_or_None)."""
    try:
        import neuralhydrology  # type: ignore
    except Exception as exc:
        return False, None, None, None, f"{type(exc).__name__}: {exc}"

    version = getattr(neuralhydrology, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as _pkg_version
            version = _pkg_version("neuralhydrology")
        except Exception:
            version = "unknown (no __version__ attribute; package metadata unavailable)"

    module_path = str(Path(neuralhydrology.__file__).resolve().parent)
    return True, neuralhydrology, version, module_path, None


# ---------------------------------------------------------------------------
# Package walk + member scan
# ---------------------------------------------------------------------------


def _walk_and_import(nh_module) -> tuple[dict[str, Any], list[dict[str, str]]]:
    modules: dict[str, Any] = {nh_module.__name__: nh_module}
    errors: list[dict[str, str]] = []
    pkg_path = getattr(nh_module, "__path__", None)
    if pkg_path is None:
        return modules, errors
    for _finder, name, _ispkg in pkgutil.walk_packages(pkg_path, prefix=nh_module.__name__ + "."):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:
            errors.append({"module": name, "error": f"{type(exc).__name__}: {exc}"})
    return modules, errors


def _record_source(obj: Callable) -> tuple[str | None, int | None, str | None]:
    try:
        file = inspect.getsourcefile(obj)
    except (TypeError, OSError):
        file = None
    try:
        _, line = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        line = None
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = None
    return file, line, sig


def _scan_members(modules: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    class_hits: list[dict] = []
    function_hits: list[dict] = []
    seen_classes: set[str] = set()

    for mod_name, module in modules.items():
        try:
            members = inspect.getmembers(module)
        except Exception:
            continue

        for name, obj in members:
            owner_module = getattr(obj, "__module__", None)

            if inspect.isclass(obj) and owner_module == mod_name:
                is_target = name in TARGET_CLASS_NAMES
                is_dataset_like = "dataset" in name.lower()
                if not (is_target or is_dataset_like):
                    continue
                key = f"{mod_name}.{name}"
                if key in seen_classes:
                    continue
                seen_classes.add(key)

                file, line, init_sig = _record_source(obj)
                methods: list[dict] = []
                try:
                    own_members = inspect.getmembers(
                        obj, predicate=lambda o: inspect.isfunction(o) or inspect.ismethod(o)
                    )
                except Exception:
                    own_members = []
                for m_name, m_obj in own_members:
                    qualname = getattr(m_obj, "__qualname__", "")
                    defined_here = qualname.split(".")[0] == name if qualname else False
                    if not defined_here:
                        continue
                    matches_kw = any(kw in m_name.lower() for kw in TARGET_MEMBER_KEYWORDS)
                    if m_name in ALWAYS_RECORD_METHODS or matches_kw:
                        m_file, m_line, m_sig = _record_source(m_obj)
                        methods.append({
                            "name": m_name, "file": m_file, "line": m_line, "signature": m_sig,
                        })

                class_hits.append({
                    "target": is_target,
                    "class": name,
                    "module": mod_name,
                    "file": file,
                    "line": line,
                    "init_signature": init_sig,
                    "methods": methods,
                })

            elif inspect.isfunction(obj) and owner_module == mod_name:
                if any(kw in name.lower() for kw in TARGET_MEMBER_KEYWORDS):
                    file, line, sig = _record_source(obj)
                    function_hits.append({
                        "function": name, "module": mod_name,
                        "file": file, "line": line, "signature": sig,
                    })

    return class_hits, function_hits


def _search_source(pkg_dir: Path, keywords: list[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {kw: [] for kw in keywords}
    try:
        py_files = sorted(pkg_dir.rglob("*.py"))
    except Exception:
        return hits
    for path in py_files:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, start=1):
            for kw in keywords:
                if len(hits[kw]) >= MAX_HITS_PER_KEYWORD:
                    continue
                if kw in line:
                    hits[kw].append(f"{path}:{i}: {line.strip()}")
    return hits


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_module_paths(path: Path, class_hits: list[dict], function_hits: list[dict]) -> None:
    lines = []
    for c in class_hits:
        tag = "TARGET-CLASS" if c["target"] else "DATASET-LIKE-CLASS"
        loc = f"{c['file']}:{c['line']}" if c["file"] else "<source unavailable>"
        lines.append(f"[{tag}] {c['module']}.{c['class']} -> {loc}")
        if c["init_signature"]:
            lines.append(f"    __init__{c['init_signature']}")
        for m in c["methods"]:
            m_loc = f"{m['file']}:{m['line']}" if m["file"] else "<source unavailable>"
            sig = m["signature"] or ""
            lines.append(f"    .{m['name']}{sig} -> {m_loc}")
    for f in function_hits:
        loc = f"{f['file']}:{f['line']}" if f["file"] else "<source unavailable>"
        lines.append(f"[FUNCTION] {f['module']}.{f['function']}{f['signature'] or ''} -> {loc}")
    if not lines:
        lines.append("(no matches -- either NeuralHydrology is unavailable, or no members "
                      "matched the target class names / keyword list in this NH version)")
    path.write_text("\n".join(lines) + "\n")


def _write_source_hits(path: Path, source_hits: dict[str, list[str]]) -> None:
    lines = []
    for kw, hits in source_hits.items():
        lines.append(f"=== {kw} ({len(hits)} hit(s), capped at {MAX_HITS_PER_KEYWORD}) ===")
        lines.extend(hits if hits else ["  (no hits)"])
        lines.append("")
    if not lines:
        lines = ["(no source search performed -- NeuralHydrology unavailable)"]
    path.write_text("\n".join(lines) + "\n")


def _write_markdown(path: Path, summary: dict, class_hits: list[dict], function_hits: list[dict]) -> None:
    lines = [
        "# NeuralHydrology 1.13 installed-code inspection (Phase A, Milestone 2K-G-G)",
        "",
        f"Generated by `scripts/inspect_neuralhydrology_stage1_mechanics.py`.",
        "",
        "## Environment",
        "",
        f"- Python executable: `{summary['python_executable']}`",
        f"- Python version: `{summary['python_version'].splitlines()[0]}`",
        f"- Platform: `{summary['platform']}`",
        f"- neuralhydrology available: **{summary['neuralhydrology_available']}**",
        f"- neuralhydrology version: `{summary['neuralhydrology_version']}`",
        f"- neuralhydrology module path: `{summary['neuralhydrology_module_path']}`",
    ]
    if summary["neuralhydrology_import_error"]:
        lines.append(f"- import error: `{summary['neuralhydrology_import_error']}`")
    lines.append("")

    if not summary["neuralhydrology_available"]:
        lines += [
            "## Result",
            "",
            "**NeuralHydrology is not importable in this Python environment.** This is expected "
            "when running locally (outside the Moriah `flashnh-moriah` env). No NH-dependent "
            "inspection was performed -- this is a syntax/smoke-check run only, not evidence "
            "for Milestone 2K-G-G. Re-run this script on Moriah after activating "
            "`flashnh-moriah` to produce real evidence.",
            "",
        ]
    else:
        lines += [
            "## Target classes found",
            "",
        ]
        if class_hits:
            for c in class_hits:
                tag = "TARGET" if c["target"] else "dataset-like"
                loc = f"{c['file']}:{c['line']}" if c["file"] else "source unavailable"
                lines.append(f"### `{c['module']}.{c['class']}` ({tag})")
                lines.append(f"- Location: `{loc}`")
                if c["init_signature"]:
                    lines.append(f"- `__init__{c['init_signature']}`")
                if c["methods"]:
                    lines.append("- Methods of interest:")
                    for m in c["methods"]:
                        m_loc = f"{m['file']}:{m['line']}" if m["file"] else "source unavailable"
                        lines.append(f"  - `.{m['name']}{m['signature'] or ''}` -> `{m_loc}`")
                lines.append("")
        else:
            lines.append("(no matching classes found in the installed package)")
            lines.append("")

        lines += ["## Functions of interest", ""]
        if function_hits:
            for f in function_hits:
                loc = f"{f['file']}:{f['line']}" if f["file"] else "source unavailable"
                lines.append(f"- `{f['module']}.{f['function']}{f['signature'] or ''}` -> `{loc}`")
        else:
            lines.append("(none found)")
        lines.append("")

        lines += [
            "## Module import errors during package walk",
            "",
        ]
        errs = summary.get("module_import_errors", [])
        if errs:
            for e in errs:
                lines.append(f"- `{e['module']}`: {e['error']}")
        else:
            lines.append("(none -- every submodule imported cleanly)")
        lines.append("")

        lines += [
            "## Source keyword search",
            "",
            "See `source_hits.txt` for full hit list "
            f"(capped at {MAX_HITS_PER_KEYWORD} hits/keyword). Hit counts:",
            "",
        ]
        for kw, n in summary.get("source_hits_counts", {}).items():
            lines.append(f"- `{kw}`: {n}")
        lines.append("")

    lines += [
        "## Scope reminder",
        "",
        "This report only describes what the installed NeuralHydrology 1.13 code looks like. "
        "It does not itself decide target-scaling, gap-policy, or lead-time implementation -- "
        "see `docs/stage1_target_scaling_gap_leadtime_feasibility.md` (Phase B) for the "
        "questions this evidence must answer, and "
        "`docs/stage1_scientific_baseline_design.md` §5/§6/§9b for the binding decisions "
        "these findings feed into.",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_command.txt").write_text(" ".join(sys.argv) + "\n")
    (out_dir / "git_commit.txt").write_text(_git_commit() + "\n")

    print("=" * 70)
    print("Flash-NH Stage 1 — NeuralHydrology 1.13 installed-code inspection")
    print(f"Out dir: {out_dir}")
    print("=" * 70)

    summary: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    nh_available, nh_module, nh_version, nh_path, nh_error = _try_import_nh()
    summary["neuralhydrology_available"] = nh_available
    summary["neuralhydrology_version"] = nh_version
    summary["neuralhydrology_module_path"] = nh_path
    summary["neuralhydrology_import_error"] = nh_error

    class_hits: list[dict] = []
    function_hits: list[dict] = []
    module_import_errors: list[dict] = []
    source_hits: dict[str, list[str]] = {}

    if nh_available:
        print(f"neuralhydrology {nh_version} at {nh_path}")
        modules, module_import_errors = _walk_and_import(nh_module)
        print(f"Imported {len(modules)} module(s); {len(module_import_errors)} import error(s)")
        class_hits, function_hits = _scan_members(modules)
        print(f"Found {len(class_hits)} class hit(s), {len(function_hits)} function hit(s)")
        source_hits = _search_source(Path(nh_path), SOURCE_SEARCH_KEYWORDS)
    else:
        print("WARNING: neuralhydrology is not importable in this Python environment.")
        print(f"  Import error: {nh_error}")
        print("  This is expected when running locally without the Moriah/h2o NH env.")
        print("  Writing a partial report; NH-dependent sections will be empty.")
        print("  Re-run on Moriah (flashnh-moriah env) to produce real Milestone 2K-G-G evidence.")

    summary["class_hits"] = class_hits
    summary["function_hits"] = function_hits
    summary["module_import_errors"] = module_import_errors
    summary["source_hits_counts"] = {k: len(v) for k, v in source_hits.items()}

    (out_dir / "nh13_inspection_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_markdown(out_dir / "nh13_inspection_summary.md", summary, class_hits, function_hits)
    _write_module_paths(out_dir / "module_paths.txt", class_hits, function_hits)
    _write_source_hits(out_dir / "source_hits.txt", source_hits)

    print("=" * 70)
    print(f"Wrote: nh13_inspection_summary.md, nh13_inspection_summary.json, "
          f"module_paths.txt, source_hits.txt, run_command.txt, git_commit.txt")
    print(f"NH available: {nh_available}")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()