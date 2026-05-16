# Agent Handoff Rules — Flash-NH

This file defines how AI coding agents should operate within the Flash-NH repository. All agents (Claude Code, ChatGPT, Copilot, Gemini Code Assist, or any future tool) must follow these rules to keep the repo clean, reproducible, and safe to collaborate on.

---

## 1. What agents may NOT commit without explicit request

The following generated artifact types must never be committed unless the user explicitly says "commit this":

- CSV files (intermediate tables, screening results, station lists)
- Parquet files (processed or cached data)
- PNG / SVG / PDF figures (hydrographs, plots, maps)
- Log files and run logs
- Raw hydrograph data files
- Cached USGS API responses
- Report bundles or zipped output directories
- Temporary diagnostics, profiling output, or scratch files

When in doubt, do not commit. Ask first.

---

## 2. What agents may commit when requested

The following file types are safe to commit when the user explicitly asks:

- Python source files (`.py`)
- Documentation and strategy documents (`.md`)
- Tests and test fixtures
- Lightweight configuration files (`.yaml`, `.toml`, `.json`, `.cfg`, `.ini`)
- `.gitignore` and other repo-management files

---

## 3. Long-running downloads and processing

Scripts that fetch data from external sources (USGS, NOAA, etc.) or perform multi-hour processing must:

- Support resumable execution via checkpoints or skip-if-exists logic
- Write intermediate state to disk so a restart does not repeat completed work
- Never re-download data that already exists locally unless forced by a flag

---

## 4. GitHub push policy

Agents must not push to GitHub (or any remote) unless the user explicitly requests it with a clear instruction such as "push this" or "push to origin". Completing a task does not authorize a push.

---

## 5. External downloads policy

Agents must not initiate expensive or time-consuming external downloads (USGS bulk pulls, NWIS queries, S3 fetches, etc.) unless the user explicitly requests it. Proposing a download plan and waiting for approval is the correct default.

---

## 6. Output directory convention

All script outputs must be written under a clearly named output directory. Preferred pattern:

```
reports/<run_name>/
```

Examples:
- `reports/flashnh_wy2024_pilot_selection_v001/`
- `reports/usgs_rbi_screening_2024/`

Outputs must not be written to the repo root, `src/`, `docs/`, or any source-code directory.

---

## 7. Required content in final replies

Every agent reply that completes a task must include:

- **Files changed**: list of files created, modified, or deleted
- **Validation commands run**: exact commands executed (e.g., `python -m pytest`, `git diff --stat`)
- **Output paths**: full or repo-relative paths to any outputs written
- **Git status**: result of `git status --short`
- **Commit hash**: only if a commit was made (include the short SHA)

---

## 8. Preferred agent usage

| Agent | Preferred role |
|---|---|
| **ChatGPT** | Strategy, scientific interpretation, prompt design, code review |
| **Claude Code** | Larger repo edits, multi-file refactors, debugging across files |
| **Copilot** | Small local fixes only while quota is constrained |
| **Gemini Code Assist** | Optional secondary reviewer for code quality or documentation |

Route tasks to the agent best suited for them. Do not duplicate expensive work across agents.

---

## 9. Reusable prompt template

Use or adapt the following template when handing off a task to an agent:

```
Project: Flash-NH
Task: <one-sentence description of what to do>

Context:
- Working directory: <repo root or relevant subdirectory>
- Relevant files: <list key files the agent should read first>
- Prior work: <brief summary of what has already been done>

Constraints:
- Do not commit generated artifacts (CSV, Parquet, PNG, logs, etc.)
- Do not push to GitHub unless I explicitly say so
- Do not run external downloads unless I explicitly say so
- Write outputs to reports/<run_name>/

When done, report:
1. Files changed
2. Validation commands run and their output
3. Output paths
4. git status --short
5. Commit hash (only if committed)
```
