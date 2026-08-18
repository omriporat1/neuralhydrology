# Repository Policy

Project: Flash-NH — near-real-time and forecast-aware hydrological modeling pipeline.

This file is the canonical policy for **Git tracking, generated artifacts, scratch/output locations, and remote-run evidence**.

## 1. Repository content

- `src/` is for application and library code.
- `scripts/` is for reusable runnable helpers, audit runners, and maintained utilities.
- `docs/` is for committed documentation.
- `data/` is for intentionally retained/downloaded samples and canonical small data assets where explicitly appropriate.
- Generated evidence/results must follow the location policy below.

Credential files must never be committed.

Large raw GRIB, NC4, Parquet, checkpoints, and similar generated/raw products must not be committed unless explicitly intended, documented, and approved.

## 2. Canonical generated-output and scratch locations

Flash-NH uses three distinct project-local patterns. They are **not interchangeable**.

### `reports/<run_name>/` — organized generated review/report outputs

Use `reports/<run_name>/` for generated report-style outputs that benefit from a stable, named directory structure, for example:

- compact review bundles;
- rendered review figures;
- generated audit/report tables;
- qualification/report outputs.

`reports/` is generally generated/untracked unless a specific curated artifact is intentionally approved for Git.

The existing `.gitignore` may allow selected curated files under `reports/`. That allowance does **not** make every report artifact commit-worthy. Curated tracked report content requires explicit intent/approval.

If a proposed review-bundle file type is currently ignored, keep it untracked unless the project intentionally changes the ignore/curation policy.

### `tmp/<run_name>/` — disposable/intermediate runtime scratch

Use `tmp/` for short-lived or intermediate working products such as:

- temporary build/evaluation outputs;
- transient exports;
- intermediate manifests/tables;
- disposable run scratch.

Treat `tmp/` as disposable and untracked.

### `.scratch_local/<run_name>/` — retained local-only scratch/evidence

Use root-level `.scratch_local/` for local-only material that should survive longer than `tmp/` but still must never be committed, especially:

- transferred h2o/Moriah compact evidence bundles;
- workflow-audit material;
- ad hoc local diagnostic evidence;
- retained temporary artifacts needed for later review.

This is the canonical retained-local-scratch location going forward.

Legacy retained scratch/evidence that already exists under another ignored project-local path (for example `scratch/.scratch_local/`) does not need to be migrated merely to satisfy this policy. Do not create new competing conventions.

Do not create Flash-NH scratch/evidence in unrelated machine-global directories unless the user explicitly approves a specific exception.

## 3. Generated artifact policy

Git tracks:

- source code;
- tests;
- documentation;
- configs;
- small intentionally curated summaries.

Git normally does **not** track:

- generated data products;
- raw downloads;
- checkpoints;
- large report tables;
- logs;
- caches;
- temporary configs;
- generated figures/animations;
- transferred evidence bundles;
- scratch scripts/drivers that are not being promoted into maintained project tooling.

Do not force-add ignored generated files.

If a large/binary artifact is needed for reproducibility, document the reason and obtain explicit approval before adding it.

## 4. Script placement

Reusable or maintained runnable code belongs under `scripts/`.

Ad hoc campaign-only drivers may remain untracked scratch when intentionally temporary.

Do not leave reusable scripts at the repository/project root merely because they began as one-off work.

Promotion of a scratch script into `scripts/` should be deliberate: clean it up, test it as appropriate, document its purpose, and then track it.

## 5. Root-directory hygiene

The project root should not become a long-term dumping ground for:

- `scratch_*`;
- `build_*`;
- campaign-only `.sbatch` files;
- temporary evidence archives;
- old patch/status files;
- generated logs.

Existing root-level untracked material should be triaged separately rather than deleted or moved blindly.

Future work should place new scratch/evidence in the canonical locations above.

## 6. Review bundle convention

For major generated review outputs, a lightweight `review_bundle` may include:

- `summary.md`;
- `summary.json` when untracked or explicitly curated by policy;
- `manifest.json`;
- `run_command.txt`;
- `git_commit.txt`;
- selected small plots only when scientifically/review useful.

The exact bundle schema may vary by task; do not create fields merely for ceremony.

Generated review bundles remain untracked unless explicitly approved.

## 7. Post-remote-run evidence policy

After a substantial h2o/Moriah/other-HPC run that produces generated outputs, create or identify a compact evidence bundle before documenting conclusions or proceeding to the next scientific milestone.

Typical compact evidence includes:

- launcher/run status;
- main run/audit logs;
- manifests/checksums;
- provenance/config identity;
- cleaning/coverage/quality audit tables;
- compact scientific result tables.

Keep large generated products remote unless explicitly needed.

Examples normally **excluded** from transfers:

- canonical NetCDF files;
- raw Parquet caches;
- GRIB files;
- checkpoints;
- large raw shards.

## 8. Local inspection requirement

Remote evidence must be pulled to the local project workspace before Claude/another agent documents or commits scientific conclusions.

Scientific conclusions in committed documentation must be based on **inspected evidence files**, not only:

- terminal summaries;
- log tails;
- `squeue`/`sacct`;
- pasted prose.

When quantitative evidence is in CSV/JSON/etc., quote/use the actual values from those files rather than paraphrasing a terminal summary.

Preferred local destinations:

- `.scratch_local/<run_name>/` for retained transferred evidence;
- `tmp/<run_name>/` for disposable transient inspection;
- `reports/<run_name>/` when the output is a structured generated report/review package.

Do not `git add` files from `tmp/` or `.scratch_local/`.

## 9. Remote code/evidence separation

Tracked production code should normally reach a cluster through Git synchronization, not ad hoc copying of dirty tracked files.

The normal pattern is:

```text
local implementation/test
-> reviewed commit
-> push
-> remote git pull --ff-only
-> run
```

Temporary dirty-code transfer is a narrow diagnostic exception and must not silently become the production/scientific provenance path.

## 10. Credentials and authentication

Passwords must never be stored in:

- scripts;
- repository files;
- committed `.env` files;
- logs;
- evidence;
- agent memory/instructions;
- documentation.

Use machine-local SSH configuration, keys, passphrases/agents, and approved host aliases.

Remote pull scripts may reference host aliases, but must not embed credentials.

## 11. Commit/push policy

Agents must not commit or push merely because a task is complete.

Commit and push require explicit user authorization according to the current task/handoff.

Before commit, confirm that only intended tracked source/docs/config/test files are staged and generated evidence remains untracked.

## 12. Cleanup policy

Do not perform broad cleanup of old scratch/evidence based only on filename age or apparent irrelevance.

For existing clutter:

1. inventory;
2. classify as active / retained evidence / obsolete / unclear;
3. move or delete only with an explicit reviewed cleanup decision.

This prevents accidental destruction of uncommitted scientific evidence.
