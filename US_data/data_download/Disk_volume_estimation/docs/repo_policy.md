# Repository Policy

Project: Flash-NH — near-real-time and forecast-aware hydrological modeling pipeline.

- `src/` is for application and library code.
- `scripts/` is for runnable helpers, audit runners, and one-off utilities.
- `docs/` is for markdown documentation.
- `reports/` is for generated audit and report artifacts.
- `data/` is for downloaded samples.
- `tmp/` is for scratch space only.
- Credential files must never be committed.
- Large raw GRIB, NC4, and sample files should not be committed unless they are explicitly intended for the repository.

## Generated artifact policy

- Git tracks source code, docs, configs, and small curated summaries.
- Git does not track generated data products, raw downloads, large report tables, logs, or caches.
- Generated run outputs should remain local or be archived/backed up separately.
- For review, each major run should create a lightweight `review_bundle` containing:
	- `summary.md`
	- `summary.json`
	- `manifest.json`
	- `run_command.txt`
	- `git_commit.txt`
	- selected small plots only if needed
- Copilot must not force-add ignored `reports/` files unless explicitly requested.

## Handling Guidance

- Keep generated outputs out of source directories.
- Prefer small, reviewable artifacts in `reports/` and documentation in `docs/`.
- If a large binary file is needed for a reproducible example, document the reason before adding it.