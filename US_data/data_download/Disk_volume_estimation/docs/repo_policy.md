# Repository Policy

- `src/` is for application and library code.
- `scripts/` is for runnable helpers, audit runners, and one-off utilities.
- `docs/` is for markdown documentation.
- `reports/` is for generated audit and report artifacts.
- `data/` is for downloaded samples.
- `tmp/` is for scratch space only.
- Credential files must never be committed.
- Large raw GRIB, NC4, and sample files should not be committed unless they are explicitly intended for the repository.

## Handling Guidance

- Keep generated outputs out of source directories.
- Prefer small, reviewable artifacts in `reports/` and documentation in `docs/`.
- If a large binary file is needed for a reproducible example, document the reason before adding it.