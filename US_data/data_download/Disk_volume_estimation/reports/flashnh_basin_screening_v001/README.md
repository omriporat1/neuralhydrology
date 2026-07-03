# reports/flashnh_basin_screening_v001/ — provenance note

This directory holds **generated** outputs of `scripts/flashnh_basin_screening.py`
(basin screening by drainage area and BFI, Milestone: basin screening). Per
`docs/repo_policy.md` → "Generated artifact policy," this directory is
git-ignored (`reports/**` in `.gitignore`); this `README.md` is committed under
the explicit `!reports/**/README.md` exception as the small curated pointer to
what lives here and how to get it back.

## `all_basins_merged.parquet` — canonical Stage-1 static attribute source

This is the file consumed by `scripts/build_stage1_nh_package.py
--attributes-csv`. Full details, checksum, and schema:
**`docs/stage1_attribute_provenance.md`** (repo root of this project, i.e.
`US_data/data_download/Disk_volume_estimation/docs/`).

Quick facts (see the linked doc for the authoritative record):
- Generated (not source, not manually curated): deterministic merge of local
  GAGES-II attribute CSVs, unfiltered (all 9,008 GAGES-II basins).
- Not tracked in git (neither this parquet nor its GAGES-II CSV inputs).
- Canonical resident copy for pipeline runs: stable h2o path
  `/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`
  (promoted 2026-07-03, Milestone 2K-G-D-A, from a `tmp/` staging copy —
  checksum-verified identical; sha256
  `06a9eeda9e94261d0b1bb9f2c2f42cb6bf11b4c02745d7ed5867ef0e0c0ad0b1`). The
  old `/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` path is
  historical/staged only — do not reference it for new work. Moriah mirror
  (not yet populated):
  `/sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`.
  Full detail: `docs/stage1_attribute_provenance.md`.
- This local file is a build fixture kept for local script development /
  regeneration only; it is not itself transferred to h2o/Moriah by any
  committed script — the h2o copy was staged manually (see decision log,
  Milestone 2K-G-B / 2K-G-D / 2K-G-D-A entries).

## Other files in this directory

`area_filtered_basins.parquet`, `area_bfi_filtered_basins.parquet`,
`candidate_basin_screening_summary.{md,json}`, `*_threshold_summary_table.*`,
`plots/*.png` — basin-screening exploration outputs, not used by any
downstream builder script. Regenerate with:

```
python scripts/flashnh_basin_screening.py
```

(reads from a local `US_data/attributes/attributes_gageii_*.csv` GAGES-II
cache — not tracked in git; obtain from the public GAGES-II release if
regenerating from scratch).