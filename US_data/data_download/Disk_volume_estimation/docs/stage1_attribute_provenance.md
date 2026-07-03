# Stage 1 static attribute source — provenance record

Milestone: 2K-G-D (Attribute Provenance + Modeling Design Gate), opened 2026-07-03.
Updated: 2K-G-D-A (Promote static attribute artifact out of `tmp`), 2026-07-03.

This is the authoritative record for the static attribute file consumed by
`scripts/build_stage1_nh_package.py --attributes-csv`. It closes the ambiguity
flagged at Milestone 2K-G-C (two different paths appeared in docs — a repo
path and an h2o-staged path — with no checksum tying them together), and
records the 2K-G-D-A promotion of the h2o-resident copy from a `tmp/` staging
path to a stable project data path.

## Decision: keep external / h2o-Moriah-resident, do not commit the parquet

Per `docs/repo_policy.md` → "Generated artifact policy": *"Git does not track
generated data products, raw downloads, large report tables, logs, or
caches."* `all_basins_merged.parquet` is a generated data product (a
deterministic merge of GAGES-II source tables — see Provenance chain below),
produced by the same class of script that the repo's "Tighten generated
artifact tracking policy" commit (`f51b34a`) deliberately untracked across
`reports/**`. At 2.97 MB it is small enough that committing it would not
violate any size limit, but doing so would contradict the established policy
for a one-off exception. **Decision: document it as a canonical,
checksum-pinned external artifact (option (b) from the 2026-06-30 decision
log entry), not commit it.**

What *is* committed instead: this doc, plus a small tracked
`reports/flashnh_basin_screening_v001/README.md` pointer (added under the
existing `!reports/**/README.md` gitignore exception) and a checksum field
(`attributes_sha256`) now written into every package's `run_provenance.json`
by the builder, so every NH package build is traceable to a specific
attribute-file byte-for-byte identity without the file itself living in git.

## Canonical path

| Location | Path | Status |
|---|---|---|
| **h2o-resident (canonical for pipeline runs)** | `/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet` | Promoted 2026-07-03 (Milestone 2K-G-D-A) from the `tmp/` staging copy below; **sha256-verified identical to the tmp copy** (see "Verification evidence" below). This is the path `--attributes-csv` should use for all package builds from this milestone onward. |
| h2o `tmp/` copy (historical/staged only, not canonical) | `/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` | Manually staged 2026-06-30 (Milestone 2K-G-B). Superseded by the stable path above as of 2026-07-03. Retained on h2o for now as the pre-promotion reference copy; not to be referenced by new work. |
| Moriah mirror (for later transfer, not yet copied) | `/sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet` | Documented destination path for when the attribute file is next transferred to Moriah alongside a package build; transfer not yet performed. |
| Local repo fixture (this machine only) | `US_data/data_download/Disk_volume_estimation/reports/flashnh_basin_screening_v001/all_basins_merged.parquet` | Present locally 2026-07-03; gitignored (`reports/**`); is the source of the checksum recorded below. |

The h2o `tmp/` copy, the promoted stable h2o copy, and the local repo fixture
are all confirmed byte-identical (see "Verification evidence" below). The
Moriah mirror has **not** been copied yet and is not confirmed.

## Checksum (verified identical across local fixture, h2o `tmp/`, and h2o stable path)

```
sha256  06a9eeda9e94261d0b1bb9f2c2f42cb6bf11b4c02745d7ed5867ef0e0c0ad0b1
size    3,037,889 bytes local (2.90 MiB); 2.9M per h2o `ls -lh` (both h2o copies)
```

Computed locally with:
```
python -c "
import hashlib
h = hashlib.sha256()
with open('reports/flashnh_basin_screening_v001/all_basins_merged.parquet','rb') as f:
    for chunk in iter(lambda: f.read(8192), b''):
        h.update(chunk)
print(h.hexdigest())
"
```

## Verification evidence (2K-G-D-A, 2026-07-03 — user-run on h2o)

The promotion copy and checksum verification were run directly on h2o (outside
this local session; results reported by the user and recorded here per
`docs/repo_policy.md` evidence conventions):

```
# Promote from tmp/ staging to the stable data path:
mkdir -p /data42/omrip/Flash-NH/data/static_attributes/gagesii_v001
cp /data42/omrip/Flash-NH/tmp/all_basins_merged.parquet \
   /data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet

# Verify identical checksum at both paths:
sha256sum /data42/omrip/Flash-NH/tmp/all_basins_merged.parquet
sha256sum /data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet
```

Result: both paths returned
`06a9eeda9e94261d0b1bb9f2c2f42cb6bf11b4c02745d7ed5867ef0e0c0ad0b1`
(matching the local fixture checksum above); `ls -lh` reported `2.9M` for
both. **The h2o stable-path copy is now checksum-verified.** This closes the
"Evidence that must be pulled" item from the original 2K-G-D version of this
doc — the tmp-vs-repo-fixture comparison is confirmed, not just assumed.

The Moriah mirror path has not been populated or verified; do so before any
Moriah-side package build that needs the attribute file directly (current
Moriah packages have been transferred pre-built from h2o, so this has not
been a blocker yet — see `docs/repo_policy.md` transfer workflow if/when a
direct Moriah-side build is needed).

## Schema

- 9,008 rows (all GAGES-II reference-quality basins in the CONUS universe —
  **not** filtered/scoped to the ~2,752–2,843 Flash-NH candidate basins; the
  builder subsets by `STAID` at load time)
- 48 columns
- Index candidate columns: `STAID` (int64; **not** `gauge_id` — the builder's
  `_load_attributes()` looks for `gauge_id`, then `STAID`, then `staid`, and
  normalizes via `_norm_staid()` to an 8-char zero-padded string, e.g.
  `3144816` → `"03144816"`; round-trip confirmed lossless)
- Required columns present: `DRAIN_SQKM` ✓, `LAT_GAGE` ✓, `LNG_GAGE` ✓,
  `BFI_AVE` ✓ (all four confirmed present; `STAID` present, `gauge_id`
  absent — expected, builder handles this)
- First 20 columns (of 48): `STAID`, `STANAME`, `DRAIN_SQKM`, `HUC02`,
  `LAT_GAGE`, `LNG_GAGE`, `STATE`, `BOUND_SOURCE`, `HCDN_2009`, `HBN36`,
  `OLD_HCDN`, `NSIP_SENTINEL`, `FIPS_SITE`, `COUNTYNAME_SITE`, `NAWQA_SUID`,
  `STREAMS_KM_SQ_KM`, `STRAHLER_MAX`, `MAINSTEM_SINUOUSITY`, `REACHCODE`,
  `ARTIFPATH_PCT`
- Remaining 28 columns: land-cover/hydrology derived fields
  (`ARTIFPATH_MAINSTEM_PCT`, `HIRES_LENTIC_PCT`, `PERDUN`, `PERHOR`,
  `TOPWET`, `CONTACT`, `RUNAVE7100`, monthly water-balance
  `WB5100_{JAN..DEC,ANN}_MM`, stream-order percentages
  `PCT_{1ST..6TH}_ORDER`/`PCT_NO_ORDER`) — all GAGES-II-native, unmodified
  by the merge script beyond column concatenation.

## Provenance chain (generated, not source, not manually curated)

1. **Source**: public GAGES-II attribute release, cached locally as 26 CSVs
   under `US_data/attributes/attributes_gageii_*.csv` on this machine (e.g.
   `attributes_gageii_BasinID.csv`, `attributes_gageii_Hydro.csv`). **Not
   tracked in git.**
2. **Merge script**: `scripts/flashnh_basin_screening.py` — deterministic
   join of the GAGES-II CSVs on `STAID`, no filtering, no manual edits. Reads
   from a hardcoded absolute path
   (`C:/PhD/Python/neuralhydrology/US_data/attributes`), so it is **not
   currently portable to h2o/Moriah** — it can only be re-run on this
   Windows machine. Writes `all_basins_merged.parquet` plus two filtered
   variants (`area_filtered_basins.parquet`,
   `area_bfi_filtered_basins.parquet`) and screening plots/summaries to
   `reports/flashnh_basin_screening_v001/`.
3. **Result**: `all_basins_merged.parquet` — classified here as **generated**
   (mechanically merged from source tables; not itself a GAGES-II source
   file, and not manually curated/edited after generation).

Because neither the GAGES-II source CSVs nor the merge script's absolute
input path are portable/tracked, the h2o/Moriah copy **cannot currently be
regenerated remotely** — it depends on the manually staged copy from
2026-06-30. This is acceptable for the current pilot scale but is a
reproducibility gap worth closing if the attribute source changes materially.

## How the builder receives this file

`scripts/build_stage1_nh_package.py --attributes-csv <path>` — required
argument, no silent default. Accepts `.parquet` or `.csv`. Use the stable
h2o path:

```
--attributes-csv /data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet
```

Do **not** pass the `tmp/` path for new work — it is historical/staged only
(see "Canonical path" table above). The builder records a `sha256` of the
file it was actually given into `run_provenance.json`
(`attributes_sha256` field), so every package's provenance is
self-verifying regardless of exactly which path supplied the file; passing
the stable path is a documentation/hygiene requirement, not something the
builder itself can enforce.

## Status: verification closed (2K-G-D-A, 2026-07-03)

The tmp-vs-stable-path checksum verification requested at the end of the
original (2K-G-D) version of this doc has been **completed and closed** —
see "Verification evidence" above. No further attribute-file verification is
required before full 2,752-basin NH package generation on this basis; that
generation is still gated on the separate scientific-baseline design-gate
decisions in `docs/stage1_scientific_baseline_design.md`, not on attribute
provenance.

Remaining open item (non-blocking): the Moriah mirror path
(`/sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`)
has not been populated or checksum-verified. Populate and verify it before
any Moriah-side build that reads the attribute file directly.
