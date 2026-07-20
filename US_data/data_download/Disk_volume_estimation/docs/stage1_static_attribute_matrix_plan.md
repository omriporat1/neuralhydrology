# Stage 1 static attribute matrix — recovery + audit plan

Milestone: **2K-G-F (Static Attribute Matrix Recovery + Audit)**, opened
2026-07-06, following the reopening of static attributes in
[docs/stage1_scientific_baseline_design.md](stage1_scientific_baseline_design.md)
§3 (2K-G-E revision). This document is the plan and inventory for that
milestone. **No final matrix has been built.** No code, config, package, or
training changed. This pass is docs + a local, uncommitted inspection only.

**Clarifications from user review (2026-07-06, same day, second pass):**
three points below were added/tightened after the first draft of this plan —
see §8 (conservative filtering philosophy stated explicitly), §4/§9 (HydroATLAS
5-basin gap promoted from a caveat to a mandatory build/audit gate), and §5/§8
(lat/lon explicitly deferred to ablation, not in `v001-core` by default).

**Milestone 2K-G-F-B (Static Attribute Source Mirror + Derived Matrix
Builder/Auditor), 2026-07-07: this plan has now been implemented.** See §11
for the concrete result. One correction to this plan's inventory: **26
distinct `attributes_gageii_*.csv` files**, not 27 as stated throughout §1–§9
below — the "27" figure (also independently repeated by the user when
describing the h2o-side mirror) was a minor miscount; the total file count of
29 (26 GAGES-II + HydroATLAS + NLDAS-2 + workbook) is correct and unaffected.
§1–§10 below are preserved as the original planning-phase record; §11
documents what was actually built.

## 1. Why this milestone exists

The current canonical static-attribute artifact —
`/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`
(48 columns, checksum `06a9eeda9e94261d0b1bb9f2c2f42cb6bf11b4c02745d7ed5867ef0e0c0ad0b1`,
documented in `docs/stage1_attribute_provenance.md`) — is a valid,
checksum-pinned provenance artifact, but it draws from only **3 of the 27**
locally available GAGES-II source tables (`BasinID`, plus a subset of
hydrology/stream-order fields). It has no topography, geology, land
cover/vegetation, snow fraction, or general climate normals. The user expects
those in the final Stage 1 modeling matrix. This milestone inventories what's
actually available, audits it against the real Stage 1 basin set, and
proposes a merge/audit policy — it does not build the final matrix.

## 2. Source inventory

Local source directory:
`C:\PhD\Python\neuralhydrology\US_data\attributes` — **29 CSVs + 1 workbook**
(matches the "~28 files" estimate). All CSVs are keyed on `STAID`, all have
exactly **9,008 rows** (the full public GAGES-II reference-quality basin
universe; not pre-filtered to Flash-NH's basin set).

| File | Family | Rows | Non-ID cols | Mergeable to Stage 1 basins? |
|---|---|---|---|---|
| `attributes_gageii_BasinID.csv` | Identity/location | 9,008 | 14 | Yes — 2,843/2,843 (100%) |
| `attributes_gageii_Bas_Classif.csv` | Reference/screening class | 9,008 | 6 | Yes |
| `attributes_gageii_Bas_Morph.csv` | Basin morphometry | 9,008 | 3 | Yes |
| `attributes_gageii_Bound_QA.csv` | Boundary QA / stream order | 9,008 | 5 | Yes |
| `attributes_gageii_Climate.csv` | Climate normals | 9,008 | 49 | Yes |
| `attributes_gageii_Climate_Ppt_Annual.csv` | Annual precip 1950-2009 | 9,008 | 60 | Yes (per-year series, not per-basin static — see §5) |
| `attributes_gageii_Climate_Tmp_Annual.csv` | Annual temp 1950-2009 | 9,008 | 60 | Yes (same caveat) |
| `attributes_gageii_FlowRec.csv` | Flow record availability | 9,008 | 115 | Yes, but mostly per-year record flags (`wy1900`…`wy2009`), not physical attributes — see §5 |
| `attributes_gageii_Geology.csv` | Geology | 9,008 | 7 | Yes |
| `attributes_gageii_Hydro.csv` | Hydrologic indices (incl. `BFI_AVE`, water balance) | 9,008 | 33 | Yes — source of most of the existing 48-col merge |
| `attributes_gageii_HydroMod_Dams.csv` | Dam/reservoir modification | 9,008 | 28 | Yes |
| `attributes_gageii_HydroMod_Other.csv` | Other anthropogenic modification | 9,008 | 12 | Yes |
| `attributes_gageii_LC06_Basin.csv` | Land cover, whole basin (NLCD 2006) | 9,008 | 19 | Yes |
| `attributes_gageii_LC06_Mains100.csv` | Land cover, 100m mainstem buffer | 9,008 | 19 | Yes |
| `attributes_gageii_LC06_Mains800.csv` | Land cover, 800m mainstem buffer | 9,008 | 19 | Yes |
| `attributes_gageii_LC06_Rip100.csv` | Land cover, 100m riparian buffer | 9,008 | 19 | Yes |
| `attributes_gageii_LC06_Rip800.csv` | Land cover, 800m riparian buffer | 9,008 | 19 | Yes |
| `attributes_gageii_LC_Crops.csv` | Cropland type breakdown | 9,008 | 22 | Yes |
| `attributes_gageii_Landscape_Pat.csv` | Landscape pattern/fragmentation | 9,008 | 4 | Yes |
| `attributes_gageii_Nutrient_App.csv` | Nutrient application | 9,008 | 2 | Yes |
| `attributes_gageii_Pest_App.csv` | Pesticide application | 9,008 | 1 | Yes |
| `attributes_gageii_Pop_Infrastr.csv` | Population/infrastructure | 9,008 | 7 | Yes |
| `attributes_gageii_Prot_Areas.csv` | Protected areas | 9,008 | 3 | Yes |
| `attributes_gageii_Regions.csv` | Region/ecoregion codes | 9,008 | 14 | Yes |
| `attributes_gageii_Soils.csv` | Soils | 9,008 | 24 | Yes |
| `attributes_gageii_Topo.csv` | Topography (slope, elevation, relief) | 9,008 | 12 | Yes |
| `attributes_hydroATLAS.csv` | BasinATLAS/HydroATLAS (climate, snow, land cover, geology, soils, anthropogenic) | 9,008 | 195 | Yes, after zero-padding — 2,838/2,843 (99.8%; gap = 5 non-standard 15-digit USGS IDs, explainable, see §4) |
| `attributes_nldas2_climate.csv` | NLDAS-2 climate | 9,008 | 9 | Yes — 2,843/2,843 (100%) after zero-padding |
| `Var description_gageii.xlsx` | GAGES-II variable dictionary | 355 described vars, sheet `variable_descriptions` | — | Reference only, not a data source |

All basin-ID keys are `STAID`. Total non-ID columns across all sources:
**780**, of which 758 (97%) are numeric-like and 22 are non-numeric (text,
codes, or membership flags — see §5).

## 3. Content summary by theme (user's requested categories)

- **Topography**: `attributes_gageii_Topo.csv` (12 numeric cols — elevation,
  slope, relief); HydroATLAS adds `ele_mt_sav/smn/smx`, `slp_dg_uav`,
  `sgr_dk_sav` (stream gradient).
- **Geology**: `attributes_gageii_Geology.csv` (7 cols, mostly categorical
  dominant-geology-class codes); HydroATLAS adds `lit_cl_smj` (lithology
  class), `kar_pc_use` (karst %), `ero_kh_uav` (erosion).
- **Land cover / land use**: 5 GAGES-II `LC06_*` files (basin + 4 buffer
  zones, 19 cols each) + `LC_Crops.csv` (22 cols); HydroATLAS adds global
  land-cover class fractions (`glc_pc_u01`…`u22`) and potential natural
  vegetation (`pnv_cl_smj`, `pnv_pc_u01`…`u15`).
- **Vegetation**: covered via GAGES-II land cover + HydroATLAS `pnv_*` (no
  separate GAGES-II vegetation file exists).
- **Snow fraction**: **not present in any GAGES-II file** — only source is
  HydroATLAS `snw_pc_syr` (annual) + `snw_pc_s01`…`s12` (monthly), confirmed
  present and populated for the Stage 1 basin set.
- **Climate / static hydrologic attributes**: `attributes_gageii_Climate.csv`
  (49 cols, normals), `Climate_Ppt_Annual`/`Climate_Tmp_Annual` (per-year
  series, see §5), `attributes_nldas2_climate.csv` (9 cols), plus HydroATLAS
  climate block (`tmp_dc_*`, `pre_mm_*`, `pet_mm_*`, `aet_mm_*`, `ari_ix_uav`,
  `cmi_ix_*` — 60+ cols of monthly/annual temperature, precipitation, PET,
  AET, aridity, moisture index). `attributes_gageii_Hydro.csv` carries the
  existing `BFI_AVE`, `PERDUN`, `PERHOR`, `TOPWET`, `CONTACT`, `RUNAVE7100`,
  `WB5100_*` water-balance fields already in the 48-col merge.
- **Anthropogenic / human modification**: `HydroMod_Dams.csv` (28 cols),
  `HydroMod_Other.csv` (12 cols), `Pop_Infrastr.csv`, `Nutrient_App.csv`,
  `Pest_App.csv`, `Prot_Areas.csv`; HydroATLAS adds `dor_pc_pva` (degree of
  regulation), `hft_ix_u93`/`u09` (human footprint index), `pop_ct_usu`,
  `urb_pc_use`, `gdp_ud_usu`, `hdi_ix_sav`.

## 4. Cross-check against the real Stage 1 basin set

Checked against `config/stage1_initial_training_basin_manifest.csv` (2,843
basins: 2,216 `TRAIN_CORE` + 627 `TRAIN_SOFT_KEEP`; this is the working
Stage 1 candidate list, upstream of the 2,752/2,754-basin final-inclusion
counts recorded elsewhere).

- **STAID format is not uniform.** 2,837 basins have standard 8-char USGS
  IDs; 5 have 15-char coordinate-based IDs (e.g. `393109104464500`); 1 has a
  9-char ID (`103366092`). All GAGES-II CSVs store `STAID` as zero-padded
  text already — **once zero-padded to (at least) 8 characters, all 2,843
  Stage 1 basins match the raw GAGES-II source tables (100% coverage)**,
  including the 6 non-standard-length IDs.
- **`attributes_hydroATLAS.csv`'s raw `STAID` column is not zero-padded**
  (e.g. `3144816`, not `03144816`) — unlike the GAGES-II CSVs. After
  `.str.zfill(8)`, HydroATLAS matches 2,838/2,843 Stage 1 basins (99.8%). The
  5 unmatched are exactly the five 15-char non-standard IDs — `.zfill(8)` is
  a no-op on strings already longer than 8 characters, so if HydroATLAS's own
  export used a different (non-zero-padded, non-fixed-width) representation
  for those specific sites, they won't join automatically.
  **This is a mandatory build/audit gate, not a loose caveat** (clarified
  2026-07-06): the matrix builder/auditor must explicitly detect these 5
  basins by ID and the build must do exactly one of — (a) resolve/match them
  under whatever ID string HydroATLAS actually uses for those sites, (b)
  intentionally retain them with missing HydroATLAS fields under a
  documented missing-value/imputation policy (not a silent NaN), or (c) fail
  the build with a clear, named-basin audit message. **A silent partial
  HydroATLAS merge — i.e. these 5 basins quietly ending up with NaN
  HydroATLAS columns with no audit flag — is not allowed.** See §9 item 2b.
- **`attributes_nldas2_climate.csv`** matches 2,843/2,843 (100%) after
  zero-padding.
- **The existing canonical 48-column parquet stores `STAID` as `int64`**
  (verified: sample values `3144816`, `3145000`, … — leading zeros stripped).
  This is already handled downstream by `build_stage1_nh_package.py`'s
  `_load_attributes()` → `_norm_staid()` (confirmed in
  `docs/stage1_attribute_provenance.md`), but **any new 2K-G-F merge/audit
  script must independently re-implement 8-char zero-padded string
  normalization** — it is not safe to assume any given source or
  intermediate file preserves leading zeros.

## 5. Column classification (from a full audit of all 780 non-ID columns, restricted to the 2,843 Stage 1 basins)

- **758 numeric-like, 22 non-numeric-like** (>90% coercible to a number, per
  column, within the Stage 1 subset).
- **Non-numeric columns (22) fall into four groups:**
  - *Free text / administrative — drop, never model inputs*: `STANAME`,
    `COUNTYNAME_SITE`, `WR_REPORT_REMARKS`, `ADR_CITATION`,
    `SCREENING_COMMENTS`, `NAWQA_SUID` (an ID, not an attribute).
  - *Sparse binary membership flags — encode as 0/1, not drop*: `HCDN_2009`,
    `HBN36`, `OLD_HCDN`, `NSIP_SENTINEL`, `ACTIVE09` (84–99% "missing" in the
    Stage 1 subset, but "missing" here means "not a member of that network,"
    not a genuine gap — see leakage/encoding note below).
  - *Categorical class codes — one-hot or embedding candidates, not raw
    numeric inputs*: `CLASS` (Ref/Non-ref, 2 values, 0% missing),
    `AGGECOREGION` (9), `HUC02` (19), `STATE` (49 — see caveat below),
    `HUC10_CHECK` (4), `GEOL_REEDBUSH_DOM`/`GEOL_REEDBUSH_SITE` (7 each),
    `GEOL_HUNT_DOM_CODE`/`GEOL_HUNT_DOM_DESC` (41 each — the `_CODE` and
    `_DESC` columns are redundant, keep one), `GEOL_HUNT_SITE_CODE` (40),
    `USDA_LRR_SITE` (20).
  - **Decision on `STATE`, `HUC02`, lat/lon, and other geographic
    identifiers (clarified 2026-07-06):** these are exactly the fields the
    California-exclusion and spatial-holdout logic in
    `docs/stage1_scientific_baseline_design.md` §8b/§8c needs to *select*
    basins, and remain available for split construction, diagnostics, and
    reporting. `STATE` and `HUC02` are **excluded from the first
    model-input static-attribute matrix** (`v001-core`), not merely
    de-prioritized. Latitude/longitude are **held out of `v001-core` by
    default** and deferred to a dedicated later ablation that tests whether
    raw geographic coordinates help or harm spatial generalization (relevant
    precisely because of the spatial-holdout/CA-exclusion design in §8b/§8c)
    — they are not included by default pending that ablation, not withheld
    permanently.
- **Per-year time-series columns are not static attributes as-is**:
  `Climate_Ppt_Annual`/`Climate_Tmp_Annual` (`PPT1950_AVG`…`PPT2006_AVG`,
  ~60 columns each) and `FlowRec`'s `wy1900`…`wy2009` (110 columns) are
  per-year values/flags, not single static descriptors. They should be
  reduced to summary statistics (e.g. long-term mean, trend, record-length
  fraction) if used at all, not included as 170 raw per-year columns.
  `wy*` in particular is a flow-record-availability indicator, not a
  physical basin attribute — likely drop entirely from the modeling matrix.
- **Missingness (within Stage 1 basins) is otherwise low**: only 6 of 780
  columns exceed 20% missing, and all 6 are the sparse membership
  flags/remarks fields listed above — no numeric physical attribute exceeds
  this threshold.
- **20 near-constant columns** (≤1 unique value across the Stage 1 subset):
  4 GAGES-II membership flags + `ACTIVE09` + 15 HydroATLAS land-cover/PNV/
  wetland class-fraction columns that are uniformly zero for this
  CONUS/non-CA-leaning basin set (e.g. tropical/coastal land-cover classes)
  — candidates for dropping as uninformative, pending a final check against
  the full non-CA CONUS basin set (this audit used the pre-exclusion 2,843
  manifest, not the final ~2,752-basin post-exclusion, post-CA-removal set).
- **Duplicate column across files**: `DRAIN_SQKM` appears in both
  `attributes_gageii_BasinID.csv` and `attributes_gageii_Bound_QA.csv`
  (identical field, harmless if one copy is dropped at merge time).

## 6. h2o / Moriah mirror status

**Not checked from this session — no network path from this machine to
h2o/Moriah exists in the Claude Code environment** (confirmed:
`ssh flashnh-h2o` fails to resolve; this is expected and consistent with the
project's established machine-role separation — h2o/Moriah commands are
user-run and reported back, per prior milestones).

**User-side commands to check whether a source-attribute mirror already
exists, and to create one if not:**

```bash
# On h2o — check whether any source-attribute directory already exists:
ssh flashnh-h2o "ls -la /data42/omrip/Flash-NH/data/static_attributes/"
ssh flashnh-h2o "ls -la /data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/ 2>&1"

# On Moriah — same check:
ssh moriah "ls -la /sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/ 2>&1"
ssh moriah "ls -la /sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/source_attributes_v001/ 2>&1"
```

If absent (expected — the 29 source CSVs currently only exist on this local
Windows machine), mirror them explicitly rather than relying on the local
path for any h2o/Moriah-side build:

```bash
# From local machine, copy the full source directory to h2o:
scp -r "C:\PhD\Python\neuralhydrology\US_data\attributes" \
    flashnh-h2o:/data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/

# Verify byte-identical checksums for every file after transfer:
# (local)
python -c "
import hashlib, glob, os
for f in sorted(glob.glob(r'C:\PhD\Python\neuralhydrology\US_data\attributes\*')):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        h.update(fh.read())
    print(h.hexdigest(), os.path.basename(f))
"
# (h2o)
ssh flashnh-h2o "cd /data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001 && sha256sum *"

# Once verified on h2o, mirror to Moriah the same way this project already
# mirrors the gagesii_v001 artifact (see docs/stage1_attribute_provenance.md):
ssh flashnh-h2o "tar czf - -C /data42/omrip/Flash-NH/data/static_attributes source_attributes_v001" \
  | ssh moriah "mkdir -p /sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes && tar xzf - -C /sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes"
ssh moriah "cd /sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/source_attributes_v001 && sha256sum *"
```

This mirroring has **not been performed** — these are the commands to run
when the user is ready; do not assume it has happened.

## 7. Proposed canonical paths

| Artifact | Path | Notes |
|---|---|---|
| Source attribute CSVs + workbook (h2o) | `/data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/` | Mirror of the local 29-CSV + 1-workbook source directory; external, not committed to git (same policy as `gagesii_v001`). |
| Source attribute CSVs + workbook (Moriah) | `/sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/source_attributes_v001/` | Mirror of the h2o copy, transferred once h2o copy is checksum-verified. |
| Derived Stage 1 modeling matrix | `/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001.parquet` | Output of the merge policy in §8. **Built and audit-PASSed on h2o 2026-07-08; see §11.6.** Checksum-pinned the same way as `gagesii_v001` (matrix sha256 `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`). |
| Existing 48-col screening merge | `/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet` | Unchanged, retained as a valid provenance artifact per `docs/stage1_attribute_provenance.md`; may be superseded as the builder's `--attributes-csv` input once `stage1_static_attributes_v001` exists, but is not deleted or invalidated by this milestone. |

Adjust naming only if it collides with an existing convention; none was
found in this repo's docs.

## 8. Proposed merge/audit policy (for the eventual matrix build — not executed here)

**Filtering philosophy for the first modeling matrix (clarified 2026-07-06):**
be **conservative by default**. Any variable suspected to be problematic,
non-physical, purely administrative, weakly useful, leakage-prone,
near-constant, high-missingness, or hard to interpret should be **excluded**
from `v001-core`, not kept on the chance the model might find signal in it.
A smaller, defensible first matrix is preferred over a maximal one; richer or
borderline variables can always be added back later as a deliberate,
documented ablation once there is a working baseline to compare against.
This philosophy governs every subsection below — where a field's status is
ambiguous, the default is exclude, not include.

**Join key.** Normalize every source file's `STAID` to an 8-character
zero-padded string before any join (`str.zfill(8)`, applied even to columns
that already look correct, to guarantee uniformity). Confirm this does not
silently collide for the 6 non-standard-length IDs (5×15-char, 1×9-char) —
those must pass through unchanged (`zfill(8)` on a string already ≥8 chars is
a no-op, which is correct behavior here, not a bug).

**Which sources to consider (candidate superset for audit — not all of it is
expected to survive into `v001-core`; see filtering philosophy above).**
- Candidate pool: all 27 GAGES-II CSVs' numeric physical-attribute columns
  (topography, geology, land cover ×5, land use/crops, soils, hydrology,
  climate normals, landscape pattern, protected areas,
  human-modification/dams, population/infrastructure, nutrient/pesticide
  application), plus HydroATLAS's numeric climate/snow/land-cover/geology/
  soils/anthropogenic block (195 cols — the **only** available snow fraction
  source) and NLDAS-2 climate (9 cols).
- Every candidate column must still clear the conservative bar above:
  near-constant columns (§5, ~20 identified), high-missingness columns
  (§5, 6 identified), fields with unclear physical interpretation, and
  fields with plausible leakage risk are excluded from `v001-core` by
  default even though they pass basic mergeability — being "available" is
  not sufficient justification for inclusion.
- Retain the existing 48-column GAGES-II screening merge's provenance
  identity where its columns are simply a subset of the richer per-file
  sources (no information is lost by superseding it with the fuller merge;
  it remains valid as a historical artifact, not deleted).
- `Climate_Ppt_Annual`/`Climate_Tmp_Annual` (120 per-year columns) and
  `FlowRec`'s `wy1900`…`wy2009` (110 per-year record flags): per the
  conservative-by-default philosophy, these are **excluded from `v001-core`
  as raw per-year columns**. If a long-term summary statistic (e.g. mean,
  trend slope, record-length fraction) is later shown to be a clearly
  interpretable, non-leakage-prone physical/record-quality descriptor, it can
  be added back as a deliberate, documented addition — not by default.

**Non-numeric / administrative / ID handling.**
- Drop free-text/administrative fields entirely from the modeling matrix:
  `STANAME`, `COUNTYNAME_SITE`, `WR_REPORT_REMARKS`, `ADR_CITATION`,
  `SCREENING_COMMENTS`, `NAWQA_SUID`, `FIPS_SITE`, `REACHCODE`.
- Encode sparse binary membership flags (`HCDN_2009`, `HBN36`, `OLD_HCDN`,
  `NSIP_SENTINEL`, `ACTIVE09`) as explicit 0/1 (blank → 0), not as
  missing-with-NaN, since blank here means "not a member," a meaningful
  value, not an unknown — but note these are >84% one-valued in the Stage 1
  subset (§5) and should be re-screened against the conservative-inclusion
  bar (weakly useful / near-constant) before being added to `v001-core`
  rather than included automatically just because they can be encoded.
- Categorical class codes (`CLASS`, `AGGECOREGION`, `HUC10_CHECK`,
  `GEOL_REEDBUSH_DOM/SITE`, `GEOL_HUNT_DOM_CODE`, `GEOL_HUNT_SITE_CODE`,
  `USDA_LRR_SITE`): candidates for one-hot encoding, but each must be
  individually justified as physically meaningful and non-redundant before
  inclusion in `v001-core`, per the conservative philosophy above — not
  included as a block by default. Drop the redundant `GEOL_HUNT_DOM_DESC`
  (text duplicate of `GEOL_HUNT_DOM_CODE`) regardless.
- `STATE`/`HUC02`: retained for split construction, diagnostics, and
  reporting only — **excluded from `v001-core` model-input features**
  (clarified 2026-07-06; see §5).
- Latitude/longitude: **held out of `v001-core` by default**, deferred to a
  dedicated ablation testing whether raw geographic coordinates help or hurt
  spatial generalization, given the spatial-holdout/CA-exclusion design in
  `docs/stage1_scientific_baseline_design.md` §8b–§8d (clarified 2026-07-06;
  see §5).

## 9. Proposed Stage 1 static-attribute audit plan (for the eventual matrix, not run yet)

Once a candidate matrix is built (out of scope for this pass), audit it for:

1. Row count and basin coverage against the final Stage 1 eligible basin set
   (post-CA-exclusion, post-`02299472`/`04073468`-exclusion).
2. Missingness per variable.
   - **2b. HydroATLAS 5-basin gap gate (mandatory, clarified 2026-07-06):**
     explicitly detect the 5 non-standard-ID basins (§4) by ID at build
     time and require the build to do exactly one of — resolve/match them,
     retain them with a documented missing/imputation policy, or fail with
     a named-basin audit message. A build that produces NaN HydroATLAS
     columns for these basins with no audit-visible flag is a **failed**
     audit, not an acceptable partial merge.
3. Missingness per basin.
4. Numeric ranges and outlier flags per variable.
5. Constant/near-constant columns (re-checked against the *final* basin set,
   not the pre-exclusion 2,843 manifest used in this inventory pass) —
   excluded from `v001-core` per the conservative-filtering policy (§8),
   not merely flagged.
6. Duplicate or redundant columns (e.g. `DRAIN_SQKM` duplication,
   `GEOL_HUNT_DOM_CODE` vs `_DESC`).
7. Categorical fields requiring encoding (list in §8) — each individually
   justified for inclusion, not included as a block by default (§8).
8. Leakage-risk fields (`STATE`, `HUC02` excluded from `v001-core` outright;
   lat/lon held for a dedicated later ablation, not included by default; and
   any other field that could let a model infer CA/spatial-holdout
   membership).
9. Unit/description coverage using `Var description_gageii.xlsx` (343/354
   GAGES-II variable names already cross-matched in this pass; HydroATLAS and
   NLDAS-2 variables are not covered by this workbook and need their own
   documentation reference — HydroATLAS variable codes follole a public
   BasinATLAS naming convention that should be cited explicitly when the
   matrix is built).
10. Checksum of every source file and of the derived matrix.
11. A manifest/provenance JSON or markdown recording all of the above,
    following the same self-verifying pattern as `run_provenance.json`.

## 10. What was and was not done in this pass

**Done:** full inventory of the local source directory (29 CSVs + workbook);
schema/content summary by theme; STAID coverage cross-check against the real
Stage 1 basin manifest (100% GAGES-II coverage, 99.8% HydroATLAS coverage
after zero-padding); confirmed checksum of the existing 48-col artifact;
identified the `int64` STAID formatting issue in that artifact; full 780-column
numeric/categorical/missingness audit restricted to the Stage 1 basin subset;
proposed canonical paths; proposed merge/audit policy; documented h2o/Moriah
mirror check-and-transfer commands for user execution.

**Not done (by design):** no final Stage 1 static-attribute matrix was built
or written to any path (parquet, csv, or otherwise); no h2o/Moriah transfer
was performed; no code, config, package, or training changed; the per-column
audit CSV produced during this inspection is a **local, uncommitted scratch
artifact** (`attr_column_audit.csv`, in the session scratchpad directory, not
under this repo), not a tracked or canonical output.

**Clarification pass (2026-07-06, second pass, same day, docs-only):** per
user review, tightened three points before commit — (1) stated the
conservative-by-default filtering philosophy explicitly in §8; (2) promoted
the HydroATLAS 5-basin gap from a caveat to a mandatory build/audit gate in
§4/§9 item 2b, disallowing any silent partial merge; (3) excluded
`STATE`/`HUC02` from `v001-core` outright and deferred lat/lon to a dedicated
ablation rather than leaving their status ambiguous (§5/§8). No matrix was
built, no code/config/package/training changed in this pass either.

## 11. Milestone 2K-G-F-B — builder/auditor implementation + local dry-run result (2026-07-07)

This section documents what was actually implemented and validated, closing
the "not built" gap noted in §10.

**h2o source mirror.** Per the user, the 29 source files were mirrored to
`/data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/` and a
`source_attributes_v001_checksums.sha256` (29 lines) was generated there
(`sha256sum * > source_attributes_v001_checksums.sha256`, directory ≈53 MB).
This session has no network path to h2o (`ssh flashnh-h2o` fails to resolve,
reconfirmed), so the mirror and its checksums could not be independently
verified from here — see §11.5 for the exact user-run verification commands.

**Scripts implemented** (both under `scripts/`, neither commits any data
output — see repo policy in `docs/repo_policy.md`):

- `scripts/build_stage1_static_attribute_matrix.py` — reads the 29-file
  source mirror + `config/stage1_initial_training_basin_manifest.csv`,
  applies the §5/§8 classification policy below in code, and writes
  `stage1_static_attributes_v001.parquet` + `_column_manifest.json` +
  `_provenance.json`.
- `scripts/audit_stage1_static_attribute_matrix.py` — independently
  re-checks the output matrix (row count/coverage, duplicate IDs,
  missingness per variable/basin, numeric ranges, constant/near-constant
  columns, duplicate-value columns, categorical/ID-name leakage into
  `model_input`, `STATE`/`HUC02`/lat-lon exclusion, HydroATLAS gap handling,
  final checksum) and writes `_audit_summary.md`. Exit 0 = PASS, 1 = FAIL.

**Concrete column-classification rules implemented** (refines §5/§8/§9 to
column-name level, discovered via direct inspection of the local source
mirror this session):

- Duplicate drop: `DRAIN_SQKM` dropped from `attributes_gageii_Bound_QA.csv`
  (kept from `BasinID.csv`).
- Admin free-text drop: `STANAME`, `COUNTYNAME_SITE`, `WR_REPORT_REMARKS`,
  `ADR_CITATION`, `SCREENING_COMMENTS`, `NAWQA_SUID`.
- Admin numeric-ID drop (numeric-looking but not physical): `FIPS_SITE`,
  `REACHCODE`, `BOUND_SOURCE`.
- Binary flags encoded 0/1 (non-blank = member): `HCDN_2009`, `HBN36`,
  `OLD_HCDN`, `NSIP_SENTINEL`, `ACTIVE09`.
- Categorical, deferred out of `v001-core` (raw values retained in a
  `categorical_deferred` column group, not one-hot-encoded this round):
  `CLASS`, `AGGECOREGION`, `HUC10_CHECK`, `GEOL_REEDBUSH_DOM/SITE`,
  `GEOL_HUNT_DOM_CODE/DESC`, `GEOL_HUNT_SITE_CODE`, `USDA_LRR_SITE`, plus
  GAGES-II `Regions.csv` dominant/site class codes (`ECO2_BAS_DOM`,
  `ECO3_BAS_DOM`, `HLR_BAS_DOM_100M`, `NUTR_BAS_DOM`, `PNV_BAS_DOM`,
  `ECO3_SITE`, `HLR100M_SITE`, `HUC8_SITE`, `NUTR_ECO_SITE` — newly
  identified this session; only `Regions.csv`'s `_PCT`-suffixed columns are
  genuine continuous fractions) and HydroATLAS's `*_cl_smj`/`*_id_smj`
  numeric-coded class/admin-division columns (10 `_cl_smj` + `gad_id_smj` —
  newly identified this session; these pass a naive `pd.to_numeric` check but
  are categorical, not ordinal).
- `STATE`, `HUC02`: `split_support` role — retained in the matrix, excluded
  from `model_input`.
- `LAT_GAGE`, `LNG_GAGE`: `diagnostic_latlon` role — retained in the matrix,
  excluded from `model_input`, reserved for a future ablation.
- Per-year series: `FlowRec.csv`'s `wy1900`…`wy2009` (110 cols) dropped
  outright — the file already carries native summaries (`FLOWYRS_1900_2009`,
  `FLOWYRS_1950_2009`, `FLOWYRS_1990_2009`, `FLOW_PCT_EST_VALUES`), which pass
  through as ordinary `model_input` columns, so no new derivation was needed.
  `Climate_Ppt_Annual`/`Climate_Tmp_Annual`'s `PPT*_AVG`/`TMP*_AVG` (60 cols
  each) have no native summary and are reduced to
  `climate_ppt_annual_mean_mm`/`_std_mm` and
  `climate_tmp_annual_mean_c`/`_std_c` (row-wise mean/std across the 1950–2009
  annual series per basin).
- Dynamic filters applied after the above, on the Stage 1 subset only:
  near-constant (`nunique(dropna=True) <= 1`) and high-missingness (>20%)
  `model_input` columns are excluded.
- Any non-numeric or unclassified column not covered by a rule above causes
  the build to fail loud (`sys.exit(1)`) rather than being silently included
  or dropped — this is a deliberate safety net against source-schema drift.

**HydroATLAS 5-basin gap — resolved, option (b) applied.** The builder
computes the observed HydroATLAS-missing basin set at build time and requires
it to equal exactly the 5 previously-audited 15-char coordinate-based STAIDs
(`393109104464500`, `394839104570300`, `401733105392404`, `402114105350101`,
`402913084285400`). If it matches, those 5 basins are retained with NaN
HydroATLAS-sourced columns and an explicit `hydroatlas_coverage_flag` column
(1 = present, 0 = known gap) written into the matrix. If the observed gap
ever differs from this expected set, the build fails loud instead of
proceeding — no silent partial merge is possible. Confirmed on the local
dry-run (§11.3): observed gap matched the expected set exactly.

**§11.3 — Local dry-run (validation of build/audit logic only; not the
canonical build).** Run against the local source mirror
`C:\PhD\Python\neuralhydrology\US_data\attributes` (checksum file not present
locally, so `--no-require-checksums` was used — the canonical h2o run must
use the default checksum-required path) into `tmp/stage1_static_attribute_matrix_v001_dryrun/`
(gitignored, not committed):

```
python scripts/build_stage1_static_attribute_matrix.py \
  --source-dir "C:/PhD/Python/neuralhydrology/US_data/attributes" \
  --manifest   config/stage1_initial_training_basin_manifest.csv \
  --out-dir    tmp/stage1_static_attribute_matrix_v001_dryrun \
  --no-require-checksums --force

python scripts/audit_stage1_static_attribute_matrix.py \
  --matrix-dir tmp/stage1_static_attribute_matrix_v001_dryrun \
  --matrix-name stage1_static_attributes_v001 \
  --manifest   config/stage1_initial_training_basin_manifest.csv
```

Result: build exit 0; matrix 2,843 rows × 531 columns (496 `model_input`, plus
`split_support`/`diagnostic_latlon`/`categorical_deferred`/flag columns);
15 near-constant HydroATLAS land-cover/PNV/wetland columns dynamically
excluded from `model_input` (all uniformly zero for this CONUS basin set);
HydroATLAS gap gate matched the expected 5-basin set exactly. Audit exit 0
(PASS), 0 errors, 0 warnings, 20 OK checks — including the checksum
round-trip (matrix sha256 in `_provenance.json` matches the recomputed file
hash) and the HydroATLAS-flagged-basin NaN check. One auditor threshold was
recalibrated during this dry-run: the numeric-range sanity check initially
flagged HydroATLAS's `gdp_ud_usu` (upstream-summed GDP, USD) at up to ≈$1.74
trillion for the largest basins as "implausibly large" at a 1e12 bound — this
is a legitimate basin-integrated economic aggregate, not a data error, so the
bound was raised to 1e13.

**§11.4 — What this dry-run does and does not establish.** It validates the
builder/auditor logic (classification rules, HydroATLAS gate, checksum
round-trip, exclusion filters) end-to-end against real data. It does **not**
constitute the canonical build — the canonical
`stage1_static_attributes_v001.parquet` must still be produced on h2o against
the h2o source mirror (with checksum verification required, not bypassed),
per §11.5. The dry-run output was left under repo `tmp/` (gitignored) and was
not copied anywhere else; it should be treated as disposable.

**Update (2026-07-08): the canonical h2o build/audit described in §11.5 has
now been run and PASSed.** See §11.6 for the result. This closes the gap
described above — a canonical, checksum-verified matrix now exists at the
path in §7.

**§11.5 — User-run commands for the canonical h2o build (not executable from
this session).**

```bash
# 1. Verify the h2o source mirror + checksums:
ssh flashnh-h2o "ls -la /data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/"
ssh flashnh-h2o "cd /data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001 && sha256sum -c source_attributes_v001_checksums.sha256"
# Confirm exactly 29 files + the checksum file itself (30 entries total in the listing),
# and no unexpected raw-data subdirectories present.

# 2. Copy the two new scripts to h2o (or pull via git if the repo is cloned there):
scp scripts/build_stage1_static_attribute_matrix.py scripts/audit_stage1_static_attribute_matrix.py \
    flashnh-h2o:/data42/omrip/Flash-NH/scripts/

# 3. Run the canonical build (checksum verification ON by default):
ssh flashnh-h2o "cd /data42/omrip/Flash-NH && python3 scripts/build_stage1_static_attribute_matrix.py \
  --source-dir /data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001 \
  --manifest   config/stage1_initial_training_basin_manifest.csv \
  --out-dir    /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001 \
  --matrix-name stage1_static_attributes_v001"

# 4. Run the auditor against the canonical output; must PASS (exit 0) before this
#    matrix is treated as usable for any downstream package build:
ssh flashnh-h2o "cd /data42/omrip/Flash-NH && python3 scripts/audit_stage1_static_attribute_matrix.py \
  --matrix-dir /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001 \
  --matrix-name stage1_static_attributes_v001 \
  --manifest   config/stage1_initial_training_basin_manifest.csv"
```

**§11.6 — Canonical h2o build/audit result (2026-07-08).** The user ran the
§11.5 commands directly on h2o (no network path exists from this session to
h2o). Results, as reported by the user:

- **Source mirror verification:** `/data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/`
  contains 30 files (29 source files + `source_attributes_v001_checksums.sha256`);
  `sha256sum -c source_attributes_v001_checksums.sha256` returned OK for all
  29 files.
- **Canonical build:** run with `--source-dir source_attributes_v001`,
  `--manifest config/stage1_initial_training_basin_manifest.csv`, `--out-dir
  stage1_static_attributes_v001`, `--matrix-name stage1_static_attributes_v001`
  (default checksum-required path, not bypassed).
- **Canonical audit: PASS.** 0 errors, 0 warnings, 20 OK checks. Matrix shape
  2,843 rows × 531 columns, 496 `model_input` columns. All Stage 1 basins
  present, no extra basins, no duplicate `gauge_id`, no non-numeric or
  ID/code-like `model_input` columns, `STATE`/`HUC02` excluded from
  `model_input` and retained as `split_support`, `LAT_GAGE`/`LNG_GAGE`
  excluded from `model_input` and retained as `diagnostic`. HydroATLAS
  coverage flag matched the expected 5-basin gap exactly (`393109104464500`,
  `394839104570300`, `401733105392404`, `402114105350101`, `402913084285400`),
  and those basins' HydroATLAS `model_input` columns are NaN as designed.
  Matrix checksum matched the provenance record.
- **Canonical artifact:** `/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001.parquet`,
  matrix sha256 `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`.
- **Output file sizes:** `stage1_static_attributes_v001.parquet` 8.8 MB;
  `stage1_static_attributes_v001_column_manifest.json` 58 KB;
  `stage1_static_attributes_v001_provenance.json` 20 KB;
  `stage1_static_attributes_v001_audit_summary.md` 1.7 KB.

These are h2o-resident data artifacts (generated outputs), not git-tracked
source files, consistent with `docs/repo_policy.md`.

**Not done in this milestone (by design, per explicit scope):** the full NH
package was not regenerated from this canonical matrix; no training was run;
no NH configs or Slurm scripts were modified; no Moriah mirror transfer of
the matrix has been performed or documented here.

## 12. Static-attribute semantic correction — sentinel decoding + role
reclassification (2026-07-20, superseding v001 for modeling)

**Context.** A comprehensive read-only semantic audit of all 496 canonical
`model_input` columns of `stage1_static_attributes_v001` (§11.6,
sha256 `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`,
preserved above as historical record — **not overwritten, not deleted**)
found a bounded set of semantic defects: 8 GAGES-II infrastructure-distance
columns carrying an undecoded `-999` "no feature within search radius"
sentinel that was being treated as a valid distance; direct gauge/basin
coordinates and gauge-record/network/QA metadata classified as `model_input`
despite not describing basin physical/hydro-environmental attributes; two
columns (`PERHOR`, `STRAHLER_MAX`) with a documented missing-value sentinel
not yet decoded; and one HydroATLAS field (`lka_pc_use`) with unresolved
catalog semantics. The audit confirmed these defects were bounded and did not
implicate the remaining ~485 columns, the HydroATLAS 5-basin gap policy, or
any target/split/training code. Full binding-decision detail:
`docs/decision_log.md` (2026-07-20 entry) and
`docs/stage1_scientific_baseline_design.md` §3.

**Implementation.** `scripts/build_stage1_static_attribute_matrix.py` gained:

- `_SENTINEL_VALUES_BY_COLUMN`: an explicit per-column sentinel map (`-999.0`
  for the 8 `RAW_*` infrastructure-distance columns, `-9999.0` for
  `NWIS_DRAIN_SQKM`/`PCT_DIFF_NWIS`/`PERHOR`, `-99.0` for `STRAHLER_MAX`).
  `_decode_column_sentinels()` converts exact matches to `NaN` **before**
  role classification and the missingness calculation, for the listed column
  only — no blanket negative-value replacement, no other column touched even
  if it happens to contain the same literal number. Non-numeric values in a
  mapped column fail the build loud (`sys.exit(1)`). Per-column replacement
  counts (including legitimate zero) are recorded in `_provenance.json` under
  `sentinel_decoding`, keyed by `<source_file>:<column>`, alongside the
  sentinel map used and an algorithm id (`stage1_static_sentinel_decode_v1`).
- Two new column roles: `diagnostic_record_network_qa` (gauge-record history
  / gauge-network membership / boundary-processing QA metadata — 12 columns:
  `FLOWYRS_1900_2009`, `FLOWYRS_1950_2009`, `FLOWYRS_1990_2009`,
  `FLOW_PCT_EST_VALUES`, `BASIN_BOUNDARY_CONFIDENCE`, `ACTIVE09`, `HBN36`,
  `HCDN_2009`, `OLD_HCDN`, `NSIP_SENTINEL`, `PCT_DIFF_NWIS`,
  `NWIS_DRAIN_SQKM`) and `deferred_ambiguous` (`lka_pc_use` only, pending
  resolution of exact HydroATLAS catalog semantics). `diagnostic_latlon` was
  extended from `{LAT_GAGE, LNG_GAGE}` to also include `LAT_CENT`,
  `LONG_CENT` (basin-centroid coordinates, same direct-location rationale).
  `NWIS_DRAIN_SQKM`/`PCT_DIFF_NWIS` still get their `-9999` sentinel decoded
  for provenance/validation even though they now land in the diagnostic role,
  not `model_input`.
- **Mechanism, not hand-classification, excludes the 8 `RAW_*` columns.**
  They are still classified `candidate_model_input` by `_classify_columns()`;
  after sentinel decoding their missingness legitimately exceeds the existing
  `>20%` threshold (24–93% missing across the 8 columns in practice — most
  basins have no dam/canal/NPDES outfall within the GAGES-II search radius),
  so the pre-existing dynamic near-constant/high-missingness filter in
  `build()` excludes them exactly as it would any other high-missingness
  column. This was verified explicitly (see local dry-run below): the 8
  columns appear in `provenance.json`'s `high_missing_excluded_model_input`
  list, not in any new hand-authored exclusion set. One consequence: the
  pre-existing "reliably numeric" gate in `_load_and_classify()` (fails the
  build if `notna().mean() < 0.90` for an unclassified numeric column) is
  bypassed specifically for sentinel-mapped columns, since their numeric-ness
  is already fail-loud-validated by `_decode_column_sentinels()` and their
  post-decode missingness is expected and legitimate, not a schema-drift
  signal.
- `PERHOR` and `STRAHLER_MAX` remain `model_input` after decoding (both stay
  under the 20% missingness threshold); `dor_pc_pva`, `dis_m3_pyr`,
  `run_mm_syr` are unchanged (still `model_input`, no sentinel, no role
  change) per the binding decision that these are valid baseline predictors
  distinct from the excluded fields.

`scripts/audit_stage1_static_attribute_matrix.py` independently mirrors the
same sentinel map and role sets (not imported from the builder, by design)
and gained hard-fail checks: direct coordinates (`LAT_GAGE`/`LNG_GAGE`/
`LAT_CENT`/`LONG_CENT`) as `model_input`; any of the 12
`diagnostic_record_network_qa` fields as `model_input`; `lka_pc_use` as
`model_input`; any of the 8 `RAW_*` infrastructure-distance columns surviving
as `model_input`; any literal mapped sentinel value remaining in a
`model_input` column; and column-manifest/matrix-column inconsistency. Plus
positive checks that `PERHOR`/`STRAHLER_MAX` remain `model_input` with zero
remaining sentinel values, and that `dor_pc_pva`/`dis_m3_pyr`/`run_mm_syr`
remain `model_input`.

**Local dry-run (validation only, not the canonical build).** Run against
the local, checksum-unverified source fixture
`C:\PhD\Python\neuralhydrology\US_data\attributes` into
`tmp/stage1_static_attribute_matrix_v002_dryrun/` (gitignored, not
committed):

```
python scripts/build_stage1_static_attribute_matrix.py \
  --source-dir "C:/PhD/Python/neuralhydrology/US_data/attributes" \
  --manifest   config/stage1_initial_training_basin_manifest.csv \
  --out-dir    tmp/stage1_static_attribute_matrix_v002_dryrun \
  --matrix-name stage1_static_attributes_v002 \
  --no-require-checksums --force

python scripts/audit_stage1_static_attribute_matrix.py \
  --matrix-dir tmp/stage1_static_attribute_matrix_v002_dryrun \
  --matrix-name stage1_static_attributes_v002 \
  --manifest   config/stage1_initial_training_basin_manifest.csv
```

Result: build exit 0; matrix 2,843 rows × 523 columns, **473 `model_input`**
(provisional at the time — this local source mirror was not
checksum-verified against the h2o mirror, so this count was not the
acceptance criterion for the canonical rebuild. **The canonical h2o rebuild
has since matched this count exactly and is authoritative — see §13.**). All
8 `RAW_*` columns excluded via
`high_missing_excluded_model_input` (15,018 total sentinel values replaced
across the 12 mapped columns). Audit exit 0 (PASS), 0 errors, 0 warnings, 32
OK checks, including all new hard-fail checks passing and the two new
positive checks (`PERHOR`/`STRAHLER_MAX` retained, no residual sentinels).

**Tests.** `tests/test_static_attribute_matrix.py` (new, 19 tests): sentinel
decoding (per-mapped-column, non-sentinel/unrelated-column non-interference,
zero-count visibility, non-numeric fail-loud, blank-value passthrough), role
classification (coordinates, record/network/QA, deferred, retained fields),
an end-to-end synthetic-fixture build proving the `RAW_*` exclusion mechanism,
and auditor regressions (PASS on a corrected fixture; hard-fail on a
surviving sentinel, a leaked coordinate, a leaked record/network/QA field,
a leaked `lka_pc_use`, a leaked `RAW_*` column).

**Status at the time this section was written (2026-07-20, commit-only
closure):** the corrected canonical rebuild had not yet been run on h2o; no
NH package was built; no training ran; Moriah was not used; the compact
static-imputation artifact had not been rerun; nothing beyond this docs+code
patch was committed. **This status is superseded — see §13 below**, which
records the canonical h2o rebuild, independent audit PASS, and compact
static-imputation v002 acceptance that followed.

## 13. Canonical v002 matrix + compact static-imputation v002 — ACCEPTED (2026-07-20)

**Canonical rebuild.** The §11.5-style h2o commands (recorded in full, with
the `stage1_static_attributes_v002` path substitution, in
`docs/decision_log.md`'s 2026-07-20 correction entry) were run on h2o and
their results reviewed and accepted by the user. Source-checksum
verification: 29/29 files PASS. Canonical path:
`/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/`
(`stage1_static_attributes_v002.parquet`,
`stage1_static_attributes_v002_column_manifest.json`,
`stage1_static_attributes_v002_provenance.json`,
`stage1_static_attributes_v002_audit_summary.md`).

**Matrix shape:** 2,843 rows × 523 total columns. Column-role breakdown:
473 `model_input` (authoritative — no longer provisional, supersedes the
473 figure reported as a local-dry-run estimate in §12 above), 2
split-support, 4 diagnostic lat/lon, 12 diagnostic record/network/QA, 1
deferred-ambiguous (`lka_pc_use`), 29 categorical-deferred, 2 flag. Sentinel
algorithm `stage1_static_sentinel_decode_v1`, 15,018 total values decoded.
The 8 infrastructure-distance `RAW_*` columns are excluded through the
existing `>20%` high-missingness mechanism, exactly as designed in §12 — not
by name. `PERHOR` and `STRAHLER_MAX` retained `model_input` with sentinels
decoded; `dor_pc_pva`/`dis_m3_pyr`/`run_mm_syr` retained unchanged. Direct
coordinate, record/network/QA, and `lka_pc_use` exclusions all verified. The
HydroATLAS 5-basin gap is unchanged from v001 and explicitly handled.

**Independent audit:** `scripts/audit_stage1_static_attribute_matrix.py` —
PASS, 0 errors, 0 warnings, 32 OK checks.

**Checksums.**

```
matrix (stage1_static_attributes_v002.parquet):
4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297

column manifest (stage1_static_attributes_v002_column_manifest.json):
02505eb4893e6848f7cbc4eabd2cdf40dd6aee64156d41744aebcbe4409f0e00

provenance (stage1_static_attributes_v002_provenance.json):
983b9f9ff187c4dfc2e8a6d7453929b31b006ff099d7682b3c1c7b348c55f022

audit summary (stage1_static_attributes_v002_audit_summary.md):
247cae508338cc51d18bc22dfd7d0124b459c5e4c12ebd07e848f66a88211f4a
```

The full 473-column `model_input` list is not reproduced here — see the
canonical column manifest above. The §11.6 v001 artifact and checksum
(`eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`) remain
preserved as the historical record of the 2026-07-08 canonical build — not
deleted, not invalid — but are superseded for modeling by v002.

**Compact static-imputation v002.** Rebuilt via
`scripts/prepare_stage1_compact_static_attributes.py` (algorithm
`stage1_static_median_imputation_v1`, primitive unchanged from the v001
run) against the accepted v002 matrix. Canonical generated output path:
`/data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002`. Input
matrix checksum matches the v002 canonical checksum above exactly. Output
shape 32 basins × 473 `model_input` columns; fit scope
development-training-only, fit population 2,307 basins; applied to the 32
accepted compact basins; all fit columns had valid medians; 168 total values
imputed, all on exactly one basin (`393109104464500`, the designated
compound-edge-case diagnostic basin — `docs/FLASHNH_CURRENT_STATE.md`); zero
remaining NaNs.

```
imputed_static_attributes.parquet:
3d476c41dda2c95481a76f7a97e288929e317b8ed0798cb4ddaa00bf4615b92e

imputed_value_mask.parquet:
61bbceb2f1643ef9184524f8c9e3c90a666396c9b44272b879c9803fcfa46796
```

`stage1_compact_static_imputation_v001` remains preserved as historical
provenance, superseded for modeling by v002.

**Not reopened / unaffected by this acceptance:** the compact selector
(`scripts/generate_stage1_compact_package_selection.py`) and canonical split
artifacts (`config/stage1_baseline_splits_v001/`) were not rerun; the
accepted 32-basin Compact Scientific Package selection remains valid as-is;
the static-imputation primitive (`src/baseline/static_preparation.py`) is
unchanged code. **Not done as of this docs-only closure:** no NH package has
been built; no training has run; nothing beyond this documentation update
has been committed.
