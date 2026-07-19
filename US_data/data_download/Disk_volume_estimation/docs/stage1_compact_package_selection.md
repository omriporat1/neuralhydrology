# Flash-NH Stage 1 — Compact Scientific Package Basin Selection

Date: 2026-07-19 (corrected same day — see "Correction history" below)
Status: **Selector implemented and locally validated against the real canonical
`split_assignment.csv` (2,307-basin `development_train` pool). This produces the
*split-based candidate* only (enrichment inputs — static attributes, qobs status —
are absent locally, so those columns are `not_evaluated`). The final accepted
Compact Scientific Package requires the fully enriched run on h2o — see
"Two selection runs, not one" below. That h2o run has not yet happened.**

### Correction history

- **policy_version 1 / algorithm_version 1** (2026-07-19, morning): initial
  implementation. A first local dry-run against the real `split_assignment.csv`
  surfaced a real defect: the 32-basin candidate's HUC02 distribution was
  `01:8, 02:9, 03:7, 04:3, 05:2, 06:1, 10L:2` — no ordinary western,
  southwestern, Pacific Northwest, or southern-Plains basin at all.
- **policy_version 2 / algorithm_version 2** (2026-07-19, same day): corrected.
  Root cause and fix are in "Method" and "Geographic diagnostics" below. This
  is a real behavioral change — the same seed now produces a different (and
  geographically unbiased) selection than algorithm_version 1 did.

---

## Purpose

The Compact Scientific Package is a small (~20–50, default **32**) diagnostic basin
subset drawn strictly from the canonical, already-frozen **`development_train`** pool
(2,307 non-California basins — see [stage1_scientific_baseline_design.md](stage1_scientific_baseline_design.md)
and `config/stage1_baseline_splits_v001/`). It exists to give a fast, reproducible,
independently checkable diagnostic set for early model iteration before committing to
the full 2,752-basin package.

This document covers **only** the deterministic basin-selection step:

- `config/stage1_compact_package_selection_v001.yaml` — versioned selection policy
- `src/baseline/compact_selection.py` — selection library
- `scripts/generate_stage1_compact_package_selection.py` — CLI entry point
- `tests/test_compact_package_selection.py` — synthetic-fixture test suite

**Explicitly out of scope for this step** (and not touched by any file above): building
per-basin NeuralHydrology NetCDF files, `FlashNHDataset`/the NH launcher, training,
building the full 2,752-basin package, or editing the frozen `scripts/build_stage1_nh_package.py`
/ `scripts/audit_stage1_nh_package.py` pair.

---

## Inputs

| Input | Required? | Description |
|---|---|---|
| `--split-assignment` | required | Canonical split assignment CSV, e.g. `config/stage1_baseline_splits_v001/split_assignment.csv`. Must contain `STAID, split_role, STATE, HUC02, area_class, hydro_class`. The selector filters to `split_role == development_train` itself — it never infers pool membership from any other artifact (e.g. the 2,752-basin target package). |
| `--policy` | optional (defaults to `config/stage1_compact_package_selection_v001.yaml`) | Versioned selection policy: seed, target count, bin thresholds, reserved-category caps. |
| `--attributes-parquet` + `--column-manifest` | optional, paired | Checksum-verified static-attribute matrix + its column-role manifest JSON (`stage1_static_attributes_v001`). Used only to annotate `DRAIN_SQKM`, `ari_ix_uav`, and the `model_input`-role missingness count. If omitted, those columns are recorded as `not_evaluated` and no basin is force-included on this basis. |
| `--qobs-status` | optional | A qobs completeness/target-status CSV (e.g. `audit/target_status.csv`, schema: `STAID, target_status, coverage_fraction, ...` — see `scripts/audit_usgs_iv_recovered_targets.py`). Accepts either a `coverage_fraction` or `qobs_coverage` column. If omitted, completeness is recorded as `not_evaluated`. |
| `--gap-manifest` | optional | A JSON list of ISO gap timestamps, recorded for **provenance only**. Archive gaps (MRMS/RTMA) are global timeline positions, not a per-basin property — see "Forcing-gap handling" below. |

`--dry-run` computes and prints the plan without writing anything. `--out-dir` is
required unless `--dry-run` is set; `--force` allows writing into a non-empty
`--out-dir`.

---

## Method

**Algorithm:** `stage1_compact_diversity_quota_selection_v1` (fixed seed `42`,
read from the policy YAML, `numpy.random.SeedSequence` used throughout — no Python
built-in `hash()`).

1. **Universe.** Filter `split_assignment.csv` to `split_role == development_train`.
   Fail if that role is absent, if any filtered row has `STATE == "CA"`, if any
   filtered STAID also appears under a forbidden role elsewhere in the file, or if
   there are duplicate STAIDs. Gauge IDs are normalized via `src/baseline/staid.py`'s
   `normalize_staid` (zero-padded to 8 chars unless already 8/9/15 chars).

2. **Reserved edge cases.** Three categories are evaluated in a fixed order, each
   contributing up to `forced_include_cap` (default 1) new reserved basins:
   - `unusual_identifier` — normalized STAID length ≠ 8 (the known 9-char and
     15-char USGS IDs in the pool).
   - `hydro_stratifier_gap` — `hydro_class` not in `{low, middle, high}` (the
     canonical "missing aridity stratifier" basins from `src/baseline/splits.py`).
     Any basin in this category is unconditionally excluded from the stratification
     grid below (it cannot occupy a valid cell) regardless of reserved status.
   - `static_missing_value_case` — `static_missing_model_input_count > 0`. Only
     evaluated when `--attributes-parquet`/`--column-manifest` are supplied;
     otherwise logged as `not_evaluated_no_attributes_input` (not an error).

   A basin may satisfy more than one predicate. **Overlap is handled
   parsimoniously**: before spending a new pick on a category, the selector checks
   whether a basin *already reserved by an earlier category* also satisfies this
   category's predicate — if so, no new basin is picked; the category is logged as
   `"covered_by_overlap"` pointing at the covering basin's STAID. This matters in
   practice: in the real 2,307-basin pool, all 5 basins with a 15-char STAID also
   have `hydro_class == "missing"` — i.e. `unusual_identifier` and
   `hydro_stratifier_gap` are the *same* 5-basin edge-case cluster, not two
   independent ones. When a category does need a new pick, candidates are ordered
   by `(-number_of_predicates_matched, STAID)` — i.e. a candidate satisfying more
   predicates simultaneously is preferred — so a single basin naturally absorbs
   overlapping edge cases instead of one near-duplicate basin per category. If a
   category has zero eligible (and zero already-covering) candidates, the run
   proceeds without it and this is recorded (not a failure) in
   `selection_summary.md`/`.json`.

3. **Diversity quota.** The remaining budget (`target_count` minus reserved picks)
   is allocated across a 3×3 `area_class × hydro_class` grid (the canonical
   split's own tercile fields — **no new tercile edges are computed**) via
   largest-remainder (Hamilton) apportionment, guaranteeing ≥1 pick per non-empty
   cell where the budget allows (deterministic trim, largest cells first, if the
   guaranteed minimums alone exceed the remaining budget — relevant mainly to small
   test fixtures).

4. **Within-cell sampling.** Each cell's eligible basins are grouped by `HUC02`.
   Each group is shuffled once with an independent, seeded
   `numpy.random.default_rng` (`SeedSequence(seed).spawn(n_cells)`), then basins
   are picked by round-robin across HUC02 groups **in a seeded permuted visitation
   order** — not ascending HUC02-code order. This is a corrected behavior
   (`algorithm_version` 2; see "Correction history" above): visiting groups in
   fixed ascending order systematically favored low-numbered HUC02 codes whenever
   a cell's quota (typically 2-6, from the largest-remainder apportionment) was
   smaller than the number of distinct HUC02 groups available in that cell (up to
   17) — which is the common case. The fix draws `rng.permutation()` over the
   sorted HUC02-key list once per cell to determine visitation order (the sort is
   only used to fix a stable *input* order for the permutation draw, not the
   visitation order itself); subsequent round-robin rounds reuse that same
   permuted order. Both the per-group candidate shuffle and the visitation-order
   permutation are drawn from the same per-cell RNG in a fixed sequence (group
   shuffles first, in sorted-key order; then the visitation permutation), so the
   result remains **byte-deterministic for identical inputs and seed**, and a
   different seed produces a different (but still deterministic) visitation order.
   This still makes geographic breadth a **soft tie-break**, not a hard per-HUC02
   quota — a hard quota would over-fragment a 32-basin target across up to 17
   HUC02s × 9 cells. `geography.distinct_huc02_soft_minimum` (default 6) is a
   warning-only threshold recorded in the summary, never a failure condition.

5. **Assembly.** Selected IDs = reserved ∪ grid picks, sorted ascending by
   `gauge_id`. Fails loudly on any count mismatch, duplicate, or out-of-pool basin.

6. **Macro-region geographic diagnostics (hard check).** See
   "Geographic diagnostics" below — this step can fail the run.

### Forcing-gap handling

MRMS/RTMA archive gaps are **global timeline positions**, not a basin property —
one basin cannot be said to "have more gaps" than another. `--gap-manifest` is
therefore recorded only as provenance (path, sha256, timestamp count) and is
**never** used to differentiate or rank basins during selection. The intent
(dimension 7 of the design) is that the *later* per-basin package build/audit for
the selected 32 basins should be checked for coverage around representative global
gap intervals — that check belongs to the package-build step, not this selector.

---

## Geographic diagnostics (macro-region + east/west breadth)

Added in `algorithm_version` 2 to catch the class of defect described in
"Correction history" above (a candidate concentrated almost entirely in
HUC01-06) *structurally*, not just by eyeballing the HUC02 table.

`config/stage1_compact_package_selection_v001.yaml`'s `geography.macro_regions`
is an **explicit, versioned HUC02 → macro-region mapping** (`macro_region_map_version:
1`) — never inferred from HUC02 numeric ordering. Every HUC02 code is listed under
exactly one of 8 macro regions:

| Macro region | HUC02 codes |
|---|---|
| `northeast_mid_atlantic` | 01, 02 |
| `southeast` | 03 |
| `great_lakes_ohio_tennessee` | 04, 05, 06 |
| `mississippi` | 07, 08 |
| `plains_missouri_south_central` | 09, 10U, 10L, 11, 12, 13 |
| `colorado_great_basin` | 14, 15, 16 |
| `pacific_northwest_california` | 17, 18 |
| `alaska_hawaii_other` | 19, 20, 21 |

`geography.east_macro_regions` / `west_macro_regions` classify 7 of these 8 as
"east" (`northeast_mid_atlantic`, `southeast`, `great_lakes_ohio_tennessee`,
`mississippi`) or "west" (`plains_missouri_south_central`, `colorado_great_basin`,
`pacific_northwest_california`); `alaska_hawaii_other` is neither (`"other"`) and is
not expected to appear in the CONUS `development_train` pool.

California is still excluded strictly by `STATE == "CA"` in `select_universe()`
(unchanged) — **not** by HUC code. HUC18 is listed in `macro_regions` for
completeness/portability, but no non-CA HUC18 basin currently exists in the
canonical split artifact; if one is ever added, this mapping (not a HUC-code
heuristic) is what would classify it.

Two checks are computed per run, over the *final* selected set:

- **`distinct_macro_region_soft_minimum`** (default 3): advisory only, like
  `distinct_huc02_soft_minimum` — recorded in the summary, never fails the run.
- **`require_east_west_spread`** (default `true`): a **hard failure**
  (`SelectionError`, not a warning) if either `n_east == 0` or `n_west == 0` in the
  final selection. This is deliberately strict — a compact package with no
  ordinary western or no ordinary eastern basin defeats its purpose as a
  nationally representative diagnostic set. If this triggers, it means the
  diversity-quota cells (area × hydro) happened to draw entirely from one side of
  CONUS for this seed/target-count combination; the response is to adjust the
  seed, target count, or macro-region policy — **not** to force equal counts
  across macro regions, since area × hydro balance remains the primary
  stratification and this check exists only to prevent pathological
  concentration, not to impose a geographic quota.

Per-basin `macro_region` / `macro_region_side` and run-level
`macro_region_counts` / `macro_region_side_counts` / `distinct_macro_regions` /
`east_west_breadth` are recorded in the CSV and manifest — see below.

---

## Output schema

`compact_basin_selection.csv` columns:

| Column | Description |
|---|---|
| `gauge_id` | Normalized STAID (leading zeros preserved) |
| `canonical_basin_role` | Always `development_train` |
| `huc02` | From the split assignment |
| `macro_region`, `macro_region_side` | From `geography.macro_regions` (see "Geographic diagnostics" above); `macro_region_side` is `east`/`west`/`other` |
| `drain_sqkm`, `aridity_value` | From the attribute matrix, if supplied |
| `area_class`, `hydro_class` | Canonical tercile classes reused from the split |
| `qobs_coverage_fraction`, `qobs_completeness_bin` | From `--qobs-status`, if supplied (`low`/`mid`/`high`/`not_evaluated`) |
| `target_status` | From `--qobs-status`, if that column is present |
| `static_missing_model_input_count`, `static_missing_bin` | From the attribute matrix, if supplied (`none`/`some`/`high`/`not_evaluated`) |
| `unusual_identifier_flag` | Boolean |
| `selection_reason` | Semicolon-joined reasons, e.g. `hydro_stratifier_gap;unusual_identifier` or `diversity_quota:area=low;hydro=middle` |

Full artifact set written under `--out-dir`:

```
<out-dir>/
  compact_basin_selection.csv
  compact_basin_ids.txt        # one gauge_id per line, sorted
  selection_summary.md         # human-readable: counts, reserved-category log, per-dimension tables, gap note
  selection_summary.json       # machine-readable subset of the manifest
  selection_manifest.json      # full manifest: policy/input provenance + sha256, algorithm id/version,
                                # seed, cell sizes/quota/repair log, per-dimension counts, artifact_sha256
  run_command.txt              # exact CLI invocation used
```

---

## Limitations

- The selector trusts `split_assignment.csv`'s own `area_class`/`hydro_class`
  fields; it does not recompute or cross-check tercile edges against the raw
  attribute matrix.
- `qobs_completeness` and `static_missingness` are informational annotations, not
  stratification-grid dimensions — a basin is never force-included solely for a
  qobs-completeness bin (only the three reserved categories force-include).
- `distinct_huc02_soft_minimum` and `distinct_macro_region_soft_minimum` are
  advisory only; a run can pass with fewer distinct HUC02s/macro-regions than the
  soft minimum if the diversity grid pulls that way. `require_east_west_spread` is
  the one geographic check that is a hard failure, not advisory (see "Geographic
  diagnostics" above).
- The local run described below is the **split-based candidate only** — it does
  not yet reflect static-attribute or qobs-completeness annotations (those inputs
  are not available outside h2o). See "Two selection runs, not one" below.

---

## Two selection runs, not one

This selector can be, and has been, run twice with two different meanings. **Do
not confuse the two:**

1. **Local split-based candidate** (this repo, no h2o access). Input is only
   `config/stage1_baseline_splits_v001/split_assignment.csv` — no
   `--attributes-parquet`/`--column-manifest`/`--qobs-status`. `drain_sqkm`,
   `aridity_value`, `qobs_completeness_bin`, and `static_missing_bin` are all
   recorded as `not_evaluated`/`NaN`, and `static_missing_value_case` cannot be
   evaluated at all (logged as `not_evaluated_no_attributes_input`). This run
   exercises the real 2,307-basin `development_train` pool and is useful for
   validating the algorithm (geographic breadth, reserved-category overlap,
   determinism) end-to-end, but it is **explicitly not the final accepted
   Compact Scientific Package** — `manifest.json`'s `"status"` field is always
   `"candidate"`.
2. **Fully enriched h2o canonical selection** (not yet run). Adds the
   checksum-pinned static-attribute matrix (for `drain_sqkm`/`aridity_value`/
   static-missingness bins and the `static_missing_value_case` reserved category)
   and the qobs/target-status table (for `qobs_completeness_bin`/`target_status`).
   This is the run whose output should actually be reviewed/accepted as the
   Compact Scientific Package basin list — see "Exact h2o command" below.

## Local validation (synthetic fixtures)

```bash
python -m py_compile src/baseline/compact_selection.py \
    scripts/generate_stage1_compact_package_selection.py \
    tests/test_compact_package_selection.py

python -m pytest tests/test_compact_package_selection.py -v
# 48 passed

python -m pytest tests/ -q --ignore=tests/test_nh_dataset.py --ignore=tests/test_nh_register.py
# 380 passed (the two ignored files require the `neuralhydrology` package,
# which is not installed in this local venv; pre-existing gap, unrelated to
# this change)
```

## Local split-based candidate run

Against the actual canonical `split_assignment.csv` (2,307 `development_train`
basins; enrichment inputs absent, so those columns are `not_evaluated` — see
"Two selection runs, not one" above):

```bash
python scripts/generate_stage1_compact_package_selection.py \
    --split-assignment config/stage1_baseline_splits_v001/split_assignment.csv \
    --policy config/stage1_compact_package_selection_v001.yaml \
    --out-dir tmp/stage1_compact_package_selection_v001 \
    --force
```

See `tmp/stage1_compact_package_selection_v001/` for the generated artifacts (not
committed — generated output, per `docs/repo_policy.md`).

---

## Exact h2o command (fully enriched canonical selection)

The checksum-pinned canonical static-attribute matrix is already built and
audit-PASS on h2o (`docs/FLASHNH_CURRENT_STATE.md`, 2026-07-08):
`/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/`
(**not** `data/static_attributes/gagesii_v001/` — that is a different, older,
smaller artifact: 48 columns from only 3 of 27 GAGES-II source tables, no
topography; see `docs/decision_log.md`. It is not an alias of
`stage1_static_attributes_v001` and must not be used here).

`--gap-manifest` is recorded for **provenance only** (never used to
differentiate basins — see "Forcing-gap handling" above) and is a
package-instance-specific `gap_timestamps.json` (written under an NH package's
own `data_dir/masks/`, per `src/baseline/nh_dataset.py`'s convention) — there is
no single global canonical path for it. Resolve
`<CANONICAL_GAP_TIMESTAMPS_JSON>` below to the `gap_timestamps.json` produced by
whichever NH package build (e.g. the Milestone 2K-G-B / 2K-G-C 50-basin package)
is being used as the current forcing-gap reference at run time; if none exists
yet, omit `--gap-manifest` entirely rather than inventing a path.

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1
cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation

python scripts/generate_stage1_compact_package_selection.py \
    --split-assignment config/stage1_baseline_splits_v001/split_assignment.csv \
    --attributes-parquet /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001.parquet \
    --column-manifest /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001_column_manifest.json \
    --qobs-status /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv \
    --gap-manifest <CANONICAL_GAP_TIMESTAMPS_JSON> \
    --policy config/stage1_compact_package_selection_v001.yaml \
    --out-dir /data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001 \
    --force
```

This is a **selection-only** command — it does not build NH packages or launch
training. Building per-basin NH NetCDF files for the selected basins is a
separate, later step.
