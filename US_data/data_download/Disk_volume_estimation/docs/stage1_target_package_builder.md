# Flash-NH Stage 1 — Target Package Builder and Auditor

Milestone: 2J-B
Date: 2026-06-16
Status: **Scripts implemented, h2o policy smoke PASS, full v001 build COMPLETE (2026-06-16)**

---

## Purpose

`scripts/build_stage1_target_package.py` reads the canonical hourly USGS IV
NetCDF files produced by `recover_usgs_iv_full_period_hourly.py` and assembles a
NeuralHydrology-compatible streamflow-target package. It applies the target policy
from `config/stage1_target_policy.yaml` so that all inclusion and cleaning decisions
are driven by config, not hard-coded in the script.

`scripts/audit_stage1_target_package.py` verifies the output package for structural
correctness, value integrity, and policy compliance.

The original 2J-B design milestone did not execute the full 2,843-basin build on h2o;
that step followed after the design was validated with a local smoke and an h2o policy smoke.
The conservative v001 target package (2,752 basins) was subsequently built and audited
on h2o on 2026-06-16 (PASS — 0 errors, 0 warnings). This document records both
the design/smoke validation and the v001 build result.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/build_stage1_target_package.py` | Builder — reads canonical NCs, applies policy, writes target NCs |
| `scripts/audit_stage1_target_package.py` | Auditor — verifies output structure, values, checksums, policy compliance |

---

## Inputs

| Input | Description |
|---|---|
| `--canonical-dir` | Directory containing `<STAID>_hourly.nc` files (flat or sharded under shard_NN/canonical/) |
| `--policy` | `config/stage1_target_policy.yaml` |
| `--status-csv` | (optional) `audit/target_status.csv` from 2,843-basin acquisition audit; required for full policy enforcement |
| `--staids` | Comma-separated STAIDs (smoke / targeted builds) |
| `--max-basins N` | Limit to first N basins (smoke mode) |
| `--exclude-staids` | Unconditionally skip listed STAIDs |
| `--allow-review-required` | Override `review_required` halt for named special-review STAIDs |
| `--force` | Overwrite existing output NCs |

---

## Output layout

```
<out-dir>/
  time_series/
    <STAID>.nc           # one per basin
  manifest.json          # basin list, counts, policy info, cleaning summary
  checksums.sha256        # SHA-256 of every time_series/*.nc
  run_provenance.json    # script, args, timestamp, policy hash
  cleaning_report.csv    # per-basin cleaning stats
```

### Per-basin NC structure

- **Coordinate:** `date` (hourly UTC, `datetime64[ns]`, no tz offset stored)
- **Variable:** `qobs_m3s` (`float32`, units `m3 s-1`, `_FillValue=-9999.0`)
- **Encoding:** `_FillValue=-9999.0` on disk; xarray decodes back to `NaN` on read
- Compatible with NeuralHydrology `GenericDataset` format

---

## Policy behavior

All policy decisions are read from `config/stage1_target_policy.yaml`. Changing
policy means editing the YAML (and bumping the version), not editing this script.

| Rule | Behavior |
|---|---|
| `include_if_historical_utility: true` | Include basins with `historical_training_utility_flag=True` (requires `--status-csv`) |
| `exclude_status: [TARGET_OPERATIONAL_REVIEW]` | Held out of first package (requires `--status-csv`) |
| `quality_review_eligible: true` | `TARGET_QUALITY_REVIEW` basins remain eligible; spike flag is advisory |
| `set_negative_to_nan: true` | All negative `qobs_m3s` values set to NaN |
| `preserve_existing_nan: true` | Existing NaN preserved; no backfilling |
| `interpolation: none` | No interpolation at any point |
| `gap_filling: none` | No gap filling at any point |
| `imputation: none` | No imputation at any point |
| `alter_positive_obs: false` | Non-negative observations are never altered |

### Smoke / permissive mode

If `--status-csv` is not provided, the builder operates in **permissive smoke mode**:
it processes all basins found in `--canonical-dir` (subject to `--max-basins` / `--staids`
limits). The `historical_training_utility_flag` and `target_status` filters are
**not applied** — they require the audit CSV from the full h2o acquisition.

Negative-cleaning and special-review rules are **always enforced**, regardless of mode.

---

## Special-review basins

Two basins have `first_package_action: review_required` in the policy:

| STAID | n_negative_values | Default action |
|---|---|---|
| `02299472` | 2,605 | `review_required` → builder HALTS |
| `04073468` | 2,054 | `review_required` → builder HALTS |

The builder will halt with a clear error message if either of these basins is
encountered in the candidate set, unless:

- `--allow-review-required 02299472,04073468` is passed (includes them with cleaning)
- `--exclude-staids 02299472,04073468` is passed (skips them entirely)
- They are not in the candidate set (absent from `--canonical-dir` or outside `--max-basins`)

This is intentional: it forces a conscious include/exclude decision rather than
silently absorbing basins with >2,000 negative streamflow values.

---

## Smoke build result (2026-06-15)

Smoke run used 5 of the 7 locally available full-period canonical NCs from Milestone 2I-B.

**Command:**

```bash
python scripts/build_stage1_target_package.py \
    --canonical-dir tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot/canonical \
    --policy config/stage1_target_policy.yaml \
    --out-dir tmp/stage1_target_package_smoke \
    --max-basins 5 \
    --force
```

**Build result: PASS**

| Basin | Valid hours | NaN hours | Neg cleaned |
|---|---|---|---|
| `01585200` | 45,584 | 136 | 0 |
| `02073000` | 45,720 | 0 | 0 |
| `02077670` | 43,309 | 2,411 | 0 |
| `02266500` | 45,717 | 3 | 0 |
| `02344700` | 45,679 | 41 | 0 |

- Negative values cleaned total: **0** (none of the 5 smoke basins have negative values)
- NaN before = NaN after = **2,591** (existing gaps preserved, no cleaning needed)
- Build time: **0.5s**

**Audit command:**

```bash
python scripts/audit_stage1_target_package.py \
    --package-dir tmp/stage1_target_package_smoke \
    --policy config/stage1_target_policy.yaml \
    --expected-basins 5
```

**Audit result: PASS — 0 errors, 0 warnings**

| Check | Result |
|---|---|
| Required package files (manifest, checksums, provenance, cleaning report) | PASS |
| Basin count (5 == 5) | PASS |
| SHA-256 checksums (5/5) | PASS |
| Per-basin NC: qobs_m3s exists, units correct | PASS |
| Per-basin NC: date coordinate hourly monotonic | PASS |
| Per-basin NC: no decoded -9999.0 values | PASS |
| Per-basin NC: no negative qobs after cleaning | PASS |
| Per-basin NC: NaN counts consistent with cleaning report | PASS |
| Special-review basins (02299472, 04073468): absent from smoke set | PASS |

---

## Full-run command (h2o)

**COMPLETE (2026-06-16).** The v001 build ran via the committed launcher script (commit `3ac51ff`).
See [Full h2o build result (2026-06-16)](#full-h2o-build-result-2026-06-16) below.

```bash
# For reference — command used for v001:
screen -S flashnh_target_v001
bash scripts/run_stage1_target_package_v001_h2o.sh
```

Pass `--force` to overwrite an existing output directory.
Run `bash scripts/run_stage1_target_package_v001_h2o.sh --help` for full options.

**Special-review basins `02299472` and `04073468` are excluded from v001 (→ 2,752 basins).**

Logs:   `/data42/omrip/Flash-NH/tmp/stage1_target_package_v001_logs/`
Output: `/data42/omrip/Flash-NH/tmp/stage1_target_package_v001/`

### Manual command reference (for partial re-runs / debugging)

The launcher is equivalent to the following manual steps:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1
cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation

python scripts/build_stage1_target_package.py \
    --canonical-dir /data42/omrip/Flash-NH/tmp/stage1_full_2843/canonical_merged \
    --policy config/stage1_target_policy.yaml \
    --status-csv /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv \
    --out-dir /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \
    --exclude-staids 02299472,04073468 \
    --force

python scripts/audit_stage1_target_package.py \
    --package-dir /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \
    --policy config/stage1_target_policy.yaml \
    --status-csv /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv \
    --expected-basins 2752
```

---

## h2o policy smoke (2026-06-15)

Smoke run on h2o against the real 2,843-basin `canonical_merged` tree with full
`--status-csv` enforcement. Validates policy filtering, negative-qobs cleaning,
and special-review halt before the full build.

**canonical_merged verified:** 2,843 flat NCs, 2,843 unique STAIDs, 0 recursive duplicates.

**Build command (5 candidates → 4 included):**

```bash
python scripts/build_stage1_target_package.py \
    --canonical-dir /data42/omrip/Flash-NH/tmp/stage1_full_2843/canonical_merged \
    --policy config/stage1_target_policy.yaml \
    --status-csv /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv \
    --out-dir /data42/omrip/Flash-NH/tmp/stage1_target_package_policy_smoke \
    --staids 01019000,01049500,01073319,08010000,01135300 \
    --force
```

**Build result: PASS**

| Basin | Status | Valid hours | NaN hours | Neg cleaned |
|---|---|---|---|---|
| `01019000` | TARGET_READY_CONTINUOUS | 45,205 | 515 | 0 |
| `01049500` | TARGET_READY_CONTINUOUS | 45,714 | 6 | 0 |
| `01073319` | TARGET_QUALITY_REVIEW | 43,091 | 2,629 | 0 |
| `08010000` | TARGET_QUALITY_REVIEW | 43,712 | 2,008 | **95** |
| ~~`01135300`~~ | TARGET_OPERATIONAL_REVIEW | — | — | excluded (hist_util=False) |

- NaN before / after: 5,063 / 5,158 (delta = 95 = `08010000` neg-cleaned)
- Valid hours total: 177,722
- Build time: 1.0s

**Audit result: PASS — 0 errors, 0 warnings**

| Check | Result |
|---|---|
| Required package files | PASS |
| Basin count (4 == 4) | PASS |
| SHA-256 checksums (4/4) | PASS |
| Per-basin NC audit (4/4) | PASS |
| `01135300` absent (89 held-out, 0 in package) | PASS |
| Special-review basins `02299472`/`04073468` absent | PASS |
| TARGET_QUALITY_REVIEW in package: 2 (`01073319`, `08010000`) | advisory |

**Special-review halt test: PASS (exit code 1)**

```bash
python scripts/build_stage1_target_package.py \
    --canonical-dir /data42/omrip/Flash-NH/tmp/stage1_full_2843/canonical_merged \
    --policy config/stage1_target_policy.yaml \
    --status-csv /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv \
    --out-dir /data42/omrip/Flash-NH/tmp/stage1_target_package_special_review_test \
    --staids 02299472 --force
```

`02299472` passes policy filter (hist_util=True, TARGET_QUALITY_REVIEW), then triggers
`review_required` halt. Builder exited 1; no NC files written. To include or exclude it
in the full build, pass `--allow-review-required 02299472` or `--exclude-staids 02299472`.

---

## Full h2o build result (2026-06-16)

Full v001 build on h2o via launcher `scripts/run_stage1_target_package_v001_h2o.sh` (commit `3ac51ff`).
Evidence bundle: `tmp/stage1_target_package_v001_evidence/` (not committed). Full result: `docs/stage1_target_package_v001_result.md`.

**Count pipeline:**

| Stage | Count |
|---|---|
| Input NC files (`canonical_merged`) | 2,843 |
| Unconditionally excluded (`--exclude-staids 02299472,04073468`) | 2 |
| Candidates | 2,841 |
| Policy excluded (`hist_util=False`, `TARGET_OPERATIONAL_REVIEW`) | 89 |
| Included and built | **2,752** |
| Failed | 0 |

**Cleaning summary:**

| Metric | Value |
|---|---|
| Basins with neg→NaN cleaning | 16 |
| Total values cleaned | 235 |
| NaN before cleaning | 3,880,507 |
| NaN after cleaning | 3,880,742 |
| Valid hours after cleaning | 121,940,698 |

Cleaned basins: `01379530` (5), `02093000` (1), `02097314` (44), `02246000` (6), `04189000` (1),
`05467000` (7), `06090500` (18), `07295000` (7), `08010000` (95), `08152000` (12),
`08380500` (1), `09484580` (4), `10349300` (4), `11065000` (6), `11135800` (9), `11465750` (15).

**Build result: PASS** — BUILD EXIT CODE: 0

**Audit result: PASS — 0 errors, 0 warnings** (18.8 s)

| Check | Result |
|---|---|
| Required package files (manifest, checksums, provenance, cleaning report) | PASS |
| Basin count (2,752 == 2,752) | PASS |
| SHA-256 checksums (2,752/2,752) | PASS |
| Per-basin NC audit (2,752/2,752) | PASS |
| Held-out basins absent (89 held-out, 0 in package) | PASS |
| Special-review basins absent (`02299472`, `04073468`) | PASS |
| TARGET_QUALITY_REVIEW in package: 1,373 | advisory |

**Policy provenance:**

- policy_name: `stage1_target_policy_v001` (version 1)
- policy_sha256: `449165686d033b9cdbd395ad70e64a3bfa82d01757021e62059f254a2a30d691`
- AUDIT EXIT CODE: 0

---

## Known limitations

1. **No `--status-csv` in smoke mode**: `historical_training_utility_flag` and
   `target_status` filters are not applied without the audit CSV. The smoke build
   uses permissive mode and includes all basins in `--canonical-dir`.

2. **No negative values in 5-basin local smoke**: The locally available pilot basins
   (from Milestone 2I-B, 7 basins) happen to have no negative `qobs` values. The
   negative-cleaning code path was not triggered in the local smoke.
   **Resolved in full v001 build (2026-06-16):** 16 basins cleaned, 235 values neg→NaN.
   Largest: `08010000` (95), `02097314` (44), `06090500` (18), `08152000` (12), `11465750` (15).

3. **Special-review STAIDs not in local smoke set**: `02299472` and `04073468` are not
   among the 7 locally available pilot basins, so the halt logic was exercised by code
   inspection only in the local smoke.
   **Resolved in h2o policy smoke (2026-06-15):** halt triggered for `02299472`
   (2,605 neg, `review_required`) — builder exited 1, no NCs written. Confirmed PASS.

4. **Canonical NC layout on h2o**: The acquisition pipeline writes NCs to both
   per-shard directories (`shard_NN/canonical/`) and a flat merged copy at
   `canonical_merged/`. Using the broad root `stage1_full_2843/` as `--canonical-dir`
   with `rglob` finds 5,686 NCs (2 × 2,843); `nc_map` silently deduplicates
   (last-wins), which is correct but ambiguous. Always use
   `stage1_full_2843/canonical_merged` as `--canonical-dir`. Confirmed flat: 2,843,
   recursive: 2,843, unique STAIDs: 2,843 (2026-06-15).

---

## Acceptance criteria for full h2o build

All criteria satisfied by v001 build (2026-06-16):

- Builder returns exit code 0 — **PASS**
- Audit returns 0 errors — **PASS (0 errors, 0 warnings)**
- Basin count matches expected (2,752) — **PASS**
- No decoded -9999 values in any NC — **PASS**
- No negative qobs in any NC after cleaning — **PASS**
- Cleaning report documents all basins with neg-cleaned > 0 — **PASS (16 basins, 235 values)**
- manifest.json, checksums.sha256, run_provenance.json all present and consistent — **PASS**

---

## What is NOT in this milestone

- Static attributes (attributes_full.csv) — deferred; streamflow targets only for 2J-B
- NeuralHydrology config file — deferred; follows Moriah transfer layout design
- Train/val/test split files — deferred
- MRMS / RTMA forcing time series — separate pipeline (spatial downloads)
- Model training — not on h2o; goes to Moriah cluster
