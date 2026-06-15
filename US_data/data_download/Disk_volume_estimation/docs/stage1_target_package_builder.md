# Flash-NH Stage 1 — Target Package Builder and Auditor

Milestone: 2J-B
Date: 2026-06-15
Status: **Scripts implemented and smoke-tested locally (5 basins, PASS)**

---

## Purpose

`scripts/build_stage1_target_package.py` reads the canonical hourly USGS IV
NetCDF files produced by `recover_usgs_iv_full_period_hourly.py` and assembles a
NeuralHydrology-compatible streamflow-target package. It applies the target policy
from `config/stage1_target_policy.yaml` so that all inclusion and cleaning decisions
are driven by config, not hard-coded in the script.

`scripts/audit_stage1_target_package.py` verifies the output package for structural
correctness, value integrity, and policy compliance.

This milestone does **not** execute the full 2,843-basin build on h2o.
That step follows after the design is validated with a local smoke.

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

## Full-run command template (h2o)

Run this after the h2o environment is activated and the audit CSV is available.
**Do not run until explicitly approved as a full build milestone.**

```bash
# Activate env
source /opt/conda/etc/profile.d/conda.sh
conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1

cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation

# Full 2,843-basin build (inside screen session)
screen -S flashnh_target_build

python scripts/build_stage1_target_package.py \
    --canonical-dir /data42/omrip/Flash-NH/tmp/stage1_full_2843 \
    --policy config/stage1_target_policy.yaml \
    --status-csv /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv \
    --out-dir /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \
    --force

# Audit after build completes
python scripts/audit_stage1_target_package.py \
    --package-dir /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \
    --policy config/stage1_target_policy.yaml \
    --status-csv /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv \
    --expected-basins 2754
```

Note: `--expected-basins 2754` reflects the policy's `effective_candidate_count`
(all basins with `historical_training_utility_flag=True`). Adjust if special-review
basins `02299472` / `04073468` are excluded (→ 2,752).

---

## Known limitations

1. **No `--status-csv` in smoke mode**: `historical_training_utility_flag` and
   `target_status` filters are not applied without the audit CSV. The smoke build
   uses permissive mode and includes all basins in `--canonical-dir`.

2. **No negative values in 5-basin smoke**: The locally available pilot basins
   (from Milestone 2I-B, 7 basins) happen to have no negative `qobs` values. The
   negative-cleaning code path is implemented and exercised by logic checks but was
   not triggered in the smoke. The full 2,843-basin build will exercise it for 18 basins.

3. **Special-review STAIDs not in smoke set**: `02299472` and `04073468` are not
   among the 7 locally available pilot basins, so the halt logic was exercised by
   code inspection only. To test the halt path explicitly:

   ```bash
   python scripts/build_stage1_target_package.py \
       --canonical-dir tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot/canonical \
       --policy config/stage1_target_policy.yaml \
       --out-dir tmp/stage1_target_package_smoke_sr_test \
       --staids 02299472 --force
   # Expected: BUILD HALTED by special-review policy
   ```

4. **Sharded h2o layout**: The builder's `_find_canonical_ncs()` function uses
   `rglob('*_hourly.nc')` to find canonical NCs recursively under `stage1_full_2843/`.
   This covers both flat and sharded layouts. First run on h2o should use a
   smoke (e.g., `--staids 02073000,01585200,02077670`) before scaling to 2,754 basins.

---

## Acceptance criteria for full h2o build

- Builder returns exit code 0
- Audit returns 0 errors
- Basin count matches expected (2,752 or 2,754 depending on special-review decision)
- No decoded -9999 values in any NC
- No negative qobs in any NC after cleaning
- Cleaning report documents all basins with neg-cleaned > 0
- manifest.json, checksums.sha256, run_provenance.json all present and consistent

---

## What is NOT in this milestone

- Static attributes (attributes_full.csv) — deferred; streamflow targets only for 2J-B
- NeuralHydrology config file — deferred; follows Moriah transfer layout design
- Train/val/test split files — deferred
- MRMS / RTMA forcing time series — separate pipeline (spatial downloads)
- Model training — not on h2o; goes to Moriah cluster
