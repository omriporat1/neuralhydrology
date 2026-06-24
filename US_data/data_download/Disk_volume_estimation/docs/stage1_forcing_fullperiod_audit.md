# Stage 1 Forcing — Full-Period Post-Run Audit Result

**Created:** 2026-06-24  
**Status:** PASS_WITH_CAVEATS  
**Audit plan:** `docs/stage1_forcing_fullperiod_postrun_audit_plan.md`  
**Audit script:** `scripts/generate_fullperiod_audit_tables.py`

---

## Purpose and Evidence Source

This document records the formal post-run audit result for the Stage 1 full-period
MRMS QPE 1h Pass1 + RTMA CONUS 2.5km basin-average forcing extraction (Milestone 2K-E).

**Evidence bundle (local, not committed):**
`tmp/stage1_forcing_fullperiod_evidence_20260624T060504Z.tar.gz`

**Generated audit tables (local, not committed):**
`tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/` — 11 CSVs,
`audit_summary.json`, and `audit_summary.md`.

All conclusions below are derived from those files, not from pasted terminal output.

---

## 1. Terminal Production Status

| Metric | Value |
|---|---|
| Months expected | 63 |
| Months with `all_pass=True` | **63 / 63** |
| Months with any failure | **0** |
| Group A (2020-10 → 2022-06) | PASS=21, FAIL=0 |
| Group B (2022-07 → 2024-01) | PASS=19, FAIL=0 |
| Group C (2024-02 → 2025-12) | PASS=23, FAIL=0 |
| Active extraction processes at evidence export | 0 |
| /data42 disk free at export | 61 TB (30% free) |
| Total wall-clock time | 182.6 hours (~7.6 days) |

---

## 2. Row-Count and Schema Audit

| Check | Result |
|---|---|
| n_basins (all 63 months) | **2,752** — uniform ✓ |
| RTMA variables (all 63 months) | **11** — uniform ✓ |
| Total MRMS rows | **125,447,168** |
| Total RTMA rows | **1,383,975,296** |
| Total combined rows | **1,509,422,464** |
| Row-count formula mismatches | **0** |
| Row-count result | **PASS** |

**Formula:** MRMS rows = `mrms_successful_hours × n_basins`;
RTMA rows = `rtma_successful_hours × n_basins × 11`.

**Schema checks (all 63 months):**

| Check | Result |
|---|---|
| `rtma_10wdir_absent` | PASS — absent in all 63 months ✓ |
| `rtma_orog_absent` | PASS — absent in all 63 months ✓ |
| No all-null variables | PASS ✓ |
| Schema result | **PASS** |

---

## 3. Forcing Gap Audit

| Metric | Value |
|---|---|
| Total missing hour-products | **138** |
| Months with any gap | 20 / 63 |
| MRMS missing hours | **136** — all `not_in_s3` |
| RTMA missing hours | **2** — 2020-11-12T09Z and T10Z, `not_in_s3` |
| Product-synchronized gaps | **0** (no hour missing in both products) |
| Isolated 1h gap runs | 45 |
| Multi-hour gap runs | 25 |

All gaps have reason `not_in_s3` — permanent upstream S3 archive absences, not
pipeline errors. No gap represents a recoverable pipeline failure.

**RTMA gap discovery:** Two RTMA hours (2020-11-12T09Z and T10Z) were absent from S3.
This was not anticipated prior to the audit. Month 2020-11 `all_pass=True` with
718/720 successful RTMA hours. The 2 missing hours do not coincide with any MRMS gap.

### 24-Hour Model-Window Impact (global, across 45,720-hour period)

| Product | Missing hours | Invalid 24h windows | Fraction lost |
|---|---|---|---|
| `mrms_qpe_1h_pass1` | 136 | **949** / 45,697 | 2.08% |
| `rtma_conus_aws_2p5km` | 2 | **25** / 45,697 | 0.05% |

Per-basin impact equals product-level impact: MRMS/RTMA source gaps affect all
2,752 basins simultaneously (source files are not per-basin).

**Basin completeness:** 0 of 5,504 basin×product pairs show any incomplete month
(across all 63 months, `fullperiod_basin_completeness.csv`).

---

## 4. Warning and Diagnostic Audit

| Source | Count |
|---|---|
| Manifest `parse_warnings` entries | **0** |
| Hourly CSV unexpected `warning_message` entries | **0** |
| `Not_in_s3` informational messages (excluded — duplicate of gap audit) | 138 |
| Diagnostic JSON files | 0 (not generated; validation embedded in manifest JSON) |
| **Warning result** | **PASS** |

---

## 5. Provenance

| Commit | Short | Months | Note |
|---|---|---|---|
| `194a489783dafecc340e5d1de382a2d1c0ff3fde` | `194a489` | 1 (2020-10) | Phase 1 run; earlier extractor version |
| `7e43760fbb6e403b7a06ac84fe5e6763677088af` | `7e43760` | 62 (2020-11 → 2025-12) | D1-optimized full-period extractor |

**Two-commit impact:** Documentation caveat only. Both commits define identical
validation check sets and both pass all 12 checks including `rtma_10wdir_absent`
and `rtma_orog_absent`. The earlier commit was used for the Phase 1 October 2020
run before the D1 optimization; the Phase 1 result was accepted and not re-run.

---

## 6. Acceptance Decision

**`PASS_WITH_CAVEATS`**

**Rationale:** 63/63 months `all_pass=True`. Zero row-count mismatches. Schema
checks PASS (10wdir absent, orog absent, no all-null). Zero unexpected warnings.
All gaps are permanent S3 archive absences (`not_in_s3`), not repairable.

**Caveats (documentation only, no rerun required):**

1. **Two-commit provenance:** 2020-10 used commit `194a489`; all other 62 months
   used `7e43760`. Both commits pass all validation checks. Impact: documentation
   caveat only.
2. **MRMS 2.08% window impact:** 949 of 45,697 possible 24h windows are affected
   by MRMS S3 archive gaps. These windows will be NaN in the raw curated product.
   Gap policy (from `docs/stage1_forcing_fullperiod_postrun_audit_plan.md §6`):
   MRMS gaps are preserved as NaN in raw and derived products — no imputation.
3. **RTMA 2-hour gap (2020-11):** Newly discovered. 25 of 45,697 windows affected.
   Month passes all validation. No corrective action required.

**No rerun required.** The acceptance status does not block curated product assembly.

---

## 7. Generated Audit Files (not committed)

All under `tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/`:

| File | Content |
|---|---|
| `fullperiod_monthly_status.csv` | Per-month extraction summary (63 rows) |
| `fullperiod_row_counts.csv` | Expected vs actual rows per month; mismatch flag |
| `fullperiod_missing_hour_products.csv` | All 138 missing hour-product entries |
| `fullperiod_gap_inventory.csv` | Gap runs classified as isolated / multi-hour |
| `fullperiod_variable_coverage.csv` | Per-RTMA-variable completeness × month (693 rows) |
| `fullperiod_basin_completeness.csv` | Per-basin completeness summary × 63 months (5504 rows) |
| `fullperiod_24h_window_impact.csv` | 24h window impact by product × month (126 rows) |
| `fullperiod_warning_inventory.csv` | Unexpected warnings (0 entries) |
| `fullperiod_diagnostic_inventory.csv` | Diagnostic file inventory (none) |
| `fullperiod_git_commit_inventory.csv` | Git commit by month group (2 entries) |
| `fullperiod_evidence_adequacy.csv` | Evidence adequacy assessment (13 items) |
| `audit_summary.json` | Machine-readable audit summary |
| `audit_summary.md` | Human-readable audit summary |

These files are gitignored (under `tmp/`) and are not committed to the repository.
To regenerate:

```bash
python -X utf8 scripts/generate_fullperiod_audit_tables.py \
    --evidence-root tmp/stage1_forcing_fullperiod_evidence_20260624T060504Z \
    --out-dir       tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z
```

---

## 8. What This Audit Does Not Do

- Does **not** create a curated forcing product or NeuralHydrology-compatible package.
- Does **not** run or modify h2o outputs.
- Does **not** certify the forcing data for model training (training remains on Moriah).

---

## 9. Next Steps

1. **Visual / event QC case selection** — select 12–24 animation cases stratified
   by gap/no-gap and event category per `docs/stage1_forcing_fullperiod_postrun_audit_plan.md §7`.
   Record in `visual_qc_case_selection.csv`.
2. **Curated forcing product v001 design** — define per-basin Parquet layout, gap-flag
   companion columns, and dataset manifest per `audit_plan.md §9`.
3. **Small forcing-to-NH smoke test** — assemble a per-basin forcing NC from a single
   monthly chunk and run a 5-basin NeuralHydrology smoke on Moriah before full assembly.
4. **Full NH package assembly on h2o** — combine v001 streamflow targets, forcing data,
   static attributes, and train/val/test splits into an audited NH-compatible package.