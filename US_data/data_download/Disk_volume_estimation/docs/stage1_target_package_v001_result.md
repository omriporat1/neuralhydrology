# Flash-NH Stage 1 — Target Package v001 Result

Date: 2026-06-16
Milestone: 2J-B
Status: **PASS — 0 errors, 0 warnings**

---

## Build metadata

| Field | Value |
|---|---|
| Build started (UTC) | 2026-06-16T08:40:00Z |
| Launcher script | `scripts/run_stage1_target_package_v001_h2o.sh` |
| Launcher commit | `3ac51ff` |
| Host | h2o.es.huji.ac.il |
| Policy file | `config/stage1_target_policy.yaml` |
| Policy name | `stage1_target_policy_v001` (version 1) |
| policy_sha256 | `449165686d033b9cdbd395ad70e64a3bfa82d01757021e62059f254a2a30d691` |
| canonical_dir | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/canonical_merged` |
| status_csv | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv` |
| exclude_staids | `02299472,04073468` |
| Output location (h2o) | `/data42/omrip/Flash-NH/tmp/stage1_target_package_v001/` |

---

## Count pipeline

| Stage | Count |
|---|---|
| Input NC files (`canonical_merged`) | 2,843 |
| Unconditionally excluded (`--exclude-staids`) | 2 |
| Candidates | 2,841 |
| Policy excluded (`hist_util=False` / `TARGET_OPERATIONAL_REVIEW`) | 89 |
| Included and built | **2,752** |
| Failed | 0 |
| BUILD EXIT CODE | 0 |

Special-review exclusions (`02299472`: 2,605 neg; `04073468`: 2,054 neg) counted under
`--exclude-staids`, not under policy exclusion. Future v002 must resolve their disposition.

---

## Acquisition audit context: negative qobs

The full 2,843-basin acquisition audit (HPC preflight, 2026-06-13) flagged **18 basins**
with negative `qobs_m3s` values. Of those:

- 2 basins (`02299472`, `04073468`) were the most heavily negative and classified
  `review_required` in the policy. They are **excluded from v001**.
- The remaining **16 basins** were eligible and included; their negative values were cleaned
  to NaN during the v001 build (235 values total).

---

## Cleaning summary

| Metric | Value |
|---|---|
| Basins with neg→NaN cleaning | 16 |
| Total values cleaned | 235 |
| NaN before cleaning | 3,880,507 |
| NaN after cleaning | 3,880,742 |
| Valid hours after cleaning | 121,940,698 |

**Per-basin cleaning:**

| STAID | neg cleaned |
|---|---|
| `01379530` | 5 |
| `02093000` | 1 |
| `02097314` | 44 |
| `02246000` | 6 |
| `04189000` | 1 |
| `05467000` | 7 |
| `06090500` | 18 |
| `07295000` | 7 |
| `08010000` | 95 |
| `08152000` | 12 |
| `08380500` | 1 |
| `09484580` | 4 |
| `10349300` | 4 |
| `11065000` | 6 |
| `11135800` | 9 |
| `11465750` | 15 |

---

## Audit result

Audit started: 2026-06-16T08:41:09Z; runtime 18.8 s.

| Check | Result |
|---|---|
| Required package files (manifest, checksums, provenance, cleaning report) | PASS |
| Basin count (2,752 == 2,752) | PASS |
| SHA-256 checksums (2,752/2,752 OK, 0 FAIL) | PASS |
| Per-basin NC audit (2,752 PASS, 0 FAIL) | PASS |
| Held-out basins absent (89 held-out, 0 in package) | PASS |
| Special-review basins absent (`02299472`, `04073468`) | PASS |
| TARGET_QUALITY_REVIEW in package: 1,373 | advisory |
| **Errors** | **0** |
| **Warnings** | **0** |
| AUDIT EXIT CODE | 0 |

---

## Package layout

```
/data42/omrip/Flash-NH/tmp/stage1_target_package_v001/
  time_series/
    <STAID>.nc           # 2,752 files
  manifest.json          # basin list, counts, policy info, cleaning summary
  checksums.sha256       # SHA-256 of every time_series/*.nc
  run_provenance.json    # script, args, timestamp, policy_sha256
  cleaning_report.csv    # per-basin cleaning stats (2,752 rows)
```

### Per-basin NC structure

- **Coordinate:** `date` (hourly UTC, `datetime64[ns]`, no tz offset stored)
- **Variable:** `qobs_m3s` (`float32`, units `m3 s-1`, `_FillValue=-9999.0`)
- **Period:** 2020-10-14T00:00:00Z – 2025-12-31T23:00:00Z (45,720 hourly steps per basin)
- Compatible with NeuralHydrology `GenericDataset` format

---

## Evidence bundle

The following compact evidence files were transferred from h2o and reside locally at
`tmp/stage1_target_package_v001_evidence/` (gitignored — not committed):

| File | Contents |
|---|---|
| `build.log` | Full build stdout/stderr (2,784 lines) |
| `audit.log` | Full audit stdout/stderr (43 lines) |
| `manifest.json` | Basin list, counts, policy info, cleaning summary |
| `checksums.sha256` | SHA-256 of all 2,752 target NCs |
| `cleaning_report.csv` | Per-basin cleaning stats (2,752 rows) |
| `run_provenance.json` | Script args, timestamp, policy_sha256 |

Per operating policy (`docs/repo_policy.md`), documentation conclusions are based on
these inspected evidence files, not terminal summaries alone.
