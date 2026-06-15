# Flash-NH Stage 1 — Target Policy (v001)

Milestone: 2J-A
Date: 2026-06-15
Config file: `config/stage1_target_policy.yaml`

---

## What this document is

`config/stage1_target_policy.yaml` is the machine-readable design input for the
Stage 1 full-period NeuralHydrology package builder.  It records the agreed
basin-inclusion, qobs-cleaning, and special-review rules.

**This file does not create or modify data.**  It is read by the package builder
at runtime so that all inclusion and cleaning decisions are driven by config rather
than hard-coded in the builder script.  Changing policy means editing the YAML and
bumping the version — not editing the builder.

---

## Source data

| Item | Value |
|---|---|
| Acquisition root (h2o) | `/data42/omrip/Flash-NH/tmp/stage1_full_2843` |
| Audit root (h2o) | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/audit` |
| Input manifest | `config/stage1_initial_training_basin_manifest.csv` |
| Acquisition commit | `e362c0e` |
| Period | `2020-10-14T00:00:00Z` → `2025-12-31T23:00:00Z` (45,720 h/basin) |
| Total basins | 2,843 |
| Canonical NCs | 2,843 / 2,843 |

---

## Inclusion policy

### Primary gate: `historical_training_utility_flag`

Basins are included if `historical_training_utility_flag == True` in
`audit/target_status.csv`.

| historical_training_utility_flag | Count | Action |
|---|---|---|
| True | 2,754 | Include (subject to special-review exceptions) |
| False | 89 | Exclude from first package |

The 89 `False` basins are exactly the 89 `TARGET_OPERATIONAL_REVIEW` basins.
No other target-status class has `historical_training_utility_flag=False`.

### `TARGET_OPERATIONAL_REVIEW` is held out

These 89 basins have late-period data gaps (last valid observation before
2025-12-31; gap to period end ≥ some threshold detected by the audit).  Their
most recent streamflow data is absent, making operational readiness uncertain.

They are **not** excluded because of poor data quality — they may have excellent
historical records.  They are held out because:
1. Including them in a first training run could bias loss functions toward
   basins with truncated recent records.
2. They are natural candidates for future historical-only training experiments.

### `TARGET_QUALITY_REVIEW` remains eligible

1,375 basins carry a `suspicious_spike_flag=True`, which drives the
`TARGET_QUALITY_REVIEW` classification.  **These basins remain eligible for
the first package.**

The spike heuristic (value > 5× p99 of the station's own distribution) is
deliberately conservative.  Manual inspection of pilot hydrographs confirmed
that rapid rises and sharp peaks are often hydrologically real events — flash
floods, ice-jam releases, and snowmelt surges regularly exceed 5× p99.  A spike
flag alone is not evidence of bad data.

The flag is advisory only.  Future targeted review of individual hydrographs may
reveal specific time windows to mask, but that review has not been done at the
scale of 1,375 basins.  Excluding them now would discard a large fraction of the
training signal for high-flow events.

---

## Target-cleaning policy

Applied by the package builder at the time of assembly.  Not applied here.

| Rule | Setting |
|---|---|
| Negative qobs → NaN | Yes — all negative streamflow values set to NaN |
| Preserve existing NaN | Yes — no backfilling |
| Interpolation | None |
| Gap filling | None |
| Imputation | None |
| Alter positive obs | No |

Negative streamflow values are physically impossible for nearly all USGS inland
gauges (tidal backwater and instrument offsets aside).  Setting them to NaN is
the correct minimal intervention: it removes the invalid values without inventing
data.

---

## Special-review policy

### Two dominant negative-qobs basins

| STAID | Negative hours | First-package action |
|---|---|---|
| `02299472` | 2,605 | `review_required` |
| `04073468` | 2,054 | `review_required` |

These two basins account for 4,659 of the total 4,894 negative hours across all
18 affected basins.  Their negative-value counts are large enough that simply
clamping to NaN would introduce substantial data gaps in a region that was
previously non-NaN.  This could interact unexpectedly with the loss function or
coverage statistics without any indication in the audit output.

The `review_required` action means the package builder must surface these basins
explicitly (raise a warning or halt, depending on implementation) unless the
caller provides an explicit override.  The intent is to force a conscious decision
— include or exclude — rather than silently absorbing them.

The decision is left to a future explicit override in the config or CLI.

### Remaining 16 negative-qobs basins

The other 16 basins have ≤ 95 negative values each.  The global `set_negative_to_nan`
cleaning rule is sufficient unless future review identifies a broader issue.

---

## Effective first-package candidate count

```
historical_training_utility_flag=True:      2,754
Less: 02299472 and 04073468 (if excluded):   − 2
                                           ------
Conservative first-package floor:          2,752
```

Whether to include `02299472` and `04073468` after review is a decision for the
package-build step, not this policy document.  The base policy count is **2,754**.

---

## Package-build status

Full package build on h2o is **blocked** by operations preflight gates
G1–G6 (see `docs/stage1_h2o_operations_preflight.md`).

| Allowed now | Blocked |
|---|---|
| Reading this policy config | Building the full 2,843-basin package on h2o |
| Writing/validating the package-builder script locally | Running the builder on h2o |
| Designing the package layout | Large MRMS/RTMA downloads |
| A small local smoke (only with explicit approval) | NeuralHydrology training |

A local smoke test (e.g. 5–10 basins on the local machine using the already-
downloaded pilot canonical NCs from Milestone 2H-D) may be approved separately
as a package-builder validation step.  It does not require h2o.

---

## Config versioning

This is `stage1_target_policy_v001`.  Future policy changes should:
1. Create a new file `config/stage1_target_policy_v002.yaml`.
2. Document the change in a `changelog` field inside the YAML and a brief note
   in this doc.
3. Never modify a previously committed `vXXX` config in place — treat them as
   immutable once a package has been built from them.
