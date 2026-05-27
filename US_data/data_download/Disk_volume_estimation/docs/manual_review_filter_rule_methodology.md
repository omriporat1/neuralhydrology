# Manual Review Filter Rule Methodology — Flash-NH Basin Screening

**Version**: 1.0
**Status**: Methodology reference — analysis is preliminary; no basins have been removed.

---

## 1. Why Manual Review Was Performed

The Flash-NH project uses WY2024 hourly streamflow data from USGS NWIS for 3,034
main-training-candidate basins selected from the GAGES-II/CAMELSH catalog. Automated
streamflow QC (hard exclusions) and USGS site-type metadata audit already removed
~300 basins. However, automated QC cannot distinguish all problematic basins from
legitimately extreme hydrology.

A stratified random sample of 73 basins was drawn for human hydrograph inspection.
The review set was intentionally enriched for suspicious cases (extreme RBI, extreme
rise rates, high Q-ratio, context-flagged basins) to calibrate whether the automated
flags reflect real problems or real flashy hydrology.

---

## 2. Separation of QC Layers

Three independent screening layers apply to the candidate universe:

| Layer | Applied at | What it checks | Outputs |
|---|---|---|---|
| Streamflow hard-QC | Build time | Data completeness (<90%), severe negatives, zero-Q50, no-RBI | EXCLUDE_HARD_QC class |
| USGS site-type metadata | Post-QC | Site type classification (tidal streams, lakes, etc.) | metadata_policy_bucket |
| Manual review + risk rules | Post-metadata | Hydrograph quality, regulation signals, sensor artifacts | preliminary_training_status |

These layers are **independent** and applied in sequence. The manual review layer does
not revisit basins already excluded by hard-QC or metadata audit. It operates only on
the 3,034 `main_training_candidate == True` basins.

---

## 3. Manual Labels as Calibration Data, Not Classifier Labels

The 73 reviewed basins are **not** a representative random sample of the 3,034 main
candidates. They are **stratified/enriched** for suspicious cases. This means:

- EXCLUDE and UNSURE rates in the review set are artificially inflated relative to
  the full population.
- The labels cannot be used to train or calibrate a statistical model for basin
  selection without correcting for the sampling design.
- Any rule derived from the review labels has uncertain precision in the full
  population until validated on a broader, unbiased sample.

The labels are used as **calibration evidence** only: they tell us whether a given
automated risk rule (e.g., high HYDRO_DISTURB_INDX) tends to agree with human
judgment on the flagged basins. A rule with high `reviewed_exclude_or_unsure_rate`
has stronger calibration support than one with low rate.

---

## 4. Decision Taxonomy

| Status label | Meaning | Policy |
|---|---|---|
| TRAIN_CORE_PRELIM | No risk flags; passes all checks | Include in training data |
| TRAIN_SOFT_KEEP_PRELIM | One moderate risk flag triggers (F, I, C, D, E) | Include but note risk; inspect in post-training residual analysis |
| HOLDOUT_REVIEW_PRELIM | Compound risk (>=2 flags), or rule G/H triggers, or reviewed UNSURE | Hold out from training pending second-pass review |
| EXCLUDE_TRAINING_PRELIM | Reviewed EXCLUDE (manual override) or metadata HARD_EXCLUDE | Exclude from training; document reason |

**Priority order**: EXCLUDE > HOLDOUT > SOFT_KEEP > CORE.

The word "PRELIM" in every status label is intentional: these are proposals, not
final decisions. No basin should be removed from training data based on these labels
alone without human sign-off.

---

## 5. Candidate Rule Philosophy

### 5.1 High Precision Over Broad Recall

The rules are designed to flag basins that are *likely* problematic with high
confidence, not to cast a wide net. A rule that catches 10 real problems and
5 false positives is preferable to one that catches 15 real problems and 50 false
positives — because false exclusions remove real hydrological signal from training.

Thresholds are set at the p95 or p99 of the full candidate distribution to ensure
only the most extreme cases are flagged. See `tables/candidate_rule_screening.csv`
for the exact threshold values used.

### 5.2 Reversible Risk Flags

Rules C through I are reversible soft flags. A basin that triggers rule F (large,
slow, low-flashiness) is flagged as TRAIN_SOFT_KEEP_PRELIM but is NOT excluded.
Post-training residual analysis may reveal whether that basin degrades model
performance — at which point it can be removed with data-driven justification.

### 5.3 Do Not Exclude Valid Hydrologic Regimes

Intermittent and ephemeral streams are real hydrological phenomena, especially in
arid and semi-arid regions. Rule G (mostly-zero suspicious) targets only the most
extreme zero-flow cases (zero_flow_fraction >= p95 AND Q50 <= 0.001 m3/s) and
assigns HOLDOUT_REVIEW rather than EXCLUDE. These basins should be reviewed by
region and season before exclusion.

Similarly, large basins with low flashiness (rule F) may represent important model
training contexts. They are flagged as SOFT_KEEP, not excluded.

### 5.4 Proxy Variables and Substitutions

GAGES-II does not include ARTIFPATH_PCT, ARTIFPATH_MAINSTEM_PCT, or HIRES_LENTIC_PCT.
The following proxies are used:

| Requested variable | Proxy used | Source | Limitation |
|---|---|---|---|
| ARTIFPATH_PCT | CANALS_PCT | GAGES-II HydroMod_Other | Canals only; excludes other artificial paths |
| HIRES_LENTIC_PCT | WATERNLCD06 | GAGES-II LC06_Basin | Basin-wide open water; includes natural lakes |
| -- | lka_pc_use | HydroATLAS | Lake area %; 99.84% coverage (3,029/3,034) after join-key fix (see note) |
| -- | dor_pc_pva | HydroATLAS | Degree of regulation; same 99.84% coverage after fix |

All proxy substitutions are documented in `tables/input_join_audit.csv` and the
analysis summary.

**HydroATLAS join-key correction**: HydroATLAS stores USGS station IDs without leading zeros
(e.g. `3144816` instead of `03144816`). An initial run using raw string matching yielded only
553/3,034 matches (18.2%). The fix applies `str.zfill(8)` normalization to all STAID columns
before joining, raising coverage to 3,029/3,034 (99.84%). The 5 unmatched candidates are
15-digit coordinate-based USGS site IDs (e.g. `401733105392404`) that are outside the
HydroATLAS coverage universe; their lka_pc_use and dor_pc_pva values remain NaN, and they
are not penalized by rules C or D (NaN comparisons evaluate to False, not True).

---

## 6. Reproducible Inputs and Outputs

### 6.1 Inputs (exact paths)

| Input | Path | Row count at analysis time |
|---|---|---|
| Manual review labels | `reports/flashnh_hydrograph_review_cards_v004_main_training_candidate/manual_review_labels.csv` | 73 |
| Review template | `reports/flashnh_hydrograph_review_cards_v004_main_training_candidate/tables/human_review_template.csv` | 73 |
| WY2024 metrics + metadata | `reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv` | 3,324 total; 3,034 main_training_candidate |
| GAGES-II attributes | `C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_gageii_*.csv` | 9,008 rows each |
| HydroATLAS | `C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_hydroATLAS.csv` | 9,008 rows; 3,029/3,034 match after join-key fix |
| NLDAS-2 climate | `C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_nldas2_climate.csv` | 9,008 rows |

### 6.2 Key Script

`scripts/analyze_manual_review_filter_rules.py`

Run with no arguments. All paths are constants at the top of the script. Deterministic output (no random seeds used in rule computation).

### 6.3 Outputs

All outputs go to `reports/flashnh_manual_review_rule_analysis_v001/` (gitignored).

| File | Purpose |
|---|---|
| `tables/input_join_audit.csv` | Row counts, match rates, proxy notes per attribute file |
| `tables/manual_review_label_summary.csv` | Decision/behavior/artifact counts and percentages |
| `tables/candidate_rule_screening.csv` | 10 rules with thresholds, reviewed evidence, full-dataset impact |
| `tables/reviewed_basin_rule_matrix.csv` | 73 reviewed basins with all rule flags and attributes |
| `tables/full_candidate_rule_matrix.csv` | 3,034 candidates with preliminary_training_status |
| `summaries/manual_review_rule_analysis_summary.md` | Narrative summary and rule table |
| `summaries/manual_review_rule_analysis_summary.json` | Machine-readable key counts and thresholds |

---

## 7. How to Report This in a Publication

The following framing is consistent with the methodology:

> "We applied a multi-stage screening workflow to identify high-quality streamflow
> records for model training. After automated data-quality exclusions (data completeness
> <90%, severe negative flow, zero median flow) and USGS site-type metadata filtering
> (removing tidal stream and lake sites), we conducted manual hydrograph inspection
> of a stratified sample of 73 basins enriched for suspicious cases. These labels were
> used as calibration evidence to evaluate candidate risk rules based on streamflow
> metrics and basin attributes (GAGES-II, HydroATLAS, NLDAS-2). Risk rules were
> conservative, using p95/p99 quantile thresholds, and resulted in risk flags rather
> than automatic exclusions. Flagged basins were held out for secondary review rather
> than being removed outright, preserving real hydrological variability in the
> training set."

---

## 8. Next Steps Before Finalizing Basin Selection

1. **Inspect** `tables/candidate_rule_screening.csv`: review each rule's
   `reviewed_exclude_or_unsure_rate` and `reviewed_false_positive_keep_rate`.
   Decide which rules are strong enough to act on.

2. **Choose** which rules to accept and at what recommendation level.
   Recommended conservative default: only rule_A (manual EXCLUDE) triggers
   EXCLUDE_TRAINING_PRELIM; rules C/D/E/G/H/J trigger HOLDOUT_REVIEW.

3. **Create a separate final basin training status script** that encodes the
   accepted rules and writes the definitive training set membership file.
   This script should be committed and versioned; the preliminary analysis
   outputs are not the final authority.

4. **Expand the manual review sample** for the HOLDOUT_REVIEW_PRELIM group
   (N=223) before finalizing exclusions. Post-training residual analysis is a
   complementary approach: train on all TRAIN_* basins, identify high-residual
   stations, then review those.

---

*This document describes methodology, not results. For quantitative results, see
`reports/flashnh_manual_review_rule_analysis_v001/summaries/manual_review_rule_analysis_summary.md`.*
