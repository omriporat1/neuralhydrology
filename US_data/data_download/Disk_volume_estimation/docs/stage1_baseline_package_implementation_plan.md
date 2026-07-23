# Stage 1 — Baseline Package Builder + Split Config Implementation Plan (Milestone 2K-G-I)

Milestone: **2K-G-I — Baseline Package Builder + Split Config Implementation**.
This document is the output of the 2K-G-I **I-0 planning pass** (2026-07-13):
repository inspection + reconciliation of the existing implementation against
the binding scientific policy signed off at 2K-G-H (commit `e860316`), and a
staged implementation plan. **No code, config, Slurm script, test, package, or
remote data was created or modified in this pass. Nothing was committed. No
remote command was run. No training was run.**

**Review patch (2026-07-13, same day, docs-only):** the five sign-off
questions this plan raised in §21 were answered by the user and are now
recorded as **SIGNED OFF** in place (§7, §9, §14, §15, §21); the validity-mask
decomposition (§12), the NH-integration wording (§13), the static-NaN audit
scope (§15), the gap-flag observation (§21), and the sub-milestone table
(§20, new I-D2-pre) were refined accordingly. No code, config, test, Slurm,
generated-output, or remote change; nothing committed.

**Correction (2026-07-13, follow-up, docs-only):** the target-hour
forcing-gap exclusion previously carried into §12 (an artifact of the 2K-G-G
counting convention) was **removed from the proposed implementation** — the
approved scientific mask is history-only
(`sample_valid = history_valid_seq & target_boundary_valid_lead &
split_valid`); the 2K-G-G loss table is now treated as a conservative upper
bound, with I-D1 producing the implementation-authoritative counts (§12, §13,
§20 I-D1, §21 updated).

**Addendum (2026-07-23, schema-support implementation) — versioned package
schema required for any future full package build.** `scripts/build_stage1_baseline_nh_package.py`
now requires an explicit `--package-schema` argument. The full-population
scientific package this plan describes has not yet been built, but when it
is, the build **must** explicitly pass `--package-schema
stage1_scientific_package_v002` (temporal coordinate `date`) — the CLI has
no default and will not infer this from basin count or output path. The
already-certified 32-basin Compact Scientific Package (Gate 4, `time`
coordinate, `stage1_compact_scientific_package_v001`) is unaffected and
remains frozen. See `docs/decision_log.md` (2026-07-23 "Versioned package
schema" entry) for the full schema/provenance/auditor/`FlashNHDataset`
contract.

Authority hierarchy (not re-decided here):
- `docs/FLASHNH_CURRENT_STATE.md` — top-level current status.
- `docs/stage1_scientific_baseline_design.md` — **binding scientific policy**
  (2K-G-H sign-off, §3/§5/§5a/§6/§8/§8b–§8d/§9a/§9b). Not reopened by this plan.
- `docs/stage1_target_scaling_gap_leadtime_feasibility.md` — authoritative
  NH 1.13.0 evidence record (Q1–Q11) and gap-window feasibility numbers.

---

## 1. Binding requirements this implementation must preserve (summary, not redesign)

1. **Target scaling.** Training target is area-normalized discharge in **mm/h
   equivalent runoff depth**, computed by the Flash-NH package builder at
   package-build time. Package target variables:
   `qobs_mm_per_h_lead01/lead03/lead06/lead12`. Raw `qobs_m3s` must not remain
   the training target. NH inverse scaling only reverses NH's internal z-score
   (returns mm/h, not m³/s; `tester.py:247-259` evidence); NH loss/validation
   curves are transformed-space training diagnostics; **official Flash-NH
   benchmark metrics are computed after full conversion back to raw m³/s.**
2. **Target inversion/audit requirements (§5a).** (a) deterministic
   m³/s → mm/h → m³/s round-trip unit tests; (b) package audit verifying
   `qobs_mm_per_h_leadXX(t) == convert(qobs_m3s(t+XXh), area)` on sampled
   basin/time points against the **original target NCs**; (c) evaluation audit
   verifying units, area, lead alignment, NaN masking, and back-conversion.
3. **Lead time.** All four leads (1/3/6/12 h) in the first package/config/sweep
   design; 6 h primary benchmark/model-selection lead, 12 h secondary, 1/3 h
   diagnostic. Implemented via package-build-time target shifting (Q9: no
   native NH `lead_time`). Lead time and `seq_length` are separate axes.
4. **`seq_length`.** Stage 1 candidates are only **12/24/48/72 h**. 168/336 h
   must not be reintroduced.
5. **Gap policy.** Hard-exclude training windows intersecting MRMS archive-gap
   hours (136 global gap hours); RTMA (2 global gap hours) may be folded into
   the same mask if free. No silent dynamic-input NaNs. `nan_handling_method`
   is fallback/ablation only; unset/default `None` is forbidden in any run that
   permits NaN dynamic inputs (Q6: unset passes raw NaN into an unprotected
   `nn.Linear`). Smoke 0/1 fill (MRMS→0 mm, RTMA→interp) is historical only.
6. **Static attributes.** [**Updated 2026-07-20** — the v001 matrix named in
   this item at I-0 planning time (2026-07-13) has since been superseded;
   see `docs/decision_log.md` 2026-07-20 entries and
   `docs/FLASHNH_CURRENT_STATE.md` for the full acceptance record. This item
   now states the current binding matrix.] Canonical
   `stage1_static_attributes_v002` (2,843 × 523, 473 `model_input`, sha256
   `4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297`), through
   NH's standard numeric static-attribute pathway. No raw categorical
   embeddings. `STATE`/`HUC02` split-support/diagnostics only;
   `LAT_GAGE`/`LNG_GAGE` diagnostics only — none of the four are model inputs.
7. **Spatial split / leakage.** California excluded completely from Stages 1–3
   (Stage 4 transfer only). Reproducible seeded stratified non-CA spatial
   holdout (~10%), stratifying at minimum on HUC02/geography, basin area, and
   hydroclimate/aridity. Explicit basin-list artifacts required for:
   development training, validation, temporal test, non-CA spatial holdout,
   California all / fine-tune train / holdout. Spatial-leakage prevention is a
   Flash-NH basin-list responsibility (Q4: NH has no concept of basin role).
8. **Temporal split.** Train 2020-10-14→2023-12-31; validation 2024; test 2025.
9. **Leakage rules (§8d).** All Stage 1–3 scalers fit only on development
   training basins + training period. Stage 4 CA scalers fit only on the CA
   fine-tune training subset.
10. **Dynamic inputs.** `v001-core` (8 variables incl. 2 gap flags), binding.

---

## 2. Repository files inspected in this pass

Documents (read in full): `docs/FLASHNH_CURRENT_STATE.md`,
`docs/stage1_scientific_baseline_design.md`,
`docs/stage1_target_scaling_gap_leadtime_feasibility.md`,
`docs/stage1_static_attribute_matrix_plan.md`,
`docs/stage1_neuralhydrology_preflight.md`, `docs/repo_policy.md`,
`docs/decision_log.md` (incl. the 2026-07-12 2K-G-G/2K-G-H entries at lines
1263–1587).

Code/config (inspected):
- `scripts/build_stage1_nh_package.py` (full read)
- `scripts/audit_stage1_nh_package.py` (full read)
- `scripts/check_stage1_nh_preflight.py` (head + config-check section)
- `scripts/build_stage1_static_attribute_matrix.py` (role/classification/
  STAID/dup-drop machinery; `_norm_staid` at lines 195–199)
- `scripts/audit_stage1_static_attribute_matrix.py` (head; independence pattern)
- `scripts/analyze_stage1_window_feasibility.py` (windowing convention + mask
  arithmetic, `_geometry_valid_windows`/`_count_valid_with_mask` lines 131–149)
- `scripts/run_stage1_smoke0_moriah.sbatch` (head; hard-pinned `catfish`/L4)
- `config/stage1_target_policy.yaml`, `config/stage1_initial_training_basin_manifest.csv`
  (header + row count), `config/stage1_pilot_basin_manifest.csv` (header)
- `.gitignore` (Disk_volume_estimation)
- `src/` layout (`src/pipeline/*`, `src/datasources/*`; import convention
  `sys.path.insert(repo_root)` + `from src.pipeline import ...`;
  `src/pipeline/provenance.py` provides `write_run_manifest`/`git_commit_hash`)
- Local gitignored evidence (read-only, for reconciliation facts):
  `tmp/stage1_target_package_v001_evidence/manifest.json` (2,752 `basins`
  verified), `tmp/stage1_static_attribute_matrix_v001_dryrun/` (local dry-run
  matrix; sha256 computed this pass: `3c3399f0ed40…` — **differs** from the
  canonical h2o `eb17aaa0…`).

Key structural facts:
- **There is no `tests/` directory in the repo** — no test infrastructure
  exists yet.
- `config/` contains only 3 files; `config/stage1_target_policy.yaml` is the
  repo's precedent for machine-readable policy config.
- `.gitignore` ignores `data/` wholesale and `**/*.parquet` globally; under
  `reports/` only `README.md`/`summary.md`/`manifest.json` are un-ignored.
  → Canonical split artifacts must live under `config/` and be
  `.txt`/`.csv`/`.json` (never parquet).

---

## 3. Current-versus-required implementation gap table

| Area | Current implementation (Smoke 0/1 era) | Binding requirement | Gap action |
|---|---|---|---|
| Temporal split | `_TRAIN_END="2022-12-31"`, `_VAL_*=2023`, `_TEST_*=2024–2025` hard-coded (`build_stage1_nh_package.py:70-74`) | Train→2023-12-31, val 2024, test 2025 | New policy YAML + new builder; never edit constants in the frozen smoke builder |
| Target | Raw `qobs_m3s` written and configured as target (builder line 344, configs line 427) | `qobs_mm_per_h_leadXX` (4 leads), mm/h | New conversion + lead-shift modules |
| Gap handling | MRMS NaN→0.0, RTMA→interp (`_apply_gap_fill`, lines 299–325); auditor asserts zero forcing NaN | Preserve NaN; hard-exclude gap-intersecting training windows | New builder writes NaN-preserving NCs; validity-mask + NH index integration |
| Static attributes | 48-col merge; `_REQUIRED_ATTR_COLS = [DRAIN_SQKM, LAT_GAGE, LNG_GAGE, BFI_AVE]` (line 93); LAT/LNG configured as NH static inputs | `stage1_static_attributes_v002` (**updated 2026-07-20**; supersedes the v001/496 figure this row originally named), 473 `model_input` cols via role manifest; lat/lon excluded from inputs | Role-driven column selection from `_column_manifest.json` |
| Basin lists | All basins in all splits (`_write_basin_lists`) | 7 real split artifacts, CA excluded, spatial holdout disjoint | New split generator/auditor/QC/promotion |
| Configs | Two smoke YAMLs, Moriah paths hard-coded, `qobs_m3s` target | 16 lead×seq configs from template+generator, policy-driven | New config generator |
| Slurm | Dead `_write_slurm` in builder (wrong invocation, `--partition=gpu`); repo sbatch hard-pinned `catfish`/L4 | Parameterized `PARTITION`/`GRES` (design doc §11) | New parameterized baseline sbatch (later sub-milestone); never call `_write_slurm` |
| STAID handling | `_norm_staid` uses `int(float(...))` then `:08d` (builder lines 179–183; same pattern in static-matrix builder 195–199) | Preserve 8/9/15-char IDs exactly, no numeric coercion | New strict string-only normalizer (see §6) |
| Auditor | Mirrors builder constants; requires zero forcing NaN; basin-list check requires `len(ln) == 8` (`audit_stage1_nh_package.py:344`) — **would reject the 6 non-8-char STAIDs that are in the 2,752 set** | Independent audit of transformed/shifted targets, NaN-preserving forcing, real splits, all STAID lengths | New independent baseline auditor |
| NH compat | `dataset: generic`, DD/MM/YYYY, `epochs`, `head: regression`, `output_activation: linear`, `attributes/attributes.csv` layout | Same (NH 1.13 unchanged) | **Reusable as-is** — port into the new config generator |

Reusable parts of the existing builder: full-period grid construction
(`_full_period_grid`), atomic NC writing pattern (`_write_basin_nc` tmp+rename),
float32/`_FillValue` encoding, `_sha256`, DD/MM/YYYY conversion, provenance
pattern (incl. `attributes_sha256` recording), manifest-writing shape.
Inseparably smoke-specific: gap fill, split constants, basin lists, smoke
YAMLs, `_write_slurm`, `_REQUIRED_ATTR_COLS`.

---

## 4. Architecture recommendation

**Recommendation: keep the smoke builder frozen; add a separate scientific
baseline builder + auditor + preflight, with shared logic in a new `src/`
library package.** This is the safest choice after inspection because:
- the smoke scripts are the historical reproducibility record for Smoke 0/1
  PASS evidence (job IDs 45370683/45370873 reference their outputs);
- nearly all scientific behavior differs (target, gap policy, splits, static
  source, configs) — a "refactor in place" would either break the historical
  scripts or become a two-mode monolith;
- the repo already uses exactly this pattern (January-pilot builder vs
  full-period builder; smoke forcing builder vs full-period forcing builder).

New scripts (thin CLIs) and frozen scripts:

| Script | Status |
|---|---|
| `scripts/build_stage1_baseline_nh_package.py` | NEW — scientific baseline builder |
| `scripts/audit_stage1_baseline_nh_package.py` | NEW — independent baseline auditor (separate script, not an extension of the smoke auditor, whose zero-forcing-NaN and 8-char assumptions are contrary to the baseline) |
| `scripts/check_stage1_baseline_nh_preflight.py` | NEW — Moriah post-transfer preflight (may import small helpers from the smoke preflight's NH-guard pattern) |
| `scripts/generate_stage1_baseline_splits.py` / `audit_stage1_baseline_splits.py` / `generate_stage1_baseline_split_qc.py` | NEW — split generator / auditor / QC |
| `scripts/generate_stage1_baseline_configs.py` | NEW — 16-config generator |
| `scripts/build_stage1_nh_package.py`, `audit_stage1_nh_package.py`, `check_stage1_nh_preflight.py`, `run_stage1_smoke0/1_moriah.sbatch` | FROZEN — historical Smoke 0/1 reproducibility; no edits |

Separate auditor and preflight: **yes** — the baseline auditor must verify
properties the smoke auditor actively forbids (forcing NaN at exactly the 136+2
gap hours) and must recompute lead-shifted targets from the *original target
package NCs*, which the smoke auditor never reads.

### Recommended source/module architecture

Follow the existing `src/<subpackage>` convention (`src/pipeline/`,
`src/datasources/`):

```
src/baseline/
  __init__.py
  staid.py            # strict string-only STAID normalization + validation
  units.py            # discharge <-> runoff-depth conversion (pure)
  lead_targets.py     # lead-shift construction (pure, xarray/pandas)
  validity_mask.py    # global gap-validity masks per seq_length x lead
  splits.py           # split-artifact loading, schema validation, disjointness
  static_attributes.py# role-driven model_input selection from column manifest
  policy.py           # load/validate config/stage1_scientific_baseline_v001.yaml
  nh_dataset.py       # (I-D2) GenericDataset subclass / lookup-table filter
tests/
  test_staid.py  test_units.py  test_lead_targets.py  test_validity_mask.py
  test_splits.py  test_static_attributes.py  test_policy.py
  fixtures/           # tiny synthetic CSled fixtures only (no real data)
```

Reuse `src/pipeline/provenance.py` (`write_run_manifest`, `git_commit_hash`)
rather than adding a new provenance module. Test runner: `pytest` (introduce
it at I-B; no test infrastructure exists today — creating `tests/` +
`pytest.ini`/config is part of the first code sub-milestone).

---

## 5. Basin-universe reconciliation

**Historical/frozen note (added 2026-07-20):** this section documents the
basin-universe reconciliation and split-generation process as it was actually
planned and executed, which correctly used the `stage1_static_attributes_v001`
matrix for `STATE`/`HUC02`/area/hydroclimate split-support fields. The split
artifacts this process produced (`config/stage1_baseline_splits_v001/`) are
now frozen and accepted, and are not being regenerated against
`stage1_static_attributes_v002` — see `docs/decision_log.md` 2026-07-20
entries. The `v001` matrix name and 531/496 figures below describe that real,
completed historical process and are preserved as-is. They are unrelated to
the *current* binding model-input static matrix (`stage1_static_attributes_v002`,
473 `model_input` columns), which is used for compact-package attribute
preparation (§15) and is stated as current in §1 item 6 and §3.

### Sources

| Universe / fact | Repository-side source | Status |
|---|---|---|
| 2,843-basin static universe | `config/stage1_initial_training_basin_manifest.csv` (columns `STAID,final_training_status`; 2,843 rows; zero-padded text) | Committed, canonical |
| 2,752-basin scientific floor | h2o target package v001 `manifest.json` (`n_basins_processed=2752`); locally mirrored in gitignored `tmp/stage1_target_package_v001_evidence/manifest.json` (`basins` list verified = 2,752 this pass) | **Not committed anywhere** — must be promoted (see below) |
| Two excluded gauges | `config/stage1_target_policy.yaml` → `special_review` (`02299472`, `04073468`, `first_package_action: review_required`) | Committed |
| Target-package inclusion | Same v001 manifest (`policy_excluded`=89 + 2 special-review; 2,843−89−2 = 2,752 ✓ verified) | h2o canonical |
| Forcing-package inclusion | h2o `v001_basin_list.csv` (derived from the target manifest by `scripts/export_v001_basin_list.py`); forcing full-period audit confirmed 2,752/2,752 | h2o canonical; local audit doc |
| STATE / HUC02 / area / hydroclimate | `stage1_static_attributes_v001` (`split_support` role for STATE/HUC02; `DRAIN_SQKM` etc. in `model_input`; HydroATLAS `ari_ix_uav` etc.) | h2o canonical (sha256 pinned); local dry-run copy is byte-different (see below) |
| California membership | `STATE == 'CA'` from the matrix's `split_support` role | Derivable; rule SIGNED OFF 2026-07-13 (§21 item 3; LAT/LNG diagnostic cross-check only) |

### Required joins

All joins key on the normalized STAID string:
1. eligible = v001 target-package basin list (2,752) — assert it is exactly
   `manifest(2,843) minus 89 hist_util=False minus {02299472, 04073468}`;
2. eligible ⨝ static matrix (`gauge_id`) — must be 2,752/2,752 (matrix covers
   all 2,843);
3. eligible ⨝ forcing product basin set — must be exactly equal (2,752 = 2,752,
   zero symmetric difference);
4. split assignment is then computed on the joined frame using
   `STATE`, `HUC02`, `DRAIN_SQKM`, and hydroclimate columns.

The split generator must **fail loud on any universe mismatch** (extra basin,
missing basin, join loss) — never intersect silently.

### STAID normalization rules (new `src/baseline/staid.py`)

Facts confirmed this pass: the 2,752 set contains **all 6 non-standard IDs**
(one 9-char `103366092`, five 15-char coordinate IDs), and all five 15-char IDs
are the HydroATLAS-gap basins. Both existing `_norm_staid` implementations
(`build_stage1_nh_package.py:179`, `build_stage1_static_attribute_matrix.py:195`)
go through `int(float(str(s)))` — currently lossless for these IDs (15 digits
< 2^53) but numerically fragile and semantically wrong. New rules:
- input must already be `str` (loaders must read ID columns with `dtype=str` —
  never let pandas infer int64, which is exactly how the old 48-col parquet
  lost leading zeros);
- `strip()`; must match `^\d+$`; else raise;
- `len < 8` → `zfill(8)`; `len in {8, 9, 15}` → unchanged; any other length →
  raise (fail loud on unknown formats);
- **no `int()`/`float()` anywhere**; round-trip property tested.
Older artifacts with integer-coerced STAIDs (the 48-col parquet) are not
inputs to the baseline pipeline, so no int-repair path is needed — but the
loader should still detect an int64 ID column and re-normalize defensively.

### Resolvable locally vs requires h2o

Resolvable locally: universe arithmetic (2,843/89/2/2,752), subset checks
against the local evidence manifest, STATE/HUC02/area availability (local
dry-run matrix or local source CSVs), split-algorithm development on fixtures.

Requires h2o (user-run, evidence pulled):
1. re-export the v001 basin list from the h2o target-package manifest and
   confirm it matches the local evidence copy (then commit it — see §6);
2. confirm forcing-product basin set == target basin set (recorded PASS in the
   2K-F-C audit; cheap to re-assert at split time);
3. pull the canonical `stage1_static_attributes_v001.parquet` (8.8 MB) +
   `_column_manifest.json` to the local machine with sha256 verification
   against `eb17aaa0…` — the local dry-run copy hashes to `3c3399f0…`
   (byte-different; presumed content-equivalent but unverified), so **canonical
   split generation must use a checksum-verified copy of the canonical matrix**
   (either run on h2o, or run locally on the verified pulled copy — the pulled
   copy is the recommended path since split generation is cheap and local
   review is faster).

---

## 6. Split artifacts: canonical vs generated, layout, promotion

### Canonical committed artifacts (small, deterministic, approved)

Location: **`config/stage1_baseline_splits_v001/`** (fits the existing
`config/stage1_*` naming; `config/` is git-tracked; `.txt`/`.csv`/`.json` are
not gitignored — `data/` and `*.parquet` are, so neither may be used):

```
config/stage1_baseline_splits_v001/
  eligible_basins_v001.txt        # the 2,752-basin scientific floor (promoted input)
  development_train.txt           # ~90% of non-CA eligible
  validation.txt                  # == development_train (temporal split; see semantics)
  temporal_test.txt               # == development_train (same)
  spatial_holdout_nonca.txt       # ~10% of non-CA eligible
  california_all.txt              # all eligible CA basins
  california_finetune_train.txt   # ~90% of CA
  california_holdout.txt          # ~10% of CA
  split_assignment.csv            # one row per eligible basin: STAID, split_role,
                                  #   STATE, HUC02, area_class, hydro_class,
                                  #   stratum_id, assignment_reason
  split_manifest.json             # seed, algorithm id+version, bin edges,
                                  #   source checksums (matrix sha256, eligible-list
                                  #   sha256, policy sha256), counts per split,
                                  #   holdout fractions, fallback log,
                                  #   HUC02-by-role counts, missing-stratifier basin
                                  #   IDs/reason, per-artifact sha256 checksums,
                                  #   explicit note that the three temporal lists
                                  #   are identical by design
```

**Temporal-list semantics:** development training, validation, and temporal
test share the same basin pool and differ only by dates (design doc §8/§8b).
Emit all three files anyway — NH configs point at per-split basin files, and
explicit files prevent a future config error — with `split_manifest.json`
recording that the three are byte-identical by design (and their common
sha256), so the duplication reads as intentional.

`eligible_basins_v001.txt` promotion: the 2,752 list is today only h2o-resident
(+ a gitignored local evidence copy). It is a small, deterministic,
scientifically load-bearing input — commit it under the split directory after
user-run h2o re-export + checksum match against the local evidence copy.

### Generated review evidence (not committed)

`tmp/stage1_baseline_splits_v001_review/` during iteration; on promotion, the
reviewed bundle moves to `reports/stage1_baseline_splits_v001/` where only
`summary.md` + `manifest.json` become tracked (existing gitignore negations);
maps/figures/tables stay untracked on disk. After approval, 2–4 key small
figures (CONUS split map, CA split map, one balance figure) may be deliberately
committed into `docs/figures/stage1_baseline_splits_v001/` for the record
(repo policy allows "selected small plots only if needed").

### Promotion workflow

```
generate candidate (seeded, deterministic)
  → machine audit (scripts/audit_stage1_baseline_splits.py, exit 0 required)
  → human review of QC package (§8)
  → user approval recorded in docs/decision_log.md
  → copy lists + assignment + manifest into config/stage1_baseline_splits_v001/
  → commit (small commit, I-A5) — checksums in split_manifest.json bind the
    committed artifacts to the generating run
```

Package builds, scaler fitting, training, and evaluation consume **only** the
committed canonical artifacts (path + checksum recorded in every
`run_provenance.json`), never a generator output directly.

---

## 7. Stratification: keep it simple

Available fields (confirmed in the matrix): `HUC02` (18 non-CA units, roughly
40–400 basins each), `STATE` (CA flag), `DRAIN_SQKM`, HydroATLAS aridity/
climate-moisture (`ari_ix_uav`, `cmi_ix_uyr`), precipitation normals
(GAGES-II Climate / `pre_mm_uyr`), snow fraction (`snw_pc_syr`).

Three candidate strategies:
1. **HUC02-proportional seeded sampling + post-hoc balance audit only** —
   simplest; area/hydroclimate balance is checked, not enforced.
2. **Composite strata: HUC02 × area tercile × aridity tercile** (~162 strata),
   deterministic seeded draw of ~10% per stratum with sparse-stratum fallback.
3. **Allocation within HUC02, balanced on area+aridity** — inside each HUC02,
   rank basins on (area class, aridity class) and use a seeded
   systematic-sample (sorted + strided with random offset), which balances
   marginals without explicit cross-strata.

**Recommendation: strategy 2, with terciles computed on the non-CA eligible
pool and a deterministic sparse-stratum fallback.** It directly implements the
binding "HUC02 + area + hydroclimate" wording, is fully explainable in one
paragraph, and reduces to strategy 1 in sparse strata. Strategy 3 is the
fallback if 2's audit shows instability. No optimizer, no iterative rebalancer.

Fixed parameters (into the policy YAML, §9) — **SIGNED OFF (2026-07-13):**
seed **42**, target holdout fraction **10%** with an acceptable overall range
of **8–12%** (per-HUC02 fractions monitored, not enforced, in the audit),
**area terciles**, **aridity terciles**, initial minimum composite-stratum
size **10**; all edges recorded verbatim in `split_manifest.json`. This
defines the **candidate** split method only — the first generated split
remains subject to the machine audit (I-A3) and human QC review (I-A4) and is
not guaranteed acceptance.

**Sparse-stratum fallback — simplified to ONE level (superseding the earlier
multi-level draft above; signed off 2026-07-13, I-A2):**
- stratum (HUC02 × area tercile × aridity tercile) with ≥ 10 basins → direct
  random ~10% holdout from that stratum;
- stratum with < 10 basins → pool **all** such sparse-stratum basins within
  the same HUC02 into **one** HUC02-level sparse pool (no intermediate
  HUC02 × area layer);
- that pool with ≥ 10 basins → random ~10% holdout from the pool; still
  < 10 → the entire pool goes to `development_train`;
- a non-sparse stratum is **never** downgraded merely because a sibling
  stratum in the same HUC02 is sparse;
- rounding: per-stratum/per-pool holdout count = `round(0.10 × n)`; there is
  **no** global top-up/trim pass and no largest-remainder optimization — the
  exact resulting global count is not a policy constant, only the 8–12%
  overall band is binding;
- every fallback decision appended to the manifest's `fallback_log`.

**Missing-hydroclimate-stratifier basins (Option B, signed off 2026-07-13):**
basins missing `ari_ix_uav` are never stratified or sampled at all — each is
assigned directly to its population's training role (`development_train` for
non-CA, `california_finetune_train` for CA) with
`assignment_reason = missing_hydroatlas_stratifier`, and can never appear in
a holdout role. v001 has 5 such basins, all non-CA. There is no
`aridity_missing` stratum and no imputation.

The same machinery, without HUC02 in the grouping key (CA is mostly HUC 18),
generates the CA 90/10 fine-tune/holdout split with a single statewide sparse
pool in place of the per-HUC02 pool; HUC-region is recorded as diagnostic
only.

---

## 8. Split auditor invariants and human-readable QC

### Independent auditor invariants (`audit_stage1_baseline_splits.py`)

- every basin in `eligible_basins_v001.txt` assigned exactly one role; no
  omissions, no extras, no duplicates within or across lists;
- eligible list == manifest(2,843) − 89 policy-excluded − 2 special-review
  (recomputed from committed inputs, not trusted from the generator);
- `02299472`/`04073468` absent from every list;
- CA basins absent from development/validation/temporal-test/spatial-holdout;
- development ∩ spatial holdout = ∅; CA fine-tune ∩ CA holdout = ∅;
- CA lists ∪-consistent with `california_all.txt`;
- the three temporal lists byte-identical (asserted, since it is by-design);
- STAIDs preserved exactly (regex `^\d{8}$|^\d{9}$|^\d{15}$`; the six known
  non-standard IDs present and unmangled);
- target and forcing eligibility respected (list ⊆ v001 basins);
- seed, algorithm version, bin edges, and source checksums present in the
  manifest and consistent with recomputation (auditor re-runs the assignment
  from the manifest parameters and requires identical output — determinism
  proof);
- holdout fraction within tolerance (overall and reported per HUC02);
- HUC02/area/hydroclimate representation tables emitted for review; rare
  strata and every fallback listed.

### Human-review QC package (generated evidence)

Maps (point maps are sufficient — 2,752 gauge points; polygons add nothing at
CONUS scale):
- CONUS map: development vs non-CA spatial holdout (color), CA greyed;
- CONUS map colored by HUC02 with holdout basins emphasized;
- California map: fine-tune vs holdout;
- optional: CONUS map colored by area class or aridity class.

Distribution comparisons (development vs spatial holdout; and CA fine-tune vs
CA holdout; and CA vs non-CA) — concise scientific set, not all 496 columns
(historical figure from the `v001` matrix used for this now-complete,
frozen split-QC pass; see the §5 historical note — unrelated to the current
`stage1_static_attributes_v002` model-input count):
`DRAIN_SQKM` (log-scale), aridity (`ari_ix_uav`), mean annual precipitation,
elevation (`ELEV_MEAN_M_BASIN` or HydroATLAS `ele_mt_sav`), snow fraction
(`snw_pc_syr`), `BFI_AVE` (or a flashiness proxy if available), target-data
completeness (qobs coverage from the target-package per-basin summary — h2o
evidence), static-attribute missingness count per basin. Overlaid ECDFs or
quantile tables, plus standardized mean differences (SMD) with |SMD| > 0.25
flagged for review (flag, not auto-fail).

Tables: counts and percentages by split; by HUC02 × split; by state; by area
class; by hydroclimate class; per-split summary statistics of the comparison
variables; rare-strata and fallback listing; missingness summary.

Review questions (verbatim, to be answered by the human reviewer):
- Are non-CA development and non-CA spatial holdout broadly comparable while
  preserving a meaningful spatial holdout?
- Does California provide the intended distinct Stage 4 domain while remaining
  internally usable and balanced for fine-tune vs holdout?
The CA-distinctness view (CA vs non-CA on the same variables) demonstrates the
first half; the CA-internal comparison the second.

Preserve after approval: `summary.md` + `manifest.json` (tracked under
`reports/stage1_baseline_splits_v001/`), plus the 2–4 committed key figures
(§6). Everything else remains generated evidence.

---

## 9. Machine-readable baseline policy — `config/stage1_scientific_baseline_v001.yaml`

**Recommendation: yes, create it (sub-milestone I-A1)** as the single
machine-readable source, modeled on `config/stage1_target_policy.yaml`
(existing precedent: policy in config, consumed by builders, never executes
transformations). Schema outline:

**Status note (added 2026-07-20):** the outline below is the **illustrative
I-0-era schema sketch** as originally drafted (2026-07-13), preserved
unedited as planning history. It was subsequently implemented as
`config/stage1_scientific_baseline_v001.yaml` (sub-milestone I-A1, and
revised in place to `policy_version: 2` on 2026-07-20 to reconcile
`static_attributes.*` with the accepted `stage1_static_attributes_v002`
matrix — see `docs/decision_log.md`). The actual policy file is the binding
artifact; where it differs from the sketch below (notably
`static_attributes.matrix_name`/`sha256`/column counts, now v002/523/473),
the actual file governs. Do not treat this code block as current schema
documentation.

```yaml
policy_name: stage1_scientific_baseline_v001
policy_version: "1"
signed_off: {milestone: 2K-G-H, date: "2026-07-12", commit: e860316}

period: {start_utc: ..., end_utc: ..., n_hours: 45720}
temporal_split: {train: [2020-10-14, 2023-12-31], validation: [...], test: [...]}

basin_universe:
  manifest: config/stage1_initial_training_basin_manifest.csv
  eligible_list: config/stage1_baseline_splits_v001/eligible_basins_v001.txt
  eligible_count: 2752
  excluded_staids: ["02299472", "04073468"]
  policy_excluded_status: [TARGET_OPERATIONAL_REVIEW]

spatial_split:
  seed: 42
  nonca_holdout_fraction: 0.10
  holdout_tolerance: 0.02
  stratification: {geography: HUC02, area: DRAIN_SQKM_terciles, hydroclimate: ari_ix_uav_terciles}
  california: {policy: excluded_stage1_3, membership_rule: STATE == CA, finetune_fraction: 0.90}
  artifacts_dir: config/stage1_baseline_splits_v001/

target:
  source_variable: qobs_m3s
  transform: {kind: area_normalized, unit: mm_per_h, formula: "q_mm_per_h = q_m3s * 3.6 / area_km2"}
  area: {column: DRAIN_SQKM, units: km2, source: stage1_static_attributes_v001}
  leads_hours: [1, 3, 6, 12]
  lead_roles: {primary: 6, secondary: 12, diagnostic: [1, 3]}
  variable_name_template: "qobs_mm_per_h_lead{lead:02d}"
  dtype: float32   # float64 internal computation

seq_lengths_hours: [12, 24, 48, 72]   # binding; 168/336 forbidden

gap_policy:
  training_windows: hard_exclude_mrms
  rtma_folding: fold_into_same_mask     # "either-gap" mask
  eval_windows: hard_exclude_all_splits   # SIGNED OFF 2026-07-13, §14
  nan_handling_method: forbidden_unset  # any NaN-input run must set it explicitly

dynamic_inputs: [mrms_qpe_1h_mm, rtma_2t_K, rtma_2d_K, rtma_2sh_kgkg,
                 rtma_10u_ms, rtma_10v_ms, mrms_qpe_1h_mm_gap, rtma_gap]

static_attributes:
  matrix_name: stage1_static_attributes_v001
  sha256: eb17aaa0...
  role_source: stage1_static_attributes_v001_column_manifest.json
  allowed_role: model_input
  forbidden_in_inputs: [STATE, HUC02, LAT_GAGE, LNG_GAGE]
  nan_policy: dev_train_median_imputation  # SIGNED OFF 2026-07-13, §15

nh: {version_expected: "1.13.0", dataset: generic, date_format: DD/MM/YYYY,
     head: regression, output_activation: linear, predict_last_n: 1}

audits: {package_lead_sample_count: ..., roundtrip_rtol: ..., expected_mrms_gap_hours: 136,
         expected_rtma_gap_hours: 2}
```

**Values that must not remain hard-coded in scripts:** temporal dates, target
variable names/leads, seq_length list, seed, holdout fraction, excluded
STAIDs, matrix checksum, area column, gap policy, v001-core input list, NH
compat expectations.

**Scientific policy vs machine paths:** the YAML holds **no machine paths**.
Data locations (h2o forcing/target dirs, Moriah package dirs, out-dirs) stay
CLI arguments, exactly as every existing builder does — this keeps the policy
portable and matches repo convention. Run-specific output paths live in
`run_provenance.json`. (A separate optional `config/machine_paths.yaml` is
*not* recommended now — CLI args have worked for 20+ scripts.)

`src/baseline/policy.py` loads and schema-validates this file (required keys,
types, binding-value checks such as `168 not in seq_lengths`), and every
builder/auditor/generator records `policy_sha256` in its provenance — same
pattern as the target-package builder's `policy_sha256`.

---

## 10. Target conversion utilities (`src/baseline/units.py`)

Formulas (float64 internally):
```
q_mm_per_h = q_m3s * 3.6 / area_km2          # = q_m3s * 3600 / (area_km2 * 1e6) * 1000
q_m3s      = q_mm_per_h * area_km2 / 3.6
```

API:
```python
discharge_m3s_to_runoff_mm_per_h(q_m3s, area_km2)
runoff_mm_per_h_to_discharge_m3s(q_mm_per_h, area_km2)
```
Behavior spec:
- accepts scalars, NumPy arrays, pandas Series, xarray DataArray (pure
  arithmetic broadcasts through all four; unit tests cover each);
- NaN q → NaN out (propagates naturally; asserted);
- zero flow → exactly 0.0; negative flow → converted arithmetically **but**
  the package source already has negatives cleaned to NaN, so the builder
  asserts no negative qobs before conversion (defense in depth: utility
  converts, builder refuses);
- area ≤ 0, NaN area, or infinite area → `ValueError` (never silent NaN) —
  area is validated once per basin at load, not per element;
- ±inf q → raise in builder-side validation (utility itself propagates);
- computation float64, package write float32; round-trip tolerance tests both.

Deterministic tests (`tests/test_units.py`): small basin (e.g. 5.3 km²), large
basin (25,000 km²), zero discharge, hand-calculated known values
(1 m³/s over 3.6 km² == 1.0 mm/h exactly), NaN preservation, array/Series/
DataArray broadcasting, invalid-area raises, m³/s→mm/h→m³/s round-trip
`rtol=1e-12` (float64), float32 write-read round-trip `rtol≈1e-6`.

Three distinct verification layers (kept separate by design):
1. **pure unit tests** (above — no data files);
2. **package artifact audit** (I-C3): sampled equality of package
   `qobs_mm_per_h_leadXX(t)` vs original target NC `qobs_m3s(t+XXh)` converted
   with the manifest-recorded area — independently reimplemented in the
   auditor (see §11);
3. **evaluation/raw-space audit** (later, evaluation milestone): metric
   scripts verify units, area join, lead alignment, NaN mask, back-conversion.

---

## 11. Lead-target construction (`src/baseline/lead_targets.py`)

For lead `L`: `target_L(t) = convert(qobs_m3s(t + L))` — i.e.
`shift(-L)` on the hourly series (values move earlier; the last `L` hours of
the period become NaN).

Spec:
- input: qobs series on the exact 45,720-hour grid; the function **validates**
  the index is exactly hourly, monotonic, duplicate-free, tz-naive UTC and
  spans the policy period — reject otherwise (no silent reindex);
- shift direction: negative pandas shift (`series.shift(-L)`); an explicit
  test pins the direction with asymmetric synthetic data so a sign error
  cannot pass;
- tail: exactly `L` trailing NaNs (asserted);
- source NaN at `t+L` → NaN at `t` (propagation asserted);
- variable naming: `qobs_mm_per_h_lead01/03/06/12` (zero-padded, from the
  policy template);
- NC metadata per variable: `units: "mm h-1"`, `lead_hours: L`,
  `source_variable: qobs_m3s`, `area_km2: <value>`, `area_source: DRAIN_SQKM`,
  long_name stating "value at timestamp t is the observation at t+Lh".

**One package with all four target variables (recommended)** — per-basin NC
carries all four `qobs_mm_per_h_leadXX` plus `qobs_m3s` retained as a
**diagnostic/audit variable only** (never configured as an NH target; keeping
it makes the package self-auditing and evaluation-side reconversion checks
cheap). Each NH config selects exactly one target. Trade-offs considered:
four extra float32 series ≈ 0.7 MB/basin ≈ 2 GB over 2,752 basins (cheap);
single package avoids 4× build/transfer/audit and keeps forcing identical
across leads; the only cost is that a config typo could select the wrong lead
— mitigated by the config generator + auditor cross-check of
`target_variables` against the config manifest.

Tests (`tests/test_lead_targets.py`): each lead 1/3/6/12; first/last valid
alignment (synthetic ramps with known values at known offsets); tail NaN count
== lead; source-NaN propagation to the correct shifted position; rejection of
irregular/duplicate/non-hourly indices; naming/metadata.

**Builder/auditor independence:** the auditor must not import
`src/baseline/lead_targets.py` (nor `units.py`) for its equality check — it
reimplements the sampled check inline as
`package_var[t] == qobs_source_nc[t+L] * 3.6 / area` (reading the original
target-package NCs and the area from the package manifest), with an
explicitly different code path (integer index offset lookup rather than
vectorized shift). Shared systematic bugs are further guarded by the pure unit
tests' hand-calculated constants (a wrong 3.6 in both places would still fail
the hand-computed test).

---

## 12. Global forcing-gap validity masks (`src/baseline/validity_mask.py`)

MRMS/RTMA archive gaps are **global timestamps** (136 + 2 hours, identical for
every basin) — no basin-by-basin scan is needed or wanted:

```
for each (seq_length, lead) in {12,24,48,72} x {1,3,6,12}:   # 16 combos
    compute one boolean validity vector over the common 45,720-hour timeline
    apply the same artifact to every basin
```

**Cost: negligible, confirmed.** `analyze_stage1_window_feasibility.py`
already performs the identical cumulative-sum computation for all 16 combos
over 45,720 steps in well under a second locally (2K-G-G evidence runs). Each
mask is a ≤45,720-entry boolean array; 16 of them total < 1 MB.

Windowing semantics (approved scientific mask — history-only forcing check):
- issue time `t` = last input step; input window is `[t − L_seq + 1, …, t]`
  (inclusive of issue time);
- validity decomposes as
  `sample_valid(t) = history_valid_seq(t) & target_boundary_valid_lead(t) & split_valid(t)`:
  - `history_valid_seq(t)` — **four `seq_length`-specific forcing-history
    masks**: sufficient history and no relevant forcing-gap hour anywhere in
    `[t − L_seq + 1, …, t]`. **Lead time does not change the historical
    forcing window** — this component is lead-independent by construction;
  - `target_boundary_valid_lead(t)` — **lead-specific boundary validity
    only**: the original target timestamp `t + lead` lies inside the research
    period (tail availability). **No forcing-gap check at `t + lead`:**
    Flash-NH's `v001-core` model uses only historical forcing through issue
    time `t` — it never reads MRMS/RTMA at the future target hour, so a
    forcing archive gap at `t + lead` does not contaminate the sample unless
    that hour also lies in the input window, which it cannot for positive
    leads;
  - `split_valid(t)` — issue time / target placement valid for the temporal
    split, applied per the §14 policy.
  The per-run artifact is the AND of the applicable components — up to 16
  final `(seq_length, lead)` sample-validity artifacts is fine, but only the
  four history masks are seq-dependent forcing components, each computed once.
  Tests assert the asymmetry between the two axes.
- Note: with package-build-time shifted targets, NH itself sees a target at
  index `t`; the `t + lead` above refers to the **original timeline** hour
  encoded there. **Basin-specific qobs availability at `t + lead` stays
  entirely separate from the global forcing mask** — it is handled through
  target NaN masking (NH's `Masked*Loss`, Q5), never folded into these masks.

**Relation to the 2K-G-G feasibility numbers (historical evidence, not
rewritten).** The accepted 2K-G-G calculation
(`analyze_stage1_window_feasibility.py:131-149`) additionally excluded windows
whose target hour `t + lead` fell on a forcing-gap hour (`target_bad` at
`i+L−1+lead`) — a conservative counting convention that is **not** part of the
approved scientific baseline mask above. Consequences:
- the published ~1.3–5.6% loss fractions may be **conservative upper bounds**
  relative to the final history-only scientific mask;
- **I-D1 must compute and document the implementation-authoritative counts**
  from the approved mask definition;
- any difference from the earlier 2K-G-G table must be **explained
  explicitly** (attributable to the removed target-hour check), not silently
  absorbed;
- runtime reconciliation on Moriah must use the **approved implementation
  mask's counts** — agreement must never be forced against the obsolete,
  broader counting convention.

Period-boundary questions (mark for Moriah verification, not assumed):
- NH warm-up: the repo's standing understanding
  (`docs/stage1_neuralhydrology_preflight.md` §5) is that NH drops the first
  `seq_length−1` samples of each period (insufficient history) rather than
  reading antecedent hours from before the period start — meaning antecedent
  history does **not** cross from one temporal split into another, and each
  split loses up to `seq_length−1` issue times at its start. **Verify against
  NH 1.13 `_create_lookup_table`/`_validate_samples` behavior on Moriah**
  before the mask artifacts encode boundary assumptions.
- Whether NH's lookup-table index is keyed by issue time exactly as assumed
  (the mapping between the mask generator's issue-time indexing and NH's
  lookup-table keys must be proven by count reconciliation on Moriah —
  against the approved implementation mask — not by reading alone).

Artifacts: the four component history masks (`history_valid_seq{L}.npz`) and
the combined `valid_issue_times_seq{L}_lead{XX}.npz` (boolean vector + the
timeline definition + policy/gap-inventory checksums) written into the package
`masks/` directory by the builder, with a human-readable
`masks_manifest.json` (counts per combo and per component — these
implementation-authoritative counts are the reference for all downstream
reconciliation, with an explicit documented comparison against the
conservative 2K-G-G table and an explanation of any difference).

Interaction with basin-specific qobs NaNs: none at mask level (masks are
global forcing-validity only); qobs NaNs remain per-basin and are handled by
NH loss masking — the package auditor reports both quantities separately so
they are never conflated.

**Runtime-vs-audit-artifact clarification (added 2026-07-20, based on direct
code inspection of `src/baseline/nh_dataset.py`,
`src/baseline/gap_mask_io.py`, and `src/baseline/validity_mask.py`, none of
which were modified by this note).** The 16-artifact `.npz`
(`history_valid_seq{L}.npz` / `valid_issue_times_seq{L}_lead{XX}.npz` /
`masks_manifest.json`) design above is a **still-proposed audit/evidence
artifact set** — it has not been built. The **currently implemented runtime
mechanism** is materially simpler and already exists:
`src/baseline/nh_dataset.py::FlashNHDataset` (a `GenericDataset` subclass,
per the §13 Option 2 design) reads a single flat timestamp list from
`masks/gap_timestamps.json` (written by `src/baseline/gap_mask_io.py` from a
gap inventory, MRMS-only by default per Policy B), builds the research
timeline from the dataset instance's own loaded dates, and computes
`history_valid` **live, per instance** via
`validity_mask.compute_history_valid` — there are no precomputed
per-`(seq_length, lead)` boolean-array files consumed at runtime. Its
target-boundary check is also period-aware
(`issue_ts + lead_delta > period_end`), which is deliberately different from
`validity_mask.compute_boundary_valid` (not period-aware). Do not read this
section as meaning only `gap_timestamps.json` is needed end-to-end: the
question of whether independently-computed, precomputed sequence/lead
validity masks or valid-timestamp arrays and their counts are still required
— for audit cross-checks and off-by-one verification separate from the
runtime filter — is **not yet decided**. That decision belongs to the
package builder/auditor implementation increment (not yet started; out of
scope for this docs-only patch), which will determine whether such
artifacts live inside the transferable package, only in the compact
package's evidence/audit bundle, or both. Whichever form is chosen, its
representation must record: the timeline definition, `seq_length`, lead,
issue-time indexing convention, source gap-inventory/policy checksums, and
valid/invalid counts per combination — so it can be reconciled against the
runtime filter's own counts. The approved rule
`sample_valid = history_valid_seq & target_boundary_valid_lead & split_valid`
is unchanged by this note; a forcing gap at the future target timestamp is
still not by itself a basis for exclusion (§12 above); basin-specific qobs
NaNs remain a separate, target/loss-masking concern, never folded into the
forcing-validity mask.

---

## 13. Applying the validity mask to NeuralHydrology

Q8 evidence: NH 1.13 has **no native mechanism** to exclude arbitrary
windows/samples; `_validate_samples` filters on insufficient history only.
Options compared:

**Option 1 — external allowed-timestamp artifact only.** NH cannot consume it
via config; without an integration layer it does nothing. Necessary (as the
auditable artifact) but not sufficient.

**Option 2 — small custom `GenericDataset` subclass / lookup-table filter
(RECOMMENDED CANDIDATE — gated on the I-D2-pre Moriah verification below,
not an already-final implementation choice).** A Flash-NH class
(`src/baseline/nh_dataset.py`) subclassing
`GenericDataset`, which after `_create_lookup_table` removes lookup entries
whose issue time is invalid for the run's `(seq_length, lead)` mask.
- custom code: small (~50–100 lines + registration shim);
- semantics: implements exactly the approved §12 history-only exclusion rule;
  per-period behavior follows the §14 signed-off uniform-exclusion policy;
- testability: high — unit-testable locally against synthetic lookup tables;
  runtime-provable via sample counts;
- risks: NH's dataset registry (`datasetzoo.get_dataset`) is an if/elif over
  built-in names — registering a custom class likely requires either (a) a
  thin Flash-NH training entry script that builds `Config`, patches/wraps the
  datasetzoo resolution, and calls NH's start-training API instead of bare
  `nh-run`, or (b) the pre-approved fork protocol
  (`docs/stage1_neuralhydrology_preflight.md` §10.2 — "custom sampler: fork or
  custom class" was already anticipated). **Which mechanism NH 1.13 actually
  supports (custom dataset hook vs wrapper vs fork) must be verified on
  Moriah** — it decides (a) vs (b), not whether Option 2 is viable.
- maintenance: pinned to NH 1.13.0 (the Moriah env is pinned; acceptable).

**Option 3 — gap-free time segmentation (config-only candidate).** Because the
gaps are global, training could be expressed as a list of contiguous gap-free
periods (MRMS's 136 hours fall in a limited number of runs; the resulting
train-period list is modest). If NH 1.13's `train_start_date`/`train_end_date`
accept lists (multi-period training), this needs **zero** custom NH code and
NH's own insufficient-history filter automatically excludes windows crossing
each segment start — equivalent to the input-span rule.
- must be verified on Moriah: list-date support in NH 1.13 `Config`, scaler
  fitting across multi-periods, validation/test multi-period behavior;
- semantic considerations: (i) segmentation naturally implements the
  history-only input-span rule (matching the approved §12 mask for training),
  but boundary/tail behavior per segment must still be proven equivalent;
  (ii) date handling/periods interactions are exactly the kind of
  silent-behavior risk the milestone is trying to avoid; (iii) couples the gap
  policy into dates, making the audit trail less direct than an explicit mask
  artifact.

**Rejected outright:** silent NaN dynamic inputs; Smoke 0/1 filling; dropping
rows from the hourly NC (breaks the shared 45,720-hour alignment); relying on
the gap flags alone.

**Recommended candidate path: Option 2**, contingent on I-D2-pre below, with
Option 3 checked opportunistically during the same Moriah session (if
list-dates are supported it becomes a documented fallback, not the baseline
mechanism — Option 2's exact semantics and provability win). The final
mechanism choice is made only after I-D2-pre evidence is in hand.

Verification split:
- **I-D2-pre — early targeted Moriah CPU inspection (BEFORE any
  `src/baseline/nh_dataset.py` implementation):** one short CPU allocation
  verifying (a) the dataset-registration mechanism (custom-class hook vs thin
  wrapper entry point vs fork), (b) lookup-table timestamp semantics
  (issue-time keying; the mapping between the mask generator's issue-time
  indexing and NH's lookup-table keys), (c) scaler handling of
  forcing NaNs that lie outside admitted samples (z-score fit over the full
  training-period series with `skipna`), and (d) whether a wrapper or a fork
  is required. Findings recorded in this doc / the decision log before I-D2
  coding starts;
- **local unit tests:** mask generation; lookup-table filtering logic against
  synthetic tables; count arithmetic vs independently computed synthetic
  expectations for the approved §12 mask definition;
- **Moriah CPU preflight:** dataset instantiation with the compact package;
  `len(dataset)` before/after filtering equals the predicted counts; datasetzoo
  registration mechanism confirmed;
- **Moriah GPU smoke:** training runs; a smoke-only assertion hook (or logged
  batch issue-times sample) shows no excluded issue time enters any batch.

**Runtime proof artifacts:** logged sample counts before/after filtering per
basin/split; sha256 of each mask artifact in `run_provenance.json`; logged
count of excluded issue times per split vs the **implementation-authoritative
expected counts from I-D1's approved mask computation** (the 2K-G-G table is a
conservative upper-bound cross-check only — agreement is never forced against
that broader, obsolete counting convention); smoke-run batch issue-time
assertion log.

---

## 14. Validation/test gap policy — SIGNED OFF (2026-07-13)

The binding §6 text covers **training windows** only; it is not explicit about
validation/test/holdout evaluation, so this was raised as an open
implementation-policy question in the I-0 draft. **User decision (2026-07-13):
option 1 + option 3's reporting convention.** Hard exclusion of
forcing-gap-contaminated issue times applies **consistently to training,
validation, temporal test, non-CA spatial-holdout evaluation, and California
fine-tune/holdout evaluation**. Excluded issue times are reported per split as
**predictions unavailable due to required forcing history intersecting an
archive gap**. This unblocks I-D1/I-D2 finalization.

Options considered (record):
1. **Exclude gap-contaminated issue times consistently from all splits
   (RECOMMENDED).** Consequence: validation model selection and all reported
   metrics are computed only on windows with complete forcing; excluded issue
   times are reported per split as "predictions unavailable due to archive
   gap" (a few % at most). Scientifically clean and symmetric; matches how the
   benchmark would honestly be described.
2. **Exclude only from training.** Consequence: with NaN-preserving forcing,
   gap-touching evaluation windows would feed NaN inputs through the LSTM →
   NaN predictions → silently dropped by `_mask_valid` at metric time. The
   *numbers* end up similar to option 1, but via fragile NaN propagation, with
   an un-audited pathway and misleading "evaluated on all hours" framing —
   and it would put NaN dynamic inputs into forward passes, brushing against
   the "no silent dynamic-input NaNs" rule.
3. **Report unavailable predictions at gap-contaminated issue times** —
   this is option 1's reporting convention; adopted as part of option 1.

---

## 15. Baseline package builder file map and behavior

```
scripts/build_stage1_baseline_nh_package.py   # thin CLI; all logic in src/baseline/
  inputs:  --forcing-dir --target-dir --out-dir
           --attributes-parquet --column-manifest
           --policy config/stage1_scientific_baseline_v001.yaml
           --splits-dir config/stage1_baseline_splits_v001/
           [--staids ... | --basin-list ...]  --expected-basins  --force  --dry-run
  steps:   validate policy + split artifacts (checksums) →
           load + validate area per basin →
           per basin: load forcing (NaN PRESERVED — no fill) + qobs →
             convert mm/h (float64) → build 4 lead targets → write NC
             (13 dynamic vars + 4 lead targets + qobs_m3s diagnostic) →
           attributes/attributes.csv (model_input columns ONLY) →
           basins/ (copied from canonical split artifacts, checksummed) →
           masks/ (still-proposed audit artifacts — see §12's 2026-07-20
             runtime-vs-audit-artifact note; the runtime filter itself only
             requires masks/gap_timestamps.json) →
           configs/ (via the config generator, §17) →
           manifests/ + run_provenance.json (policy/matrix/split/mask checksums)
scripts/audit_stage1_baseline_nh_package.py   # independent; reads original target
                                              # NCs for lead-equality sampling; accepts
                                              # 8/9/15-char STAIDs; asserts forcing NaN
                                              # counts == 136/2 per basin (NOT zero)
scripts/check_stage1_baseline_nh_preflight.py # Moriah-side structural + NH checks
```

Frozen for historical reproducibility: all Smoke 0/1 scripts (§4 table).

### Static attribute integration

**Updated 2026-07-20** — this subsection is forward-looking builder spec
(not yet implemented); it now names the current binding matrix. It
originally named `stage1_static_attributes_v001`/496 columns at I-0 planning
time (2026-07-13); see `docs/decision_log.md` 2026-07-20 entries for the
supersession record.

- Column selection: **strictly** `role == "model_input"` from
  `stage1_static_attributes_v002_column_manifest.json` — never dtype
  inference. `attributes/attributes.csv` contains **only** `gauge_id` + those
  473 columns, so `STATE`/`HUC02`/`LAT_GAGE`/`LNG_GAGE`/categorical/
  admin/coverage-flag columns cannot leak into NH inputs even via config
  error. The NH configs' `static_attributes` list is generated from the same
  manifest and cross-checked by the auditor. For the accepted 32-basin
  Compact Scientific Package specifically, `attributes/attributes.csv` is
  exactly the accepted 32×473 prepared static matrix (development-train-fit
  median imputation already applied and accepted — see §16).
- **Canonical basin area: `DRAIN_SQKM`** (GAGES-II BasinID, km²; the
  Bound_QA duplicate was already dropped at matrix build). The builder must
  fail loud if any HydroATLAS area-like column would be accidentally used;
  the audit records the exact area column name + value per basin in the
  package manifest, and the evaluation layer must read area **from the
  package manifest**, never re-join externally — this guarantees builder and
  evaluator use identical values.
- Area audit: units (km², sanity vs known basin sizes), positivity,
  completeness for all 2,752, exact source column, matrix sha256, and a
  conversion-consistency spot check (recompute one lead target from area).
- **Static-attribute NaN policy — SIGNED OFF (2026-07-13); numbers below are
  the original v001-era sign-off figures, preserved as historical record of
  the decision. The policy itself (median imputation fit on development-train
  only, applied unchanged elsewhere) carries over unchanged to v002 and has
  already been executed and accepted for the 32-basin Compact Scientific
  Package (168 imputed values, all on basin `393109104464500`, zero
  remaining NaNs — see `docs/FLASHNH_CURRENT_STATE.md`); a full 2,752-basin
  v002 missingness audit analogous to the ~195-column figure below has not
  yet been documented in this plan.** The problem is
  broader than the 5 HydroATLAS-gap basins (all of which are in the 2,752
  eligible set, making ~195 HydroATLAS `model_input` columns NaN for them,
  under v001):
  matrix policy allows up to 20% missingness per column, so the builder must
  **audit missingness across all 473 `model_input` columns (v002) over the
  development-training basin set**, not just the known HydroATLAS gap.
  Decided policy:
  - impute static `model_input` NaNs with **medians fitted only on
    development-training basins** (leakage-safe under §8d), and apply those
    fitted values **unchanged** to all other Stage 1–3 splits (validation,
    temporal test, spatial holdout);
  - audit missing/imputed counts **per column and per split**; record the
    fitted median values and their checksum in the package manifest /
    provenance;
  - **fail the build** if any `model_input` column is all-NaN over the
    development-training set;
  - **no missingness-indicator columns as model inputs in v001** unless
    separately approved;
  - Stage 4 may later fit its own medians using **only the California
    fine-tune-training subset** (§8c/§8d exception pattern).
  NH's behavior on NaN static attributes is still verified on Moriah
  (I-D2-pre / preflight) as a defensive check — post-imputation the package
  should contain none.
- **Zero-variance trainability projection — mechanism implemented
  2026-07-23 (`docs/decision_log.md` same-date entry).** After
  development-only median imputation above, some `model_input` columns may
  be exactly constant over the development-training population and cannot
  be normalized/trained on. `fit_zero_variance_projection` /
  `apply_zero_variance_projection` / `build_zero_variance_manifest`
  (`src/baseline/static_preparation.py`) fit this **once**, on the
  2,307-basin development-training population only, using exact
  post-imputation constancy (no near-zero-variance threshold), and freeze
  the retained/excluded column list for unchanged reuse across validation,
  temporal-test, and spatial holdout. **This does not change the package
  contract** — the authoritative static matrix and package remain 473
  `model_input` columns; the frozen retained-column list is consumed later
  by config generation. The 32-basin compact-smoke 13-column zero-variance
  exclusion (§ "Two findings" in the 2026-07-23 compact-integration-smoke
  closure) is compact-smoke-specific historical evidence and is **not**
  reused here. The real 2,307-basin excluded-column list has **not** been
  computed by this patch (requires h2o access to the full matrix).

---

## 16. Early h2o compact package and early Moriah smoke (revised sequencing)

**Updated 2026-07-20** — the compact basin subset described in this section
was a **~12–15-basin illustrative proposal** at I-0 planning time
(2026-07-13); the basin set is **no longer undecided**. A deterministic
32-basin Compact Scientific Package selection has since been accepted (see
`docs/stage1_compact_package_selection.md` and the acceptance record in
`docs/FLASHNH_CURRENT_STATE.md`): all 32 basins drawn from
`development_train` (no California or spatial-holdout basins), spanning 13
distinct HUC02s and 7 macro-regions (east/west split 19/13), including the
diagnostic basins `393109104464500` (HydroATLAS-gap / compound edge case;
under v002 imputation this basin carries all 168 imputed static values in
the accepted compact matrix) and `05568800` (lowest qobs completeness,
≈0.8746). This selection and its evidence bundle
(`/data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001_evidence`
on h2o) are frozen and **must not be regenerated or redesigned** here; the
paragraphs below describe the package this basin set is built into, not a
still-open selection question.

Moriah integration is pulled **forward**: it runs as soon as one
scientifically correct configuration exists, before the 16-combo expansion.

**Package architecture (provisional, frozen for this rollout):** one
physical package covering all 32 accepted basins. Every basin NetCDF
contains the eight accepted `v001-core` dynamic inputs, the diagnostic/
provenance `qobs_m3s` series, and all four shifted-target mm/h series
(`qobs_mm_per_h_lead01/03/06/12`) — never only the compact-configuration
lead. `attributes/attributes.csv` is exactly the accepted 32×473 prepared
static matrix (development-train-fit median imputation already applied; 168
imputed values, all on basin `393109104464500`; zero remaining NaNs). Each
lead-specific NH config selects exactly one of the four transformed targets
via `target_variables`; raw `qobs_m3s` is never configured as an NH training
target in any config, consistent with §1 item 1 and §11.

Compact configuration: **lead = 6 h, seq_length = 24 h** (primary benchmark
lead; the seq value continuous with Smoke 0/1 evidence — no better candidate
emerged from inspection). This is the **first** integration config, not the
final config set: after this smoke passes, the complete 16 lead×seq
combinations (§17) are generated against the same 32-basin package. Do not
describe the final design as only these four leads' worth of configs, or as
only this one config — the rollout is (1) this one lead06/seq24 config
first, then (2) the full 16-config expansion once (1) passes.

Sequence: build compact package on h2o → h2o audit (I-C3 auditor) → `scp -O`
to Moriah → CPU preflight (NH loads package; dataset counts match mask
predictions; static attributes load under the chosen NaN policy) → short GPU
smoke (few epochs) verifying: NH trains on `qobs_mm_per_h_lead06` with finite
loss; excluded issue times never appear in batches; predictions after NH
inverse scaling are in mm/h; a tiny Flash-NH script converts a sample back to
m³/s and matches the original magnitudes. Only after this smoke passes is the
16-combo config expansion (I-E2) and any full 2,752-basin build authorized.

No implementation, package build, remote run, or training has occurred as of
this docs-only patch (2026-07-20); the package/builder/auditor code itself is
still not implemented.

---

## 17. Config generation across 16 lead × seq combinations

One canonical template + `scripts/generate_stage1_baseline_configs.py`:
- template holds everything shared (NH 1.13 compat keys, dates from the policy
  YAML, v001-core `dynamic_inputs`, generated `static_attributes` list, basin
  files, seed hyperparameters from design doc §9c);
- per-run overrides: `seq_length`, `target_variables:
  [qobs_mm_per_h_lead{XX}]`, experiment name
  (`flashnh_stage1_baseline_seq{L}_lead{XX}`), dataset key for the custom
  gap-filtering dataset, mask artifact reference;
- generator emits all 16 + `config_manifest.json` (sha256 per config, policy
  checksum, template version).

Committed vs generated: **template + generator + a unit test are committed;
the 16 rendered configs are generated at package-build time** into the
package `configs/` (matching the smoke-package convention and repo generated-
artifact policy) and checksummed in the config manifest. Rendering at
training-launch time is rejected (config must be frozen and audited with the
package it belongs to).

Every config: exactly one target variable; **never `qobs_m3s`** (auditor
check); revised dates; development basin files; a separate evaluation config
variant for the spatial holdout (test basin file = `spatial_holdout_nonca.txt`,
test period 2025 — evaluation-only); CA absent everywhere;
`nan_handling_method` **absent** in the baseline configs (inputs in included
windows are complete by construction; the auditor asserts the key is not set
to a wrong value and that no config relies on unset-NaN behavior); provenance
fields (git commit, package/policy checksums) in a comment header block.

---

## 18. W&B onboarding (separate mini-milestone I-W1, after the first Moriah smoke)

W&B must not block I-F1 (first scientific integration smoke runs without W&B).
I-W1 scope, afterwards:
- concepts walk-through for the project owner (project = per major package
  version, e.g. `flashnh-stage1-scientific-baseline-v001`; run = one training;
  sweep = §9c/§9d later);
- config + metric logging (resolved NH config, loss/validation curves, LR,
  epoch timing), artifacts for small manifests only;
- authentication: `wandb login` once per machine writing to the user-private
  netrc; on Moriah, the API key is provided via the user's protected home file
  or `WANDB_API_KEY` exported in the user's private shell profile — **never**
  in repo files, scripts, sbatch files, docs, logs, Claude memory, or
  committed env files; sbatch scripts reference the variable, never the value;
- network: verify from a Moriah compute node whether outbound HTTPS works; if
  not, `WANDB_MODE=offline` with run dirs under the Flash-NH project root and
  `wandb sync` from the login node (or locally from pulled evidence) — resume
  behavior tested once;
- Slurm integration: job ID, node, partition, GRES, GPU type logged as run
  config; resource telemetry via `nvidia-smi` sampling where available;
- provenance to log (eventually): git commit, package manifest checksum, split
  manifest checksum, static matrix checksum, baseline policy checksum, lead,
  seq_length, target variable, Slurm fields, runtime/epoch timing, LR,
  training loss, validation metrics, raw-space Flash-NH metrics;
- minimal first W&B smoke: re-run the I-F1 compact configuration for 2–3
  epochs with W&B enabled (online or offline+sync) before any sweep.

---

## 19. Local / h2o / Moriah responsibility split

| Work | Local | h2o | Moriah |
|---|---|---|---|
| Repo inspection, planning, docs | ✔ | | |
| Policy YAML schema + loader + tests | ✔ | | |
| Pure unit tests (units, leads, staid, masks) | ✔ | | |
| Split algorithm development (fixtures) | ✔ | | |
| Canonical split generation | ✔ (on checksum-verified pulled matrix) or h2o | ✔ (alternative) | |
| Split QC figures/tables generation | ✔ (small data) | (✔ if generated there) | |
| Basin-universe reconciliation (canonical re-export, checksums) | review | ✔ | |
| Full static-matrix / forcing / target reads | | ✔ | |
| Baseline package building + auditing + checksums/manifests | dry-run only | ✔ | |
| Config rendering tests / dry runs | ✔ | | |
| Storage + runtime evidence bundles | pull+review | ✔ | ✔ |
| NH import/runtime checks, dataset-index integration checks | | | ✔ (CPU alloc) |
| CPU preflight, GPU training smoke, W&B smoke, sweeps | | | ✔ |

Moriah rules honored throughout: no Python imports or heavy scans on login
nodes; Slurm allocations for anything importing NH; GPU allocations for
training; `scp -O` for transfers. h2o is not a training platform.

---

## 20. Sub-milestones and commit sequence

Each is one small commit (or one commit + one evidence pull). "AC" =
acceptance criteria.

| ID | Purpose | Files created/changed | Local tests | h2o | Moriah | Depends on | AC / evidence | Canonical artifacts |
|---|---|---|---|---|---|---|---|---|
| I-0 | This plan | `docs/stage1_baseline_package_implementation_plan.md` | git checks only | — | — | — | plan approved by user | doc |
| I-B | Target conversion utilities + test infra | `src/baseline/{__init__,staid,units}.py`, `tests/`, pytest config | `pytest` green (units, staid, round-trips) | — | — | I-0 | all §10 tests pass; no other file touched | src+tests |
| I-A1 | Machine-readable policy | `config/stage1_scientific_baseline_v001.yaml`, `src/baseline/policy.py`, `tests/test_policy.py` | schema-validation tests | — | — | I-0 (§21 sign-offs recorded 2026-07-13) | loader rejects bad/binding-violating values | policy YAML |
| I-A2 | Split generator | `scripts/generate_stage1_baseline_splits.py`, `src/baseline/splits.py`, tests on fixtures | determinism + fixture tests | matrix pull/verify or run there; eligible-list re-export | — | I-A1; matrix checksum | candidate lists + assignment + manifest generated deterministically | (generated only) |
| I-A3 | Split auditor | `scripts/audit_stage1_baseline_splits.py` | auditor self-tests on corrupted fixtures | — | — | I-A2 | exit 0 on candidate; exit 1 on each seeded corruption | — |
| I-A4 | Split QC products | `scripts/generate_stage1_baseline_split_qc.py` | render smoke on candidate | — | — | I-A2 | QC bundle under tmp/ review dir; human review done | — |
| I-A5 | Canonical split promotion | `config/stage1_baseline_splits_v001/*`, decision-log entry, key figures | auditor re-run on committed copies | — | — | I-A3, I-A4, user approval | committed lists match approved candidate checksums | **split artifacts** |
| I-C1 | Lead-target utilities | `src/baseline/lead_targets.py`, tests | §11 tests | — | — | I-B | alignment/direction/tail tests pass | src+tests |
| I-D1 | Validity masks | `src/baseline/validity_mask.py`, tests | counts vs synthetic expectations for the approved §12 mask | — | — | I-A1 | implementation-authoritative counts computed + documented per combo; documented comparison against the conservative 2K-G-G table with any difference explicitly explained | src+tests |
| I-C2 | Baseline builder | `scripts/build_stage1_baseline_nh_package.py`, `src/baseline/static_attributes.py` | `--dry-run` + tiny synthetic-fixture build | — | — | I-A5, I-C1, I-D1 | local synthetic build passes its own manifest checks | builder |
| I-C3 | Independent package auditor | `scripts/audit_stage1_baseline_nh_package.py` | corrupted-fixture tests (wrong lead, wrong area, filled gap) | — | — | I-C2 | detects each seeded corruption; passes clean build | auditor |
| I-D2-pre | Targeted Moriah CPU inspection for NH integration (§13) | doc/decision-log findings only | — | — | short CPU alloc: registration hook, lookup-table timestamp semantics, scaler NaN handling, wrapper-vs-fork | I-D1 | findings recorded; mechanism (a)/(b) chosen | evidence bundle |
| I-D2 | NH sample-index integration | `src/baseline/nh_dataset.py` (+ entry shim), tests on synthetic lookup tables | filter-logic tests | — | CPU alloc: count check on compact data | I-D1, I-D2-pre | dataset counts match predictions on synthetic + compact data | src |
| I-E1 | Compact h2o package | (no new code; runbook in docs) | — | build + audit compact package | — | I-C2, I-C3 | h2o audit PASS; evidence pulled | h2o package |
| I-F1 | Early Moriah smoke | parameterized baseline sbatch (`PARTITION`/`GRES` vars per §11 policy) | — | — | CPU preflight + short GPU smoke | I-E1, I-D2 | finite loss on lead06 target; excluded times absent from batches; mm/h→m³/s spot conversion OK | evidence bundle |
| I-E2 | Full config generation | `scripts/generate_stage1_baseline_configs.py`, template, tests | render + manifest tests | regenerate package configs | — | I-F1 | 16 configs + manifest validated | generator |
| I-W1 | W&B onboarding + smoke | docs + minimal logging hooks | — | — | W&B-enabled compact rerun | I-F1 | run visible/synced with provenance fields | docs |
| I-F2 | Sweep readiness | sweep YAML skeleton, checklist | — | — | — | I-E2, I-W1 | user go/no-go for full 2,752 package + sweep | sweep config |

Full 2,752-basin package generation and the W&B sweep remain **separately
authorized** after I-F2 — not part of this milestone's commits.

**I-A3 status (2026-07-16):** implemented (`src/baseline/split_audit.py`,
`scripts/audit_stage1_baseline_splits.py`, `tests/test_split_audit.py`, 32
tests, all passing) as an *independent* auditor — it does not call
`build_split_assignment` and reimplements population reconstruction,
tercile fitting, stratum/pool routing, counts/fractions/HUC02 summaries,
and manifest/checksum reconciliation from scratch. Run against the real
I-A2 candidate (`tmp/stage1_baseline_splits_v001_candidate`) and its repeat
directory: **PASS, 0 errors, 0 warnings, 146 OK checks**, byte-identical
repeat comparison confirmed. This is a machine audit result only — it is
not a human QC review (I-A4) and does not constitute split promotion
(I-A5). I-A3 is now considered closed; no further auditor hardening is
planned unless a future check fails. I-A4 is scoped down to a minimal
visual sanity check (one CONUS map, one California map, two distribution
plots) to answer a single question — obvious geographic clustering,
missing regions, or severe area/aridity imbalance — not a general QC
framework. The candidate remains unpromoted; I-A5 remains pending human
sign-off on I-A4.

**I-A4 status (2026-07-16): human visual QC PASS.** Reviewed
`scripts/generate_stage1_baseline_split_qc.py`'s four plots against the
same real candidate: non-CA spatial holdout is broadly distributed across
the major CONUS basin clusters; the 19-basin California holdout has
reasonable north/central/south representation; non-CA development vs.
holdout drainage-area ECDFs broadly overlap; development vs. holdout
aridity ECDFs are nearly coincident (five missing-aridity basins omitted
from that plot only, not imputed, all remain in development training).
No visible clustering, missing region, or severe imbalance was found.
Generated plots stay under `tmp/stage1_baseline_splits_v001_qc/`
(gitignored, uncommitted). With I-A3 and I-A4 both PASS, **I-A5 canonical
promotion is the next and final split sub-milestone.**

**I-A5 status (2026-07-16): promotion COMPLETE.** The accepted I-A2
candidate was byte-copied (no regeneration, no reordering, no manual
edits) from `tmp/stage1_baseline_splits_v001_candidate/` to the canonical
path **`config/stage1_baseline_splits_v001/`**. Source basis: I-A3
independent audit PASS (0 errors, 0 warnings) and I-A4 human visual QC
PASS. All 10 expected artifacts are present and SHA-256-identical between
candidate and canonical copy. The committed I-A3 auditor was re-run
against the canonical directory (repeat evidence unchanged) and returned
**PASS, 0 errors**, with unchanged role counts
(`development_train`/`validation`/`temporal_test` 2307 each,
`spatial_holdout_nonca` 250, `california_all` 195,
`california_finetune_train` 176, `california_holdout` 19) and unchanged
holdout fractions (non-CA 0.09777, CA 0.09744). The five nonstandard
15-digit missing-aridity STAIDs were verified present unchanged. **The
Stage 1 baseline split design is now frozen** for the first Stage 1
baseline; do not reopen it absent a concrete scientific or correctness
problem. Next work is the baseline NH package-builder implementation.

**NH config-generation + structural-preflight status (2026-07-22, local
implementation increment, precedes I-E2/I-F1):** following Gate 4
certification of the 32-basin Compact Scientific Package,
`src/baseline/nh_config_generation.py` +
`scripts/generate_stage1_nh_config.py` render **one** config only — lead06,
seq_length 24, single target `qobs_mm_per_h_lead06`, the 8 approved dynamic
inputs in binding order, all 473 static `model_input` attributes, and the
frozen temporal split (train 2020-10-14→2023-12-31 / validation 2024 / test
2025) over the same 32 certified compact basins in every period. This is
deliberately narrower than §17's full 16-config generator and §20's I-E2
row (which remain future work, not started); it exists to prove the
config-generation + preflight mechanics on the smallest real case before
expanding. A companion two-layer structural preflight
(`src/baseline/nh_structural_preflight.py` +
`scripts/check_stage1_nh_config_preflight.py`) validates the generated
bundle file-only (Layer 1) and, against synthetic fixtures only, exercises
real `FlashNHDataset` construction for train/validation/test (Layer 2) —
this is preflight validation, not I-F1's Moriah smoke, and was never run
against the real h2o package or on Moriah. 38 tests added and passing (see
`docs/decision_log.md`'s 2026-07-22 entry for full detail, including a
discovered-and-fixed NH 1.13 upstream mutable-default-argument scaler bug
found while writing these tests). No h2o/Moriah access, no training, no
Slurm, no W&B. I-D2 (`src/baseline/nh_dataset.py`/`FlashNHDataset`) remains
the prerequisite this increment builds on, not something it redoes.

---

## 21. Risks, hidden failure modes, and open questions

### Confirmed repository facts (from this inspection)

- Old temporal dates live in `build_stage1_nh_package.py:70-74` (frozen smoke
  script — must not be consumed by any baseline code path).
- Raw `qobs_m3s` is the target in both generated smoke configs and the smoke
  auditor's expectations.
- Smoke gap fill lives only in `_apply_gap_fill` (same frozen script); the
  smoke auditor **requires** zero forcing NaN — reusing it on a baseline
  package would invert PASS/FAIL semantics.
- Both existing `_norm_staid` implementations coerce via `int(float(...))`;
  currently lossless for all 2,843 IDs but a latent corruption pattern.
- The 2,752 eligible set contains all 6 non-standard-length STAIDs; the smoke
  auditor's `len == 8` basin-list check would reject them.
- All 5 HydroATLAS-gap basins are in the 2,752 set → NaN static attributes
  will reach NH unless a policy is chosen (§15).
- The local dry-run static matrix is byte-different from the canonical h2o
  matrix (`3c3399f0…` vs `eb17aaa0…`) — content equivalence unverified; do not
  treat the local copy as canonical.
- The 2,752-basin list exists only on h2o + gitignored local evidence — not
  committed anywhere.
- `.gitignore` blocks `data/` and all `*.parquet` — split artifacts must be
  `config/` + text formats.
- No `tests/` directory exists.
- Under Policy B, the two gap-flag inputs (`mrms_qpe_1h_mm_gap`, `rtma_gap`)
  are **constant across all admitted samples** (every included window contains
  only non-gap hours), and a feature that is constant over the admitted sample
  distribution carries **no discriminative signal** for training. This is a
  consequence of two binding decisions interacting, recorded here as a
  diagnostic observation — v001-core is not reopened.

### Failure modes the design above explicitly guards

Old dates surviving (policy YAML is the only date source; auditor checks
config dates against it) · raw target remaining (auditor forbids `qobs_m3s`
in `target_variables`) · smoke fill reused (baseline auditor asserts exactly
136/2 NaN forcing hours per basin) · silent input NaNs / unset
`nan_handling_method` (hard exclusion + config audit) · wrong area
field/units/divergent builder-vs-evaluator area (single manifest-recorded
`DRAIN_SQKM`; evaluator reads the manifest) · lead shift off-by-one/wrong
direction/tail-NaN errors (pinned asymmetric-fixture tests + independent
auditor recomputation from original NCs) · builder/auditor shared bug
(independent code paths + hand-calculated constants) · static role leakage
(STATE/HUC02/lat-lon/categoricals physically absent from package
attributes.csv) · CA leakage / holdout in scaler fit / val-test basins in
training (split auditor disjointness + builder consumes only committed
artifacts + NH `is_train` contract for the temporal half) · missing
seed/non-determinism (manifest-recorded seed; auditor re-runs assignment) ·
sparse-stratum instability / overcomplex stratification (deterministic
fallback ladder, everything logged) · STAID coercion/leading-zero/non-standard
loss (strict `staid.py`, `dtype=str` loading, length whitelist) ·
2,843-vs-2,752 join mixing (fail-loud universe reconciliation) ·
forcing/target basin mismatch (equality assertion) · config drift / policy
duplication (single policy YAML + generator + checksummed config manifest) ·
generated artifacts committed accidentally (gitignore already blocks
parquet/nc; promotion is an explicit copy step) · remote paths in portable
config (policy YAML carries no machine paths) · gap-mask/NH indexing mismatch
and invalid issue times reaching the dataloader (Moriah count reconciliation +
batch assertions, §13) · val/test gap-policy inconsistency (signed-off uniform policy,
§14) · W&B credentials in repo / W&B blocking the first smoke (I-W1 sequenced
after I-F1; key only via user-private env).

### User sign-offs recorded 2026-07-13 (raised as open questions by the I-0 draft)

1. **Validation/test gap policy — SIGNED OFF (§14).** Hard exclusion of
   forcing-gap-contaminated issue times applies consistently to training,
   validation, temporal test, non-CA spatial-holdout evaluation, and
   California fine-tune/holdout evaluation; excluded issue times are reported
   as predictions unavailable due to required forcing history intersecting an
   archive gap. Unblocks I-D1/I-D2.
2. **Static-attribute NaN policy — SIGNED OFF (§15).** Medians fitted only on
   development-training basins; applied unchanged to all other Stage 1–3
   splits; missing/imputed counts audited per column and per split across all
   473 `model_input` columns (**updated 2026-07-20** to the current
   `stage1_static_attributes_v002` count — the original 2026-07-13 sign-off
   named 496/v001; the decision itself is unchanged, see §15); fitted values
   + checksum recorded; build fails if any `model_input` column is all-NaN
   over development training; no missingness-indicator inputs in v001 unless
   separately approved; Stage 4 may later fit its own medians on the CA
   fine-tune-training subset only. Unblocks I-C2.
3. **California membership — SIGNED OFF.** `STATE == "CA"` from the canonical
   split-support field. `LAT_GAGE`/`LNG_GAGE` are diagnostic cross-checks
   only: anomalies are flagged for human review in the split QC, never
   silently reassigned. Unblocks I-A2.
4. **Spatial-holdout parameters — SIGNED OFF (§7).** Seed 42, target 10%,
   acceptable overall range 8–12%, area terciles, aridity terciles, initial
   minimum composite-stratum size 10. This is the **candidate** split method,
   subject to machine audit and human QC — not a guarantee that the first
   generated split is accepted. Unblocks I-A2.
5. **`eligible_basins_v001.txt` commit — APPROVED, conditionally.** Committed
   under `config/stage1_baseline_splits_v001/` **only after** canonical h2o
   re-export, checksum verification, forcing-target basin-set equality
   checking, and comparison with the local evidence copy. Gates I-A5.

### Items requiring h2o evidence

- Canonical matrix pull + sha256 verification for local split generation
  (or run split generation on h2o instead).
- Re-export + checksum of the 2,752 list from the h2o target-package manifest.
- Forcing-vs-target basin-set equality re-assertion.
- HUC02/STATE value sanity in the canonical matrix (dtype/padding of HUC02).
- Compact package build + audit (I-E1).

### Items requiring Moriah evidence

- **I-D2-pre (bundled targeted CPU inspection, §13/§20):** dataset-
  registration mechanism for a custom `GenericDataset` subclass (custom-class
  hook vs wrapper entry point vs fork); lookup-table timestamp semantics /
  warm-up behavior at period starts (issue-time mapping; antecedent history
  not crossing split boundaries); scaler handling of forcing NaNs outside
  admitted samples; whether `train_start_date`/`train_end_date` accept lists
  (Option 3 viability as fallback).
- NH behavior on NaN static attributes (defensive check; post-imputation the
  package should contain none — §15).
- Compact-package CPU preflight + GPU smoke (I-F1), including
  sample-count reconciliation and batch issue-time assertions.
- Moriah compute-node network egress for W&B online mode (I-W1).

---

## 22. Recommended next Claude Code prompt (exact scope)

**Sub-milestone I-B: target conversion utilities + STAID normalizer + test
infrastructure.** One commit:
- create `src/baseline/__init__.py`, `src/baseline/units.py`,
  `src/baseline/staid.py`;
- create `tests/` with `tests/test_units.py`, `tests/test_staid.py` and
  minimal pytest configuration, following the §10 test list (hand-calculated
  values, round-trips, NaN propagation, broadcasting across
  scalar/ndarray/Series/DataArray, invalid-area rejection, float32 tolerance;
  STAID: 8/9/15-char preservation, zfill-under-8, non-digit rejection,
  no-numeric-coercion property);
- touch nothing else — no builder, no config, no split code, no Slurm, no
  remote work.

Why this first: it has **zero** open dependencies (the conversion formula and
naming are binding; the API takes `area_km2` as an argument, so the
area-source decision does not block it); it establishes the `src/baseline/` +
`tests/` skeleton every later sub-milestone builds on (the repo currently has
no test infrastructure at all); and it is the smallest reviewable,
independently correct commit. With the five §21 sign-offs now recorded
(2026-07-13), I-A1 (policy YAML) follows immediately after I-B with no
remaining open inputs.

The I-B prompt should also carry the standing constraints: do not modify the
frozen smoke scripts; do not create generated outputs outside `tmp/`; do not
run remote commands; single small commit after tests pass locally.
