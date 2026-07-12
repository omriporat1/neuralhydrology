# Stage 1 — Target Scaling + Gap Policy + Lead-Time Feasibility Report

Milestone: **2K-G-G**, opened 2026-07-08 (Phase A).

## Status

**Phase A scaffold complete. Phase B complete for all evidence gathering
planned so far: authoritative Moriah NH 1.13.0 evidence obtained and
analyzed for NH-mechanics questions Q1-Q9, all now answered (no remaining
`REQUIRES TARGETED SOURCE INSPECTION` markers — the 3 that were blocked in
the first follow-up round were closed in a second follow-up round,
2026-07-12, once Moriah SSH access was restored; see "Evidence follow-up
(2026-07-12, part 2)" below). Window-feasibility evidence against the real
gap inventory (Q10-Q11) is also answered (2026-07-12, part 1 follow-up).
No final target-scaling, gap-policy, or lead-time implementation decision
has been made — those remain explicitly `PENDING`, gated on a Flash-NH
policy decision, not on missing evidence.**

`scripts/inspect_neuralhydrology_stage1_mechanics.py` was run on Moriah on
a compute node (`glacier-30`, not the login node) against the
`flashnh-moriah` conda environment. Confirmed facts: NeuralHydrology
package version `1.13.0` (via package metadata; the package has no
`__version__` attribute), Python `3.11.15`, module path
`/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/lib/python3.11/site-packages/neuralhydrology`.
Evidence bundle pulled to local `tmp/nh13_inspection_moriah_20260712T100047Z/`
per `docs/repo_policy.md` (gitignored, not committed). A separate local
1.11.0 inspection (`tmp/nh13_inspection_local_20260712T085939Z/`) also
exists but is version-mismatched relative to Moriah's 1.13.0 and is treated
as non-authoritative background only — no finding below relies on it.

Sections below distinguish clearly between:
- what is **inherited / already decided** (binding, from
  `docs/stage1_scientific_baseline_design.md`),
- what is **scaffolding created in Phase A** (scripts, commands, this doc),
- what is now **answered from Moriah 1.13.0 evidence** (Phase B, this
  update) — see "Findings" below, with question numbers matching "Questions
  to answer in Phase B",
- what is **still `PENDING` or `REQUIRES TARGETED SOURCE INSPECTION`** —
  either evidence not yet gathered (window-feasibility numbers) or keyword
  hits without enough surrounding source context to state a claim as fact.

Do not treat a `PENDING` or `REQUIRES TARGETED SOURCE INSPECTION` marker as
an answer — those remain open. Everything else in "Findings" below is
sourced from the Moriah 1.13.0 evidence bundle and is authoritative per the
"Inspection environment policy" below.

## Purpose

Answer, from the actual installed NeuralHydrology 1.13 code (not public
docs, not assumptions — see "Inspection environment policy" below for
where that code may be inspected), the questions gating §5 (target
normalization) and §6 (forcing gap policy) of
`docs/stage1_scientific_baseline_design.md`, and quantify sample/window loss
across the Stage 1 `seq_length` x lead-time design space (§9a/§9b of the
same document).

## Inspection environment policy

Local inspection of a pinned NeuralHydrology 1.13 installation is allowed
and encouraged for code review, source navigation, and preliminary
interpretation. However, final Phase B conclusions must be verified against
the actual `flashnh-moriah` environment, because that is the runtime used
for Smoke 0/1 and future Stage 1 training. If local and Moriah installations
differ, Moriah is authoritative.

In practice: `scripts/inspect_neuralhydrology_stage1_mechanics.py` may be
run against any environment with `neuralhydrology` importable — local or
Moriah — to produce a **preliminary** `nh13_inspection_summary.{md,json}`.
Before any Phase B finding in this document is recorded as final, the same
claim must be re-verified against `flashnh-moriah` on Moriah, and the NH
version/path recorded by each run (already captured in
`nh13_inspection_summary.json`) compared.

**Moriah usage discipline (consolidated, 2026-07-12 self-check).** Applies
to all Moriah interaction for this milestone, not just the commands listed
in "Exact Moriah commands" below:
- Simple `sed`/`grep` reads over already-installed source files are
  acceptable directly on the Moriah **login node** — no allocation
  needed. This includes the targeted single-file inspections used to
  answer Q2/Q5/Q6/Q7 above.
- Do **not** run Python imports, recursive source-tree walks/greps,
  package builds, training, large file transfers, or other heavy
  filesystem scans on the login node. Any recursive search across the
  whole `neuralhydrology` package tree (see the optional `Sampler`-class
  check in "Q8" above) must be scoped to specific files or moved into a
  short CPU allocation.
- Any Python code that imports or walks NeuralHydrology (e.g.
  `scripts/inspect_neuralhydrology_stage1_mechanics.py`) must run inside a
  short CPU allocation, not directly on the login node.
- Local inspection of a non-Moriah NeuralHydrology install is useful for
  exploration only when the local version is clean and confirmed
  version-matched to Moriah's `1.13.0`; Moriah remains authoritative
  regardless (see local-1.11.0 caveat in "Status" above).
- On Windows/VS Code, `ssh moriah` has been confirmed to work through
  PowerShell; the Bash tool's own hostname/alias resolution may differ
  and should not be assumed to work identically — verify per-session
  rather than assuming the prior session's connectivity result still
  holds.
- Moriah file transfer requires legacy SCP mode (`scp -O` /
  `scp -O -r`) — the default modern `scp`/SFTP subsystem is unavailable
  on Moriah as of 2026-07-12.

## Non-goals (Phase A)

- Does not run training.
- Does not build the full 2,752-basin (or 2,843-basin universe) NH package.
- Does not modify package builders (`scripts/build_stage1_nh_package.py`),
  scientific NH configs, or Slurm training templates
  (`scripts/run_stage1_smoke{0,1}_moriah.sbatch`).
- Does not commit generated data or outputs under `tmp/`.
- Does not finalize target-scaling, gap-policy, or lead-time implementation
  decisions. Those remain gated on Phase B, which requires the actual
  Moriah evidence produced by the tools introduced here.
- Does not assume Moriah inspection has succeeded. Phase A only creates the
  tooling and commands; a human must run them.

## Inherited binding decisions (context, not re-decided here)

From `docs/stage1_scientific_baseline_design.md` (14 binding decisions,
2026-07-06 revision) — quoted here only as the constraints this milestone's
findings must respect, not re-opened:

- Stage 1 dynamic inputs are `v001-core` (8 variables): `mrms_qpe_1h_mm`,
  `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms`,
  `mrms_qpe_1h_mm_gap`, `rtma_gap`. (§1)
- `seq_length` candidates are **only** 12 / 24 / 48 / 72 h. 168/336 h are
  **not** Stage 1 candidates. (§9a)
- Lead time is a separate axis from `seq_length`: primary **6 h**, secondary
  **12 h**; 1 h / 3 h are diagnostic-only. (§9b)
- Log-transform target scaling is **rejected**. Leading candidate is
  area-normalized / specific discharge, pending feasibility — this is
  exactly what Phase B must resolve. (§5)
- Evaluation metrics are always computed in raw `m^3/s` after
  inverse-transforming model output; raw-space NSE is primary. (§7)
- Smoke 0/1 forcing gap fill (MRMS gaps → 0.0 mm, RTMA gaps → linear
  interpolation) is **technical-only**, explicitly not approved for
  scientific training. (§6)
- MRMS archive gaps: 136 h/basin; RTMA gaps: 2 h/basin, in the full-period
  curated forcing product (45,720 h/basin, 2020-10-14 → 2025-12-31). These
  gaps are archive-level (`not_in_s3`) absences and are the same absolute
  hours for every basin, not basin-specific.
- Leakage prevention: all Stage 1–3 scaling/normalization statistics are fit
  only on development-training basins/period, never on validation, temporal
  test, spatial holdout, or California data. (§8d) Any target-scaling
  mechanism found feasible in Phase B must be checked against this rule too
  (e.g., is a basin-area normalization "static" and leakage-free, or does a
  scaler need explicit training-only fitting?).

## Phase A deliverables (created in this patch)

| File | Purpose |
|---|---|
| `scripts/inspect_neuralhydrology_stage1_mechanics.py` | Read-only inspection of installed NH 1.13 code (local and/or Moriah; Moriah authoritative) |
| `scripts/analyze_stage1_window_feasibility.py` | Geometry / gap-exclusion window-loss estimator (no NH required) |
| `docs/stage1_target_scaling_gap_leadtime_feasibility.md` | This document |

Both scripts were syntax-checked and smoke-tested locally (see "Local
validation" below). Neither requires Moriah to run in degraded/geometry-only
mode; the inspection script additionally requires an environment with the
`neuralhydrology` package installed to produce real findings — this may be
a local NH 1.13 install or the `flashnh-moriah` env on Moriah (see
"Inspection environment policy" above). If run locally, record the NH
version and install path (already written to
`nh13_inspection_summary.json` by the script) so results can be compared
against Moriah.

## Exact Moriah commands to run

Known paths:
- Moriah env: `/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah`
- Moriah project root: `/sci/labs/efratmorin/omripo/Flash-NH`
- Repo working directory on Moriah:
  `/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation`

```bash
# 1. SSH to Moriah, then get an interactive CPU/light allocation if not
#    already in one (no GPU needed for this inspection):
ssh moriah
cd /sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation
git pull

# 2. Activate the Moriah NH environment (module context first, per
#    docs/stage1_neuralhydrology_preflight.md §10.6):
module load spack/all
module load miniconda3/24.3.0-gcc-iqeknet
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah

# 3. Confirm Python / NH paths and version:
which python
python --version
python -c "import neuralhydrology, sys; print(neuralhydrology.__file__); print(getattr(neuralhydrology, '__version__', 'no __version__ attr'))"

# 4. Run the read-only NH inspection helper:
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="/sci/labs/efratmorin/omripo/Flash-NH/tmp/nh13_inspection_${TS}"
mkdir -p "$OUT_DIR"
python scripts/inspect_neuralhydrology_stage1_mechanics.py --out-dir "$OUT_DIR"

# 5. Run the window feasibility script in geometry-only mode (no NH import,
#    safe to run in the same environment or a plain Python one):
FEAS_OUT="/sci/labs/efratmorin/omripo/Flash-NH/tmp/stage1_window_feasibility_${TS}"
python scripts/analyze_stage1_window_feasibility.py \
    --period-start 2020-10-14 --period-end 2025-12-31 \
    --seq-lengths 12,24,48,72 --lead-times 1,3,6,12 \
    --out-dir "$FEAS_OUT"

# 6. OPTIONAL — real-gap run, only after the Flash-NH full-period forcing
#    gap-run audit table (`fullperiod_gap_inventory.csv`, columns:
#    product, gap_start_utc, gap_end_utc, gap_length_hours, gap_type,
#    reason, product_synchronized) has been copied or located on Moriah.
#    Do not fabricate one; skip this step if no such file exists yet.
#    The script auto-detects this "gap-run-intervals" schema (as opposed
#    to the older "timestamp-rows" schema) and maps product names
#    robustly (any value containing "mrms" -> mrms pool, "rtma" -> rtma
#    pool) — see the script's module docstring for both schemas.
#
#    Example expected location once copied to Moriah:
GAP_CSV="/sci/labs/efratmorin/omripo/Flash-NH/tmp/evidence_inputs/fullperiod_gap_inventory.csv"
if [ -f "$GAP_CSV" ]; then
  python scripts/analyze_stage1_window_feasibility.py \
      --period-start 2020-10-14 --period-end 2025-12-31 \
      --seq-lengths 12,24,48,72 --lead-times 1,3,6,12 \
      --gap-inventory-csv "$GAP_CSV" \
      --out-dir "${FEAS_OUT}_with_gaps"
else
  echo "No real gap inventory found at $GAP_CSV -- copy/locate"
  echo "fullperiod_gap_inventory.csv there first, or run this step later."
fi

# 7. List generated evidence files:
find "$OUT_DIR" -type f
find "$FEAS_OUT" -type f
[ -d "${FEAS_OUT}_with_gaps" ] && find "${FEAS_OUT}_with_gaps" -type f
```

Per `docs/repo_policy.md` ("h2o/Moriah Remote Run Evidence Policy"), pull
the evidence bundle to a local `tmp/` directory (e.g. via `scp`) before any
conclusion from it is documented or committed. Do not `git add` anything
under `tmp/`.

## Expected evidence files

From `scripts/inspect_neuralhydrology_stage1_mechanics.py --out-dir <D>`:
- `<D>/nh13_inspection_summary.md`
- `<D>/nh13_inspection_summary.json`
- `<D>/module_paths.txt`
- `<D>/source_hits.txt`
- `<D>/run_command.txt`
- `<D>/git_commit.txt`

From `scripts/analyze_stage1_window_feasibility.py --out-dir <D>`:
- `<D>/window_feasibility.csv`
- `<D>/window_feasibility_summary.md`
- `<D>/window_feasibility_summary.json`

## Questions to answer in Phase B (framed here, not answered)

From the NH-inspection evidence:
1. Can NH 1.13's `GenericDataset` support area-normalized discharge
   cleanly — as a target transform, a custom static-attribute-driven
   scaling, or does it require a subclass/custom dataset?
2. Is the target transform reversible for raw `m^3/s` evaluation metrics
   using NH's native scaler save/load/inverse-transform machinery, or does
   Flash-NH need a custom inverse-transform step at evaluation time?
3. Where, in the installed code, are scalers actually fit and saved (which
   class/method, what file)?
4. Can scalers be fit only on training basins/period (per the §8d leakage
   rule), using NH's existing config options, or does this require custom
   code?
5. How are `qobs` NaNs handled in the loss and in evaluation metrics —
   confirm the loss-masking behavior already assumed in
   `docs/stage1_neuralhydrology_preflight.md` §8.1 against the actual code
   (not just prior behavior observed in Smoke 0/1 runs).
6. Are dynamic-input NaNs allowed at all, and if so under what
   configuration?
7. What does `nan_handling_method` actually do in the installed NH 1.13
   (exact code path), not what upstream docs/changelogs claim?
8. Can windows/samples intersecting an MRMS/RTMA gap hour be excluded
   natively by the `generic` dataset's batch sampler (e.g. via a config
   option), or does this require the package builder to emit prefilled
   values + flags and/or a custom sample mask / custom sampler class?
9. How should 6 h and 12 h lead time actually be implemented in an NH 1.13
   config — is this `predict_last_n`, a dedicated forecast/horizon config
   block, or does it require restructuring the target coordinate at
   package-build time?

From the window-feasibility evidence:
10. What is the sample/window-loss estimate across the full `seq_length` x
    lead-time matrix (12/24/48/72 h x 1/3/6/12 h) — both geometry-only and,
    once a real gap inventory is available, under MRMS/RTMA/either-gap
    window exclusion?
11. Does any `seq_length` x lead-time combination have a surprisingly large
    loss fraction that should influence the sweep design in
    `docs/stage1_scientific_baseline_design.md` §9c?

Cross-cutting:
12. What evidence is still missing before final Stage 1 package generation
    and before §5/§6 of the scientific baseline design doc can be signed
    off?

## Findings — Phase B (Moriah NH 1.13.0 evidence, 2026-07-12)

Evidence source for everything in this section unless otherwise noted:
`tmp/nh13_inspection_moriah_20260712T100047Z/{nh13_inspection_summary.md,
nh13_inspection_summary.json, source_hits.txt}`, `flashnh-moriah` env,
NH `1.13.0`, git commit `28883a6b4c50942a9da3223b3a863a1955444d4c` (matches
the committed Phase A scaffold). `source_hits.txt` caps each keyword search
at 40 hits; where a claim below rests on a hit without enough surrounding
context, it is marked `REQUIRES TARGETED SOURCE INSPECTION` with the
smallest follow-up command.

### NH 1.13 mechanics — answered

**Class/method inventory (`BaseDataset`/`GenericDataset`).**
`neuralhydrology.datasetzoo.basedataset.BaseDataset` (`basedataset.py:28`)
is the real implementation: `_load_or_create_xarray_dataset` ->
`_load_data` -> per-basin `_load_basin_data`/`_load_attributes` -> scaling
-> `_create_lookup_table` (sample index) -> `__getitem__`/`__len__` at
inference time. `neuralhydrology.datasetzoo.genericdataset.GenericDataset`
(`genericdataset.py:13`) is a thin `BaseDataset` subclass that overrides
only `_load_attributes` and `_load_basin_data` (`genericdataset.py:69,75`,
~6-10 lines each) — it adds no normalization, no target transform, and no
sample-filtering logic beyond what `BaseDataset` already does. Its
docstring (`genericdataset.py:25`) states that values "marked as NaN's"
are the values `GenericDataset` recognizes as missing — i.e. Flash-NH's
forcing gap convention must use actual NaN (not a numeric sentinel) for
`GenericDataset` to treat a gap hour as missing.

**Q3 — Where are scalers fit and saved?** Answered. In
`BaseDataset.__init__` (`basedataset.py:78-107`): if `is_train=True` and no
`scaler` argument is passed, `self._compute_scaler = True`; if
`is_train=False` (validation/test) and no `scaler` is passed, it raises
`ValueError("During evaluation of validation or test period, scaler
dictionary has to be passed")` (`basedataset.py:79`). Dynamic/target
feature scaling is z-score by default: `xr = (xr - center) / scale`
(`basedataset.py:758`), with `scale = xr.std(skipna=True)`,
`center = xr.mean(skipna=True)` computed only when `_compute_scaler` is
True (`basedataset.py:763-765`); per-feature override branches exist for
fixed-zero-center, median-center, and min-center variants
(`basedataset.py:773-777`). Static attributes get an analogous
mean/std scaler (`basedataset.py:725-736`, with a deprecated
`camels_attr_means/stds` alias). The fitted scaler dict is written to
`<train_dir>/train_data_scaler.yml` via `_dump_scaler()`
(`basedataset.py:254-266`), and reloaded via
`neuralhydrology.datautils.utils.load_scaler(run_dir)`
(`datautils/utils.py:29`).

**Q4 — Can scalers be fit train-only per §8d, without custom code?**
Answered, with a scope refinement added 2026-07-12: **yes, but only for
the *temporal* half of §8d, and only if Flash-NH constructs its basin
lists correctly before NH ever sees them — NH does not know what a
"spatial holdout basin" or "California basin" is.**

The `is_train`/`scaler`-argument contract above (a scaler is computed
exactly once, on whichever dataset instantiation passes `is_train=True`,
and every other instantiation must be passed that same scaler dict or NH
raises `ValueError`) enforces one specific thing: **NH will only ever fit
a scaler on the dataset it was told is the training dataset.** That
dataset is defined entirely by which **basin list and period** Flash-NH's
launch code passes in when it constructs the `is_train=True`
`GenericDataset`. NH has no concept of "spatial holdout" or "California" —
those are Flash-NH-side basin-list categories (§8d), not NH config
options.

Two leakage modes follow from this and must be kept distinct:
- **Temporal leakage** (fitting a scaler using timesteps from validation
  or test *periods*, for basins that legitimately belong in training) —
  this is what the `is_train`/`scaler`-argument contract above actually
  protects, automatically, as long as the training-period dataset's date
  range excludes validation/test periods (a period-slicing detail Flash-NH
  already controls upstream of NH, not evidenced here).
- **Spatial leakage** (fitting a scaler using basins that should never
  contribute to training statistics at all — spatial holdout basins,
  California basins) — **NH provides zero automatic protection for this.**
  If Flash-NH's training-basin list accidentally includes a spatial
  holdout or California basin, NH will compute the scaler over it without
  complaint; nothing in `BaseDataset`/`GenericDataset` filters basins by
  role. Spatial-leakage prevention is entirely a Flash-NH responsibility:
  the basin list passed to the `is_train=True` dataset instantiation must
  already have non-CA spatial-holdout and California basins removed
  *before* NH is invoked at all.

Conclusion: it is correct to say "NH enforces train-dataset-only scaling"
(no custom scaler-fitting code is needed for that mechanism). It is
**not** correct to say "NH enforces §8d spatial holdout" — that claim
would conflate what NH's contract does (protect a dataset it is told is
"train") with what Flash-NH's basin-list construction must independently
guarantee (that the dataset NH is told is "train" is actually the correct
set of basins). This is an interpretation refinement of the original
Q4 answer, not a reversal of it, and does not change the earlier
conclusion that no custom *scaler-fitting* code is needed — it clarifies
that correct *basin-list* code upstream of NH is still Flash-NH's
responsibility and is unverified by this evidence round.

**Q1 — Can `GenericDataset` support area-normalized discharge natively?**
Answered: **not as a config-level transform.** Area/temporal-resolution
based discharge normalization exists in NH 1.13 only as
`neuralhydrology.datasetzoo.lamah._normalize_discharge(ser, area,
temporal_resolution)` (`lamah.py:287`), called from the `LamaH`
dataset-subclass's own `_load_basin_data` (`lamah.py:124,219`) — i.e. it is
a dataset-subclass-specific unit-conversion pattern, not a `GenericDataset`
config flag, and `GenericDataset` itself has no equivalent hook (confirmed
above: its only overrides are `_load_attributes`/`_load_basin_data`, both
minimal). Two implementation paths follow from this: (a) write a
Flash-NH-specific `Dataset` subclass replicating the `lamah.py` pattern, or
(b) precompute area-normalized/specific discharge **at package-build
time** (i.e. the NC/CSV target column Flash-NH's builder writes is already
`qobs / area`, so plain `GenericDataset` + NH's default z-score scaler
apply on top with zero custom dataset code). (b) is much lower-effort and
consistent with Flash-NH's existing builder-centric architecture
(`scripts/build_stage1_nh_package.py`) — recommended as the leading
candidate for §5, though not decided here (Phase B supplies mechanism
facts, not the final policy decision).

**Q2 — Is the target transform reversible via NH's native scaler
machinery?** Fully answered (2026-07-12, part 2 follow-up — exact
arithmetic confirmed against `evaluation/tester.py:247-259` on Moriah).
Confirmed exact code (`_evaluate` scope, inside the per-basin/per-frequency
loop):

```python
feature_scaler = self.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values
feature_center = self.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values
y_freq = y[freq] * feature_scaler + feature_center
# rescale predictions
if y_hat[freq].ndim == 3 or (len(feature_scaler) == 1):
    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
elif y_hat[freq].ndim == 4:
    feature_scaler = np.expand_dims(feature_scaler, (0, 1, 3))
    feature_center = np.expand_dims(feature_center, (0, 1, 3))
    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
```

This confirms, exactly as hypothesized: both observations (`y_freq`) and
predictions (`y_hat_freq`) are unscaled with the same formula,
`raw = scaled * feature_scale + feature_center` — the direct algebraic
inverse of the forward z-score at `basedataset.py:758`
(`scaled = (raw - center) / scale`). There is no separate/different
formula for predictions vs. observations — both go through this identical
inline arithmetic (the `ndim==4` branch only handles a multi-distribution/
multi-target shape difference via `np.expand_dims`, not a different
formula). The literal string `inverse_transform` has **zero** hits
anywhere in the installed 1.13.0 source — there is **no public
inverse-transform API**; this is inline arithmetic private to `tester.py`
(no equivalent helper exists elsewhere for Flash-NH to reuse directly;
replicating this reversal outside of `Tester` would mean copying this
exact 3-line formula, not calling a documented NH function).

Critically, the caveat from the first Phase B pass still holds and is now
confirmed rather than inferred: **this reversal only undoes NH's z-score.**
If Flash-NH precomputes area-normalized/specific discharge at
package-build time (Q1 recommendation), NH's native reversal (the formula
above) returns predictions in specific-discharge units, not raw m^3/s —
converting back to raw m^3/s (required for §7's raw-space NSE) needs an
**additional Flash-NH-side step** (multiply by basin area) applied after
NH's native unscale, using the same area value the builder used. This is
not automatic and is not provided by NH. Evidence:
`tmp/nh13_targeted_inspection_moriah_20260712T120839Z/tester_240_270.txt`.

**Q5 — How are target (`qobs`) NaNs handled in loss and evaluation
metrics?** Fully answered (2026-07-12, part 2 follow-up). Evaluation side
(unchanged from the first pass): `neuralhydrology.evaluation.metrics._mask_valid(obs, sim)`
(`metrics.py:38`) is called at the start of every individual metric
function (NSE, KGE, peak-timing, FDC signatures, etc. — 13+ call sites,
e.g. `metrics.py:84,120,184,223,261,305,341,389,451,506,584,667,735`),
masking out any timestep where either `obs` or `sim` is NaN before the
metric is computed.

Sample/loss side: **confirmed.** `neuralhydrology/training/loss.py`
defines `BaseLoss` (`loss.py:13`) and six concrete subclasses —
`MaskedMSELoss` (`:175`), `MaskedRMSELoss` (`:196`), `MaskedNSELoss`
(`:217`), `MaskedGMMLoss` (`:265`), `MaskedCMALLoss` (`:311`),
`MaskedUMALLoss` (`:349`) — every one of which masks target NaNs
per-element inside its own `_get_loss` before computing the loss, e.g.
`MaskedMSELoss._get_loss` (`loss.py:191-193`):
```python
mask = ~torch.isnan(ground_truth['y'])
loss = 0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask])**2)
```
and the mixture-density losses use an analogous per-sample mask,
`~torch.isnan(ground_truth['y']).any(1).any(1)` (`loss.py:299,330,370`).
This confirms `BaseDataset._validate_samples`'s lack of a target-NaN
exclusion criterion (Q5's earlier finding) is intentional, not an
oversight: NH's design is "include the sample in the lookup table,
exclude the NaN element(s) from the loss reduction at train time" —
exactly analogous to `_mask_valid` on the evaluation side, using
`torch.isnan` element-wise boolean masking rather than `torch.nanmean`.
**A target NaN cannot silently contaminate training** through the default
loss classes — every NH-provided loss class is masked by construction (the
"Masked" prefix is not decorative; there is no unmasked loss variant in
this file). Evidence:
`tmp/nh13_targeted_inspection_moriah_20260712T120839Z/loss_py_nan_grep.txt`.

**Q6 — Are dynamic-input NaNs allowed, and how dangerous are they?** Fully
answered (2026-07-12, part 2 follow-up — default behavior now confirmed).
Dynamic-input NaNs are explicitly supported by a dedicated mechanism:
`nan_handling_method` (`modelzoo/inputlayer.py`), which has **three**
real, distinct handling modes confirmed in code (one more than the first
pass found) — `'input_replacing'` (`inputlayer.py:79-81` in `__init__`,
`inputlayer.py:229-230` in `forward`, adds an explicit NaN-flag feature
rather than masking), `'masked_mean'` (`inputlayer.py:226-227`, computes
the embedding mean over only the non-NaN group members), and `'attention'`
(`inputlayer.py:261-291`, builds a boolean mask over NaN positions per
group, zeroes NaN inputs before embedding to avoid NaN gradients, and uses
the mask again in the attention key/query so NaN positions are
effectively down-weighted rather than corrupting the embedding).
`GenericDataset`'s own docstring (`genericdataset.py:25`) confirms NaN is
the expected missing-data sentinel it recognizes.

**Default (unset) behavior — confirmed, and it is dangerous.**
`Config.nan_handling_method` (`utils/config.py:608-613`) returns `None`
when the key is absent from the run config (`self._cfg.get("nan_handling_method", None)`,
falsy-check `if not method: return None`). When `self.nan_handling_method`
is `None`, both the `__init__` sizing branch (`inputlayer.py:73-88`) and
the `forward()` dispatch branch (`inputlayer.py:210-221`) fall through
every `if`/`elif` and hit the **final unconditional `else`**, which
performs **no NaN handling whatsoever**:
```python
else:
    x_d = torch.cat([data[self._x_d_key][k] for k in itertools.chain(*features)], dim=-1).transpose(0, 1)
    dynamics_out = self.dynamics_embeddings[0](x_d)
```
Raw dynamic-input tensors (including any NaN values) are concatenated and
passed directly into a plain embedding layer (`self.dynamics_embeddings[0]`,
an `nn.Linear`-based module with no NaN-awareness) with no masking, no
zeroing, and no flagging. Standard tensor/autograd NaN propagation applies
from there: a `nn.Linear` fed a NaN input produces NaN output for that
element, which propagates through every downstream layer and, at the loss
step, would either (a) be masked out if it lands on a target-adjacent
position covered by one of the `Masked*Loss` classes (Q5) — but the
`Masked*Loss` classes only mask the **target** (`ground_truth['y']`), not
intermediate hidden activations — or (b) corrupt the gradient of the
*entire batch* via backpropagation, since NaN gradients are not
implicitly localized to the offending sample in a shared-batch matrix
multiply. **This confirms unset `nan_handling_method` is not a safe silent
default for NaN-valued dynamic inputs**; explicit configuration
(`masked_mean`, `attention`, or `input_replacing`) is required before
Flash-NH's scientific (non-technical-fill) path can safely leave MRMS/RTMA
gap hours as literal NaN in the dynamic inputs. This is directly relevant
to §6 (gap policy): it is a real, available alternative to hard
sample/window exclusion, but only if explicitly configured — never as an
implicit fallback. Evidence:
`tmp/nh13_targeted_inspection_moriah_20260712T120839Z/inputlayer_nan_handling.txt`.

**Q7 — What does `nan_handling_method` actually do?** Fully answered — see
Q6. Confirmed a model-input-embedding-layer mechanism operating on
**dynamic input** NaNs only, with three evidenced modes (`masked_mean`,
`attention`, `input_replacing`), no safe implicit default (unset ==
`None` == raw NaN passed into an unprotected `nn.Linear` embedding); it is
not a sample-exclusion mechanism and not the target-NaN masking mechanism
(that's `_mask_valid` for evaluation, and the confirmed `Masked*Loss`
family in `training/loss.py` for training — see Q5).

**Q9 — Native lead-time config, or package-level target shifting?**
Answered: **no native fit for Flash-NH's purely-historical inputs; recommend
package-build-time target shifting**, confirming the framing already
recorded in Milestone 2K-G-D memory (seq_length is not the lead-time
driver) directly from source rather than inference. Evidence: the literal
string `lead_time` has **zero** hits anywhere in NH 1.13.0. `horizon` has
only 2 hits, both descriptive prose inside forecast-model docstrings
(`modelzoo/multihead_forecast_lstm.py:15`, `modelzoo/stacked_forecast_lstm.py:34`
— "shifted in time by the forecast horizon (7 days...)"), not a config
key. NH does have a native **hindcast/forecast** architecture
(`forecast_seq_length`, `forecast_overlap` config properties;
`hindcast_counter`/`forecast_counter` bookkeeping at `basedataset.py:89-94`;
dedicated model classes `HandoffForecastLSTM`, `MultiHeadForecastLSTM`,
`SequentialForecastLSTM`, `StackedForecastLSTM`), but this architecture
requires **forecast-known future dynamic inputs** (a `forecast_inputs`
config list — e.g. NWP forecast fields available at prediction time for
the forecast window). Flash-NH's `v001-core` inputs are purely historical
MRMS/RTMA reanalysis with no forecast-input equivalent, so this native
mechanism does not fit without inventing synthetic forecast inputs.
Separately, `predict_last_n` (confirmed at `basedataset.py:908`,
`tester.py:233-296,425-459,529-530,577-578`) selects the **trailing N
already-time-aligned** timesteps of the target series for loss/eval — it
is not an offset-into-future mechanism by itself. Conclusion: implementing
Flash-NH's 6 h/12 h lead time most likely requires shifting the target
(`qobs`/specific discharge) series by `lead_time` hours relative to the
aligned forcing window **at package-build time**, so that "timestep t" in
the NC file already encodes "target at t + lead_time", then using plain
`seq_length`/`predict_last_n=1` semantics on top — not a native NH config
flag.

**Q8 — Native exclusion of gap-intersecting windows/samples?** Answered:
**no native mechanism found; this is a design fork, not just an evidence
gap.** `_validate_samples` (`basedataset.py:861-908`), the only
sample-inclusion filter found in the codebase, filters purely on
insufficient-history (see Q5) — no gap-mask, blacklist, or arbitrary
excluded-timestamp parameter was found in any of the 15 keywords searched
(note: "exclude"/"blacklist"/"valid_mask" were not in the searched keyword
set — a genuine gap; see follow-up below). `GenericDataset` adds no
override that would provide this either (Q1). Two feasible paths for
Flash-NH, not mutually exclusive: (a) rely on `nan_handling_method`
(Q6/Q7) for gap hours *within* a window — this masks/down-weights the gap
hour's contribution but does not drop the window; (b) hard-exclude
gap-intersecting windows via a custom `Sampler`/`Dataset` subclass filtering
the lookup table by the gap mask, which is exactly the loss fraction
`scripts/analyze_stage1_window_feasibility.py` estimates (Q10/Q11, now
answered — see "Window/sample feasibility" below). Which of (a)/(b) — or a
combination — is used is a Flash-NH design decision that should be
informed by the actual window-loss numbers, not decided by Phase B
evidence alone.
**Optional follow-up, not run this round** (only needed if (b) is
pursued): a check for whether NH exposes a pluggable sampler class Flash-NH
could subclass instead of writing dataset-index filtering from scratch —
`grep -n 'class.*Sampler\|BatchSampler'` over the `neuralhydrology`
package. Per the Moriah usage discipline in "Inspection environment
policy" above, this must **not** be run as an unbounded recursive `grep -r`
over the whole package tree from the login node (that is the kind of
heavy filesystem scan the discipline note rules out); if pursued, scope it
to specific known files (e.g. `basedataset.py`, `data/__init__.py`, any
`sampler*.py`) or run it from within a short CPU allocation instead of the
login node.

### Window/sample feasibility (from `window_feasibility.{csv,md,json}`)

**Q10/Q11 — answered (2026-07-12 follow-up), using the real
`fullperiod_gap_inventory.csv`.** This script requires no NeuralHydrology
import (pure pandas/numpy geometry + gap-mask arithmetic), so it was run
locally against the real gap-inventory evidence already pulled to
`tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/fullperiod_gap_inventory.csv`
from an earlier milestone (Milestone 2K-G-E full-period forcing postrun
audit, 2026-06-24) rather than re-running on Moriah — no new Moriah
round-trip was needed for this sub-question, consistent with the user's
instruction that only NeuralHydrology-importing code must run on Moriah.

**A real, non-trivial bug was found and fixed in the script before these
numbers could be trusted — disclosed here for the record.** The real gap
CSV's timestamps are ISO-8601 with a `Z` suffix (e.g.
`2020-10-14T00:00:00Z`), which `pandas.to_datetime` parses as tz-aware
UTC; the hourly period index the script builds internally
(`_build_hourly_index`) is tz-naive. Comparing a tz-aware gap-hour set
against the tz-naive index via `.isin()` silently matches **zero** rows —
no error, no warning. The first run against the real CSV
(`tmp/stage1_window_feasibility_real_gaps_20260712T103728Z/`, superseded,
not used below) consequently reported `mrms_gap_loss_fraction_mean=0.0`
and `rtma_gap_loss_fraction_mean=0.0` for every one of the 16
`seq_length` x `lead_time` rows — identical to the geometry-only
(no-gap-exclusion) numbers. That is implausible on its face given the
known 136 MRMS + 2 RTMA archive-gap hours are scattered across the full
2020-2025 period, not confined to the period boundary. This was root
caused via a scratch debug script that isolated the internal gap-mask
boolean arrays and found them all-`False` despite the loader correctly
reporting `expanded gap hours by product={'mrms': 136, 'rtma': 2}` in its
own metadata — i.e. the gap hours were counted correctly but never
matched against the hourly index. The two prior Phase A synthetic smoke
CSVs never had a `Z` suffix, so this bug was invisible to those regression
tests.

Fixed in `scripts/analyze_stage1_window_feasibility.py` by adding a
`_to_naive_utc()` helper (`pd.to_datetime(series, utc=True).dt.tz_localize(None)`)
and applying it at all three timestamp-parsing call sites
(`_expand_gap_runs_to_timestamps`, the `timestamp_rows`-schema branch of
`_load_gap_masks`, and `_load_target_missing_masks`). Verified via
`python -m py_compile` (clean) and full regression re-runs of both
pre-existing synthetic fixtures — output byte-identical to before the fix
(`tmp/smoke_window_feasibility_inputs/`: `mrms: 136, rtma: 2`;
`tmp/smoke_gap_run_schema/fullperiod_gap_inventory_synthetic.csv`:
`mrms: 5, rtma: 1, unspecified: 1`) — no regression. The real-gap run was
then repeated with the fix applied, producing the authoritative numbers
below (console-confirmed correct gap detection:
`expanded gap hours by product={'mrms': 136, 'rtma': 2, 'unspecified': 0}`,
matching the known archive-gap counts recorded in "Inherited binding
decisions" above).

**Corrected results** (`tmp/stage1_window_feasibility_real_gaps_20260712T104021Z/window_feasibility_summary.md`,
period `2020-10-14`–`2025-12-31`, 45,720 hourly steps, no
`--target-availability-csv` supplied so target-availability columns are
`not_computed`):

| seq_length (h) | lead_time (h) | geometry loss frac | MRMS-gap loss frac | RTMA-gap loss frac | either-gap loss frac |
|---:|---:|---:|---:|---:|---:|
| 12 | 1  | 0.026% | 1.306% | 0.031% | 1.337% |
| 12 | 3  | 0.031% | 1.346% | 0.033% | 1.378% |
| 12 | 6  | 0.037% | 1.370% | 0.033% | 1.403% |
| 12 | 12 | 0.050% | 1.396% | 0.033% | 1.429% |
| 24 | 1  | 0.053% | 2.147% | 0.057% | 2.204% |
| 24 | 3  | 0.057% | 2.186% | 0.059% | 2.245% |
| 24 | 6  | 0.063% | 2.211% | 0.059% | 2.270% |
| 24 | 12 | 0.077% | 2.235% | 0.059% | 2.294% |
| 48 | 1  | 0.105% | 3.779% | 0.110% | 3.889% |
| 48 | 3  | 0.109% | 3.819% | 0.112% | 3.930% |
| 48 | 6  | 0.116% | 3.843% | 0.112% | 3.955% |
| 48 | 12 | 0.129% | 3.863% | 0.112% | 3.975% |
| 72 | 1  | 0.158% | 5.358% | 0.162% | 5.521% |
| 72 | 3  | 0.162% | 5.398% | 0.164% | 5.562% |
| 72 | 6  | 0.168% | 5.423% | 0.164% | 5.587% |
| 72 | 12 | 0.182% | 5.436% | 0.164% | 5.601% |

Full machine-readable data (not committed, pulled evidence only):
`tmp/stage1_window_feasibility_real_gaps_20260712T104021Z/{window_feasibility.csv,window_feasibility_summary.json,window_feasibility_summary.md}`.
The prior all-zero run directory
(`tmp/stage1_window_feasibility_real_gaps_20260712T103728Z/`) is stale/
superseded and should not be cited — it remains on local disk under
gitignored `tmp/` only and is not evidence for anything past this note.

**Interpretation (facts only, no policy decision):** MRMS-gap loss
dominates RTMA-gap loss by roughly two orders of magnitude at every
`seq_length`/`lead_time` combination (e.g. at `seq_length=72, lead_time=12`:
5.44% MRMS vs. 0.16% RTMA), consistent with the underlying gap-hour counts
(136 MRMS vs. 2 RTMA). Loss grows monotonically and non-trivially with
`seq_length` — from ~1.3% at `seq_length=12` to ~5.4-5.6% at
`seq_length=72` (either-gap) — and grows only slightly with `lead_time`
within a fixed `seq_length` (e.g. at `seq_length=72`: 5.52% at
`lead_time=1` vs. 5.60% at `lead_time=12`). These are the numbers Q11 asks
about: none is so large (all remain under 6% even at the largest
`seq_length=72`/`lead_time=12` combination) that it would rule out any of
the four `seq_length` candidates outright on window-availability grounds
alone, but the loss is clearly non-negligible and `seq_length`-dependent
enough that it should inform (not silently be ignored by)
`docs/stage1_scientific_baseline_design.md` §9c's sweep design — see the
gap-policy decision framework below for how these numbers bear on the
MRMS-exclusion-vs-NaN-handling choice.

### Gap-policy decision (RTMA interpolation)

`PENDING` as a final decision — §6 of the scientific baseline design doc
requires this to be explicitly recorded. Phase B now supplies two
mechanism-level facts that narrow the decision space (see Q6 and Q8 above):
NH has a real, working per-input NaN-handling mechanism
(`nan_handling_method`) that can absorb dynamic-input gap hours without
hard-excluding the window, but no native mechanism was found for
excluding gap-intersecting **windows** wholesale. The RTMA-interpolation
vs. leave-as-NaN-and-use-`nan_handling_method` choice is now a real,
evidenced tradeoff rather than an assumption — not decided here.

**Gap-policy decision framework (2026-07-12 follow-up) — comparison only,
not a decision.** Two candidate policies for MRMS/RTMA gap hours, informed
by the Q6/Q8 mechanism facts and the corrected Q10/Q11 loss numbers above:

- **Policy A — leave gap hours as true NaN + explicit `nan_handling_method`.**
  Uses `GenericDataset`'s native NaN recognition (Q6) plus an explicit,
  deliberately-configured `nan_handling_method` (`masked_mean`,
  `attention`, or `input_replacing`) — **mandatory**, not optional: Q6's
  2026-07-12 confirmation shows the unset/default behavior is confirmed
  dangerous (raw NaN passed directly into an unprotected `nn.Linear`
  embedding, with no masking), so this policy is only safe if
  `nan_handling_method` is set explicitly in every Flash-NH scientific
  training config; there is no safe way to "just leave it as NaN" without
  also setting this config key.
  - Pros: uses existing NH machinery end-to-end, no custom
    `Dataset`/`Sampler` code; does not drop any window from the training
    index, so the full 45,720-hour period contributes samples at every
    `seq_length`/`lead_time`.
  - Cons: a model trains on windows with partial missing forcing history
    rather than clean exclusion — the gap hour's *effect* on the sample is
    down-weighted/masked at the embedding layer, not absent from the
    sample; this is scientifically closer to imputation-by-omission than
    to "this sample never saw a gap"; requires a config-review/lint step
    (or equivalent safeguard) to guarantee `nan_handling_method` is never
    accidentally left unset in a scientific run, since the failure mode
    (silent NaN propagation corrupting the batch gradient) is easy to miss
    without this evidence.

- **Policy B — hard-exclude any window intersecting an MRMS/RTMA gap hour**,
  via package-builder-time sample-mask/lookup-table filtering or a custom
  `Dataset`/`Sampler` (Q8: no native NH config flag does this).
  - Pros: scientifically cleaner for MRMS gaps specifically — no training
    sample has any missing-forcing evidence in its window at all; easier
    to state and defend in a benchmark-paper methods section ("windows
    intersecting a known archive gap were excluded") than to characterize
    NH's internal NaN-masking behavior to a reviewer.
  - Cons: not native in NH per the Q8 evidence — requires custom
    lookup-table filtering or a custom sampler at the package-builder or
    dataset level; sample loss is not free and **grows with `seq_length`
    and, more weakly, with `lead_time`** — the corrected numbers above show
    either-gap loss ranging from ~1.3% (`seq_length=12`) to ~5.6%
    (`seq_length=72`), i.e. this policy's cost is `seq_length`-dependent
    engineering + `seq_length`-dependent data loss, not a fixed cost.

**Reading the corrected numbers into this framework (still not a
decision):** the corrected Q10/Q11 loss fractions are modest in absolute
terms (under 6% even at the most expensive `seq_length=72,
lead_time=12` combination) but are clearly non-trivial and
`seq_length`-dependent rather than negligible-and-flat. This is exactly
the regime the user asked to flag explicitly: loss is small enough that
hard MRMS-window exclusion (Policy B) is not obviously infeasible on data-
availability grounds at any of the four candidate `seq_length` values, but
large enough — and `seq_length`-dependent enough — that it is not "free"
either, and the engineering cost (custom sample filtering, since Q8 found
no native mechanism) is real. Both policies remain viable candidates;
this document does not select between them.

**RTMA may warrant a separate policy from MRMS — flagged, not decided.**
The MRMS/RTMA loss asymmetry is stark: MRMS-gap loss is roughly two orders
of magnitude larger than RTMA-gap loss at every `seq_length`/`lead_time`
(e.g. 5.44% vs. 0.16% at `seq_length=72, lead_time=12`), tracking the
underlying 136 MRMS vs. 2 RTMA archive-gap-hour counts. This asymmetry
means a combined "either-gap" exclusion policy is almost entirely driven
by MRMS, and a policy that treats RTMA differently from MRMS (e.g. RTMA
interpolation retained as today's Smoke-0/1 technical-only fill, but
promoted to a scientific-path decision for RTMA specifically, while MRMS
gets its own separate policy — either NaN+`nan_handling_method` or hard
exclusion) is a structurally reasonable option given how small the RTMA
contribution is. This document flags the option without deciding it.

### Target-scaling decision

`PENDING` as a final decision — §5 remains open. Phase B evidence
(Q1/Q2 above) shows area-normalized/specific discharge is not a
`GenericDataset` config flag but is straightforward to implement at
package-build time, with the caveat that raw-m^3/s reversal at evaluation
time would need an extra Flash-NH-side step beyond NH's native scaler
unscale. This narrows the choice among (a) area-normalized/specific
discharge (package-build-time, extra reversal step), (b) raw `qobs_m3s`
with NH default z-score (no extra reversal step, but no basin-area
normalization benefit), or (c) per-basin standardization — but does not
select among them.

## Local validation (Phase A, this patch)

Both scripts were syntax-checked (`python -m py_compile`) and smoke-run
locally (no NeuralHydrology installed locally):

- `inspect_neuralhydrology_stage1_mechanics.py --out-dir tmp/smoke_nh13_inspection`
  — exits 0, correctly detects `neuralhydrology` is not importable, writes
  all 6 output files with a clear "NOT AVAILABLE" note instead of crashing.
- `analyze_stage1_window_feasibility.py --period-start 2020-10-14
  --period-end 2025-12-31 --out-dir tmp/smoke_window_feasibility` — exits 0,
  computes `total_hours=45720` (matches the known full-period step count
  exactly), writes 16 rows (4 seq_lengths x 4 lead_times) to
  `window_feasibility.csv`.
- Same script re-run with synthetic (locally-generated, not real) gap
  inventory and target-availability CSVs to exercise those optional code
  paths — exits 0, produces gap-loss and target-availability columns that
  scale sensibly with window size (larger `seq_length` → larger gap-overlap
  loss fraction, as expected).
- **Phase A hardening patch (2026-07-08):** the gap-inventory loader now
  auto-detects both the original "timestamp-rows" schema and the real
  Flash-NH "gap-run-intervals" schema (`gap_start_utc`/`gap_end_utc`/
  `product`, e.g. `fullperiod_gap_inventory.csv`), expanding each gap run
  inclusively into hourly timestamps and mapping product names robustly
  (substring match on "mrms"/"rtma"). Re-run against a small synthetic
  gap-run-schema CSV (`tmp/smoke_gap_run_schema/`) with 3 gap runs (2 MRMS
  spanning 3 and 2 hours, 1 RTMA spanning 1 hour) plus one unrecognized
  product label — output correctly reported
  `expanded gap hours by product={'mrms': 5, 'rtma': 1, 'unspecified': 1}`.
  Also re-ran the original timestamp-rows synthetic CSVs
  (`tmp/smoke_window_feasibility_inputs/`) as a regression check — output
  unchanged: `mrms: 136, rtma: 2` (matching the known real archive-gap
  counts). `window_feasibility_summary.json` now also includes a
  `gap_inventory_meta` block recording the detected schema, basin mode,
  and expanded gap-hour counts by product.

None of this constitutes Milestone 2K-G-G evidence — it only confirms the
Phase A tooling itself runs correctly. All outputs from these local smoke
runs remain under `tmp/` (gitignored, not committed).

## Evidence follow-up (2026-07-12, part 1) — Moriah access blocker for tasks 1-3

A first follow-up evidence round attempted to close the three remaining
`REQUIRES TARGETED SOURCE INSPECTION` items (exact `tester.py`
inverse-scaling arithmetic, `training/loss.py` target-NaN masking, default
`nan_handling_method` behavior) using the exact `sed`/`grep` commands
already recorded next to each item above. **All three were unresolved in
this part-1 round**: this working session had no SSH/network path to
Moriah at the time
(`ssh -o ConnectTimeout=5 -o BatchMode=yes moriah "echo CONNECTIVITY_OK"`
failed with "Could not resolve hostname moriah"). This blocker was later
fixed (see "part 2" below) and all three items are now closed.

Q10/Q11 (window/sample-loss numbers) **did** get closed in part 1 without
needing Moriah, because `scripts/analyze_stage1_window_feasibility.py`
requires no NeuralHydrology import and a real gap-inventory CSV was
already available locally from an earlier milestone's evidence pull — see
"Window/sample feasibility" above, including the timezone bug found and
fixed in that script.

## Evidence follow-up (2026-07-12, part 2) — Moriah access restored, all 3 items closed

Moriah SSH access was fixed on the user's local Windows/VS Code setup
(confirmed via `ssh moriah "hostname"` -> `moriah-gw-01`; the default
modern `scp`/SFTP subsystem is still unavailable on Moriah, but plain SSH
command execution works, which is all these three tasks needed — no file
transfer was required, only inline `sed`/`grep` output over an SSH
session). The three exact commands supplied by the user (see tasks 1-3 in
`docs/decision_log.md`'s 2026-07-12 part-2 entry for the literal command
text) were run directly against
`/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/lib/python3.11/site-packages/neuralhydrology`
on the Moriah login node (lightweight source-file reads only — no compute
allocation used, consistent with the user's instruction that plain
`sed`/`grep` over installed source needs no allocation). Raw command
output saved to
`tmp/nh13_targeted_inspection_moriah_20260712T120839Z/{tester_240_270.txt,loss_py_nan_grep.txt,inputlayer_nan_handling.txt}`
(gitignored, not committed) per `docs/repo_policy.md`. Findings written up
in place at Q2, Q5, Q6/Q7 above — all three previously-`REQUIRES TARGETED
SOURCE INSPECTION` items are now fully answered with cited file:line
evidence; none remain open in this document.

## Not done in this update (Phase B, 2026-07-12, parts 1+2)

- No final target-scaling, gap-policy, or lead-time implementation decision
  was made — Phase B supplies mechanism-level facts (what NH 1.13.0 can and
  cannot do natively); §5/§6/§9b decisions in
  `docs/stage1_scientific_baseline_design.md` remain open. The gap-policy
  decision framework compares Policy A vs. Policy B and reads the real
  loss numbers into that comparison, but does not select between them, and
  does not decide the RTMA-vs-MRMS split question either.
- No package builder, scientific NH config, or Slurm template was modified.
- No training was run; no NH package was generated; nothing under `tmp/`
  was committed (including the two `window_feasibility_real_gaps_*` run
  directories and the `nh13_targeted_inspection_moriah_*` evidence
  directory produced across both parts of this follow-up).