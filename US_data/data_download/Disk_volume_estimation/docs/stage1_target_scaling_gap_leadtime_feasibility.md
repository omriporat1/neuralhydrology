# Stage 1 — Target Scaling + Gap Policy + Lead-Time Feasibility Report

Milestone: **2K-G-G**, opened 2026-07-08 (Phase A).

## Status

**Phase A scaffold created; Moriah evidence pending.**

This document is a scaffold, not a findings report. No NeuralHydrology 1.13
code has been inspected yet in this session — neither locally (no NH 1.13
install available in this session's environment) nor on Moriah (no network
path from this environment to Moriah). Sections below distinguish clearly
between:
- what is **inherited / already decided** (binding, from
  `docs/stage1_scientific_baseline_design.md`),
- what is **scaffolding created in Phase A** (scripts, commands, this doc),
- what is **pending** — every conclusion, finding, or number that requires
  actually running the Phase A tooling (locally and/or on Moriah — see
  "Inspection environment policy" below) and inspecting its output.

Do not treat any placeholder in the "Findings" sections below as an answer.
They are explicitly marked `PENDING` until the tooling is run and the
evidence is inspected. A local-only run may fill in a **preliminary**
finding, but it is not a substitute for the Moriah commands in this
document — see policy below.

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

## Findings — PENDING

### NH 1.13 mechanics (from `nh13_inspection_summary.{md,json}`)

`PENDING` — not yet run, locally or on Moriah, in this session. Per the
"Inspection environment policy" above: a local run may be recorded here as
**preliminary**, but a matching `flashnh-moriah` run on Moriah is required
before this finding is treated as final.

### Window/sample feasibility (from `window_feasibility.{csv,md,json}`)

`PENDING` — geometry-only local smoke test confirms the tool runs and
produces plausible numbers (see "Local validation" below), but that local
run is not Moriah evidence and used only the default period/seq_length/
lead_time grid with no real gap inventory. The authoritative run must
happen on Moriah (or at least from a location with access to a real gap
inventory CSV) and be reported here.

### Gap-policy decision (RTMA interpolation)

`PENDING` — §6 of the scientific baseline design doc requires this decision
to be explicitly recorded once Phase B evidence exists. Not decided here.

### Target-scaling decision

`PENDING` — §5 remains open. Phase A does not select among (a)
area-normalized/specific discharge, (b) raw `qobs_m3s` with NH default
z-score, or (c) per-basin standardization.

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

## Not done in this patch

- No NeuralHydrology 1.13 code was inspected, locally or on Moriah.
- No target-scaling, gap-policy, or lead-time implementation decision was
  made or recorded as final.
- No package builder, scientific NH config, or Slurm template was modified.
- No training was run; no NH package was generated.