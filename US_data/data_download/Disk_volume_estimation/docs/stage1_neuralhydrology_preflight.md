# Flash-NH Stage 1 — NeuralHydrology Package Preflight

**Created:** 2026-06-09 (Milestone 2G — January 2023 pilot)
**Updated:** 2026-06-30 (Milestone 2K-G-A — full-period pilot design + corrections)

**2G status:** COMPLETE (2026-06-09) — January 2023 pilot package built and audited.
**2K-G-A status:** DESIGN COMPLETE — full-period pilot design frozen with corrections (2026-06-30).
Next: 2K-G-B — implement package builder using 5-basin corrected pilot as input.

---

## Part I — Milestone 2K-G-A: Full-Period Pilot Design

### 1. Context and architecture recap

Three machines, strict role separation:

| Machine | Role |
|---|---|
| **Local PC** | Code, commits, planning, docs — no large data |
| **h2o** (`h2o.es.huji.ac.il`) | Source data, forcing/target products, package building |
| **Moriah** (`/sci/labs/efratmorin/omripo/Flash-NH`) | NeuralHydrology training only (GPU + Slurm) |

The bridge from the curated forcing library to a trainable NH package requires:

1. A **package builder** that merges the forcing Parquets + target NCs into per-basin
   NeuralHydrology-compatible NetCDF files and writes static attributes / basin lists.
2. A compact **pilot package** (5 basins, full period) transferred to Moriah.
3. NH environment + config YAML + Slurm job script on Moriah for Smoke 0.

---

### 2. Inputs already available

| Product | Location (h2o) | Status |
|---|---|---|
| Curated forcing library v001 | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001/` | Corrected rebuild running (schema fix 2K-F-C-B) |
| Target package v001 (qobs) | `/data42/omrip/Flash-NH/tmp/stage1_target_package_v001/` | PASS — 2,752 basins × 45,720 h |
| Static attributes (full, 237 cols) | Local: `C:/PhD/Python/.../attributes/` | Available; 9,008-basin HydroATLAS table |
| Basin list v001 | target package manifest | 2,752 STAIDs |

**Forcing schema after 2K-F-C-B correction (13 columns):**

| # | Column | Type | Units |
|---|---|---|---|
| 1 | `mrms_qpe_1h_mm` | float32 | mm |
| 2 | `rtma_2t_K` | float32 | K |
| 3 | `rtma_2d_K` | float32 | K |
| 4 | `rtma_2sh_kgkg` | float32 | kg/kg |
| 5 | `rtma_sp_Pa` | float32 | Pa |
| 6 | `rtma_10u_ms` | float32 | m/s |
| 7 | `rtma_10v_ms` | float32 | m/s |
| 8 | `rtma_tcc_pct` | float32 | % |
| 9 | `rtma_vis_m` | float32 | m |
| 10 | `rtma_gust_ms` | float32 | m/s |
| 11 | `rtma_ceil_m` | float32 | m |
| 12 | `mrms_qpe_1h_mm_gap` | bool | — |
| 13 | `rtma_gap` | bool | — |

---

### 3. NH-compatible package format

**Format decision: NeuralHydrology `GenericDataset` — single NC per basin.**

One NetCDF file per basin contains all dynamic variables (forcings) plus the target
(`qobs_m3s`) on a shared `date` coordinate. This matches the format proven in Milestone
2G and is directly compatible with NH's built-in `GenericDataset` without a custom
dataset class.

```
stage1_pilot_v001/
  time_series/
    01019000.nc          # one per basin
    01022500.nc
    ...
  attributes.csv         # 5 rows × (gauge_id + retained attribute cols)
  basins/
    smoke0_train.txt     # one STAID per line
    smoke0_val.txt
    smoke0_test.txt
  configs/
    stage1_smoke0.yml    # NH YAML config
  slurm/
    smoke0.sh            # Moriah Slurm job script
```

**Per-basin NC structure:**

| Field | Value |
|---|---|
| Coordinate | `date` (hourly UTC, `datetime64[ns]`, no tz offset) |
| Dynamic vars | Forcing columns (depends on smoke level — see §5/§6) |
| Gap flags | `mrms_qpe_1h_mm_gap`, `rtma_gap` (retained as auxiliary dynamic vars) |
| Target | `qobs_m3s` (float32, units `m3 s-1`, `_FillValue=-9999.0`) |
| Encoding | `float32`; `_FillValue=-9999.0`; NaN preserved for target |

**Why keep gap flags as dynamic vars?**
They give NH an explicit signal that the forcing value at that timestep is filled
(not observed). This is information the model can learn to act on. Include them as
standard dynamic inputs; do not hide them.

---

### 4. Package builder (to be implemented: Milestone 2K-G-B)

A new script `scripts/build_stage1_nh_package.py` is needed. It is **not yet
implemented**. Scope:

1. Read per-basin forcing Parquets from the curated library
   (`time_series/{STAID}.parquet`, wide format, `valid_time_utc` as index)
2. Read per-basin target NCs from the target package
   (`time_series/{STAID}.nc`, `date` coordinate, `qobs_m3s` variable)
3. Apply gap-fill policy to forcing NaN values (see §7 — gap handling)
4. Merge onto a shared `date` index (45,720 hourly UTC steps)
5. Write per-basin NCs in GenericDataset format
6. Write `attributes.csv` from the static attribute table for the selected basins
7. Write basin list text files for the specified splits
8. Write a manifest/provenance JSON

This builder is distinct from the January 2023 pilot builder
(`build_stage1_neuralhydrology_january_pilot.py`), which:
- read from monthly chunk Parquets (not the curated library),
- covered January 2023 only,
- used a different schema (included `rtma_weasd_kgm2`, `rtma_10si_ms`, etc.).

The new builder must use the corrected v001 schema.

---

### 5. Pilot basin and period selection

**Pilot basins (5):** First 5 STAIDs from the corrected forcing pilot:
`01019000`, `01022500`, `01033000`, `01038000`, `01049500`

These are the same basins used in the corrected 5-basin full-period forcing pilot.
They span the northeastern US and cover a range of basin sizes.

**Full period:** 2020-10-14T00Z – 2025-12-31T23Z (45,720 hourly steps per basin)

**Suggested train/val/test splits:**

| Split | Period | Hours |
|---|---|---|
| Train | 2020-10-14 – 2022-12-31 | ~19,128 h (~26 months) |
| Validation | 2023-01-01 – 2023-12-31 | 8,760 h |
| Test | 2024-01-01 – 2025-12-31 | 17,544 h |

Rationale:
- Train: covers the 2020–2022 calibration period; avoids test contamination.
- Val: 2023 is useful for early generalization monitoring; contains varied seasonal patterns.
- Test: 2024–2025 is the quasi-operational period; held out entirely until final evaluation.
- NH warm-up is handled internally by discarding the first `seq_length` steps; no manual
  warm-up period needed in the split definition.

---

### 6. Smoke 0 — Rain-only technical smoke

**Purpose:** verify that NH loads the package, runs a forward pass, computes loss,
and completes training epochs without crashing. Not a scientific baseline.

**Basin count:** 5
**Period:** full (2020-10-14 – 2025-12-31)
**NH model:** `cudalstm` (default; uses GPU on Moriah)
**Hidden size:** 64 (smallest practical; reduce memory footprint)
**Sequence length:** 24 h (1 day — minimal for plumbing verification)
**Predict last n:** 1 (single-step prediction per window)
**Batch size:** 256
**Epochs:** 1–2 (enough to confirm finite loss; not a scientific run)

**Dynamic inputs (1 variable):**
```yaml
dynamic_inputs:
  - mrms_qpe_1h_mm
  - mrms_qpe_1h_mm_gap   # auxiliary gap flag
```

**Target:**
```yaml
target_variables:
  - qobs_m3s
```

**Static attributes (minimal, 4):**
```yaml
static_attributes:
  - DRAIN_SQKM
  - LAT_GAGE
  - LNG_GAGE
  - BFI_AVE
```

**Pass criteria for Smoke 0:**
- NH loads without import error on Moriah
- All 5 basin NCs load without error
- Training loss is finite (not NaN) after epoch 1
- NH produces run outputs (epoch loss logs, model checkpoint)
- Slurm job completes with exit code 0

---

### 7. Smoke 1 — Minimal meteorology smoke

**Purpose:** verify that the six core meteorological forcing variables load, normalize,
and train correctly alongside precipitation.

**Dynamic inputs (6 + 2 gap flags):**
```yaml
dynamic_inputs:
  - mrms_qpe_1h_mm
  - rtma_2t_K
  - rtma_2d_K
  - rtma_2sh_kgkg
  - rtma_10u_ms
  - rtma_10v_ms
  - mrms_qpe_1h_mm_gap
  - rtma_gap
```

**`rtma_sp_Pa` in Smoke 1:** **Hold for Smoke 2.**
Surface pressure (Pa, ~70,000–101,325 range) introduces a large-magnitude feature
that may require careful normalization. The 6 variables above are sufficient to
establish that RTMA meteorology loads and trains. Include `rtma_sp_Pa` in the
per-basin NC file (so it is available), but exclude it from `dynamic_inputs` in the
Smoke 1 NH config. Add it in Smoke 2 with explicit normalization review.

**Sequence length for Smoke 1:** step up from Smoke 0. Use `seq_length: 72` (3 days)
or `seq_length: 168` (7 days) as the next technical step. `seq_length: 336` (14 days)
is a later candidate for hyperparameter testing once Smoke 1 passes; do not use it
as the first meteorology smoke default.

**Pass criteria for Smoke 1:** same as Smoke 0, plus:
- All 6 RTMA variables have sensible normalization statistics (mean/std logged by NH)
- `rtma_2d_K` non-null counts match expected 45,718/45,720 per basin (confirms dewpoint
  mapping fix carried through to the NH package correctly)
- No NaN loss after epoch 1

---

### 8. NaN / gap handling policy

#### 8.1 Target NaNs (`qobs_m3s`)

NH **handles missing target values by loss-masking**. Timesteps where `qobs_m3s` is NaN
are ignored (masked out) during loss computation in training and validation — they do not
contribute to the gradient. No pre-filling of target NaN is required or recommended.
Preserve NaN in the per-basin NC exactly as it appears in the target package v001.

Expected target NaN rates from the v001 build:
- 3,880,742 NaN hours total across 2,752 basins
- Per basin: ~1,410 NaN hours average out of 45,720 (3.1%)
- After cleaning: 235 negative values were set to NaN (16 basins)

#### 8.2 Dynamic input NaNs (forcing gaps)

Raw dynamic-input NaNs are risky unless handled deliberately through package
preprocessing or NH `nan_handling_method`. For Smoke 0/1, we prefer preprocessed
forcing inputs with explicit gap flags rather than relying on raw NaNs reaching the
LSTM. This makes the gap-handling policy auditable and transparent.

Known gap rates in the corrected v001 forcing library:
- MRMS gaps: 136 hours / basin (0.30% of 45,720)
- RTMA gaps: 2 hours / basin (0.004% of 45,720)

**Gap-fill policy for the NH pilot package (Smoke 0/1 technical policy — not final
scientific training policy):**

| Source | Gap hours | Fill strategy | Rationale |
|---|---|---|---|
| MRMS QPE | 136 | Fill with **0.0 mm** for Smoke 0/1 | S3 archive absence; plumbing smoke only. Gap flag retained. **See below for final training.** |
| RTMA all-vars | 2 | Fill with **linear interpolation** between adjacent hours | 2 hours; both neighbors always available. Gap flag retained. Review before final training. |

**MRMS gap policy note — not final scientific training policy.**
Precipitation is the primary forcing driver. Filling MRMS archive gaps with 0.0 mm
is a safe, auditable technical choice for Smoke 0/1, but it must not be carried
unchanged into scientific baseline training. Before baseline training, evaluate:
- **Window/sample exclusion:** exclude any training window that contains an MRMS gap
  hour in its input sequence or prediction horizon. Do not remove rows from the per-basin
  NC file (the 45,720-hour `date` coordinate must stay aligned between forcing and
  `qobs_m3s`); instead, exclude those windows during NH batch sampling.
- **Or:** use a deliberately tested NH `nan_handling_method` (e.g., `masked_mean`)
  so the gap-handling behavior is explicit and reproducible.

The gap flags (`mrms_qpe_1h_mm_gap`, `rtma_gap`) are retained as explicit dynamic
inputs so the model can potentially learn to act on data-quality information.

**Do not use NH `nan_handling_method`** as the primary gap strategy for Smoke 0.
Pre-fill in the package builder is more transparent and auditable. If a pre-fill bug
is discovered after the package is on Moriah, `nan_handling_method: mean` is a valid
fallback; do not combine pre-fill + `nan_handling_method`.

#### 8.3 Normalization of gap flags

Gap flag columns (`bool` → `float32` 0.0/1.0 in the NC) will be normalized by NH.
At 0.30% and 0.004% True rates, the mean is nearly 0 and std is very small. NH will
z-score normalize by default. This is acceptable; the near-constant signal still
provides the gap location. If normalization produces instability, consider including
gap flags but setting them to bypass normalization via a custom scaler config.

---

### 9. Data locations and machine roles

**h2o — source and build:**
```
/data42/omrip/Flash-NH/tmp/
  stage1_forcing_fullperiod/
    stage1_basin_hourly_forcings_v001/
      time_series/{STAID}.parquet    ← forcing source
  stage1_target_package_v001/
    time_series/{STAID}.nc           ← target source
```

The package builder script (`scripts/build_stage1_nh_package.py`) runs on h2o because
it needs direct access to both ~6.9 GB forcing library and ~400 MB target package.
It writes a compact pilot package (5 basins × ~5 MB each → ~25 MB) that is then
transferred to Moriah.

**Moriah — training:**

```
/sci/labs/efratmorin/omripo/Flash-NH/
  repos/
    neuralhydrology/        ← clean upstream NH clone
    flashnh/                ← this repo clone (configs, scripts, docs)
  envs/
    flashnh-moriah/         ← NH training conda env (PyTorch + CUDA)
  data/
    stage1_pilot_v001/      ← pilot NH package (5 basins, from h2o)
      time_series/          ← per-basin NCs
      attributes.csv
      basins/               ← split txt files
      configs/              ← NH YAML configs
  runs/
    stage1_smoke0/          ← NH run output (epoch logs, checkpoints)
    stage1_smoke1/
  logs/
    slurm-*.out             ← Slurm job output logs
  slurm/
    smoke0.sh               ← Slurm job scripts
    smoke1.sh
  evidence/
    smoke0_evidence.tar.gz  ← compact evidence bundles for local documentation
```

**Transfer path (h2o → Moriah):**
```bash
# On Moriah after ssh:
scp -r omripo@h2o.es.huji.ac.il:/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/ \
    /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/
```
Or via h2o push: `scp -r /data42/.../stage1_nh_pilot_v001/ moriah:/sci/.../`.
The pilot package is small enough (~25 MB) that `scp` is appropriate; no rsync needed.

---

### 10. NeuralHydrology setup on Moriah

#### 10.1 Setup strategy: clean upstream NH first

**Do not use the old Flash-NH fork.** Start with:
```bash
cd /sci/labs/efratmorin/omripo/Flash-NH/repos
git clone https://github.com/neuralhydrology/neuralhydrology.git
```

Reasons to prefer clean upstream:
- The old fork pre-dates the full-period curated forcing architecture
- NH has been updated with more robust `nan_handling_method` options since the fork
- Starting clean avoids inheriting stale patches and merge debt
- All Flash-NH customizations belong in config YAMLs and the `src/flashnh/` layer in
  this repo, not in NH internals (until a specific limitation forces a fork)

#### 10.2 Fork-readiness without forking

Maintain the ability to fork without doing it prematurely. Keep all custom logic in:
- NH config YAML (`dynamic_inputs`, `static_attributes`, loss, seq_length, etc.)
- Package builder script (`scripts/build_stage1_nh_package.py` in this repo)
- A future `src/flashnh/` custom dataset class if needed (not yet)

Only create a Flash-NH fork when a **specific NH internal limitation** blocks progress.
Decision points that may trigger a fork:

| Scenario | Fork required? | Alternative |
|---|---|---|
| Custom event-weighted loss | Likely yes | NH allows custom loss class via config; try first |
| Gap-aware masking of dynamic inputs | Possibly | Pre-fill in package builder covers Smoke 0/1 |
| Custom normalization (per-event, per-basin) | Possibly | NH `custom_normalization` in config; try first |
| Custom sampler (overweight flood events) | Likely yes | No config-only path; fork or custom class |
| Custom dataset class (e.g., multi-product) | Yes | Add as NH plugin or fork after testing |
| Performance bottleneck in NH DataLoader | Unlikely short-term | Profile first |

Fork protocol (when needed):
1. Create `github.com/omripo/neuralhydrology-flashnh` from clean upstream HEAD
2. Keep upstream `main` as a tracking remote
3. All Flash-NH patches go on a `flashnh-stage1` branch
4. Merge upstream `main` periodically to stay current

#### 10.3 Moriah environment

**Name:** `flashnh-moriah`
**Location:** `/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/`
**Python:** 3.11 (match h2o env for consistency)
**Install approach (Slurm job recommended for heavy installs):**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=flashnh-env-install
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/env-install-%j.out

source /opt/conda/etc/profile.d/conda.sh   # adjust to Moriah conda init
conda create -y -n flashnh-moriah python=3.11 -c conda-forge
conda activate flashnh-moriah
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # confirm CUDA ver
pip install neuralhydrology
pip install pandas pyarrow xarray netcdf4
```

**CUDA version:** confirm with `nvidia-smi` on a Moriah GPU node before installing.
Common options: cu118 (CUDA 11.8) or cu121 (CUDA 12.1). Ask lab wiki or `sinfo --long`.

**Do not run environment installation on the Moriah login node.** Submit as a Slurm job
or use an interactive session: `sinteractive --gres=gpu:1 --mem=16G --time=01:00:00`
(check Moriah/HURCS wiki for the exact `sinteractive` command).

#### 10.4 NH YAML config skeleton (Smoke 0)

```yaml
# configs/stage1_smoke0_nh.yml
experiment_name: flashnh_stage1_smoke0
run_dir: /sci/labs/efratmorin/omripo/Flash-NH/runs

train_basin_file: /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/basins/smoke0_train.txt
validation_basin_file: /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/basins/smoke0_val.txt
test_basin_file: /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/basins/smoke0_test.txt

data_dir: /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001

train_start_date: "2020-10-14"
train_end_date: "2022-12-31"
validation_start_date: "2023-01-01"
validation_end_date: "2023-12-31"
test_start_date: "2024-01-01"
test_end_date: "2025-12-31"

dynamic_inputs:
  - mrms_qpe_1h_mm
  - mrms_qpe_1h_mm_gap

target_variables:
  - qobs_m3s

static_attributes:
  - DRAIN_SQKM
  - LAT_GAGE
  - LNG_GAGE
  - BFI_AVE

model: cudalstm
hidden_size: 64
seq_length: 24           # 1 day — minimal for plumbing verification
predict_last_n: 1        # single-step prediction per window
batch_size: 256
num_epochs: 2            # 1–2 epochs; not a scientific run

log_interval: 10
log_tensorboard: False
save_validation_results: True

# NaN in targets: NH handles natively (loss masked at NaN target timesteps)
# NaN in dynamic inputs: pre-filled in package builder (0.0 for MRMS, interp for RTMA)
# Do NOT set nan_handling_method here for Smoke 0; inputs are pre-cleaned
```

#### 10.5 Slurm job script skeleton (Smoke 0)

```bash
#!/usr/bin/env bash
# slurm/smoke0.sh
#SBATCH --job-name=flashnh-smoke0
#SBATCH --partition=gpu            # confirm partition name on Moriah
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00            # generous for 5-basin 1–2-epoch smoke with seq_length=24
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/smoke0-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/smoke0-%j.err

source /opt/conda/etc/profile.d/conda.sh
conda activate /sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah

NH_REPO=/sci/labs/efratmorin/omripo/Flash-NH/repos/neuralhydrology
CONFIG=/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/configs/stage1_smoke0_nh.yml

python "${NH_REPO}/neuralhydrology/nh_run.py" train --config-file "${CONFIG}"
```

**Check Moriah/HURCS wiki** (`https://wiki.rcs.huji.ac.il/hurcs`) for:
- actual GPU partition names (`sinfo -s`)
- GPU memory per node (A100, V100, etc.)
- per-job time limits
- storage quotas on `/sci/labs/efratmorin/`

---

### 11. Static attributes for the pilot

For Smoke 0/1 (5 basins), static attributes are available via the HydroATLAS +
GAGES-II + derived pipeline from Milestone 2G.

**Minimum required for Smoke 0:** `gauge_id`, `DRAIN_SQKM`, `LAT_GAGE`, `LNG_GAGE`, `BFI_AVE`
- All 5 pilot STAIDs are in the HydroATLAS table (50/50 match in 2G pilot, and the 5
  pilot basins are a subset of those 50)

**For the full 2,752-basin build:**
- Need to verify all 2,752 v001 STAIDs are in HydroATLAS and GAGES-II
- Build `attributes.csv` for all 2,752 basins
- This is Milestone 2K-G-C (static attribute assembly), not 2K-G-A
- The 2G Milestone machinery (HydroATLAS join, GAGES-II merge) is reusable

---

### 12. Blocking gaps before Smoke 0 can run

| Gap | Status | Milestone |
|---|---|---|
| Corrected forcing library v001 (full 2,752 basins) | Running on h2o | Not blocking 2K-G-B pilot |
| 5-basin pilot forcing pilot (5 basins, corrected) | **PASS** on h2o (2026-06-30) | 2K-F-C-B ✓ |
| Package builder `build_stage1_nh_package.py` | Not yet implemented | **2K-G-B — can start now** |
| 5-basin NH pilot package built on h2o | Not yet | 2K-G-B |
| Transfer pilot package to Moriah | Not yet | 2K-G-C |
| Moriah NH environment installed | Not yet | 2K-G-D |
| NH YAML config (Smoke 0) finalized | Draft in §10.4 | 2K-G-D |
| Slurm job script written and tested | Draft in §10.5 | 2K-G-D |
| Moriah GPU partition name confirmed | Unknown — check wiki | 2K-G-D |
| Moriah CUDA version confirmed | Unknown — check `nvidia-smi` | 2K-G-D |

**Full 2,752-basin rebuild is NOT a prerequisite for 2K-G-B.**
The 5-basin NH pilot package can be built from the already-passing corrected 5-basin
forcing pilot. Full-scale NH package generation (all 2,752 basins) must wait for the
corrected full rebuild PASS, but the pilot builder implementation and 5-basin test
can proceed immediately.

---

### 13. Next concrete actions (after this preflight)

In order:

1. **Implement `scripts/build_stage1_nh_package.py`** (Milestone 2K-G-B — unblocked now).
   Input for the 5-basin pilot: the already-passing corrected 5-basin forcing pilot
   (`time_series/01019000.parquet` … `01049500.parquet`) + target package v001 NCs.
   Output: 5-basin NH pilot package with the gap-fill policy from §8.
   **Do not wait for the full 2,752-basin forcing rebuild** to start this step.

2. **Build and audit the 5-basin NH pilot package on h2o.**
   Transfer the compact package (~25 MB) to Moriah.

3. **(Parallel, can overlap)** Monitor corrected full-period rebuild on h2o.
   Check: `tail -f ${PRODUCT}/build.log`. When complete: run auditor with `--full-period`,
   copy evidence bundle locally, then document as the corrected v001 certification.

4. **Confirm Moriah GPU partition name and CUDA version.**
   SSH into Moriah: `sinfo -s`, `nvidia-smi` (on a GPU node via `srun --gres=gpu:1`).
   Update Slurm script partition and PyTorch CUDA version before env install.

5. **Install `flashnh-moriah` conda env on Moriah** (via Slurm job or `sinteractive`).
   Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

6. **Run Smoke 0 on Moriah** (Slurm job, 1–2 epochs, 5 basins, `seq_length: 24`).
   Pass criteria: finite loss after epoch 1, checkpoint written, Slurm exit 0.

7. **Run Smoke 1** (`seq_length: 72` or `168`; add 5 RTMA vars; verify `rtma_2d_K` non-null).

8. **After corrected full rebuild PASS:** extend package builder to all 2,752 basins
   for full-scale NH package generation.

---

## Part II — Milestone 2G: January 2023 Pilot (historical reference)

**Status:** COMPLETE (2026-06-09) — retained for reference; superseded for full-period work.

---

### Scripts

| Script | Purpose |
|---|---|
| `scripts/build_stage1_neuralhydrology_january_pilot.py` | Builds the January 2023 NH package |
| `scripts/audit_stage1_neuralhydrology_january_pilot.py` | Preflight auditor; exits 0 on PASS, 1 on FAIL |

Run order:
```bash
python scripts/build_stage1_neuralhydrology_january_pilot.py
python scripts/audit_stage1_neuralhydrology_january_pilot.py
```
Builder runtime: ~8 s. Auditor runtime: ~20 s.

---

### Package layout (2G)

```
tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/
  package/
    time_series/          {STAID}.nc  (50 files)
    attributes/           attributes_full.csv, attributes_smoke.csv
    basin_lists/          all_basins.txt, no_streamflow_basins.txt
                          january_2023_smoke/{train,val,test}_basins.txt
                          january_2023_smoke_streamflow_only/{train,val,test}_basins.txt
    configs/              smoke_v1.yml  [DRAFT — not yet run]
    manifests/            dataset_manifest.json, variable_schema.csv,
                          static_attribute_audit.csv, hydroatlas_join_audit.csv,
                          missingness_report.csv
    README.md
  audit/
    audit_report.json
    audit_report.md
    per_basin_summary.csv
    per_variable_missingness.csv
    static_attribute_audit.csv
  design/
    stage1_2f_neuralhydrology_package_design.md
    preflight_audit_plan.md
    variable_schema_proposal.csv
    static_attribute_schema_proposal.csv
    hydroatlas_asset_investigation.md
    streamflow_recovery_plan.md
```

All outputs under `tmp/` (gitignored). No generated files committed.

---

### NetCDF structure (2G)

- Layout: NeuralHydrology GenericDataset-style. One file per basin.
- Time coordinate: `date` (744 hourly UTC steps, 2023-01-01T00Z – 2023-01-31T23Z)
- Encoding: `float32` for all dynamic variables; `float64` for `date`
- `_FillValue`: -9999.0

Dynamic variables included in 2G (note: schema since corrected in 2K-F-C-B):

| Variable | Units | Source | Smoke config |
|---|---|---|---|
| `mrms_qpe_1h_mm` | mm | MRMS QPE 1h Pass1 | yes |
| `rtma_2t_K` | K | RTMA CONUS 2.5km | yes |
| `rtma_2d_K` | K | RTMA CONUS 2.5km | yes (V1) |
| `rtma_2sh_kgkg` | kg kg⁻¹ | RTMA CONUS 2.5km | yes (V1) |
| `rtma_sp_Pa` | Pa | RTMA CONUS 2.5km | no (V2: wide only) |
| `rtma_10u_ms` | m s⁻¹ | RTMA CONUS 2.5km | yes |
| `rtma_10v_ms` | m s⁻¹ | RTMA CONUS 2.5km | yes |
| `rtma_10si_ms` | m s⁻¹ | RTMA CONUS 2.5km | no (wide only) |
| `rtma_i10fg_ms` | m s⁻¹ | RTMA CONUS 2.5km | no (wide only) |
| `rtma_tcc_pct` | % | RTMA CONUS 2.5km | no (V3: wide only) |
| `qobs_m3s` | m³ s⁻¹ | CAMELSH hourly | target |

`ceil` and `vis` were excluded from 2G outputs (diagnostic_only per extraction metadata).
**Note:** `rtma_10si_ms` and `rtma_i10fg_ms` are renamed/corrected in the v001 schema
(now `rtma_gust_ms` from source `i10fg`; `10si` is a scalar wind speed not in v001).
V1/V2/V3 design decisions documented in `design/stage1_2f_neuralhydrology_package_design.md`.

---

### Static attributes (2G)

- `attributes_full.csv`: 50 rows × 238 columns (237 attribute cols + `gauge_id`)
- `attributes_smoke.csv`: 50 rows × 6 columns
- Sources: 14 manifest/physical cols, 30 GAGES-II, 193 HydroATLAS
- HydroATLAS -999/-9999 replaced with NaN; 50/50 pilot match after STAID normalization
- Nullable cols (expected, S2 decision): `max_abs_hourly_jump_over_Q50` (1 NaN),
  `q95_q50_ratio` (1 NaN), `wet_cl_smj` (14 NaN)

---

### Streamflow coverage (2G)

| Category | Count |
|---|---|
| Full Jan 2023 `qobs_m3s` | 20 |
| Partial `qobs_m3s` (some NaN) | 8 |
| All-NaN `qobs_m3s` | 22 |

The 22 all-NaN STAIDs were recovered from USGS IV in Milestone 2H.

---

### Audit result (2G)

```
Errors:   0
Warnings: 1  (nullable columns — expected; see Null policy)
```

PASS. No NeuralHydrology model training was run. No generated files committed.
