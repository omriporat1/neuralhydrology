# Flash-NH Stage 1 — h2o Operations Preflight

Last updated: 2026-06-15
Status: **KEY POLICIES ANSWERED — CPU/storage work conditionally unblocked; training designated for Moriah cluster**

---

## Purpose

This document is an operations and governance preflight before Flash-NH Stage 1
proceeds to:

- Large spatial-data bulk downloads (NOAA MRMS, RTMA/URMA, ECMWF, NWM, etc.)
- Basin-average preprocessing over 2,843 basins × 5+ years
- NeuralHydrology model training

The completed USGS IV target acquisition (2,843 basins, 4-shard launcher,
~3 h wall clock, 6.6 GB output, commit `e362c0e`) validated h2o for
**moderate network-bound sequential workloads** under `screen`-based job
management.

Following policy clarification (2026-06-15), h2o is designated as the platform for:

- Large spatial-data bulk downloads (MRMS, RTMA/URMA, etc.)
- Basin-average preprocessing and intermediate data assembly
- Canonical dataset storage and eventual promotion to shared lab storage

**NeuralHydrology model training is not planned on h2o.** h2o has no usable GPU
and is not a training platform. Training will run on the Moriah cluster — see
the [Moriah Cluster (Training Destination)](#moriah-cluster-training-destination) section below.

---

## Known Facts and Partial Answers

### Confirmed from preflight / project owner (updated 2026-06-15)

| Item | Status | Known value |
|---|---|---|
| Host | Confirmed | `h2o.es.huji.ac.il` |
| Personal work root | Confirmed | `/data42/omrip/Flash-NH` |
| Repo clone | Confirmed | `/data42/omrip/Flash-NH/repos/flash-nh` |
| Generated outputs | Confirmed | `/data42/omrip/Flash-NH/tmp` |
| Machine purpose | **Answered** | Storage, downloads, preprocessing, and data assembly; **not** NeuralHydrology training |
| Scheduler in PATH | **Answered** | No scheduler **by design** (not a gap — intentional); `screen` is the agreed background job manager |
| Session managers | Confirmed | `screen` available and in use; `tmux` unverified |
| `/data42` backup | **Answered** | **Not backed up.** Processed outputs must be exported locally or promoted deliberately |
| `/data42/omrip` auto-delete | **Partially answered** | Not auto-deleted; formal retention lifetime still unknown |
| Total volume guidance | **Answered** | Try not to exceed ~20 TB overall across all Flash-NH data on `/data42` |
| Raw data reproducibility | **Answered** | Raw spatial downloads are redownloadable, but request specs, manifests, and checksums must be documented |
| Shared data root | **Partially answered** | `/data42/hydrolab/Data/Flash-NH_data/` — subfolders allowed following reproducibility policy; exact write-access and naming TBD |
| `/data42` capacity / quota | Partially known | Large free capacity observed; formal per-user quota unknown |
| GPU availability | **Answered** | **No usable GPU** (`nvidia-smi` not found; `torch` not installed in env; PI confirms h2o likely has no GPU) |
| CUDA / PyTorch | **Answered** | Not applicable on h2o; `torch` not installed; not needed for preprocessing workloads |
| CPU/process fair-use | **Partially answered** | CPU compute allowed; see Compute Etiquette section below |
| Training location | **Answered** | Moriah GPU cluster — not h2o; see Moriah section below |
| Python environment | **Answered** | Installed at `/data42/omrip/Flash-NH/envs/flashnh-stage1`; Python 3.11.15; 7/7 smoke PASS (2026-06-15); activation requires `source /opt/conda/etc/profile.d/conda.sh` first |
| Promotion approval | **Partially answered** | Subfolders under `/data42/hydrolab/Data` allowed with reproducibility policy; formal write-access and convention confirmation still needed |

### Key cautions from partial answers

- **`/data42` is not backed up.** Any curated outputs that must be preserved long-term
  must be either:
  - transferred to a local machine (e.g. via `scp` or `rsync`), or
  - promoted to `/data42/hydrolab/Data/Flash-NH_data/` once naming and write-access are confirmed, or
  - documented with checksums so they can be regenerated if lost.

- **~20 TB total volume target.** Estimated Flash-NH spatial-data footprint
  (see §Large External Download Policy) could reach 7–15 TB. Raw downloads, processed
  intermediates, and ML datasets combined must stay well under 20 TB.
  Raw downloads that have been processed and checksummed should be considered for
  deletion to free space.

- **Raw downloads are redownloadable; specs are not.** Every bulk download must be
  accompanied by a `manifest.json` or equivalent that captures the exact source URL
  patterns, parameter codes, spatial bounds, temporal range, and file count, so the
  download can be reproduced exactly. Checksums on downloaded files are strongly
  recommended.

- **`/data42/omrip` is not auto-deleted**, but it is also not backed up. Do not treat
  it as permanent storage — treat it as a working space that requires deliberate export
  or promotion for anything that needs to survive long-term.

---

## h2o Compute Etiquette

h2o has no scheduler by design. CPU compute is allowed under the following rules:

1. **Start with smoke tests.** Validate a new workload with 1–4 basins or a short time
   window before launching full-scale jobs.

2. **CPU ceiling: 50–60%.** After smoke tests pass, scale to no more than 50–60% total
   CPU use. On a 64-core machine that means ≤ 32–38 active workers; start with 16–32.

3. **Use 12–24 hour chunks.** Break large workloads into resumable chunks so that a
   failure or interruption does not require a full restart. Chunk boundaries should be
   logged with checksums so completed chunks can be skipped on re-run.

4. **Notify for long or heavy jobs.** Before launching a job expected to run more than
   a few hours or use more than ~30% CPU, notify the PI / machine owner. A short
   Slack message or email is sufficient.

5. **Run under `screen`.** All background jobs must be launched inside a named `screen`
   session so they survive SSH disconnect and can be monitored or killed if needed.

6. **Monitor load.** Check `uptime` or `htop` before expanding parallelism. If average
   load is already above ≈ 0.7 × nproc, hold off until it drops.

---

## Moriah Cluster (Training Destination)

NeuralHydrology model training is planned for the Moriah GPU cluster at HUJI.

| Item | Value |
|---|---|
| Cluster name | Moriah |
| Personal root | `/sci/labs/efratmorin/omripo/PhD` |
| Proposed repo clone | `/sci/labs/efratmorin/omripo/PhD/repos/flash-nh` |
| Proposed data root | `/sci/labs/efratmorin/omripo/PhD/Data/Flash-NH_data/` |
| GPU access | Yes (GPU cluster; CUDA/PyTorch available) |
| Scheduler | Standard HPC scheduler (SLURM expected; confirm before use) |

### Data transfer from h2o to Moriah

Assembled NeuralHydrology packages (forcing time series, target NCs, attributes CSV,
splits) must be transferred from h2o to Moriah before training can begin. The
recommended approach:

1. Assemble the NH package on h2o (preprocessing, builder script).
2. Create a compact audit export bundle (see `docs/repo_policy.md`).
3. Transfer the full assembled package directory to Moriah via `scp`, `rsync`, or
   shared-filesystem access if available.
4. Record transfer checksums. Both endpoints (h2o and Moriah) should have matching
   `checksums.sha256` files.

**Moriah GPU environment is a separate design task** from the h2o preprocessing
environment. The two envs have different dependency trees:
- h2o env: data downloads, xarray, geopandas, rasterio — no GPU libraries needed
- Moriah env: PyTorch + NeuralHydrology + CUDA — no download tooling needed

---

## Environment Strategy

**h2o preprocessing env: INSTALLED AND SMOKE-TESTED (2026-06-15)**

| Item | Status |
|---|---|
| h2o preprocessing env path | `/data42/omrip/Flash-NH/envs/flashnh-stage1` ✅ |
| h2o env spec file (repo) | `envs/environment-stage1-h2o.yml` ✅ committed |
| Python version | `3.11.15` ✅ |
| Env size | `7.0 G` (see CUDA torch caveat below) |
| All smoke checks | **7/7 PASS** ✅ |
| Smoke log | `/data42/omrip/Flash-NH/tmp/env_smoke_20260615T120918Z/env_smoke.log` |
| Documentation | `docs/stage1_environment.md` ✅ |
| Moriah training env | Separate design; after Moriah access confirmed |
| Moriah env spec file (repo) | `envs/environment-stage1-moriah.yml` (future) |

**Install workaround (h2o-specific):** `mamba` was broken (`libmamba.so.4` load error);
default conda solver hit a permission error on `/opt/conda/pkgs/cache/`. Successful command:

```bash
export CONDA_PKGS_DIRS=/home/omrip/.conda/pkgs
conda env create --solver classic \
    --file envs/environment-stage1-h2o.yml \
    --prefix /data42/omrip/Flash-NH/envs/flashnh-stage1
```

**Activation on h2o** requires sourcing conda first (non-login shells):

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1
which python   # verify: must show .../flashnh-stage1/bin/python
```

**Activation caveat (observed 2026-06-18):** The shell prompt may show `(flashnh-stage1)` while
`which python` still points to `/opt/conda/envs/iacpy3_2025/bin/python`. Always verify `which python`
after activation. If it shows the wrong path, run the `source` + `conda activate` sequence again
explicitly. Clean reactivation resolves it. Do not start any job until `which python` confirms
the correct prefix.

**Additional package — py7zr (added 2026-06-18):** `py7zr` was installed into `flashnh-stage1`
using the same solver workaround required during initial env creation:

```bash
export CONDA_PKGS_DIRS=/home/omrip/.conda/pkgs
conda install --solver classic py7zr
```

**CUDA torch caveat:** `neuralhydrology` pip-installed `torch==2.12.0+cu130` and NVIDIA
packages, making the env 7.0 G. `torch.cuda.is_available()` = `False` on h2o — CUDA is
inert. Future spec revision will use `--no-deps` or a CPU-only torch to keep the env lean.
**h2o is not for NeuralHydrology training** — training remains designated for Moriah.

See `docs/stage1_environment.md` for full install notes, smoke test results, and mitigation plan.

---

## Questions for PI / Machine Owner

Status legend: ✅ Answered  ⚠️ Partially answered  ❓ Still open

### 1. Machine Role and Intended Use

- ✅ What is h2o for? → Storage, downloads, preprocessing, and data assembly. Not training.
- ❓ Are there other active users whose workloads compete for CPU/RAM/disk?
- ✅ Is h2o the intended home for preprocessing? → Yes; training goes to Moriah.

---

### 2. Scheduler Availability and Direct-Run Policy

- ✅ Is there a job scheduler? → No scheduler by design. `screen` is the agreed approach.
- ✅ What is the policy for multi-hour background jobs? → `screen`; `tmux` is likely fine but unverified.
- ❓ Is there a maximum number of simultaneous background processes per user?

---

### 3. Fair-Use CPU, Process, and Wall-Time Limits

- ⚠️ Is there an agreed per-user CPU core limit? → Informal: stay ≤ 50–60% total CPU. Formal quota unknown.
- ❓ Is there a wall-time limit on unattended processes?
- ❓ Are there times of day or days of week when heavy CPU use is restricted?
- ⚠️ Should compute-heavy jobs be announced? → Yes; notify PI/machine owner before long or heavy jobs.

---

### 4. Storage Quotas, Backup, and Purge

- ✅ Is `/data42` backed up? → **No. Not backed up.**
- ⚠️ Total volume guidance: try not to exceed ~20 TB overall.
- ❓ Is there a formal per-user quota on `/data42` beyond the informal ~20 TB guidance?
- ❓ Are there inode limits?
- ⚠️ Purge / auto-delete policy for `/data42/omrip`? → Not auto-deleted; formal lifetime unknown.
- ❓ What is the expected lifetime of data in `/data42/omrip`?

---

### 5. Acceptable Storage Locations

| Data type | Current/planned location | Status |
|---|---|---|
| Raw USGS IV Parquet cache | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/*/raw_cache/` | In-place; deletable once canonical NCs verified |
| Canonical hourly NC files | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/*/canonical/` | Promote to shared space or export locally |
| MRMS / RTMA raw GRIB2 | Not yet downloaded | Stage under `/data42/omrip/Flash-NH/tmp/` before processing |
| Basin-average Parquet | Not yet created | Under personal tmp; promote cleaned product to shared space |
| Final NeuralHydrology inputs | Not yet created | Promote to `/data42/hydrolab/Data/Flash-NH_data/04_ml_datasets/` |

- ⚠️ `/data42/hydrolab/Data/Flash-NH_data/` is the proposed shared root — subfolders
  allowed following reproducibility policy; formal write-access and exact naming TBD.

---

### 6. Large External Download Policy

Planned bulk downloads (rough estimates; may be refined):

| Source | Estimated raw volume | Notes |
|---|---|---|
| MRMS QPE 1h (2020–2025) | ~1–2 TB | CONUS composite; ~45,000 hourly files |
| RTMA 2.5 km CONUS (2020–2025) | ~2–5 TB | 7–8 variables × hourly |
| URMA QPE follow-up (optional) | ~0.5 TB | QPE only |
| NWM retrospective (optional) | TBD | Depends on variable selection |
| **Total (estimated)** | **~4–9 TB raw** | After processing, basin-avg intermediates add ~1–3 TB |

- ✅ Raw downloads are redownloadable — but request specs and manifests must be documented.
- ❓ Are there network egress or bandwidth limits from h2o to external sources?
- ❓ Should large downloads be scheduled at off-peak hours?
- ❓ Does the institution require data transfer agreements (ECMWF)?

---

### 7. GPU Availability and Training Policy

- ✅ Does h2o have usable GPUs? → **No.** `nvidia-smi` not found; `torch` not installed; PI confirms h2o likely has no GPU.
- ✅ Should NeuralHydrology training run on h2o? → **No.** Training designated for Moriah cluster.
- ❓ Moriah scheduler details (confirm SLURM partition names and GPU queue policy before first training run).
- ❓ Are there CUDA driver / PyTorch version constraints on Moriah?

---

### 8. Software Environment Policy

- ✅ Production should use `flashnh-stage1` under `/data42/omrip/Flash-NH/envs/flashnh-stage1`,
  not the shared `iacpy3_2025` env. Confirmed in use from 2K-A (2026-06-18).
- ✅ Env spec `envs/environment-stage1-h2o.yml` committed.
- ✅ Project-local conda env confirmed creatable under `/data42/omrip` (done successfully).
- ⚠️ `mamba` broken on h2o (`libmamba.so.4` load error); use `conda --solver classic` with
  `CONDA_PKGS_DIRS=/home/omrip/.conda/pkgs` for all installs. Applies to both env creation
  and subsequent package additions (e.g., `py7zr` added 2026-06-18).
- ❓ Is `micromamba` available on h2o as an alternative?
- ❓ What is the intended lifetime of `iacpy3_2025`?
- ⚠️ **PS1 transfer helper broken (2026-06-18):** `scripts/prepare_stage1_forcing_inputs_h2o.ps1`
  fails to parse on Windows PowerShell 5.1 (8 AST errors). It is not needed for 2K-B or 2K-C
  (grid JSONs and CAMELSH shapefile are already transferred and verified on h2o). Fix in a
  separate small commit before relying on it for future transfer operations.
- ✅ **Launcher activation bug — patched and verified (2026-06-18):** Both
  `scripts/run_stage1_forcing_smoke_h2o.sh` and `scripts/run_stage1_forcing_fullperiod_h2o.sh`
  previously raised `CondaError: Run 'conda init' before 'conda activate'` when invoked as
  `bash script.sh` in a non-interactive shell. Root cause: `conda activate` requires the shell
  *function* registered by `conda.sh`, not just the conda binary being in PATH, and non-interactive
  shells do not source `~/.bashrc`. The 2K-B smoke was therefore run via direct extractor
  invocation. Fixed in commit `ccb2631`: `conda.sh` is now sourced unconditionally and
  `conda activate` is non-fatal (`|| true`); PATH-prepend is the authoritative env-selection
  mechanism; Python version check is the fatal gate. **Verified on h2o (2026-06-18):**
  `bash scripts/run_stage1_forcing_smoke_h2o.sh` completed end-to-end via the launcher,
  reporting `/data42/omrip/Flash-NH/envs/flashnh-stage1/bin/python (Python 3.11.15)` and PASS.

---

### 9. Monitoring and Notification

- ⚠️ Notification rule: notify PI / machine owner before heavy CPU or disk workloads (> few hours or > ~30% CPU).
- ❓ Is there a standard way to be notified when a long job completes or fails?
- ❓ Is there a shared job registry for active h2o workloads?

---

### 10. Data Promotion to Shared Lab Storage

- ⚠️ Shared root: `/data42/hydrolab/Data/Flash-NH_data/` — subfolders allowed with provenance; write access not yet formally confirmed.
- ❓ What is the promotion process (who approves, what metadata is required)?
- ❓ Are there naming or versioning conventions required by the lab?
- ❓ Is there a retention policy for data in shared space?

---

## Proposed Shared Data Layout

Subfolders under `/data42/hydrolab/Data` are approved following the project
reproducibility policy. Promotion of curated datasets should proceed with full
provenance (manifest, checksums, provenance JSON, git commit) as described below.
**Confirm write access before the first promotion.**

```
/data42/hydrolab/Data/Flash-NH_data/
│
├── 00_raw/                  # Downloaded source files (MRMS, RTMA, URMA, etc.)
│   ├── mrms_qpe_1h/
│   ├── rtma_conus_2p5km/
│   └── urma_qpe/
│
├── 01_standardized_grids/   # Reprojected / regridded rasters in a common CRS
│
├── 02_basin_geometries/     # Shapefile / GeoPackage basin boundaries and weights
│   └── stage1_2843basins_v001/
│       ├── basins.gpkg
│       ├── rtma_weights.parquet
│       └── manifest.json
│
├── 03_basin_timeseries/     # Basin-average hourly time series (forcing)
│   └── stage1_basin_hourly_forcings_v001/
│       ├── <STAID>_forcing_hourly.nc   (per basin)
│       ├── manifest.json
│       ├── checksums.sha256
│       ├── dataset_config.yaml
│       ├── source_git_commit.txt
│       └── run_provenance.json
│
├── 04_ml_datasets/          # Assembled NeuralHydrology-format datasets
│   ├── stage1_targets_usgs_iv_v001/        # Cleaned target NC files (2,843 basins)
│   │   ├── <STAID>_hourly.nc
│   │   ├── manifest.json
│   │   ├── checksums.sha256
│   │   └── run_provenance.json
│   └── stage1_neuralhydrology_fullperiod_v001/  # Assembled NH package
│       ├── time_series/
│       ├── attributes/
│       ├── manifest.json
│       ├── checksums.sha256
│       ├── dataset_config.yaml
│       ├── source_git_commit.txt
│       └── run_provenance.json
│
├── 05_splits/               # Train/val/test basin splits (frozen, versioned)
│   └── stage1_split_v001/
│       ├── train_basins.txt
│       ├── val_basins.txt
│       ├── test_basins.txt
│       └── split_config.yaml
│
├── 06_qc_reports/           # Audit outputs promoted for lab reference
│   └── stage1_target_audit_v001/
│       ├── target_status.csv
│       ├── gap_audit.csv
│       └── audit_summary.md
│
├── 07_experiments/          # Training run configs and result summaries (not model weights)
│
├── 08_logs/                 # Promoted run logs (compact, not raw per-station)
│
├── 09_manifests/            # Cross-dataset manifests and lineage records
│
└── tmp/                     # Scratch space under shared root (not guaranteed retained)
```

### What counts as a curated output

**Include** in shared space:
- Final basin geometries and weight tables
- Cleaned, versioned hourly target NC files
- Basin-average forcing time series
- Frozen ML datasets (assembled NH packages)
- Frozen splits and split configs
- QC summaries and audit CSVs (compact)
- Manifests, checksums, provenance records

**Do not include** in shared space:
- Raw scratch caches (Parquet API responses, raw GRIB2 files)
- Per-station acquisition logs
- Temporary downloads or exploratory intermediates
- Model checkpoint weights (unless explicitly archived)

---

## Candidate Naming Convention

Each versioned promoted dataset uses a `v00N` suffix for the first stable version:

```
stage1_targets_usgs_iv_v001/
stage1_neuralhydrology_fullperiod_v001/
stage1_basin_hourly_forcings_v001/
stage1_2843basins_geometries_v001/
stage1_split_seed42_v001/
```

Every promoted dataset directory must contain at minimum:
- `manifest.json` — file list, row counts, version, source commit, generation date
- `checksums.sha256` — SHA-256 of every data file in the dataset
- `source_git_commit.txt` — exact git commit hash of the script that produced it
- `run_provenance.json` — arguments, environment, timestamp, input dataset hashes
- `dataset_config.yaml` — where applicable (NH datasets, splits)

---

## Read-Only h2o Discovery Commands

Run these on h2o to collect system facts before any heavy work begins.
**All commands are read-only and safe to run.**

```bash
#!/bin/bash
# h2o discovery — read-only, safe to run
# Save output: bash h2o_discovery.sh | tee /data42/omrip/Flash-NH/tmp/h2o_discovery_$(date +%Y%m%d).txt

echo "=== IDENTITY ==="
hostname -f; date -u; whoami

echo "=== UPTIME / LOAD ==="
uptime

echo "=== CPU ==="
nproc
lscpu | grep -E "^(Architecture|CPU\(s\)|Thread|Core|Socket|Model name|NUMA)"

echo "=== MEMORY ==="
free -h

echo "=== STORAGE ==="
df -hT /data42 /data42/hydrolab/Data $HOME /tmp 2>/dev/null

echo "=== /data42 MOUNT ==="
findmnt /data42 2>/dev/null || mount | grep data42

echo "=== SCHEDULER / SESSION TOOLS ==="
for cmd in sbatch squeue sinfo qsub qstat bsub bjobs screen tmux nohup; do
    loc=$(command -v $cmd 2>/dev/null) && echo "  $cmd: $loc" || echo "  $cmd: NOT FOUND"
done

echo "=== ENVIRONMENT / CONTAINER TOOLS ==="
for cmd in module conda mamba micromamba apptainer singularity docker; do
    loc=$(command -v $cmd 2>/dev/null) && echo "  $cmd: $loc" || echo "  $cmd: NOT FOUND"
done

echo "=== GPU ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "  nvidia-smi: NOT FOUND"
fi

echo "=== PYTORCH CUDA CHECK ==="
python - <<'PYEOF' 2>/dev/null || echo "  torch not importable or no CUDA"
import torch
print(f"  torch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM")
PYEOF

echo "=== RESOURCE LIMITS ==="
ulimit -a

echo "=== PYTHON ==="
which python && python --version

echo "=== REPO STATE ==="
cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation \
    && git log --oneline -5 \
    && git status --short
```

Save to `/data42/omrip/Flash-NH/tmp/h2o_discovery_YYYYMMDD.txt` and transfer locally
for reference. Do not commit the output file.

---

## Decision Gates Before Next Major Stages

| Gate | Condition to clear | Status |
|---|---|---|
| **G1: Spatial bulk download** | Storage location agreed; off-peak preference known; volume target acknowledged | **CONDITIONALLY UNBLOCKED** — `/data42/omrip` not auto-deleted; 20 TB informal target acknowledged; staging under `/data42/omrip/Flash-NH/tmp/` is acceptable; formal per-user quota still open |
| **G2: Basin-average preprocessing** | CPU fair-use policy known; storage location agreed | **CONDITIONALLY UNBLOCKED** — CPU compute allowed with etiquette (≤50–60% CPU; start 16–32 workers; notify for heavy jobs); staging location agreed in principle |
| **G3: NeuralHydrology training** | Training platform designated | **NOT PLANNED ON h2o** — h2o has no usable GPU; training is designated for Moriah cluster; Moriah scheduler details still need confirmation before first training run |
| **G4: Shared-data promotion** | `/data42/hydrolab/Data` subfolders approved; provenance policy followed | **CONDITIONALLY UNBLOCKED** — subfolders allowed with reproducibility provenance; formal write-access and exact naming convention confirmation still needed |
| **G5: Scheduler / parallelism** | Background-job policy documented | **RESOLVED** — no scheduler by design; `screen`-based job management is the agreed approach |
| **G6: `/data42` purge policy** | Auto-delete policy clarified | **PARTIALLY RESOLVED** — `/data42/omrip` not auto-deleted; formal retention lifetime still unknown |

Notes:
- G1 and G2 are conditionally unblocked, meaning bulk downloads and preprocessing may
  begin with compute-etiquette rules followed. Start with smoke tests, scale up gradually.
- G3 is reclassified from BLOCKED to NOT PLANNED ON h2o. The blocker is now "confirm
  Moriah scheduler + env before first training run" rather than "decide training location."
- G4 is conditionally unblocked pending formal write-access confirmation. Do not promote
  until a test write is confirmed to succeed.
- G5 is fully resolved: screen-based job management is the established pattern.
- G6 is partially resolved: no auto-delete on `/data42/omrip`, but formal lifetime unknown.
  Treat personal tmp as impermanent and export/promote curated outputs.

---

## Recommended Immediate Project Status

**Bulk CPU/storage work on h2o is conditionally unblocked** (etiquette rules apply).
**Training is not planned on h2o** — that goes to Moriah.
**h2o preprocessing env is installed and smoke-tested** — use `flashnh-stage1` for all future h2o work.

Unblocked work that can proceed now:

1. ~~**h2o environment setup**~~ — **DONE (2026-06-15)**. Env at
   `/data42/omrip/Flash-NH/envs/flashnh-stage1`, all 7 smoke checks PASS.
2. **Target-cleaned builder design** — design the script that consumes the 2,843
   canonical NC files + `config/stage1_target_policy.yaml` and produces the
   NeuralHydrology-format target dataset. Local code design, no heavy execution.
3. **Moriah transfer layout design** — define the exact directory structure and
   transfer procedure for moving assembled NH packages from h2o to Moriah.
4. **Push pending commits** — push local commits ahead of origin.

Blocked pending additional confirmation:

5. **Spatial bulk downloads (MRMS, RTMA/URMA)** — wait for smoke-test sign-off under
   etiquette rules and confirm per-user quota is acceptable before TB-scale downloads.
6. **Shared-data promotion** — confirm write access to `/data42/hydrolab/Data/Flash-NH_data/`
   before first promotion.
7. **NeuralHydrology training** — blocked on Moriah scheduler confirmation, env setup,
   and assembled NH package transfer.
