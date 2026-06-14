# Flash-NH Stage 1 — h2o Operations Preflight

Last updated: 2026-06-14
Status: **PARTIALLY ANSWERED — heavy execution still blocked pending full policy clarification**

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
management. It did not validate h2o for:

- Large sustained downloads (TB-scale spatial raster archives)
- Multi-hour CPU-heavy preprocessing at high parallelism
- GPU training workloads
- Promotion to shared lab storage

---

## Known Facts and Partial Answers

### Confirmed from preflight / project owner

| Item | Status | Known value |
|---|---|---|
| Host | Confirmed | `h2o.es.huji.ac.il` |
| Personal work root | Confirmed | `/data42/omrip/Flash-NH` |
| Repo clone | Confirmed | `/data42/omrip/Flash-NH/repos/flash-nh` |
| Generated outputs | Confirmed | `/data42/omrip/Flash-NH/tmp` |
| Machine purpose | **Partially answered** | Intended to support multi-hour to multi-day computations |
| Scheduler in PATH | Confirmed (observed) | None detected (`sbatch`, `squeue`, `sinfo`, `qsub`, `bsub` absent); full scheduler policy still unknown |
| Session managers | Partially confirmed | `screen` available and used for 2,843-basin run; `tmux` unverified |
| `/data42` backup | **Answered** | **Not backed up.** Processed outputs must be exported/transferred locally or promoted deliberately |
| Total volume guidance | **Answered** | Try not to exceed ~20 TB overall across all Flash-NH data on `/data42` |
| Raw data reproducibility | **Answered** | Raw spatial downloads are redownloadable, but request specs, manifests, and checksums must be documented for reproducibility |
| Likely shared data root | **Partially answered** | `/data42/hydrolab/Data` is the likely location; write-access and promotion policy still TBD |
| `/data42` capacity / quota | Partially known | Large free capacity observed; per-user quota unknown |
| Purge / auto-delete policy | Unknown | Not yet answered |
| Python environment | Partially known | `iacpy3_2025` shared env used for smoke/recovery; project-local env strategy not yet decided |
| GPU availability | Unknown | Check with `nvidia-smi`; see discovery commands below |
| CUDA / PyTorch | Unknown | Depends on GPU availability result |
| CPU/process fair-use | Unknown | Not yet answered |
| Training location | Unknown | h2o vs external GPU cluster not yet decided |
| Promotion approval | Unknown | Process for promoting to `/data42/hydrolab/Data` not yet answered |

### Key cautions from partial answers

- **`/data42` is not backed up.** Any curated outputs that must be preserved long-term
  must be either:
  - transferred to a local machine (e.g. via `scp` or `rsync`), or
  - promoted to `/data42/hydrolab/Data` once that policy is confirmed, or
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

---

## Questions for PI / Machine Owner

Status legend: ✅ Answered  ⚠️ Partially answered  ❓ Still open

### 1. Machine Role and Intended Use

- ⚠️ Is h2o a shared research server or a personal workstation?
  *Partial: intended for multi-hour to multi-day computations; shared status unclear*
- ❓ Are there other active users whose workloads compete for CPU/RAM/disk?
- ❓ Is h2o the intended long-term home for Flash-NH compute, or a stepping stone?

---

### 2. Scheduler Availability and Direct-Run Policy

- ❓ Is there a job scheduler (SLURM, PBS/Torque, LSF, SGE) installed but not in
  the default PATH? If so, how is it accessed?
- ❓ If no scheduler: what is the agreed policy for multi-hour background jobs —
  `screen`, `nohup`, `tmux`, or something else?
- ❓ Is there a maximum number of simultaneous background processes per user?

---

### 3. Fair-Use CPU, Process, and Wall-Time Limits

- ❓ Is there an agreed per-user CPU core limit for background jobs?
- ❓ Is there a wall-time limit on unattended processes?
- ❓ Are there times of day or days of week when heavy CPU use is restricted?
- ❓ Should compute-heavy jobs be announced to other users?

---

### 4. Storage Quotas, Backup, and Purge

- ✅ Is `/data42` backed up? → **No. Not backed up.**
- ⚠️ Total volume guidance: try not to exceed ~20 TB overall.
- ❓ Is there a per-user quota on `/data42` beyond the informal ~20 TB guidance?
- ❓ Are there inode limits?
- ❓ Is there a purge or auto-delete policy for files under `/data42/omrip`?
- ❓ What is the expected lifetime of data in `/data42/omrip`?

---

### 5. Acceptable Storage Locations

| Data type | Current/planned location | Status |
|---|---|---|
| Raw USGS IV Parquet cache | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/*/raw_cache/` | In-place; deletable once canonical NCs verified |
| Canonical hourly NC files | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/*/canonical/` | Promote to shared space or export locally |
| MRMS / RTMA raw GRIB2 | Not yet downloaded | Needs location decision before download |
| Basin-average Parquet | Not yet created | Acceptable under personal tmp? |
| Final NeuralHydrology inputs | Not yet created | Promote to shared space |

- ⚠️ `/data42/hydrolab/Data` is the likely shared data root — write access and naming
  convention not yet confirmed.
- ❓ Quotas or approval requirements for promotion?

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
- ❓ Is there a preferred staging area for raw downloads?
- ❓ Does the institution require data transfer agreements (ECMWF)?

---

### 7. GPU Availability and Training Policy

- ❓ Does h2o have one or more GPUs? (Check `nvidia-smi` — see discovery commands)
- ❓ Is GPU access shared or reserved per user?
- ❓ Is there a queuing mechanism for GPU access?
- ❓ Should NeuralHydrology training run on h2o or an external GPU cluster?
- ❓ Are there CUDA driver / PyTorch version constraints?

---

### 8. Software Environment Policy

- ⚠️ Production should use a project-designated environment, not rely indefinitely
  on the shared `iacpy3_2025` env. Specific strategy (project venv, new conda env,
  container) not yet decided.
- ❓ Can a project-local conda env be created under `/data42/omrip`?
- ❓ Is `mamba`, `micromamba`, or `conda` preferred?
- ❓ Are `apptainer`/`singularity` or `docker` available and encouraged?
- ❓ What is the intended lifetime of `iacpy3_2025`?

---

### 9. Monitoring and Notification

- ❓ Is there a standard way to be notified when a long job completes or fails?
- ❓ Should other lab members be notified before heavy CPU or disk workloads?
- ❓ Is there a shared job registry for active h2o workloads?

---

### 10. Data Promotion to Shared Lab Storage

- ⚠️ Likely shared root: `/data42/hydrolab/Data` — not yet confirmed.
- ❓ What is the promotion process (who approves, what metadata is required)?
- ❓ Are there naming or versioning conventions required?
- ❓ Is there a retention policy for data in shared space?

---

## Proposed Shared Data Layout

**This is a proposal only.** Nothing should be promoted to `/data42/hydrolab/Data`
until PI/shared-data policy is confirmed and write access is granted.

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
| **G1: Spatial bulk download** | Storage quota confirmed; download location agreed; off-peak policy confirmed | **BLOCKED** |
| **G2: Basin-average preprocessing** | CPU/process fair-use limits confirmed; storage location agreed | **BLOCKED** |
| **G3: NeuralHydrology training** | GPU availability confirmed; training location agreed; environment pinned | **BLOCKED** |
| **G4: Shared-data promotion** | `/data42/hydrolab/Data` write access and naming convention confirmed | **BLOCKED** |
| **G5: Scheduler / parallelism** | Scheduler availability or direct-run policy documented | **BLOCKED** |
| **G6: `/data42` purge policy** | Purge and lifetime policy for `/data42/omrip` clarified | **BLOCKED** |

Notes on partial answers:
- The ~20 TB informal volume target is **acknowledged but not a formal quota**. G1 and G2 remain blocked until a formal or agreed-upon per-user limit is known.
- The confirmed `/data42` no-backup policy means G4 is doubly important: promotions to shared space are the primary data protection path.
- Scheduler (G5) remains fully open; `screen`-based launcher is acceptable for the scale of the USGS acquisition but must be re-evaluated for TB-scale downloads.

---

## Recommended Immediate Project Status

**Heavy h2o execution is on hold.** Work that can proceed **without h2o**:

1. **Target-policy configuration** — define basin inclusion/exclusion rules and
   negative-qobs handling for the NeuralHydrology package builder. Code-only, local.
2. **Package-builder script design** — design the script that consumes the 2,843
   canonical NC files and produces the NeuralHydrology-format package. Local design,
   no heavy execution.
3. **Respond to open preflight questions** — particularly scheduler policy, CPU limits,
   GPU availability, and `/data42/hydrolab/Data` write-access.
4. **Run read-only h2o discovery** — execute the command block above on h2o and
   save output locally. Takes < 1 min, no load.
