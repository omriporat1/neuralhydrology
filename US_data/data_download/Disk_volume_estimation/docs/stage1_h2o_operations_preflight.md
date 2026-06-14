# Flash-NH Stage 1 — h2o Operations Preflight

Date: 2026-06-14
Status: **OPEN — awaiting PI/machine-owner responses before heavy execution**

---

## Purpose

This document is an operations and governance preflight before Flash-NH Stage 1
proceeds to:

- Large spatial-data bulk downloads (NOAA MRMS, RTMA/URMA, ECMWF, NWM, etc.)
- Basin-average preprocessing over 2,843 basins × 5+ years
- NeuralHydrology model training

The completed USGS IV target acquisition (2,843 basins, 4-shard launcher,
~3 h wall clock, 6.6 GB output, commit `e362c0e`) validated h2o for
**moderate network-bound sequential workloads** with `screen`-based job
management. It did not validate h2o for:

- Large sustained downloads (TB-scale spatial raster archives)
- Multi-hour CPU-heavy preprocessing at high parallelism
- GPU training workloads
- Promotion to shared lab storage

Before any of those workloads are launched, the questions below must be
answered by the PI or machine owner.

---

## Known Facts (from preflight)

| Item | Known value |
|---|---|
| Host | `h2o.es.huji.ac.il` |
| Personal work root | `/data42/omrip/Flash-NH` |
| Repo clone | `/data42/omrip/Flash-NH/repos/flash-nh` |
| Generated outputs | `/data42/omrip/Flash-NH/tmp` |
| Scheduler in PATH | None detected (`sbatch`, `squeue`, `sinfo`, `qsub`, `bsub` all absent) |
| Interactive session managers | `screen` confirmed available; `tmux` unverified |
| `/data42` capacity | Large free capacity observed during preflight; quota unknown |
| Backup / purge policy | Unknown |
| Python environment | Shared system conda env (`iacpy3_2025`) used for recovery; project-local venv not yet created |
| PROJ warning | Non-blocking warning at import; unresolved |
| GPU availability | Unknown |
| Training workloads policy | Unknown |

---

## Questions for PI / Machine Owner

### 1. Machine Role and Intended Use

- [ ] Is h2o a shared research server or a personal workstation?
- [ ] Are there other active users whose workloads compete for CPU/RAM/disk?
- [ ] Is h2o the intended long-term home for Flash-NH compute, or a temporary
      stepping stone to a larger cluster?

---

### 2. Scheduler Availability and Direct-Run Policy

- [ ] Is there a job scheduler (SLURM, PBS/Torque, LSF, SGE) installed but
      not in the default PATH? If so, how is it accessed?
- [ ] If no scheduler: what is the policy for running multi-hour background
      jobs — `screen`, `nohup`, `tmux`, or something else?
- [ ] Is there a maximum number of simultaneous background processes per user?

---

### 3. Fair-Use CPU, Process, and Wall-Time Limits

- [ ] Is there an agreed per-user CPU core limit for background jobs?
- [ ] Is there a wall-time limit on unattended processes?
- [ ] Are there times of day or days of week when heavy CPU use is restricted?
- [ ] Should compute-heavy jobs be announced to other users?

---

### 4. Storage Quotas, Backup, and Purge

- [ ] Is there a per-user disk quota on `/data42`? What is it?
- [ ] Are there size or inode limits?
- [ ] Is `/data42` backed up? How frequently?
- [ ] Is there a purge or auto-delete policy for files in `/data42/omrip/Flash-NH/tmp`?
- [ ] What is the expected lifetime of the data in `/data42/omrip`?

---

### 5. Acceptable Storage Locations

The following data categories need permanent or long-lived homes:

| Data type | Current location | Question |
|---|---|---|
| Raw USGS IV Parquet cache | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/*/raw_cache/` | Keep here or move? |
| Canonical hourly NC files | `/data42/omrip/Flash-NH/tmp/stage1_full_2843/*/canonical/` | Suitable long-term location? |
| MRMS / RTMA raw GRIB2 (TB-scale) | Not yet downloaded | Where should bulk spatial data land? |
| Basin-average intermediate Parquet | Not yet created | Acceptable under `/tmp`? |
| Final curated NeuralHydrology inputs | Not yet created | Promote to shared space? |

- [ ] Is there a shared lab data directory (e.g. `/data42/hydrolab/Data`) available for
      curated outputs, and if so, what is the write-access policy?
- [ ] Are there quotas or approval requirements for data promotion to shared space?

---

### 6. Large External Download Policy

Planned bulk downloads include:

| Source | Estimated raw volume | Frequency |
|---|---|---|
| MRMS QPE 1 h (2020–2025) | ~1–2 TB | Once |
| RTMA 2.5 km CONUS (2020–2025) | ~2–5 TB | Once |
| URMA QPE follow-up (optional) | ~0.5 TB | Once |
| NWM retrospective (optional) | TBD | TBD |

- [ ] Are there network egress or bandwidth limits from h2o to external sources?
- [ ] Should large downloads be scheduled at off-peak hours?
- [ ] Is there a preferred staging area for raw downloads before processing?
- [ ] Does the institution require data transfer agreements for NOAA or ECMWF data?

---

### 7. GPU Availability and Training Policy

- [ ] Does h2o have one or more GPUs? If so, what model(s) and VRAM?
- [ ] Is GPU access shared or reserved per user?
- [ ] Is there a queuing mechanism for GPU access (even without a formal scheduler)?
- [ ] Are NeuralHydrology training runs expected to run on h2o, or should training
      be offloaded to a GPU cluster?
- [ ] Are there CUDA driver / PyTorch version constraints on the shared environment?

---

### 8. Software Environment Policy

- [ ] Is there a required or preferred Python/conda environment for shared use?
- [ ] Can a project-local venv or conda env be created under `/data42/omrip`?
- [ ] Is `mamba`, `micromamba`, or `conda` preferred for environment management?
- [ ] Are `apptainer`/`singularity` or `docker` containers available and encouraged
      for reproducibility?
- [ ] Should the `iacpy3_2025` shared environment be used indefinitely, or only
      for development/smoke testing?

---

### 9. Monitoring and Notification

- [ ] Is there a standard way to be notified when a long job completes or fails
      (e.g. email, Slack, webhook)?
- [ ] Should other lab members be notified before running a workload that uses
      significant CPU or disk?
- [ ] Is there a shared log or job registry where active h2o workloads are tracked?

---

### 10. Data Promotion to Shared Lab Storage

- [ ] What is the process for promoting curated data from personal tmp to a shared
      lab directory?
- [ ] Who approves promotions to `/data42/hydrolab/Data` or equivalent?
- [ ] Are there naming or metadata conventions required for promoted datasets?
- [ ] Is there a retention policy for data in shared space (vs. personal tmp)?

---

## Read-Only h2o Discovery Commands

Run these on h2o to gather facts. None of these launch heavy jobs.

```bash
# --- identity and machine basics ---
echo "=== hostname ===" && hostname -f
echo "=== date ===" && date -u
echo "=== whoami ===" && whoami
echo "=== uptime ===" && uptime
echo "=== nproc ===" && nproc

# --- CPU detail ---
echo "=== lscpu ===" && lscpu | grep -E "^(Architecture|CPU\(s\)|Thread|Core|Socket|Model name|NUMA)"

# --- memory ---
echo "=== free -h ===" && free -h

# --- storage ---
echo "=== df -hT (key mounts) ===" && df -hT /data42 /tmp $HOME 2>/dev/null

# --- /data42 mount info ---
echo "=== findmnt /data42 ===" && findmnt /data42 2>/dev/null || mount | grep data42

# --- scheduler and session tools ---
for cmd in sbatch squeue sinfo qsub qstat bsub bjobs screen tmux nohup; do
    loc=$(command -v $cmd 2>/dev/null) && echo "$cmd: $loc" || echo "$cmd: NOT FOUND"
done

# --- environment and container tools ---
for cmd in module conda mamba micromamba apptainer singularity docker nvidia-smi; do
    loc=$(command -v $cmd 2>/dev/null) && echo "$cmd: $loc" || echo "$cmd: NOT FOUND"
done

# --- GPU ---
nvidia-smi 2>/dev/null || echo "nvidia-smi: NOT AVAILABLE"

# --- resource limits ---
echo "=== ulimit -a ===" && ulimit -a

# --- Python ---
echo "=== python ===" && which python && python --version

# --- repo state ---
cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation
echo "=== git log --oneline -5 ===" && git log --oneline -5
echo "=== git status --short ===" && git status --short
```

Save output to a file for reference:

```bash
bash <(cat above commands) 2>&1 | tee /data42/omrip/Flash-NH/tmp/h2o_discovery_$(date +%Y%m%d).txt
```

---

## Decision Gates Before Next Major Stages

The following gates must be **explicitly cleared** (answer documented here or
in a follow-up note) before the corresponding work begins on h2o.

| Gate | Condition to clear | Status |
|---|---|---|
| **G1: Spatial bulk download** | Storage quota confirmed; download location agreed; download policy confirmed | BLOCKED |
| **G2: Basin-average preprocessing** | CPU fair-use limits confirmed; storage location agreed | BLOCKED |
| **G3: NeuralHydrology training** | GPU availability confirmed; training location agreed; environment pinned | BLOCKED |
| **G4: Shared-data promotion** | Shared directory access confirmed; naming convention agreed | BLOCKED |
| **G5: Scheduler-based parallelism** | Scheduler availability confirmed or direct-run policy documented | BLOCKED |

Until G1–G5 are cleared, all heavy h2o execution is on hold.

---

## Recommended Immediate Project Status

The USGS IV target acquisition for all 2,843 basins is **structurally complete**.
The next planned work that can proceed **without h2o** is:

1. **Target-policy configuration design** — define the basin inclusion/exclusion
   rules and negative-qobs handling for the NeuralHydrology package builder (local,
   code-only, no downloads).
2. **Package-builder script design** — design the script that consumes the 2,843
   canonical NC files and produces the NeuralHydrology-format package (local design,
   no heavy execution).
3. **Response to operations preflight** — answer the questions above.

**Do not initiate spatial downloads, preprocessing, or training until the decision
gates above are resolved.**
