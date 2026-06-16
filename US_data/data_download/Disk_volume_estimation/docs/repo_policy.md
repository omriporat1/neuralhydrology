# Repository Policy

Project: Flash-NH — near-real-time and forecast-aware hydrological modeling pipeline.

- `src/` is for application and library code.
- `scripts/` is for runnable helpers, audit runners, and one-off utilities.
- `docs/` is for markdown documentation.
- `reports/` is for generated audit and report artifacts.
- `data/` is for downloaded samples.
- `tmp/` is for scratch space only.
- Credential files must never be committed.
- Large raw GRIB, NC4, and sample files should not be committed unless they are explicitly intended for the repository.

## Generated artifact policy

- Git tracks source code, docs, configs, and small curated summaries.
- Git does not track generated data products, raw downloads, large report tables, logs, or caches.
- Generated run outputs should remain local or be archived/backed up separately.
- For review, each major run should create a lightweight `review_bundle` containing:
	- `summary.md`
	- `summary.json`
	- `manifest.json`
	- `run_command.txt`
	- `git_commit.txt`
	- selected small plots only if needed
- Copilot must not force-add ignored `reports/` files unless explicitly requested.

## Handling Guidance

- Keep generated outputs out of source directories.
- Prefer small, reviewable artifacts in `reports/` and documentation in `docs/`.
- If a large binary file is needed for a reproducible example, document the reason before adding it.

## Post-h2o Run Export Policy

After any substantial h2o (or other HPC) run that produces generated outputs,
create a compact **audit export bundle** before documenting results or proceeding
to the next milestone.

### What to include / exclude

| Include | Exclude |
|---|---|
| Launcher summary JSON and Markdown | Canonical NetCDF files |
| Audit CSV files (gap, quality, coverage, target status) | Raw Parquet caches |
| Main audit / run log | Per-station acquisition logs |
| Shard manifests | GRIB / large binaries |

### Policy

- Name bundles `<run_name>_audit_export.tar.gz` and place under a local `tmp/`
  subdirectory (gitignored).
- **Do not commit** the archive or extracted files to the repo.
- Scientific conclusions in committed documentation must be based on **inspected
  audit CSVs**, not terminal summaries or log tails alone.
- Quote specific numbers from the CSV files rather than paraphrasing logs.

See `docs/stage1_hpc_transition_preflight.md` → "Post-h2o Run Export Policy"
for the canonical example export command.

---

## h2o/Moriah Remote Run Evidence Policy

These rules apply to any run executed on a remote cluster (h2o, Moriah, or similar)
where generated outputs are too large to commit.

### Evidence bundle requirements

- h2o/Moriah launchers must create or identify a **compact evidence bundle** before
  the run concludes: logs, manifests, audit files, checksums, cleaning reports,
  and provenance JSON. Large generated outputs (NetCDF, Parquet, GRIB) stay remote.
- Evidence bundles must be **pulled to the local machine** before Claude documents
  or commits any conclusions. Claude must inspect the local evidence files directly —
  terminal summaries and pasted log tails are not sufficient.
- Large generated artifacts (NetCDF, Parquet, GRIB, raw shards) must **not be
  transferred or committed** unless explicitly approved by the user.

### Credential and authentication policy

- Passwords must **never be stored** in scripts, repo files, `.env` files, logs,
  Claude memory, or documentation.
- Preferred future authentication method: **SSH key + passphrase + ssh-agent + host alias**.
- Pull scripts may reference host aliases such as `flashnh-h2o` (configured in `~/.ssh/config`),
  but must not embed credentials, IP addresses intended to be private, or passwords.

### Transfer workflow (example using host alias)

```bash
# Define alias in ~/.ssh/config (not committed):
# Host flashnh-h2o
#   HostName h2o.es.huji.ac.il
#   User omrip
#   IdentityFile ~/.ssh/id_ed25519_h2o

# Pull compact evidence to local tmp/:
mkdir -p tmp/stage1_target_package_v001_evidence
scp "flashnh-h2o:/data42/omrip/Flash-NH/tmp/stage1_target_package_v001_logs/build.log" \
    tmp/stage1_target_package_v001_evidence/
scp "flashnh-h2o:.../manifest.json" tmp/stage1_target_package_v001_evidence/
# ... etc. for audit.log, checksums.sha256, cleaning_report.csv, run_provenance.json
```

Do not `git add` any files under `tmp/`.