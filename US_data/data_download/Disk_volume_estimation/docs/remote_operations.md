# Flash-NH — Remote Operations

This file is the canonical home for **stable host-level operating facts** for h2o and Moriah/HURCS.

It intentionally excludes temporary job IDs, current campaign paths, one-off debugging incidents, and general scientific policy.

Troubleshooting notes should be added only when an issue is repeatable enough to be useful beyond one incident.

## 1. General remote-work principles

- Use remote systems for the work they are intended to perform; do not move large canonical data locally without a scientific/operational need.
- Keep tracked production code synchronized through Git whenever practical.
- Preserve compact provenance/evidence for substantial runs.
- Follow `docs/repo_policy.md` for local evidence destinations and what may be committed.

For tracked production code, `docs/repo_policy.md` is the canonical source for the Git-sync path.

Direct transfer of dirty tracked production files to a remote host is a narrow diagnostic exception. If used, keep it explicit, narrow, reversible, and restore canonical Git state before scientific closure.

## 2. Moriah / HURCS

### 2.1 Canonical project/runtime base

```text
/sci/labs/efratmorin/omripo/Flash-NH
```

The Git clone and the Flash-NH runtime/data base are conceptually distinct. Do not assume every runtime artifact belongs inside the tracked clone.

### 2.2 Login-node discipline

Do not run resource-intensive work on login nodes, including:

- model training or substantial inference/evaluation;
- heavy Python imports/computation;
- installs/package builds;
- large transfers;
- long-running jobs;
- recursive/heavy filesystem scans.

Lightweight operations are acceptable when genuinely cheap:

- navigation and small text inspection;
- `git status`, `git pull`, `git log`;
- `sbatch`, `squeue`, `sacct`;
- brief environment/version checks.

Use Slurm for compute/resource-intensive work.

Prefer job arrays when many equivalent jobs are required rather than large manual submission loops.

Project environments/runtime data belong on lab/project storage, not home storage.

### 2.3 SSH login-shell behavior

Non-login remote command execution may not expose the expected Slurm PATH.

When invoking Moriah Slurm commands remotely over SSH, use a login-shell wrapper when needed, e.g.:

```bash
ssh moriah 'bash -lc "sbatch ..."'
```

If a direct remote command cannot find `sbatch` or another expected environment tool, verify login-shell initialization before assuming paths/installations changed.

### 2.4 File transfer

Moriah SCP requires legacy mode because the default SFTP subsystem is unavailable in this environment.

Use:

```bash
scp -O <source> moriah:<destination>
scp -O -r <source_dir> moriah:<destination_dir>
```

and the same `-O` mode for Moriah -> local transfers.

### 2.5 PowerShell-to-SSH quoting caution

When constructing SSH commands inside PowerShell, be careful with PowerShell interpolation, especially `$()` inside double-quoted strings.

PowerShell may evaluate command substitutions locally before the command reaches Moriah.

Prefer quoting/command construction that makes the remote-shell evaluation boundary explicit, and inspect the final command when provenance-sensitive values (for example Git SHAs) are involved.

This is a troubleshooting/command-construction caution, not a scientific rule.

## 3. h2o

### 3.1 Known conda bootstrap

Use:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1
```

Do not rediscover the conda installation unless this documented path actually fails.

### 3.2 Data/evidence discipline

Large canonical/generated products should normally remain on h2o.

Transfer only the compact evidence needed for local scientific inspection unless the task explicitly requires larger data.

Follow `docs/repo_policy.md` for local evidence destinations.

## 4. Windows/local SSH aliases

Known OpenSSH aliases used by the project:

```text
flashnh-h2o
moriah
```

Do not embed passwords or credentials in repo files/scripts. SSH configuration and private keys remain machine-local.

Moriah file transfer uses `scp -O` even when the host alias is used.

## 5. Known troubleshooting notes (not universal rules)

### 5.1 Deep Windows paths and ephemeral environments

Very deep temporary paths can break some package installs on Windows due to path-length limitations.

If an ephemeral local environment install fails with path-length symptoms, prefer a short project-local or explicitly approved path rather than redesigning the project environment.

Do not generalize one package-install failure into a requirement that every environment use a short global path.

### 5.2 Agent/harness scheduling

Agent-specific scheduling/wakeup behavior is not a Flash-NH infrastructure guarantee.

Do not promise unattended continuation based solely on a harness wakeup mechanism unless it has been verified for the current agent/tooling. When reliable bounded continuation matters, design the task around observable job state and explicit continuation logic.

## 6. Items intentionally not codified as standing rules

The following should not be treated as permanent operational requirements unless re-verified:

- a blanket claim that h2o file transfer must use heredoc paste;
- one-off temporary remote dirty-code workarounds;
- campaign-specific Slurm partitions/resources;
- temporary job IDs or run directories.
