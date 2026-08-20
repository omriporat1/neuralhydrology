# Phase-B Track A — Batch-Size Operational Qualification

Status: preparation only. **OPERATIONAL QUALIFICATION ONLY — NOT SCIENTIFIC PERFORMANCE EVIDENCE.**

The provisional Sweep-v1 batch-size values `128`, `256`, and `512` must each
complete an independent Moriah L4 smoke at `hidden_size=256` before they are
used in scientific HPO. The prepared workflow generates both an intended
Sweep-v1 configuration with `max_updates_per_epoch=50000` and a separately
labeled smoke configuration with one epoch and eight optimizer updates. The
smoke does not validate, calculate NSE/KGE, initialize W&B, or select a
batch-size winner.

Each submitted job uses the real Flash-NH PT / lead-6 / sequence-72 / learned
`[128,32]` static-embedding / CudaLSTM configuration and the validated
development-only basin membership. It runs only through Slurm with one L4
GPU. PASS means `neuralhydrology.nh_run.start_run` completes and records eight
training updates without an exception; FAIL records the error excerpt. The
evidence includes the reviewed commit, both generated configs and their
hashes, Slurm identity, CUDA device, peak allocated GPU memory, elapsed time,
and explicit PASS/FAIL.

After this preparation is committed, reviewed, pushed, and synchronized to
Moriah, the separately authorized submission form is:

```bash
EXPECTED_COMMIT=<reviewed_commit> sbatch --export=ALL,EXPECTED_COMMIT \
  scripts/run_phase_b_batch_size_operational_qualification_moriah.sbatch 128
```

Repeat once for `256` and `512`; do not use arrays until separately reviewed.
The default remote evidence root is
`/sci/labs/efratmorin/omripo/Flash-NH/evidence/phase_b_batch_size_operational_qualification_only/`.
