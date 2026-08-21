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

Attempt 1 is preserved as **invalid preflight commit-guard evidence**: its
three jobs reached an L4 node but the historical launcher resolved a stale
clone and refused before configuration generation, CUDA, data loading, model
initialization, or training. It is not evidence against any batch size.

Attempt 2 is a separately reviewed combined sequential retry. It reserves one
Catfish L4 allocation and invokes the reviewed single-batch launcher for
`128`, `256`, and `512` in that deterministic order. Each invocation is a
fresh shell/Python process, generates separate configs/checksums and evidence,
and may fail without preventing later batch attempts. The combined summary
returns nonzero for any individual failure; it contains no scientific metric,
comparison, or ranking.

After Attempt 2 is committed, reviewed, pushed, and synchronized to Moriah,
the separately authorized single submission form is:

```bash
EXPECTED_COMMIT=<reviewed_attempt2_commit> sbatch --export=ALL,EXPECTED_COMMIT \
  scripts/run_phase_b_batch_size_operational_retry_moriah.sbatch
```

The Attempt-2 evidence root is
`/sci/labs/efratmorin/omripo/Flash-NH/evidence/phase_b_batch_size_operational_qualification_attempt2_combined_v001/`;
it is distinct from the immutable Attempt-1 root
`/sci/labs/efratmorin/omripo/Flash-NH/evidence/phase_b_batch_size_operational_qualification_only/`.

## Closure

Attempt 1 (45901431/2/3) was invalid: a stale clone failed before config/CUDA.
Attempt 2 (45904704) was invalid: ambient Python lacked yaml/torch and the
aggregate hit a readonly-variable defect. CPU preflight 45904829 passed exact
runtime imports and all three 50k/8 configs without training. Attempt 3
(45904830) validly passed batch 128, then timed out while 256 loaded data;
512 was not attempted. Attempt 4 validly passed 256 (45904976) and 512
(45904977). Their Slurm 127 occurred only after PASS, from optional post-run
`nvidia-smi`; the launcher now records that diagnostic absence as a warning.

**[DECIDED FOR SWEEP V1]** `{128,256,512}` is operationally qualified under
the reviewed H256/L4 eight-update envelope: PT, sequence 72, lead 6h,
`[128,32]` tanh static embedding, output dropout 0.25, intended 50,000 update
cap, and one-epoch/eight-update smoke. This is not a scientific ranking,
throughput result, long-run stability claim, or final Sweep-v1 performance.
