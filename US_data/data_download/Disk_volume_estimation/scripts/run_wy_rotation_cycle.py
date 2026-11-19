#!/usr/bin/env python
"""Run ONE water-year rotation run (a campaign cycle or a canary), score-blind:
NeuralHydrology training of the sealed cycle config, then NH evaluation of the
checkpoint year (NH ``validation`` period) for EVERY epoch over all 2,307
development basins with no metrics, then a receipts/keys/shapes verification.

It computes, prints and stores no skill score: no in-loop validation
(``validate_every: null``), ``metrics=[]`` at evaluation, no metrics key. The raw ``validation_results.p`` per epoch is left for the later,
separately approved checkpoint rule. The assessment year (NH ``test``) is never
evaluated here. Intended caller: a Moriah sbatch (the canary's is
``.scratch_local/wy_rotation_runs_v001/run_wy_rotation_canary_moriah.sbatch``).

W&B (optional, ``--wandb-policy``): stage A of the adopted design only
(``src/baseline/wy_rotation_tracking.py``): NH's per-epoch training loss, epoch
and evaluation timing, peak memory, receipt status and the explained-missing
basin list, plus the small evidence files. No score of any kind. A W&B
failure degrades tracking (recorded under ``receipt["wandb"]``) and never stops
the run. Without the flag nothing is tracked.

Evidence written to the config dir: ``wy_rotation_run_receipt.json``
(STARTED -> COMPLETE/FAILED; provenance, Slurm ids, timing, peak memory) and
``wy_rotation_score_blind_evidence.json`` (verify_score_blind_run).

Exclusive lock: the runner's first write is an atomic ``mkdir`` of
``<config_dir>/wy_rotation_run.lock`` (owner: Slurm job id, host, pid, time),
removed on exit, on the fresh and the resume path alike. A second runner on the
same config dir is refused with the owner named. A lock left by a SIGKILLed job
is never removed automatically: check that the job is dead, then remove it by hand.

Repeats are refused, never silently retrained or overwritten. A run starts only
on a config dir with no receipt, no evidence, no receipt archive and no
non-empty ``runs/`` or ``wandb/``; otherwise the runner exits non-zero and
leaves nothing behind (its lock is removed again). A COMPLETE run is never
rerun (a new run needs a new attempt id and manifest).

The one opt-in exception is ``--resume-from-job JOBID``: it re-evaluates an
already fully trained run whose receipt is FAILED or STARTED and records
``slurm_job_id == JOBID`` (``none`` when the earlier job had none). Same run id
and manifest bytes, exactly one NH run dir with the flat 1..N checkpoint set;
it never trains. The earlier receipt/evidence are moved to
``<name>.job<JOBID>.json`` first and the new receipt records
``resumed_from_job``, ``previous_receipt_sha256`` and ``previous_git_commit``.
Which epochs are re-evaluated is decided in ``_resume_evaluation_policy`` and
recorded as ``resume_evaluation_policy``: on the earlier job's commit, existing
pickles that load are kept (one that does not load, e.g. truncated by a kill
mid-write, is deleted and re-evaluated); on another commit every pickle is
deleted and every epoch re-evaluated. ``resume_reevaluated_epochs`` gives each
re-evaluated epoch and why. The lock refuses a resume while the earlier job
still holds the config dir (only a runner that takes this lock holds it; older
runners took none). The runner cannot otherwise tell whether that job is alive,
so naming the job stays the operator's confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline import pilot_orchestration as po  # noqa: E402
from src.baseline import wy_rotation_runs as rot  # noqa: E402
from src.baseline.wy_rotation_split import load_split_contract, verify_basin_population  # noqa: E402

_CANONICAL_MORIAH_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
RECEIPT_NAME = "wy_rotation_run_receipt.json"
EVIDENCE_NAME = "wy_rotation_score_blind_evidence.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", type=Path, required=True, help="campaign_manifest.json or canary_manifest.json")
    p.add_argument("--index", type=int, default=None, help="campaign row 0..4; default $SLURM_ARRAY_TASK_ID; "
                                                           "ignored for a canary manifest")
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--expected-runtime-python", default=_CANONICAL_MORIAH_PYTHON)
    p.add_argument("--wandb-policy", type=Path, default=None,
                   help="W&B tracking policy YAML (e.g. config/wy_rotation_wandb_policy_v001.yaml); "
                        "default: no tracking")
    p.add_argument("--resume-from-job", default=None, metavar="JOBID",
                   help="re-evaluate (never retrain) the fully trained run of the FAILED/STARTED receipt written by "
                        "Slurm job JOBID ('none' if that job had no Slurm id); default: refuse any existing run")
    return p


class RunRefused(SystemExit):
    """An operator-facing refusal raised before anything is written."""

    def __init__(self, message: str):
        super().__init__(f"refusing to run: {message}")


_RESUMABLE_STATUSES = ("FAILED", "STARTED")


def _archive_name(name: str, job: str) -> str:
    stem, suffix = name.rsplit(".", 1)
    return f"{stem}.job{job}.{suffix}"


def _existing_run_artifacts(config_dir: Path) -> list[str]:
    """What an earlier attempt at this run left in its config dir."""
    found = [name for name in (RECEIPT_NAME, EVIDENCE_NAME) if (config_dir / name).exists()]
    for name in (RECEIPT_NAME, EVIDENCE_NAME):
        found += sorted(p.name for p in config_dir.glob(_archive_name(name, "*")))
    for sub in ("runs", "wandb"):
        path = config_dir / sub
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            found.append(f"{sub}/")
    return found


def _receipt_archives(config_dir: Path) -> list[dict]:
    """The archived receipts in ``config_dir``: job id (from the file name), file name,
    and the status / resumed_from_job read from the file (None when unreadable)."""
    stem, suffix = RECEIPT_NAME.rsplit(".", 1)
    out = []
    for path in sorted(config_dir.glob(_archive_name(RECEIPT_NAME, "*"))):
        job = path.name[len(stem) + len(".job"):-len("." + suffix)]
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            content = None
        if not isinstance(content, dict):
            content = {}
        out.append({"job": job, "name": path.name, "status": content.get("status"),
                    "resumed_from_job": (None if content.get("resumed_from_job") is None
                                         else str(content["resumed_from_job"]))})
    return out


def _archive_restore_hint(config_dir: Path, job: Optional[str] = None) -> str:
    """A refusal hint that names the archived receipt to restore, or "" without archives.

    A resume archives the earlier job's receipt as ``<stem>.job<JOB>.json`` before it
    writes its own; killed in between, it leaves no receipt. The archive to restore is
    the head of the resume chain: the one archive no other archive names as its
    resumed_from_job. ``job`` (the --resume-from-job argument) is chosen only when it
    is that head; an older job of the chain is named as stale and the head is chosen
    instead. When the head is not unique, every archive is listed and none chosen."""
    archives = _receipt_archives(config_dir)
    if not archives:
        return ""
    listing = ", ".join(f"{a['name']} (job {a['job']}, status {a['status'] or 'unreadable'})" for a in archives)
    referenced = {a["resumed_from_job"] for a in archives if a["resumed_from_job"] is not None}
    heads = [a for a in archives if a["job"] not in referenced]
    chosen = heads[0] if len(heads) == 1 else None
    stale = ""
    if job is not None and any(a["job"] == job for a in archives) and (chosen is None or chosen["job"] != job):
        stale = (f" Job {job} is not the head of the resume chain"
                 + (f"; the head is job {chosen['job']}." if chosen is not None else "."))
    if chosen is None:
        return (f". Archived receipts here: {listing}.{stale} No single one is the latest of a resume chain: "
                "check with sacct which job last ran for this run before restoring any by hand")
    evidence = config_dir / _archive_name(EVIDENCE_NAME, chosen["job"])
    evidence_part = (f" and {evidence.name} back to {EVIDENCE_NAME}" if evidence.exists()
                     else f" (there is no {evidence.name})")
    return (f". Archived receipts here: {listing}.{stale} If a resume was killed after archiving one and before "
            f"writing its own receipt, check that no job for this run is alive (sacct/squeue), then restore by "
            f"hand: rename {chosen['name']} back to {RECEIPT_NAME}{evidence_part}, and rerun with "
            f"--resume-from-job {chosen['job']}")


def _refuse_existing_run(config_dir: Path, run_id: str) -> None:
    found = _existing_run_artifacts(config_dir)
    if not found:
        return
    receipt_path = config_dir / RECEIPT_NAME
    status, job = None, None
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        status, job = previous.get("status"), previous.get("slurm_job_id")
    if status == "COMPLETE":
        hint = "it is COMPLETE; a rerun needs a new attempt id and manifest"
    elif status in _RESUMABLE_STATUSES:
        hint = (f"its receipt is {status} (job {'none' if job is None else job}); if training finished, "
                f"re-evaluate with --resume-from-job {'none' if job is None else job}, otherwise a retrain "
                "needs a new attempt id and manifest")
    else:
        hint = "inspect it by hand; nothing was written" + _archive_restore_hint(config_dir)
    raise RunRefused(f"{run_id} already has {', '.join(found)} in {config_dir}: {hint}")


def _check_resume(config_dir: Path, *, run_id: str, manifest_sha256: str, epochs: list, job: str) -> dict:
    """Every check of --resume-from-job, before anything is moved or written."""
    if not job.replace("_", "").isalnum():
        raise RunRefused(f"--resume-from-job {job!r} is not a Slurm job id (or 'none')")
    receipt_path = config_dir / RECEIPT_NAME
    if not receipt_path.is_file():
        # e.g. a resume killed after archiving the receipt, before writing its own
        hint = _archive_restore_hint(config_dir, job)
        raise RunRefused(f"--resume-from-job {job}: no {RECEIPT_NAME} in {config_dir}{hint}")
    previous_bytes = receipt_path.read_bytes()
    previous = json.loads(previous_bytes.decode("utf-8"))
    status = previous.get("status")
    if status not in _RESUMABLE_STATUSES:
        raise RunRefused(f"--resume-from-job {job}: the receipt is {status}, only {'/'.join(_RESUMABLE_STATUSES)} "
                         "can be resumed (a COMPLETE run is never rerun)")
    recorded = "none" if previous.get("slurm_job_id") is None else str(previous["slurm_job_id"])
    if recorded != job:
        raise RunRefused(f"--resume-from-job {job}: the receipt was written by job {recorded}")
    if previous.get("run_id") != run_id or previous.get("manifest_sha256") != manifest_sha256:
        raise RunRefused(f"--resume-from-job {job}: the receipt's run id/manifest sha256 "
                         f"({previous.get('run_id')}, {previous.get('manifest_sha256')}) are not this run's")
    nh_run_dir = po._try_discover_nh_run_dir(config_dir, run_id)
    if nh_run_dir is None:
        raise RunRefused(f"--resume-from-job {job}: no single NH run dir of {run_id} under {config_dir / 'runs'}; "
                         "a resume never trains (a retrain needs a new attempt id and manifest)")
    checkpoints = po.discover_physical_checkpoints(nh_run_dir)
    if sorted(checkpoints) != epochs or any(c.owning_run_dir != nh_run_dir for c in checkpoints.values()):
        raise RunRefused(f"--resume-from-job {job}: checkpoint set {sorted(checkpoints)} under {nh_run_dir} is not "
                         f"the flat 1..{len(epochs)} set of a completed training; a resume never trains")
    for name in (RECEIPT_NAME, EVIDENCE_NAME):
        source, archive = config_dir / name, config_dir / _archive_name(name, job)
        if source.is_file() and archive.exists() and archive.read_bytes() != source.read_bytes():
            raise RunRefused(f"--resume-from-job {job}: a different {archive.name} already exists")
    return {"resumed_from_job": job, "previous_status": status,
            "previous_receipt_sha256": hashlib.sha256(previous_bytes).hexdigest(),
            "previous_git_commit": previous.get("git_commit")}, nh_run_dir


def _resume_evaluation_policy(previous_git_commit, git_commit) -> dict:
    """The ONE place that decides which epochs a resume re-evaluates. The policy is the
    user's decision, docs/decision_log.md entry 2026-11-18 "Rotation runs: no Slurm
    re-queue, resume across a code change, ...". To change it, swap it here (e.g. raise
    RunRefused to refuse a cross-commit resume instead).

    Same commit as the earlier job: keep every existing pickle that loads, evaluate
    the missing and unloadable ones. Different commit, or an earlier receipt with no
    git_commit (counted as different): re-evaluate every epoch, so all pickles of
    the run come from one commit."""
    if previous_git_commit and previous_git_commit == git_commit:
        return {"mode": "reuse_loadable", "previous_git_commit": previous_git_commit, "git_commit": git_commit,
                "reason": "same git commit as the earlier job: only missing or unloadable pickles are evaluated"}
    return {"mode": "reevaluate_all", "previous_git_commit": previous_git_commit, "git_commit": git_commit,
            "reason": "the git commit differs from the earlier job's (or it was not recorded): every epoch is "
                      "re-evaluated so all pickles of the run come from one commit"}


def _resume_reevaluated_epochs(resume: dict, records: list) -> dict:
    """Epoch -> why this resume evaluated it. Epochs whose existing pickle was kept are
    not listed. Only what this job saw is claimed: an earlier job (e.g. a killed resume)
    may already have deleted a pickle, so "none present" never means "never written"."""
    if resume["resume_evaluation_policy"]["mode"] == "reevaluate_all":
        # every epoch, whatever was on disk; which pickles this job found and deleted
        # is deleted_pickle_epochs in resume_evaluation_policy
        return {str(int(rec["epoch"])): "re-evaluated: commit differs, reevaluate_all"
                for rec in records if not rec.get("skipped_existing")}
    out = {}
    for rec in records:
        if rec.get("skipped_existing"):
            continue
        epoch = int(rec["epoch"])
        if rec.get("redone_existing"):
            out[str(epoch)] = f"existing pickle {rec['redone_existing']}"
        else:
            out[str(epoch)] = "no pickle present when this resume started"
    return out


LOCK_NAME = "wy_rotation_run.lock"
_LOCK_OWNER = "owner.json"


def _acquire_run_lock(config_dir: Path) -> dict:
    """Take this run's exclusive lock: an atomic ``mkdir`` of ``config_dir/LOCK_NAME``
    (atomic on Moriah's shared FS too, like the campaign sbatch's own row lock in
    ``campaign/``, a different path). It is the runner's first write, on fresh and resume
    paths alike, and is removed when the runner exits. A runner killed with SIGKILL
    leaves it behind on purpose: nobody can tell from here that the job is dead."""
    import uuid

    path = config_dir / LOCK_NAME
    owner = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "pid": os.getpid(),
             "acquired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "token": uuid.uuid4().hex}
    try:
        path.mkdir()
    except FileExistsError:
        try:
            held = json.loads((path / _LOCK_OWNER).read_text(encoding="utf-8"))
            if not isinstance(held, dict):
                raise ValueError("owner record is not a JSON object")
            who = (f"job {held.get('slurm_job_id') or 'none'} on host {held.get('host')} (pid {held.get('pid')}), "
                   f"since {held.get('acquired_utc')}")
        except (OSError, ValueError):
            who = "an owner whose record is missing or unreadable"
        raise RunRefused(f"{path} exists: another runner holds this config dir: {who}. If that job is still "
                         "running, let it finish. A lock left by a job killed with SIGKILL (or a lost node) is "
                         f"never removed automatically: check that the job is dead (sacct/squeue), then remove "
                         f"{path} by hand") from None
    # written aside, then renamed: a reader never sees half-written JSON. If the write
    # fails (quota, I/O), take the lock back down before raising: it is outside main's
    # try/finally, and an empty lock dir would block every later runner.
    tmp = path / (_LOCK_OWNER + ".tmp")
    try:
        tmp.write_text(json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path / _LOCK_OWNER)
    except BaseException:
        for leftover in (tmp, path / _LOCK_OWNER):
            try:
                leftover.unlink()
            except OSError:
                pass
        try:
            path.rmdir()
        except OSError:
            pass
        raise
    return owner


def _release_run_lock(config_dir: Path, owner: dict) -> None:
    """Remove the lock if it is still this runner's (same token); otherwise leave it.
    Never raises: it runs in main's ``finally`` and must not replace the run's own
    exception. A lock it cannot remove, or whose owner record is missing, unreadable or
    not a JSON object (malformed), is left in place with a warning."""
    path = config_dir / LOCK_NAME
    try:
        held = json.loads((path / _LOCK_OWNER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        held = None
    if not isinstance(held, dict) or held.get("token") != owner["token"]:
        print(f"warning: {path} is not this runner's lock any more; left in place", file=sys.stderr)
        return
    try:
        (path / _LOCK_OWNER).unlink()
        path.rmdir()
    except OSError as exc:
        print(f"warning: could not remove {path} ({exc!r}); left in place: check that this job has ended, "
              "then remove it by hand", file=sys.stderr)


def _archive_previous(config_dir: Path, job: str) -> dict:
    """Move the earlier job's receipt/evidence aside (checked by _check_resume)."""
    archived = {}
    for name in (RECEIPT_NAME, EVIDENCE_NAME):
        source = config_dir / name
        if source.is_file():
            os.replace(source, config_dir / _archive_name(name, job))
            archived[name] = _archive_name(name, job)
    return archived


def _write_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    # On Windows a scanner can briefly hold the just-written target, and the atomic
    # replace then fails with "Access is denied"; a few short retries cover that.
    # Linux (Moriah) never takes this branch, and a lasting error is still raised.
    for attempt in range(_REPLACE_TRIES):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


_REPLACE_TRIES = 6


def _defect_marker(exc: BaseException) -> dict:
    """Omri, 2026-11-19 (gate-4 Q2): a ``PermissionError`` anywhere in the failure's
    exception chain is not a retriable environment fault. It may mean another job
    owns this run directory, so the FAILED receipt says the run stopped as a defect
    and must be inspected before anyone resumes it (``--resume-from-job`` stays an
    explicit operator act). The bounded Windows-only replace retry in
    :func:`_write_json` is separate: it never runs on Moriah, and a lasting error is
    still raised and lands here."""
    seen, cur = set(), exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, PermissionError):
            return {"stopped_as_defect": f"{cur!r}: another job may own this run directory; inspect it "
                                         "(sacct/squeue, file owners) before any resume"}
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return {}


def _peak_memory() -> dict:
    out = {}
    try:
        import resource

        out["max_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            out["cuda_max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    return out


def _tester_factory(nh_run_dir: Path):
    from neuralhydrology.utils.logging_utils import setup_logging

    log_path = nh_run_dir / "output.log"
    if not po.root_logger_has_file_handler(log_path):
        setup_logging(str(log_path))
    return rot._default_tester_factory(nh_run_dir)


def _start_tracker(policy, *, row, campaign_row, config, campaign, manifest_sha256, provenance):
    from src.baseline import wy_rotation_tracking as trk

    layout = trk.run_layout(cycle=int(row["cycle"]), config=config, is_canary=bool(row.get("is_canary")))
    wandb_config = trk.run_config(row=row, campaign_row=campaign_row, config=config, campaign=campaign,
                                  manifest_sha256=manifest_sha256, provenance=provenance,
                                  slurm=trk.slurm_ids(os.environ))
    return trk.init_rotation_tracker(policy, flashnh_run_id=row["run_id"], config=wandb_config, layout=layout,
                                     wandb_dir=Path(row["config_dir"]) / "wandb")


def _track(tracker, operation: str, fn) -> None:
    """Run one tracker call on the runner's main path; a tracking problem only
    degrades tracking and never fails the scientific run."""
    if tracker is None:
        return
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        from src.baseline.wandb_tracking import _mark_degraded

        _mark_degraded(tracker.run, operation, exc)


def _close_tracker(tracker, receipt, receipt_path, *, evidence=None, files=(), failed: bool) -> None:
    """Summary + small files + finish; records the tracking state in the receipt.
    Never lets a tracking problem replace the run's own outcome."""
    try:
        tracker.log_receipt_summary(receipt, evidence)
        receipt["wandb"] = tracker.receipt_record()
        _write_json(receipt_path, receipt)
        for path in (*files, receipt_path):
            tracker.save_file(path)
        tracker.finish(failed=failed)
    except Exception as exc:  # noqa: BLE001
        receipt.setdefault("wandb", {})["close_error"] = repr(exc)
    receipt["wandb"] = {**receipt.get("wandb", {}), **tracker.receipt_record()}
    _write_json(receipt_path, receipt)


def _finish_unfinished_tracker(tracker, receipt, receipt_path, *, failed: bool) -> None:
    """The runner's ``finally``: a W&B run that no earlier step finished (a
    signal or error inside _close_tracker, or a close that failed before
    finish) is finished here. The receipt's status is never changed here. A
    signal arriving during this finish still propagates after the receipt
    records it."""
    interrupt = None
    try:
        tracker.finish(failed=failed)
    except BaseException as exc:  # noqa: BLE001
        receipt.setdefault("wandb", {})["finally_finish_error"] = repr(exc)
        interrupt = None if isinstance(exc, Exception) else exc
    try:
        receipt["wandb"] = {**receipt.get("wandb", {}), **tracker.receipt_record(), "finished_in_finally": True}
        _write_json(receipt_path, receipt)
    except Exception:  # noqa: BLE001 -- the receipt on disk already holds the run's outcome
        pass
    if interrupt is not None:
        raise interrupt


def _finish_orphan_wandb_run() -> None:
    """A signal inside wandb.init after the run started leaves no tracker:
    finish the process's active W&B run as failed (no-op without one)."""
    wandb = sys.modules.get("wandb")
    run = getattr(wandb, "run", None) if wandb is not None else None
    if run is not None:
        try:
            run.finish(exit_code=1)
        except Exception:  # noqa: BLE001 -- best effort; W&B's own exit hook remains
            pass


def _raise_on_sigterm(signum, frame):
    # Slurm sends SIGTERM at the time limit / on scancel: turn it into an
    # exception so the receipt records FAILED instead of staying STARTED.
    raise SystemExit(f"terminated by signal {signum}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    from src.baseline.sweep_v1_runtime_contract import RuntimeContractError, verify_commit_and_interpreter

    try:
        provenance = verify_commit_and_interpreter(repo_root=ROOT, expected_commit=args.expected_commit,
                                                   expected_runtime_python=args.expected_runtime_python)
    except RuntimeContractError as exc:
        raise SystemExit(f"runtime provenance check failed, refusing to run: {exc}") from exc

    policy = None
    if args.wandb_policy is not None:
        from src.baseline.wandb_tracking import load_tracking_policy

        policy = load_tracking_policy(args.wandb_policy)  # a bad policy file is an operator error: fail now

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("is_canary"):
        checked = rot.verify_canary_manifest(manifest)
        campaign = checked["campaign"]
        campaign_row = manifest["campaign_row"]
    else:
        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        checked = rot.verify_campaign_row(manifest, index)
        campaign = manifest
        campaign_row = checked["row"]
    config_dir = Path(checked["row"]["config_dir"])
    epochs = list(range(1, int(checked["config"]["epochs"]) + 1))
    manifest_sha256 = rot.sha256_file(args.manifest)
    lock = _acquire_run_lock(config_dir)  # the first write: from here on this process owns the config dir
    try:
        return _run_locked(args, provenance=provenance, policy=policy, checked=checked, campaign=campaign,
                           campaign_row=campaign_row, epochs=epochs, manifest_sha256=manifest_sha256)
    finally:
        _release_run_lock(config_dir, lock)


def _run_locked(args, *, provenance, policy, checked, campaign, campaign_row, epochs, manifest_sha256) -> int:
    row, config = checked["row"], checked["config"]
    config_dir = Path(row["config_dir"])
    receipt_path = config_dir / RECEIPT_NAME
    resume = None
    if args.resume_from_job is None:
        _refuse_existing_run(config_dir, row["run_id"])
    else:
        resume, resume_run_dir = _check_resume(config_dir, run_id=row["run_id"], manifest_sha256=manifest_sha256,
                                               epochs=epochs, job=str(args.resume_from_job))
        decision = _resume_evaluation_policy(resume["previous_git_commit"], provenance.get("git_commit"))
        if decision["mode"] == "reevaluate_all":
            # before the earlier receipt is archived: a kill in between leaves that receipt (and its
            # commit) in place, so the next resume decides the same way and no pickle mix survives
            decision["deleted_pickle_epochs"] = rot.delete_results_pickles(resume_run_dir, epochs)
            decision["deleted_pickle_epochs_meaning"] = (
                "pickles present when this job started and deleted by this job (an earlier, killed resume "
                "may have deleted others)")
        resume["resume_evaluation_policy"] = decision
        resume["archived"] = _archive_previous(config_dir, resume["resumed_from_job"])

    receipt = {"status": "STARTED", "run_id": row["run_id"], "cycle": row["cycle"],
               "is_canary": bool(row.get("is_canary")), "epochs": len(epochs),
               "max_updates_per_epoch": config["max_updates_per_epoch"], "started_utc": time.time(),
               "manifest": str(args.manifest), "manifest_sha256": manifest_sha256, **provenance,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
               "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "host": socket.gethostname(),
               **(resume or {})}
    _write_json(receipt_path, receipt)
    tracker = None
    evidence = None
    completed = False
    try:
        if policy is not None:  # inside the try: a signal or error during wandb.init still records FAILED
            tracker = _start_tracker(policy, row=row, campaign_row=campaign_row, config=config, campaign=campaign,
                                     manifest_sha256=receipt["manifest_sha256"], provenance=provenance)
            receipt["wandb"] = tracker.receipt_record()
            _write_json(receipt_path, receipt)
        contract = load_split_contract(campaign["split_contract"]["path"])
        population = verify_basin_population(contract, ROOT)

        if resume is not None:  # checked before anything was written: one NH run dir, fully trained
            nh_run_dir = po.discover_nh_run_dir(config_dir, row["run_id"])
            receipt["trained_in_this_job"] = False
        else:  # a fresh config dir (_refuse_existing_run): train
            t0 = time.monotonic()
            if tracker is not None:
                tracker.attach_live_loss_handler()
            try:
                po.default_train_chunk(po.TrainChunkRequest(
                    is_first_chunk=True, config_path=checked["config_path"], nh_run_dir=None, current_epoch=None,
                    logical_target_epoch=len(epochs), additional_epochs=len(epochs)))
            finally:
                if tracker is not None:
                    tracker.detach_live_loss_handler()
            receipt["train_wall_s"] = round(time.monotonic() - t0, 1)
            receipt["train_peak_memory"] = _peak_memory()
            nh_run_dir = po.discover_nh_run_dir(config_dir, row["run_id"])
            receipt["trained_in_this_job"] = True
        receipt["nh_run_dir"] = str(nh_run_dir)
        checkpoints = po.discover_physical_checkpoints(nh_run_dir)
        if sorted(checkpoints) != epochs or any(c.owning_run_dir != nh_run_dir for c in checkpoints.values()):
            raise rot.WyRotationRunsError(
                f"checkpoint set {sorted(checkpoints)} under {nh_run_dir} is not the flat 1..{len(epochs)} set of a "
                "completed training; continuing an interrupted run needs a separate decision")
        receipt["epoch_end_timestamps"] = rot.log_epoch_timestamps(nh_run_dir / "output.log")
        if sorted(receipt["epoch_end_timestamps"]) != epochs:
            receipt.setdefault("warnings", []).append(
                f"output.log has epoch-end lines for {sorted(receipt['epoch_end_timestamps'])}, expected {epochs}")
        _write_json(receipt_path, receipt)
        _track(tracker, "rotation_backfill", lambda: tracker.backfill_from_output_log(nh_run_dir / "output.log"))

        t0 = time.monotonic()
        receipt["evaluation"] = rot.evaluate_all_epochs(nh_run_dir, epochs, tester_factory=_tester_factory)
        receipt["evaluation_wall_s"] = round(time.monotonic() - t0, 1)
        receipt["evaluation_peak_memory"] = _peak_memory()
        if resume is not None:
            receipt["resume_reevaluated_epochs"] = _resume_reevaluated_epochs(resume, receipt["evaluation"])
        _write_json(receipt_path, receipt)
        _track(tracker, "rotation_evaluation_records", lambda: tracker.log_evaluation_records(receipt["evaluation"]))

        evidence = rot.verify_score_blind_run(nh_run_dir, cycle=row["cycle"], epochs=epochs, population=population)
        _write_json(config_dir / EVIDENCE_NAME, evidence)
        if not evidence["passed"]:
            raise rot.WyRotationRunsError(f"score-blind verification failed: {evidence['problems']}")
        receipt.update(status="COMPLETE", finished_utc=time.time())
        _write_json(receipt_path, receipt)
        completed = True  # from here on, nothing may turn the receipt back into FAILED
        if tracker is not None:
            _close_tracker(tracker, receipt, receipt_path, evidence=evidence, failed=False,
                           files=(checked["config_path"], config_dir / EVIDENCE_NAME))
        print(json.dumps({"run_id": row["run_id"], "status": "COMPLETE", "epochs_evaluated": len(epochs),
                          "n_basins_per_epoch": sorted({e["n_basins"] for e in evidence["per_epoch"].values()}),
                          "n_basins_missing_explained": len(evidence["missing_basins"])}))
        return 0
    except BaseException as exc:  # recorded, then re-raised: never a silent pass
        if completed:  # e.g. SIGTERM while W&B finishes: the scientific outcome stands
            receipt["post_complete_error"] = repr(exc)
            _write_json(receipt_path, receipt)
            raise
        receipt.update(status="FAILED", finished_utc=time.time(), error=repr(exc), traceback=traceback.format_exc(),
                       **_defect_marker(exc))
        _write_json(receipt_path, receipt)
        if tracker is not None and not tracker.run.finished:
            # only this job's evidence is uploaded, never a stale file of an earlier job
            _close_tracker(tracker, receipt, receipt_path, evidence=evidence, failed=True,
                           files=(config_dir / EVIDENCE_NAME,) if evidence is not None else ())
        raise
    finally:  # whatever happened above, W&B tracking is closed exactly once
        if tracker is not None and not tracker.run.finished:
            _finish_unfinished_tracker(tracker, receipt, receipt_path, failed=not completed)
        elif tracker is None and policy is not None:
            _finish_orphan_wandb_run()


if __name__ == "__main__":
    raise SystemExit(main())
