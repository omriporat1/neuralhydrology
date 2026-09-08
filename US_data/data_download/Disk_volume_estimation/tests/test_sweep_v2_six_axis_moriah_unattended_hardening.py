"""Focused, fake-only tests for the two observed Moriah unattended failures.

These exercise the ACTUAL maintained consumer artifacts -- not a helper that
could theoretically reconstruct the needed state:

  Failure A (Slurm client env before ``sbatch``):
      scripts/submit_sweep_v2_six_axis_wandb_agent_moriah.sh
      run under a synthetic non-login-like environment whose ``PATH`` lacks
      ``sbatch`` and whose ``SLURM_CONF`` is unset, with a FAKE ``sbatch``.

  Failure B (W&B CLI credential inside the submitted job):
      the credential-availability fragment of
      scripts/run_sweep_v2_six_axis_wandb_agent_moriah.sbatch, taken verbatim
      between its sentinel markers and run down to the REAL
      ``wandb agent ...`` line with a FAKE ``wandb`` executable and a
      synthetic ``$HOME/.netrc``.

No real Slurm, no real W&B, no network.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SUBMIT_WRAPPER = ROOT / "scripts" / "submit_sweep_v2_six_axis_wandb_agent_moriah.sh"
AGENT_SBATCH = ROOT / "scripts" / "run_sweep_v2_six_axis_wandb_agent_moriah.sbatch"

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash unavailable for the shell-path demonstration")


def _exact_fake_exec_skip(*, env: dict, path: str, command_name: str,
                          direct_ref: "str | None" = None) -> "str | None":
    """Exact-path executable/PATH prerequisite for one POSIX shell-integration test.

    The maintained wrapper / job resolve a *fake* ``sbatch`` / ``wandb`` through a
    specific primitive: an ``[ -x "<dir>/<name>" ]`` guard (``direct_ref``) and,
    after prepending that directory, a bare ``command -v <name>`` / PATH exec.
    Whether a freshly created ``#!`` fake satisfies that depends on the host
    shell's executable semantics for the *exact* path representation and the
    *exact* sanitized environment the test will use -- not on OS name, generic
    ``bash`` presence, a probe file in a different directory/representation, or a
    less-sanitized environment.

    This runs that precise check with the SAME ``bash``, the SAME ``env``, and
    the SAME ``PATH`` string the maintained code will see:

      * when ``direct_ref`` is given: ``test -x "<direct_ref>"`` (the literal
        ``[ -x ... ]`` guard) must hold;
      * ``command -v <command_name>`` must resolve under ``path``; the resolved
        file must itself be ``-x``; and, when ``direct_ref`` is given, must be
        the very same file.

    Returns ``None`` when the primitive is representable here (the test then runs
    normally and any later maintained-code failure is a real failure); otherwise
    a precise capability reason for skipping BEFORE any maintained code runs.
    """
    if BASH is None:
        return "bash unavailable for the shell-path demonstration"
    probe_env = dict(env)
    probe_env["PATH"] = path
    script = (
        'name="$1"; ref="$2"\n'
        'if [ -n "$ref" ]; then test -x "$ref" || exit 11; fi\n'
        'r="$(command -v "$name")" || exit 12\n'
        'test -x "$r" || exit 13\n'
        'if [ -n "$ref" ] && ! [ "$r" -ef "$ref" ]; then exit 14; fi\n'
    )
    res = subprocess.run(
        [BASH, "-c", script, "_", command_name, direct_ref or ""],
        text=True, capture_output=True, check=False, env=probe_env,
    )
    if res.returncode == 0:
        return None
    stage = {
        11: "the `[ -x <fake> ]` guard is false for the normalized fake path",
        12: f"`command -v {command_name}` does not resolve under the sanitized PATH",
        13: f"the resolved `{command_name}` is not `-x`",
        14: f"`command -v {command_name}` resolves to a different file than the fake",
    }.get(res.returncode, f"prerequisite check exited {res.returncode}")
    return (
        f"this shell environment cannot present the normalized synthetic "
        f"{command_name!r} executable with the executable/PATH semantics this "
        f"POSIX integration test requires ({stage}); the maintained wrapper/job "
        f"code has not been run. The platform-independent tests in this module "
        f"still run."
    )

# Recognizable, distinct synthetic key values. The launcher does not police
# key format (W&B is authoritative for that); these are just non-empty markers
# whose exact value must never appear in captured output.
SYNTH_NETRC_KEY = "synthetic-netrc-wandb-key-value-do-not-log"
SYNTH_ENV_KEY = "synthetic-env-wandb-key-value-do-not-log"
assert SYNTH_NETRC_KEY != SYNTH_ENV_KEY


def _make_exe(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    # Some Windows shells ignore the Python chmod above; ask bash to set the bit
    # too. Where the `#!` line already suffices this is a harmless no-op. If the
    # file still cannot be used with the executable/PATH semantics the maintained
    # code needs, the exact-path prerequisite check (`_exact_fake_exec_skip`) in
    # the affected tests skips them before any maintained code runs.
    if BASH is not None:
        subprocess.run(
            [BASH, "-c", 'chmod +x "$1"', "_", str(path)],
            capture_output=True, check=False,
        )


def _sanitized_path() -> str:
    """The PATH handed to the wrapper: usable shell, but no ``sbatch`` anywhere.

    This is exactly the detached / non-login parent's state -- Moriah's
    interactive-only Slurm ``profile.d`` hook never ran, so ``sbatch`` is not
    resolvable until the wrapper restores the configured Slurm bin. (On the
    test hosts a real Slurm client is not installed anyway.)
    """
    keep = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        if (Path(entry) / "sbatch").exists() or (Path(entry) / "sbatch.exe").exists():
            continue
        keep.append(entry)
    return os.pathsep.join(keep)


def _posix(path: Path) -> str:
    """A Windows path expressed the way a POSIX caller's environment holds it.

    ``FLASHNH_MORIAH_SLURM_BIN`` is prepended to ``PATH`` by the wrapper
    (``export PATH="${_slurm_bin}:${PATH}"``). A raw ``C:\\...`` value cannot go
    on ``PATH`` in Git Bash -- the drive-letter colon is read as a ``PATH``
    separator -- so the test converts it here, exactly as the real
    ``/vol/slurm/moriah/bindir/bin`` value already is a plain POSIX path.
    """
    if os.name != "nt":
        return str(path)
    converted = subprocess.run(
        [BASH, "-lc", 'cygpath -u "$1"', "_", str(path)],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert converted and not converted[1:2] == ":", f"cygpath produced a non-POSIX path: {converted!r}"
    return converted


# ---------------------------------------------------------------------------
# Failure A -- the tracked submission boundary restores the Slurm client env.
# ---------------------------------------------------------------------------


def _run_submit(tmp_path: Path, *, provide_fake_sbatch: bool, env_extra: dict | None = None):
    manifest = tmp_path / "production_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    slurm_bin = tmp_path / "slurm_bin"
    slurm_bin.mkdir()
    record = tmp_path / "sbatch_invocation.txt"
    if provide_fake_sbatch:
        _make_exe(
            slurm_bin / "sbatch",
            f'{{\n'
            f'  echo "ARGS: $*"\n'
            f'  echo "SELF: $0"\n'
            f'  echo "SLURM_CONF: ${{SLURM_CONF:-<unset>}}"\n'
            f'  echo "WANDB_API_KEY_PRESENT: ${{WANDB_API_KEY:+yes}}"\n'
            f'}} > "{record}"\n'
            'echo "987654"\n'
            "exit 0\n",
        )

    slurm_conf = tmp_path / "slurm.conf"
    slurm_conf.write_text("ClusterName=fake\n", encoding="utf-8")

    # The fake Slurm bin is NEVER placed on PATH. It is made known to the
    # wrapper only through FLASHNH_MORIAH_SLURM_BIN -- the same override the
    # real /vol/slurm/moriah/bindir/bin restore uses -- so the wrapper's own
    # restoration branch is the only way `sbatch` becomes reachable.
    sanitized_path = _sanitized_path()
    env = {
        "PATH": sanitized_path,
        "HOME": str(tmp_path / "home"),
        "FLASHNH_MORIAH_SLURM_BIN": _posix(slurm_bin),
        "FLASHNH_MORIAH_SLURM_CONF": _posix(slurm_conf),
    }
    if "SYSTEMROOT" in os.environ:  # keep bash/coreutils working on Windows
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    env.pop("SLURM_CONF", None)
    if env_extra:
        env.update(env_extra)

    if provide_fake_sbatch:
        # Exact-path prerequisite: the wrapper's `[ -x "${_slurm_bin}/sbatch" ]`
        # guard and its post-restore `command -v sbatch` must both hold for THIS
        # fake, in THIS env, on the PATH the wrapper builds
        # (`"${_slurm_bin}:${PATH}"`). If not, skip now -- before the wrapper
        # runs -- rather than let a guard the wrapper is right to apply look
        # like a wrapper failure.
        reason = _exact_fake_exec_skip(
            env=env,
            path=env["FLASHNH_MORIAH_SLURM_BIN"] + ":" + sanitized_path,
            command_name="sbatch",
            direct_ref=env["FLASHNH_MORIAH_SLURM_BIN"] + "/sbatch",
        )
        if reason:
            pytest.skip(reason)

    result = subprocess.run(
        [BASH, str(SUBMIT_WRAPPER), str(manifest)],
        cwd=str(tmp_path), text=True, capture_output=True, check=False, env=env,
    )
    return result, record, manifest, slurm_bin, slurm_conf, sanitized_path


def _sbatch_resolves(path_value: str) -> bool:
    """Whether ``sbatch`` resolves from ``path_value`` alone (no overrides)."""
    env = {"PATH": path_value}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return subprocess.run(
        [BASH, "-c", "command -v sbatch"],
        text=True, capture_output=True, check=False, env=env,
    ).returncode == 0


def test_submission_boundary_restores_slurm_env_and_calls_sbatch(tmp_path):
    result, record, manifest, slurm_bin, slurm_conf, sanitized_path = _run_submit(
        tmp_path, provide_fake_sbatch=True
    )

    # Entry precondition: on the PATH actually handed to the wrapper -- and
    # without the FLASHNH_MORIAH_SLURM_BIN override -- `sbatch` does NOT
    # resolve. The fake sbatch is reachable ONLY after the wrapper restores
    # the configured Slurm bin.
    assert not _sbatch_resolves(sanitized_path), (
        "sbatch was already resolvable on the wrapper's entry PATH; the "
        "restoration branch would not be exercised"
    )
    assert (slurm_bin / "sbatch").is_file(), "test fake sbatch missing"

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("987654"), result.stdout
    assert record.is_file(), "the maintained wrapper never reached sbatch"

    captured = record.read_text(encoding="utf-8")
    # The real consumer reached sbatch with the required submission shape.
    assert "--parsable" in captured
    assert f"FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST={manifest}" in captured
    assert "run_sweep_v2_six_axis_wandb_agent_moriah.sbatch" in captured
    # Finding 3: the wrapper no longer injects independent W&B project/entity
    # values -- the submitted job derives them from the strict manifest.
    assert "WANDB_PROJECT=" not in captured
    assert "WANDB_ENTITY=" not in captured
    # ... and it repaired exactly the state a detached parent had lost: the
    # `sbatch` that actually ran is the one reached only via the restored PATH
    # (its own $0 is inside the configured Slurm bin), and SLURM_CONF was
    # exported for it.
    assert "/slurm_bin/sbatch" in captured.replace("\\", "/"), captured
    assert "slurm_bin/sbatch" in result.stderr.replace("\\", "/"), result.stderr
    assert f"SLURM_CONF: {_posix(slurm_conf)}" in captured


def test_submission_boundary_forwards_an_existing_wandb_api_key(tmp_path):
    result, record, *_ = _run_submit(
        tmp_path, provide_fake_sbatch=True, env_extra={"WANDB_API_KEY": SYNTH_ENV_KEY}
    )
    assert result.returncode == 0, result.stderr
    captured = record.read_text(encoding="utf-8")
    # --export=ALL carries an already-configured key through to the job.
    assert "WANDB_API_KEY_PRESENT: yes" in captured
    # The key value itself is never echoed by the wrapper.
    assert SYNTH_ENV_KEY not in result.stdout and SYNTH_ENV_KEY not in result.stderr


def test_submission_boundary_fails_clearly_when_sbatch_is_unreachable(tmp_path):
    result, record, *_ = _run_submit(tmp_path, provide_fake_sbatch=False)
    assert result.returncode != 0
    assert "sbatch" in result.stderr.lower()
    assert not record.is_file()


def test_submission_wrapper_sets_no_independent_wandb_project_or_entity():
    # Finding 3: the wrapper must not carry its own WANDB_PROJECT / WANDB_ENTITY
    # defaults or overrides -- the strict manifest is authoritative and the job
    # resolves them. The wrapper only forwards the caller's env via --export=ALL.
    text = SUBMIT_WRAPPER.read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    # No assignment, default, override, or export of the W&B project/entity.
    for name in ("WANDB_PROJECT", "WANDB_ENTITY"):
        assert f"{name}=" not in code, f"wrapper assigns {name}"
        assert f"export {name}" not in code, f"wrapper exports {name}"
        assert f"${{{name}:-" not in code and f"${{{name}:=" not in code, f"wrapper defaults {name}"
    assert '--export="ALL,FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST=${MANIFEST}"' in code


# ---------------------------------------------------------------------------
# Finding 3 -- the canonical submission path cannot silently use a W&B
#              project/entity that contradicts the validated manifest.
# ---------------------------------------------------------------------------

import importlib.util as _importlib_util  # noqa: E402

_PROD_SEAM_PATH = ROOT / "scripts" / "create_sweep_v2_six_axis_wandb_bridge_production_sweep.py"

_AUTH_BEGIN = "# >>> flashnh wandb project/entity manifest authority"
_AUTH_END = "# <<< flashnh wandb project/entity manifest authority <<<"


def _load_prod_seam():
    spec = _importlib_util.spec_from_file_location("prod_seam_hardening", _PROD_SEAM_PATH)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _authority_fragment() -> str:
    lines = AGENT_SBATCH.read_text(encoding="utf-8").splitlines()
    begin = next(i for i, l in enumerate(lines) if l.startswith(_AUTH_BEGIN))
    end = next(i for i, l in enumerate(lines) if l.startswith(_AUTH_END))
    assert begin < end, "manifest-authority sentinel block not found in order"
    return "\n".join(lines[begin : end + 1]) + "\n"


def _run_authority_fragment(tmp_path, *, manifest_project, manifest_entity, inherited: dict):
    script = "\n".join([
        "set -euo pipefail",
        f'_MANIFEST_PROJECT="{manifest_project}"',
        f'_MANIFEST_ENTITY="{manifest_entity}"',
        _authority_fragment(),
        'echo "REACHED project=${WANDB_PROJECT} entity=${WANDB_ENTITY:-<none>}"',
    ]) + "\n"
    script_path = tmp_path / "authority_harness.sh"
    script_path.write_text(script, encoding="utf-8")
    env = {"PATH": os.environ.get("PATH", "")}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    env.update(inherited)
    return subprocess.run(
        [BASH, str(script_path)], text=True, capture_output=True, check=False, env=env
    )


def test_validate_launch_emit_env_is_the_manifest_authoritative_project_entity(tmp_path):
    # Producer side: the maintained validation/launch seam emits the manifest's
    # OWN wandb_project / wandb_entity (no reconstruction, no default).
    seam = _load_prod_seam()
    manifest_path = tmp_path / "launch_manifest.json"
    seam.write_production_manifest(
        manifest_path=manifest_path, wandb_sweep_id="prodsweep99", expected_commit="a" * 40,
        proposal_order=9, execution_generation=1,
        output_root_base="/sci/labs/efratmorin/omripo/Flash-NH/evidence/hardening_x",
    )
    out = subprocess.run(
        [sys.executable, str(_PROD_SEAM_PATH), "validate-launch",
         "--manifest-path", str(manifest_path), "--emit", "env"],
        cwd=str(ROOT), text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert out.returncode == 0, out.stderr
    emitted = dict(
        line.split("=", 1) for line in out.stdout.splitlines() if "=" in line
    )
    assert emitted["WANDB_SWEEP_ID"] == "prodsweep99"
    assert emitted["WANDB_PROJECT"] == "flashnh-stage1"
    assert emitted["WANDB_ENTITY"] == "omri-porat1-huji"


def test_authority_block_adopts_manifest_project_entity_when_env_is_absent(tmp_path):
    result = _run_authority_fragment(
        tmp_path, manifest_project="flashnh-stage1", manifest_entity="omri-porat1-huji",
        inherited={},
    )
    assert result.returncode == 0, result.stderr
    assert "REACHED project=flashnh-stage1 entity=omri-porat1-huji" in result.stdout


def test_authority_block_accepts_a_matching_inherited_value(tmp_path):
    result = _run_authority_fragment(
        tmp_path, manifest_project="flashnh-stage1", manifest_entity="omri-porat1-huji",
        inherited={"WANDB_PROJECT": "flashnh-stage1", "WANDB_ENTITY": "omri-porat1-huji"},
    )
    assert result.returncode == 0, result.stderr
    assert "REACHED project=flashnh-stage1 entity=omri-porat1-huji" in result.stdout


def test_authority_block_refuses_a_contradicting_inherited_project(tmp_path):
    result = _run_authority_fragment(
        tmp_path, manifest_project="flashnh-stage1", manifest_entity="omri-porat1-huji",
        inherited={"WANDB_PROJECT": "some-other-project"},
    )
    assert result.returncode == 1
    assert "REFUSING" in result.stderr and "contradicts the validated manifest project" in result.stderr
    assert "REACHED" not in result.stdout


def test_authority_block_refuses_a_contradicting_inherited_entity(tmp_path):
    result = _run_authority_fragment(
        tmp_path, manifest_project="flashnh-stage1", manifest_entity="omri-porat1-huji",
        inherited={"WANDB_ENTITY": "some-other-entity"},
    )
    assert result.returncode == 1
    assert "REFUSING" in result.stderr and "contradicts the validated manifest entity" in result.stderr
    assert "REACHED" not in result.stdout


# ---------------------------------------------------------------------------
# Vertical producer -> consumer contract proof (fake-only).
#
# One test that wires the ACTUAL artifacts end to end:
#
#   real submit wrapper
#     -> fake `sbatch` (only emulates the submission boundary: reads the real
#        `--export` spec, applies it, and runs the script it was handed)
#     -> the ACTUAL tracked run_sweep_v2_six_axis_wandb_agent_moriah.sbatch
#     -> the REAL `validate-launch --emit env` (strict manifest loader)
#     -> the ACTUAL shell parse/adopt of sweep id + project + entity
#     -> the credential block (synthetic inherited WANDB_API_KEY)
#     -> fake `wandb` (records argv + env, stops before any agent behaviour)
#
# Nothing here reimplements manifest parsing or emulates Slurm generally, and
# no real Slurm / W&B / network contact occurs.
# ---------------------------------------------------------------------------


def test_vertical_submit_to_agent_uses_manifest_derived_identity(tmp_path):
    seam = _load_prod_seam()

    manifest = tmp_path / "vertical_production_manifest.json"
    seam.write_production_manifest(
        manifest_path=str(manifest),
        wandb_sweep_id="prodsweepvert7",
        expected_commit="b" * 40,
        proposal_order=9,
        execution_generation=1,
        output_root_base="/sci/labs/efratmorin/omripo/Flash-NH/evidence/hardening_vertical",
    )
    manifest_arg = manifest.as_posix()  # absolute for Windows Python AND ok for bash `test -f`

    slurm_bin = tmp_path / "slurm_bin"
    slurm_bin.mkdir()
    wandb_bin = tmp_path / "wandb_bin"
    wandb_bin.mkdir()
    sbatch_log = tmp_path / "fake_sbatch.log"
    wandb_record = tmp_path / "wandb_invocation.txt"
    wandb_calls = tmp_path / "wandb_calls.log"

    # Fake `sbatch`: emulate ONLY the submission boundary. Parse the real
    # wrapper's `--export=ALL,KEY=VALUE` spec, export the KEY=VALUE items, then
    # run the exact script it was told to submit under bash. No manifest logic.
    _make_exe(
        slurm_bin / "sbatch",
        f'script=""\n'
        f'for a in "$@"; do\n'
        f'  case "$a" in\n'
        f'    --parsable) ;;\n'
        f'    --export=*)\n'
        f'      spec="${{a#--export=}}"\n'
        f'      IFS="," read -ra _items <<< "$spec"\n'
        f'      for it in "${{_items[@]}}"; do\n'
        f'        case "$it" in\n'
        f'          ALL) ;;\n'
        f'          *=*) export "${{it%%=*}}"="${{it#*=}}" ;;\n'
        f'        esac\n'
        f'      done\n'
        f'      ;;\n'
        f'    -*) ;;\n'
        f'    *) script="$a" ;;\n'
        f'  esac\n'
        f'done\n'
        f'echo "FAKE_SBATCH_SCRIPT: $script" >> "{sbatch_log}"\n'
        f'echo "FAKE_SBATCH_MANIFEST: ${{FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST:-<unset>}}" >> "{sbatch_log}"\n'
        f'exec bash "$script"\n',
    )

    # Fake `wandb`: record what the real job hands the agent, then stop.
    _make_exe(
        wandb_bin / "wandb",
        f'{{\n'
        f'  echo "ARGS: $*"\n'
        f'  echo "WANDB_PROJECT: ${{WANDB_PROJECT:-<unset>}}"\n'
        f'  echo "WANDB_ENTITY: ${{WANDB_ENTITY:-<unset>}}"\n'
        f'  echo "WANDB_API_KEY_LEN: ${{#WANDB_API_KEY}}"\n'
        f'}} > "{wandb_record}"\n'
        f'echo called >> "{wandb_calls}"\n'
        "exit 0\n",
    )

    home = tmp_path / "home"
    home.mkdir()

    env = {k: v for k, v in os.environ.items() if k != "SLURM_CONF"}
    # sbatch must NOT resolve on entry: the wrapper restores it from
    # FLASHNH_MORIAH_SLURM_BIN. The fake `wandb` dir IS on PATH -- that is the
    # submitted job's own PATH, where `wandb agent` is looked up.
    env["PATH"] = os.pathsep.join([str(wandb_bin), _sanitized_path()])
    env["HOME"] = str(home)
    env["FLASHNH_MORIAH_SLURM_BIN"] = _posix(slurm_bin)
    env["FLASHNH_MORIAH_SLURM_CONF"] = _posix(tmp_path / "slurm.conf")
    (tmp_path / "slurm.conf").write_text("ClusterName=fake\n", encoding="utf-8")
    # The actual .sbatch resolves the repo + interpreter from these overrides
    # rather than the Moriah defaults.
    env["REPO_WORKDIR"] = _posix(ROOT)
    env["CANONICAL_PYTHON"] = _posix(Path(sys.executable))
    # One synthetic inherited key: keep this vertical test about wiring, not
    # about re-testing the .netrc fallback (covered separately above).
    env["WANDB_API_KEY"] = SYNTH_ENV_KEY
    for stale in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_SWEEP_ID",
                  "FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST", "FLASHNH_SWEEP_V2_BRIDGE_SELFTEST"):
        env.pop(stale, None)

    assert not _sbatch_resolves(env["PATH"]), "sbatch resolved on entry; restore branch not exercised"

    # Exact-path prerequisites for BOTH fakes this vertical path relies on,
    # checked before any maintained code runs: the wrapper's fake `sbatch`
    # (`[ -x ]` guard + post-restore `command -v`) and the job's fake `wandb`
    # (`command -v` on the job PATH). Once both hold, every later failure below
    # is a real maintained-code failure, never a skip.
    for reason in (
        _exact_fake_exec_skip(
            env=env,
            path=env["FLASHNH_MORIAH_SLURM_BIN"] + ":" + env["PATH"],
            command_name="sbatch",
            direct_ref=env["FLASHNH_MORIAH_SLURM_BIN"] + "/sbatch",
        ),
        _exact_fake_exec_skip(env=env, path=env["PATH"], command_name="wandb"),
    ):
        if reason:
            pytest.skip(reason)

    result = subprocess.run(
        [BASH, str(SUBMIT_WRAPPER), manifest_arg],
        cwd=str(tmp_path), text=True, capture_output=True, check=False, env=env,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert wandb_record.is_file(), "the actual .sbatch never reached `wandb agent`"

    # The manifest path supplied to the real wrapper reached the actual sbatch consumer.
    assert f"FAKE_SBATCH_MANIFEST: {manifest_arg}" in sbatch_log.read_text(encoding="utf-8")
    assert f"V2_BRIDGE_MANIFEST: {manifest_arg}" in result.stdout

    # The REAL strict validator derived the synthetic sweep id from the manifest.
    assert "VALIDATED_SWEEP_ID: prodsweepvert7" in result.stdout

    rec = wandb_record.read_text(encoding="utf-8")
    # Exactly one agent invocation, and it is the manifest-derived identity.
    assert wandb_calls.read_text(encoding="utf-8").count("called") == 1
    assert rec.splitlines()[0] == "ARGS: agent --count 1 prodsweepvert7", rec
    # Manifest project + entity (present in this fixture) reach the agent env.
    assert "WANDB_PROJECT: flashnh-stage1" in rec, rec
    assert "WANDB_ENTITY: omri-porat1-huji" in rec, rec
    # The credential path succeeded without real W&B, and the value never leaked.
    assert f"WANDB_API_KEY_LEN: {len(SYNTH_ENV_KEY)}" in rec, rec
    for stream in (result.stdout, result.stderr, rec):
        assert SYNTH_ENV_KEY not in stream


# ---------------------------------------------------------------------------
# Failure B -- the submitted job makes a usable W&B key available to the
#              REAL ``wandb agent`` invocation, or fails before it.
# ---------------------------------------------------------------------------

_BEGIN = "# >>> flashnh wandb-agent credential availability"
_END = "# <<< flashnh wandb-agent credential availability <<<"


def _credential_fragment() -> str:
    lines = AGENT_SBATCH.read_text(encoding="utf-8").splitlines()
    begin = next(i for i, l in enumerate(lines) if l.startswith(_BEGIN))
    end = next(i for i, l in enumerate(lines) if l.startswith(_END))
    agent = next(i for i, l in enumerate(lines) if l.startswith("wandb agent "))
    assert begin < end < agent
    return "\n".join(lines[begin : end + 1] + [lines[agent]]) + "\n"


def _run_fragment(tmp_path: Path, *, netrc_key: str | None, env_key: str | None,
                  require_exec: bool = False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "wandb_invocation.txt"
    _make_exe(
        bin_dir / "wandb",
        f'{{\n'
        f'  echo "ARGS: $*"\n'
        f'  echo "KEYLEN: ${{#WANDB_API_KEY}}"\n'
        f'}} > "{record}"\n'
        "exit 0\n",
    )

    home = tmp_path / "home"
    home.mkdir()
    if netrc_key is not None:
        (home / ".netrc").write_text(
            f"machine api.wandb.ai\n  login flashnh\n  password {netrc_key}\n", encoding="utf-8"
        )

    script = (
        "set -euo pipefail\n"
        f'CANONICAL_PYTHON="{sys.executable}"\n'
        'VALIDATED_SWEEP_ID="synthsweep01"\n'
        + _credential_fragment()
    )
    script_path = tmp_path / "fragment_harness.sh"
    script_path.write_text(script, encoding="utf-8")

    env = {
        "PATH": os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "HOMEDRIVE": "",
        "HOMEPATH": "",
    }
    for passthrough in ("SYSTEMROOT", "PATHEXT", "TEMP", "TMP", "PYTHONPATH"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    if env_key is not None:
        env["WANDB_API_KEY"] = env_key

    if require_exec:
        # Exact-path prerequisite for the fragment's terminal
        # `wandb agent --count 1 ...`: the fake `wandb` must resolve and be `-x`
        # on THIS env's PATH. Only the two tests that assert the agent line is
        # actually reached opt in; the "fails before the agent" tests do not
        # (they pass regardless of the fake's executability) and stay
        # unconditional.
        reason = _exact_fake_exec_skip(env=env, path=env["PATH"], command_name="wandb")
        if reason:
            pytest.skip(reason)

    result = subprocess.run(
        [BASH, str(script_path)], cwd=str(tmp_path), text=True, capture_output=True, check=False, env=env
    )
    return result, record


def test_existing_env_key_is_used_verbatim_and_not_logged(tmp_path):
    result, record = _run_fragment(
        tmp_path, netrc_key=SYNTH_NETRC_KEY, env_key=SYNTH_ENV_KEY, require_exec=True
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert record.is_file(), "the real 'wandb agent' line was never reached"
    captured = record.read_text(encoding="utf-8")
    assert "agent --count 1 synthsweep01" in captured
    # The pre-existing env key wins over the netrc entry and reaches the agent.
    assert f"KEYLEN: {len(SYNTH_ENV_KEY)}" in captured
    for stream in (result.stdout, result.stderr):
        assert SYNTH_ENV_KEY not in stream and SYNTH_NETRC_KEY not in stream


def test_netrc_key_is_resolved_and_exported_to_the_agent(tmp_path):
    result, record = _run_fragment(
        tmp_path, netrc_key=SYNTH_NETRC_KEY, env_key=None, require_exec=True
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert record.is_file(), "the real 'wandb agent' line was never reached"
    captured = record.read_text(encoding="utf-8")
    assert "agent --count 1 synthsweep01" in captured
    assert f"KEYLEN: {len(SYNTH_NETRC_KEY)}" in captured
    for stream in (result.stdout, result.stderr):
        assert SYNTH_NETRC_KEY not in stream


def test_no_key_anywhere_fails_before_the_agent(tmp_path):
    result, record = _run_fragment(tmp_path, netrc_key=None, env_key=None)
    assert result.returncode == 1
    assert "FATAL" in result.stderr
    assert not record.is_file(), "'wandb agent' must not run without a usable key"


def test_empty_netrc_password_is_treated_as_no_credential_and_fails_before_the_agent(tmp_path):
    # An api.wandb.ai .netrc entry with an empty password is not a usable
    # credential: the fragment must refuse before the agent rather than export
    # an empty WANDB_API_KEY. (The launcher does not otherwise police key
    # format -- W&B is authoritative for whether a non-empty key authenticates.)
    result, record = _run_fragment(tmp_path, netrc_key="", env_key=None)
    assert result.returncode == 1
    assert "FATAL" in result.stderr
    assert not record.is_file()
