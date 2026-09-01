"""Vertical producer/consumer + launcher-contract + retry-lineage focused
tests for the frozen Phase-B Sweep-v2 six-axis random-control arm.

The sibling ``tests/test_sweep_v2_six_axis_random_control.py`` covers the
frozen manifest generator/validator and its identity grammar. This file
covers the *production consumer* vertically instead:

  * one frozen manifest row -> ``prepare_random_control_proposal_v2`` ->
    ``write_prepared_proposal_v2`` -> persisted config/provenance ->
    synthetic (no-training) execution seam -> interpreted completion result;
  * the adapter's refusal cases (manifest checksum mismatch, a row that is
    not an exact frozen row, arm crossing);
  * the Bayesian production front door still rejects ``random_control``;
  * explicit retry-lineage resolution for generation 1 and later attempts,
    routed through the established v2 retry contract;
  * ``retry_of_trial_id`` reaches the durable execution provenance and the
    review records;
  * static contract checks on the W&B-free trial runner and its Moriah
    launcher (runtime provenance, explicit runner arguments, and the
    absence of any W&B-agent / Bayesian-controller contact).

Never starts real NeuralHydrology training and never contacts W&B: the
execution seam is the injected synthetic executor already used by
``tests/test_sweep_v2_six_axis_execution.py``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.baseline import sweep_v2_six_axis_random_control as rc
from src.baseline.sweep_v2_six_axis_campaign import (
    FORBIDDEN_PRODUCTION_SWEEP_IDS,
    SweepV2CampaignError,
    _AXES_V2,
    validate_v2_proposal_shape,
)
from src.baseline.sweep_v2_six_axis_config import V2_METRIC_NAME
from src.baseline.sweep_v2_six_axis_execution import execute_prepared_trial_v2
from src.baseline.sweep_v2_six_axis_production_adapter import (
    PreparationPathsV2,
    SweepV2PreparationError,
    prepare_bayesian_proposal_v2,
    prepare_random_control_proposal_v2,
    write_prepared_proposal_v2,
)
import scripts.run_sweep_v2_six_axis_random_control_trial as runner
from tests.test_sweep_v2_six_axis_execution import (
    _baseline_v2,
    _fake_result,
    _paths_v2,
    _write_real_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / (
    "stage1_phase_b_sweep_v2_six_axis_random_control_v001_random_control_manifest.json"
)
RUNNER_SCRIPT = ROOT / "scripts" / "run_sweep_v2_six_axis_random_control_trial.py"
SBATCH_SCRIPT = ROOT / "scripts" / "run_sweep_v2_six_axis_random_control_moriah.sbatch"
_BAYESIAN_CONTROLLER_RUN_ID = "wta85z3b"
_DUMMY_PATHS = PreparationPathsV2(*(Path("does-not-exist"),) * 6)


def _committed_row(index: int = 0) -> dict:
    return json.loads(MANIFEST_PATH.read_bytes())["rows"][index]


def _clean_axes(row: dict) -> dict:
    return {axis: row[axis] for axis in _AXES_V2}


# --------------------------------------------------------------------------
# Vertical producer/consumer: frozen row -> prepare -> persist -> synthetic
# execution seam -> interpreted completion.
# --------------------------------------------------------------------------

def test_frozen_row_prepares_executes_and_interprets_through_synthetic_seam(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    paths = _paths_v2(tmp_path / "prep", monkeypatch)
    row = _committed_row(0)

    prepared = prepare_random_control_proposal_v2(
        row=row, manifest_path=MANIFEST_PATH, paths=paths, execution_generation=1
    )
    assert prepared.proposal["search_arm"] == "random_control"
    assert prepared.trial_id.endswith("__attempt001")
    assert prepared.evidence["search_arm"] == "random_control"
    assert prepared.evidence["hyperparameters"]["seq_length"] == row["seq_length"]

    record = write_prepared_proposal_v2(prepared, tmp_path / "prepared_out")
    assert Path(record["generated_nh_config_path"]).is_file()
    assert record["search_arm"] == "random_control"

    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {e: 0.10 + 0.02 * e for e in epochs}
    fake_result = _fake_result(nh_run_dir, checkpoint_epochs=epochs, screening_scores=scores, n_basins=400)

    output_dir = tmp_path / "trial_out"
    outcome = execute_prepared_trial_v2(
        prepared_record=record, output_dir=output_dir,
        expected_screening_population=400, execute_prepared_run_fn=lambda: fake_result,
    )

    assert outcome["valid"] is True
    trial = outcome["review_records"]["trial_summary"]
    assert trial["search_arm"] == "random_control"
    assert trial["fixed_support_metric_name"] == V2_METRIC_NAME
    assert trial["objective_score"] is not None
    assert trial["retry_of_trial_id"] is None

    provenance = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["execution_status"] == "VALID"
    assert provenance["search_arm"] == "random_control"
    assert (output_dir / "review_records.json").is_file()


# --------------------------------------------------------------------------
# Adapter refusal cases.
# --------------------------------------------------------------------------

def test_adapter_rejects_manifest_checksum_mismatch(tmp_path):
    tampered = tmp_path / "manifest_tampered.json"
    tampered.write_bytes(MANIFEST_PATH.read_bytes() + b" ")
    with pytest.raises(SweepV2PreparationError, match="SHA-256"):
        prepare_random_control_proposal_v2(
            row=_committed_row(0), manifest_path=tampered, paths=_DUMMY_PATHS, execution_generation=1
        )


def test_adapter_rejects_row_that_is_not_an_exact_frozen_row(tmp_path):
    mutated = {**_committed_row(0), "seq_length": 999}
    with pytest.raises(SweepV2PreparationError, match="exact committed manifest row"):
        prepare_random_control_proposal_v2(
            row=mutated, manifest_path=MANIFEST_PATH, paths=_DUMMY_PATHS, execution_generation=1
        )


def test_bayesian_production_front_door_rejects_random_control(tmp_path):
    row = _committed_row(0)
    crossing = {**_clean_axes(row), "proposal_order": row["proposal_order"], "search_arm": "random_control"}
    with pytest.raises(SweepV2CampaignError, match="bayesian"):
        prepare_bayesian_proposal_v2(proposal=crossing, paths=_DUMMY_PATHS)


def test_proposal_shape_rejects_arm_crossing_in_both_directions():
    axes = _clean_axes(_committed_row(0))
    bayes = {**axes, "proposal_order": 1, "search_arm": "bayesian"}
    control = {**axes, "proposal_order": 1, "search_arm": "random_control"}
    # Matching arm: accepted.
    validate_v2_proposal_shape(bayes, expected_arm="bayesian")
    validate_v2_proposal_shape(control, expected_arm="random_control")
    # Crossed arm: rejected, in both directions.
    with pytest.raises(SweepV2CampaignError):
        validate_v2_proposal_shape(control, expected_arm="bayesian")
    with pytest.raises(SweepV2CampaignError):
        validate_v2_proposal_shape(bayes, expected_arm="random_control")


# --------------------------------------------------------------------------
# Explicit retry lineage (established v2 retry contract).
# --------------------------------------------------------------------------

def test_retry_lineage_generation_one_has_no_predecessor():
    row = _committed_row(0)
    retry_of, expected_trial_id = runner.resolve_random_control_retry_lineage(
        row, execution_generation=1, retry_of_trial_id=None
    )
    assert retry_of is None
    assert expected_trial_id == row["trial_id_attempt001"]
    assert expected_trial_id.endswith("__attempt001")


def test_retry_lineage_generation_one_rejects_a_supplied_predecessor():
    row = _committed_row(0)
    with pytest.raises(SystemExit, match="no retry predecessor"):
        runner.resolve_random_control_retry_lineage(
            row, execution_generation=1, retry_of_trial_id=row["trial_id_attempt001"]
        )
    with pytest.raises(SystemExit, match="prior attempt"):
        runner.resolve_random_control_retry_lineage(
            row, execution_generation=1, retry_of_trial_id=None, prior_attempt_generations=[1]
        )


def test_retry_lineage_later_generation_retains_identity_and_advances_attempt():
    row = _committed_row(3)
    retry_of, expected_trial_id = runner.resolve_random_control_retry_lineage(
        row, execution_generation=2, retry_of_trial_id=row["trial_id_attempt001"]
    )
    assert retry_of == row["trial_id_attempt001"]
    # configuration/proposal identity retained; only the attempt suffix advances.
    stem = row["trial_id_attempt001"].rsplit("__attempt", 1)[0]
    assert expected_trial_id == f"{stem}__attempt002"
    assert row["configuration_id"] in expected_trial_id
    assert row["proposal_id"].rsplit("__", 1)[-1] in expected_trial_id  # proposalNNN token

    # A third generation after a recorded second attempt.
    _, third = runner.resolve_random_control_retry_lineage(
        row, execution_generation=3, retry_of_trial_id=row["trial_id_attempt001"],
        prior_attempt_generations=[2],
    )
    assert third == f"{stem}__attempt003"


def test_retry_lineage_later_generation_requires_matching_predecessor():
    row = _committed_row(3)
    with pytest.raises(SystemExit, match="frozen attempt001 trial_id"):
        runner.resolve_random_control_retry_lineage(
            row, execution_generation=2, retry_of_trial_id="some-unrelated-trial-id"
        )


def test_retry_lineage_rejects_generation_already_attempted():
    row = _committed_row(3)
    with pytest.raises(SystemExit, match="already"):
        runner.resolve_random_control_retry_lineage(
            row, execution_generation=2, retry_of_trial_id=row["trial_id_attempt001"],
            prior_attempt_generations=[2],
        )


def test_retry_of_trial_id_reaches_provenance_and_review_records(tmp_path, monkeypatch):
    """The execution seam threads an explicit retry predecessor into the
    durable execution provenance and the review records (arm-agnostic)."""
    fx = _baseline_v2(tmp_path, monkeypatch)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    output_dir = tmp_path / "trial_out"
    predecessor = "stage1_phase_b_sweep_v2_six_axis_common120_v001__random_control__proposal001__x__attempt001"
    outcome = execute_prepared_trial_v2(
        prepared_record=fx["record"], output_dir=output_dir,
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
        retry_of_trial_id=predecessor, slurm_job_id="46000001",
    )
    assert outcome["review_records"]["trial_summary"]["retry_of_trial_id"] == predecessor
    provenance = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["retry_of_trial_id"] == predecessor
    on_disk = json.loads((output_dir / "review_records.json").read_text(encoding="utf-8"))
    assert on_disk["trial_summary"]["retry_of_trial_id"] == predecessor
    assert on_disk["operations"]["slurm_job_id"] == "46000001"


# --------------------------------------------------------------------------
# Static contract: the W&B-free trial runner.
# --------------------------------------------------------------------------

def test_trial_runner_is_wandb_free_and_names_no_controller_or_forbidden_sweep():
    src = RUNNER_SCRIPT.read_text(encoding="utf-8")
    assert "import wandb" not in src and "from wandb" not in src
    assert "wandb.init" not in src and "wandb agent" not in src
    assert _BAYESIAN_CONTROLLER_RUN_ID not in src
    for forbidden in FORBIDDEN_PRODUCTION_SWEEP_IDS:
        assert forbidden not in src
    assert "oz5p4csb" not in src


def test_trial_runner_pins_runtime_provenance_and_threads_retry_lineage():
    src = RUNNER_SCRIPT.read_text(encoding="utf-8")
    assert "verify_commit_and_interpreter" in src
    assert "derive_exact_retry_identity_v2" in src
    for flag in ("--expected-commit", "--expected-runtime-python",
                 "--retry-of-trial-id", "--prior-attempt-generation"):
        assert flag in src
    assert "required=True" in src  # --expected-commit is a required argument
    assert "retry_of_trial_id=resolved_retry_of" in src
    assert 'slurm_job_id=os.environ.get("SLURM_JOB_ID")' in src


def test_trial_runner_selftest_hook_resolves_without_execution(capsys):
    row = _committed_row(1)
    argv = ["--row-index", "1", "--output-root", "unused",
            "--expected-commit", "0" * 40, "--execution-generation", "1"]
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["run_sweep_v2_six_axis_random_control_trial.py", *argv]
    try:
        import os as _os
        _os.environ["FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST"] = "resolve_only"
        try:
            code = runner.main()
        finally:
            _os.environ.pop("FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST", None)
    finally:
        _sys.argv = old
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selftest"] == "resolve_only"
    assert payload["retry_of_trial_id"] is None
    assert payload["expected_trial_id"] == row["trial_id_attempt001"]


def test_trial_runner_rejects_a_non_full_commit_identity(capsys):
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["run_sweep_v2_six_axis_random_control_trial.py",
                 "--row-index", "0", "--output-root", "unused", "--expected-commit", "deadbeef"]
    try:
        with pytest.raises(SystemExit, match="40-hex"):
            runner.main()
    finally:
        _sys.argv = old


# --------------------------------------------------------------------------
# Static contract: the Moriah launcher (never submitted here).
# --------------------------------------------------------------------------

def _launcher_code_text() -> str:
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    return "\n".join(
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    )


def test_launcher_preserves_resources_job_name_and_log_paths():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    for directive in ("#SBATCH --partition=catfish", "#SBATCH --gres=gpu:l4:1",
                      "#SBATCH --cpus-per-task=8", "#SBATCH --mem=128G", "#SBATCH --time=08:00:00",
                      "#SBATCH --job-name=flashnh-sweep-v2-six-axis-random-control"):
        assert directive in text
    assert "sweep-v2-six-axis-random-control-%j.out" in text
    assert "sweep-v2-six-axis-random-control-%j.err" in text


def test_launcher_requires_row_output_and_full_expected_commit():
    code = _launcher_code_text()
    assert 'set -euo pipefail' in code
    assert '"${ROW_INDEX:?' in code
    assert '"${OUTPUT_ROOT:?' in code
    assert '"${EXPECTED_COMMIT:?' in code
    assert "[0-9a-f]{40}" in code  # cheap full-commit-identity format check


def test_launcher_enters_canonical_workdir_and_uses_canonical_interpreter():
    code = _launcher_code_text()
    assert 'test -d "${REPO_WORKDIR}"' in code
    assert 'cd "${REPO_WORKDIR}"' in code
    assert 'test -x "${CANONICAL_PYTHON}"' in code
    assert 'test -f "${MANIFEST_PATH}"' in code
    assert "envs/flashnh-moriah/bin/python" in code
    assert "/sci/labs/efratmorin/omripo/Flash-NH" in SBATCH_SCRIPT.read_text(encoding="utf-8")
    # The authoritative HEAD / dirty-tree pin is delegated to Python, not shell.
    assert "git rev-parse" not in code
    assert "git status" not in code


def test_launcher_delegates_to_runner_with_explicit_arguments():
    code = _launcher_code_text()
    exec_lines = [line for line in code.splitlines() if line.strip().startswith("exec ")]
    assert len(exec_lines) == 1
    joined = code[code.index("exec "):]
    assert 'exec "${CANONICAL_PYTHON}" scripts/run_sweep_v2_six_axis_random_control_trial.py' in joined
    for arg in ('--expected-commit "${EXPECTED_COMMIT}"',
                '--expected-runtime-python "${CANONICAL_PYTHON}"',
                '--row-index "${ROW_INDEX}"',
                '--output-root "${OUTPUT_ROOT}"',
                '--execution-generation "${EXECUTION_GENERATION}"',
                '--manifest-path "${MANIFEST_PATH}"',
                '--package-root "${PACKAGE_ROOT}"',
                '--screening-basin-ids "${SCREENING_BASIN_IDS}"',
                '--fixed-support-contract-path "${FIXED_SUPPORT_CONTRACT_PATH}"'):
        assert arg in joined
    assert 'RETRY_ARGS' in joined
    assert '--retry-of-trial-id "${RETRY_OF_TRIAL_ID}"' in code


def test_launcher_passes_all_three_scientific_policies_explicitly():
    """The launcher must not rely on the runner's CLI defaults for the
    baseline / v2-overlay / pilot policy paths -- it resolves each to its
    canonical committed location and passes it explicitly."""
    code = _launcher_code_text()
    joined = code[code.index("exec "):]
    # Canonical committed locations (identical to the runner CLI defaults and
    # to sweep_v2_six_axis_wandb_bridge_registration._REAL_* constants).
    assert 'BASELINE_POLICY_PATH="${REPO_WORKDIR}/config/stage1_scientific_baseline_v001.yaml"' in code
    assert ('POLICY_OVERLAY_PATH="${REPO_WORKDIR}/config/'
            'stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml"') in code
    assert 'BASE_PILOT_POLICY_PATH="${REPO_WORKDIR}/config/stage1_lead06_pilot_v001.yaml"' in code
    for arg in ('--baseline-policy-path "${BASELINE_POLICY_PATH}"',
                '--policy-overlay-path "${POLICY_OVERLAY_PATH}"',
                '--base-pilot-policy-path "${BASE_PILOT_POLICY_PATH}"'):
        assert arg in joined
    # And each canonical policy file actually exists in this checkout.
    for name in ("stage1_scientific_baseline_v001.yaml",
                 "stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml",
                 "stage1_lead06_pilot_v001.yaml"):
        assert (ROOT / "config" / name).is_file()


def test_launcher_pins_canonical_paths_with_no_environment_override():
    """FLASHNH_BASE / REPO_WORKDIR / CANONICAL_PYTHON are assigned directly to
    the canonical Moriah values -- a submitter cannot redirect the checkout or
    the interpreter through the submission environment."""
    code = _launcher_code_text()
    assert "FLASHNH_BASE=/sci/labs/efratmorin/omripo/Flash-NH" in code
    assert 'REPO_WORKDIR="${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation"' in code
    assert 'CANONICAL_PYTHON="${FLASHNH_BASE}/envs/flashnh-moriah/bin/python"' in code
    # No ${VAR:-default} fallback / override syntax for any of the three.
    for var in ("FLASHNH_BASE", "REPO_WORKDIR", "CANONICAL_PYTHON"):
        assert "${%s:-" % var not in code
        assert "${%s:=" % var not in code
        assert "${%s:?" % var not in code
    # No alternate/fallback location string for the checkout or interpreter.
    assert code.count("/repos/flash-nh/US_data/data_download/Disk_volume_estimation") >= 1
    assert "envs/flashnh-moriah/bin/python" in code


def test_launcher_refuses_and_unsets_inherited_selftest_hook():
    """Static: the production launcher rejects a non-empty
    FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST before doing any work, then
    defensively unsets it, and never forwards it to the runner."""
    code = _launcher_code_text()
    hook = "FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST"
    refusal_at = code.index('if [ -n "${%s:-}" ]; then' % hook)
    assert 'echo "REFUSING' in code[refusal_at:refusal_at + 400]
    assert "exit 1" in code[refusal_at:refusal_at + 400]
    assert ("unset %s" % hook) in code
    # The refusal + unset both precede the exec of the runner.
    assert refusal_at < code.index("exec ")
    assert code.index("unset %s" % hook) < code.index("exec ")
    # The hook is never passed as a runner argument / re-exported.
    joined = code[code.index("exec "):]
    assert hook not in joined
    assert ("export %s" % hook) not in code


@pytest.mark.skipif(shutil.which("bash") is None, reason="no POSIX shell available")
def test_launcher_shell_syntax_is_valid():
    result = subprocess.run([shutil.which("bash"), "-n", str(SBATCH_SCRIPT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="no POSIX shell available")
def test_launcher_behaviourally_refuses_inherited_selftest(tmp_path):
    """Behavioural: with the self-test hook inherited, the launcher exits
    nonzero with the refusal message before reaching the canonical checkout
    (no MISSING-path diagnostics), so ``resolve_only`` cannot leak into a
    real allocation."""
    env = {
        **os.environ,
        "FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST": "resolve_only",
        "ROW_INDEX": "0",
        "OUTPUT_ROOT": str(tmp_path / "out"),
        "EXPECTED_COMMIT": "0" * 40,
    }
    result = subprocess.run([shutil.which("bash"), str(SBATCH_SCRIPT)],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "REFUSING" in result.stderr
    assert "FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST" in result.stderr
    assert "MISSING" not in result.stdout and "MISSING" not in result.stderr


def test_launcher_has_no_loop_array_self_submission_or_wandb_contact():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code = _launcher_code_text()
    assert "--array" not in text
    # No shell loop construct over scientific candidates.
    assert "; do" not in code and "; do\n" not in code
    for line in code.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("for ", "while ", "until ", "do ", "done"))
    assert [line for line in code.splitlines() if "sbatch " in line] == []
    assert "wandb agent" not in code and "wandb login" not in code
    assert "--count" not in code
    assert "WANDB_API_KEY" not in text and ".netrc" not in text
    assert _BAYESIAN_CONTROLLER_RUN_ID not in text
    for forbidden in FORBIDDEN_PRODUCTION_SWEEP_IDS:
        assert forbidden not in text
