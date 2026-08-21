"""Static guards for the explicit Moriah runtime and CPU preflight."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SINGLE = ROOT / "scripts" / "run_phase_b_batch_size_operational_qualification_moriah.sbatch"
COMBINED = ROOT / "scripts" / "run_phase_b_batch_size_operational_retry_moriah.sbatch"
CPU = ROOT / "scripts" / "run_phase_b_batch_size_operational_cpu_preflight_moriah.sbatch"


def test_qualification_python_is_explicit_and_import_preflighted():
    for script in (SINGLE, COMBINED, CPU):
        text = script.read_text(encoding="utf-8")
        assert 'MORIAH_PYTHON="${FLASHNH_MORIAH_PYTHON:-' in text
        assert '"${MORIAH_PYTHON}" --version' in text
        assert 'import yaml, torch, neuralhydrology' in text
        assert 'MORIAH_PYTHON_IMPORT_PREFLIGHT' in text
        assert "conda activate" not in text
        assert "python -u" not in text


def test_combined_summary_cannot_reassign_readonly_parent_attempt_id():
    text = COMBINED.read_text(encoding="utf-8")
    assert 'readonly PARENT_ATTEMPT_ID=' in text
    assert 'ATTEMPT_ID="${PARENT_ATTEMPT_ID}"' in text
    assert 'readonly ATTEMPT_ID=' not in text
    assert 'PARTIAL_FAILURE_OR_FAIL' in text
    assert 'set +e' in text


def test_cpu_preflight_is_glacier_cpu_only_and_stops_before_training():
    text = CPU.read_text(encoding="utf-8")
    assert '#SBATCH --partition=glacier' in text
    assert '#SBATCH --cpus-per-task=2' in text
    assert '#SBATCH --mem=8G' in text
    assert '#SBATCH --time=00:10:00' in text
    assert 'readonly BATCH_SIZES=(128 256 512)' in text
    assert "attempt3_cpu_preflight_summary.json" in text
    assert "50000" in text and "operational_updates') != 8" in text
    executable = [line for line in text.splitlines() if not line.startswith("#")]
    executable_text = "\n".join(executable).lower()
    for forbidden in ("start_run", "nvidia-smi", "nse", "kge", "wandb"):
        assert forbidden not in executable_text
    assert not any(line.lstrip().startswith("sbatch ") for line in executable)
