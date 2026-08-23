"""Deterministic-generation and visual-smoke tests for the Sweep-v1 durable
review-rendering layer plus its synthetic fixture driver.

These tests render actual PNGs/PDFs into pytest tmp_path directories (never
into the durable `.scratch_local/` packet) and check structural properties
(file existence, non-zero size, no exceptions, expected filenames/titles).
They do not replace the mandatory manual visual inspection of the real
generated packet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.baseline import sweep_v1_review_analysis as analysis
from src.baseline import sweep_v1_review_rendering as rendering
import scripts.build_sweep_v1_visualization_fixture as fixture_script
import scripts.build_sweep_v1_visualization_fixture_v002 as fixture_script_v002


# ------------------------------------------------------------------
# module-scoped fixture tables (built once; fixture generation is the
# expensive step -- 36 Bayesian + 12 random-control trials with rejection-
# sampled trajectories)
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_tables() -> dict[str, pd.DataFrame]:
    return fixture_script.build_fixture()


@pytest.fixture(scope="module")
def rendered_checkpoint12(tmp_path_factory, fixture_tables) -> tuple[Path, dict[str, str]]:
    output_dir = tmp_path_factory.mktemp("checkpoint_12_render")
    generated = rendering.render_checkpoint_packet(
        output_dir, trial_df=fixture_tables["trial_df"], trajectory_df=fixture_tables["trajectory_df"],
        proposal_df=fixture_tables["proposal_df"], operations_df=fixture_tables["operations_df"],
        checkpoint_label="checkpoint_12", checkpoint_valid_bayesian_count=13,
        random_control_count=12, synthetic=True)
    return output_dir, generated


@pytest.fixture(scope="module")
def fixture_tables_v002() -> dict[str, pd.DataFrame]:
    return fixture_script_v002.build_fixture()


# ------------------------------------------------------------------
# deterministic fixture generation
# ------------------------------------------------------------------

def test_build_fixture_is_deterministic_across_calls():
    first = fixture_script.build_fixture()
    second = fixture_script.build_fixture()
    for key in ("trial_df", "trajectory_df", "proposal_df", "operations_df"):
        pd.testing.assert_frame_equal(first[key], second[key])


def test_fixture_trial_counts_match_contract(fixture_tables):
    trial_df = fixture_tables["trial_df"]
    valid = analysis.valid_trials(trial_df)
    assert (valid["search_arm"] == "bayesian").sum() == fixture_script.N_BAYESIAN_VALID
    assert (valid["search_arm"] == "random_control").sum() == 12
    failed = trial_df[trial_df["workflow_status"] != "pass"]
    assert len(failed) == 3  # 2 bayesian retry slots + 1 random-control retry slot


def test_fixture_retry_rows_share_configuration_not_trial_id(fixture_tables):
    trial_df = fixture_tables["trial_df"]
    failed = trial_df[trial_df["workflow_status"] != "pass"]
    for _, failed_row in failed.iterrows():
        retried = trial_df[(trial_df["configuration_id"] == failed_row["configuration_id"])
                            & (trial_df["workflow_status"] == "pass")]
        assert len(retried) == 1
        assert retried.iloc[0]["trial_id"] != failed_row["trial_id"]
        assert retried.iloc[0]["execution_generation"] > failed_row["execution_generation"]


def test_fixture_has_no_missing_proposal_order(fixture_tables):
    trial_df = fixture_tables["trial_df"]
    assert trial_df["proposal_order"].notna().all()


def test_fixture_trajectories_cover_exactly_epochs_1_to_12_for_valid_trials(fixture_tables):
    trial_df, trajectory_df = fixture_tables["trial_df"], fixture_tables["trajectory_df"]
    valid = analysis.valid_trials(trial_df)
    for trial_id in valid["trial_id"]:
        epochs = sorted(trajectory_df.loc[trajectory_df["trial_id"] == trial_id, "epoch"])
        assert epochs == list(range(1, 13))


# ------------------------------------------------------------------
# expected output filenames / no-exception rendering
# ------------------------------------------------------------------

_EXPECTED_FIGURE_BASENAMES = [
    "fig01_search_progress", "fig02_compute_efficiency", "fig03_objective_distribution",
    "fig04_five_axis_response", "fig05_boundary_occupancy", "fig06_proposal_drift",
    "fig07_representative_trajectories", "fig08_best_epoch_late_gain", "fig09_best_vs_final",
    "fig10_parallel_coordinates", "fig11_pairwise_interactions", "fig12_operations",
    "fig13_boundary_evolution", "fig14_boundary_band_sensitivity", "fig15_top_configurations",
]


def test_render_checkpoint_packet_creates_all_expected_files(rendered_checkpoint12):
    output_dir, generated = rendered_checkpoint12
    for basename in _EXPECTED_FIGURE_BASENAMES:
        png = output_dir / f"{basename}.png"
        pdf = output_dir / f"{basename}.pdf"
        assert png.exists() and png.stat().st_size > 0, f"missing/empty {png}"
        assert pdf.exists() and pdf.stat().st_size > 0, f"missing/empty {pdf}"
    board_png = output_dir / "decision_board.png"
    board_pdf = output_dir / "decision_board.pdf"
    assert board_png.exists() and board_png.stat().st_size > 0
    assert board_pdf.exists() and board_pdf.stat().st_size > 0


def test_render_checkpoint_packet_creates_derived_data_and_readme(rendered_checkpoint12):
    output_dir, generated = rendered_checkpoint12
    for name in ("derived_trial_slice.csv", "derived_operations_slice.csv",
                 "derived_boundary_pressure_table.csv", "derived_boundary_pressure_table.json",
                 "derived_boundary_pressure_evolution.csv", "derived_boundary_band_sensitivity.csv",
                 "derived_top_configurations.csv", "derived_categorical_occupancy.csv", "README.md"):
        path = output_dir / name
        assert path.exists() and path.stat().st_size > 0, f"missing/empty {path}"


def test_render_checkpoint_packet_return_value_lists_all_generated_paths(rendered_checkpoint12):
    _, generated = rendered_checkpoint12
    expected_keys = {"decision_board", "readme", "derived_trial_slice_csv", "derived_operations_slice_csv",
                      "derived_boundary_pressure_csv", "derived_boundary_pressure_json",
                      "derived_boundary_pressure_evolution_csv", "derived_boundary_band_sensitivity_csv",
                      "derived_top_configurations_csv", "derived_categorical_occupancy_csv"}
    expected_keys |= {f"fig{str(i).zfill(2)}_{name}" for i, name in enumerate(
        ["search_progress", "compute_efficiency", "objective_distribution", "five_axis_response",
         "boundary_occupancy", "proposal_drift", "representative_trajectories", "best_epoch_late_gain",
         "best_vs_final", "parallel_coordinates", "pairwise_interactions", "operations",
         "boundary_evolution", "boundary_band_sensitivity", "top_configurations"], start=1)}
    assert expected_keys <= set(generated.keys())
    for path_str in generated.values():
        assert Path(path_str).exists()


def test_render_all_three_checkpoints_end_to_end_no_exceptions(tmp_path, fixture_tables):
    results = fixture_script.render_all_checkpoints(tmp_path, fixture_tables)
    assert set(results.keys()) == {"checkpoint_12", "checkpoint_24", "final"}
    for label in results:
        checkpoint_dir = tmp_path / label
        assert (checkpoint_dir / "decision_board.png").exists()
        assert (checkpoint_dir / "README.md").exists()


def test_final_checkpoint_operations_slice_contains_all_three_failures(tmp_path, fixture_tables):
    results = fixture_script.render_all_checkpoints(tmp_path, fixture_tables)
    ops_csv = pd.read_csv(tmp_path / "final" / "derived_operations_slice.csv")
    assert (ops_csv["workflow_status"] != "pass").sum() == 3


# ------------------------------------------------------------------
# previous_boundary_df / checkpoint-evolution wiring (§10, §21)
# ------------------------------------------------------------------

def test_render_checkpoint_packet_without_previous_boundary_df_reports_first_checkpoint(rendered_checkpoint12):
    output_dir, _ = rendered_checkpoint12
    text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Previous checkpoint available for evolution tracking: no (first checkpoint)." in text


def test_render_checkpoint_packet_threads_previous_boundary_df(tmp_path, fixture_tables):
    trial_df = fixture_tables["trial_df"]
    first_slice = analysis.checkpoint_slice(trial_df, 13, 12)
    first_boundary_df = analysis.derive_boundary_pressure_table(first_slice)

    output_dir = tmp_path / "checkpoint_24"
    generated = rendering.render_checkpoint_packet(
        output_dir, trial_df=trial_df, trajectory_df=fixture_tables["trajectory_df"],
        proposal_df=fixture_tables["proposal_df"], operations_df=fixture_tables["operations_df"],
        checkpoint_label="checkpoint_24", checkpoint_valid_bayesian_count=25,
        random_control_count=12, synthetic=True, previous_boundary_df=first_boundary_df)

    text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Previous checkpoint available for evolution tracking: yes." in text

    evolution_csv = pd.read_csv(Path(generated["derived_boundary_pressure_evolution_csv"]))
    assert "tier_previous" in evolution_csv.columns and "direction" in evolution_csv.columns
    assert (evolution_csv["direction"] != "n/a (first checkpoint)").any()


def test_fig13_boundary_evolution_reflects_first_vs_later_checkpoint(tmp_path, fixture_tables):
    trial_df = fixture_tables["trial_df"]
    first_slice = analysis.checkpoint_slice(trial_df, 13, 12)
    first_boundary_df = analysis.derive_boundary_pressure_table(first_slice)

    no_previous = rendering.fig13_boundary_evolution(
        tmp_path / "no_prev", analysis.derive_boundary_pressure_table(first_slice), None, True)
    with_previous = rendering.fig13_boundary_evolution(
        tmp_path / "with_prev",
        analysis.derive_boundary_pressure_table(analysis.checkpoint_slice(trial_df, 25, 12)),
        first_boundary_df, True)
    assert no_previous.exists() and no_previous.stat().st_size > 0
    assert with_previous.exists() and with_previous.stat().st_size > 0


# ------------------------------------------------------------------
# README content sanity
# ------------------------------------------------------------------

def test_readme_contains_synthetic_banner(rendered_checkpoint12):
    output_dir, _ = rendered_checkpoint12
    text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert rendering.SYNTHETIC_BANNER in text


def test_readme_reports_nonzero_failed_attempts_for_checkpoint_covering_retry_slot(tmp_path, fixture_tables):
    # Bayesian retry slots are at proposal_order 5 and 22 (see build_fixture());
    # a checkpoint at valid_count=13 spans slot 5, so its operations slice --
    # and therefore the README's failed-attempt count -- must be nonzero.
    output_dir = tmp_path / "checkpoint_12"
    rendering.render_checkpoint_packet(
        output_dir, trial_df=fixture_tables["trial_df"], trajectory_df=fixture_tables["trajectory_df"],
        proposal_df=fixture_tables["proposal_df"], operations_df=fixture_tables["operations_df"],
        checkpoint_label="checkpoint_12", checkpoint_valid_bayesian_count=13,
        random_control_count=12, synthetic=True)
    text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Failed/incomplete attempts in this slice (excluded from scientific curves): " in text
    assert "excluded from scientific curves): 0" not in text


# ------------------------------------------------------------------
# v002 fixture (§2, §20, §21): strengthening LR-lower-boundary narrative +
# determinism
# ------------------------------------------------------------------

def test_build_fixture_v002_is_deterministic_across_calls():
    first = fixture_script_v002.build_fixture()
    second = fixture_script_v002.build_fixture()
    for key in ("trial_df", "trajectory_df", "proposal_df", "operations_df"):
        pd.testing.assert_frame_equal(first[key], second[key])


def test_fixture_v002_trial_counts_match_contract(fixture_tables_v002):
    trial_df = fixture_tables_v002["trial_df"]
    valid = analysis.valid_trials(trial_df)
    assert (valid["search_arm"] == "bayesian").sum() == fixture_script_v002.N_BAYESIAN_VALID
    assert (valid["search_arm"] == "random_control").sum() == 12


def test_fixture_v002_does_not_alter_real_random_control_manifest(fixture_tables_v002):
    from src.baseline import sweep_v1_campaign as sweep
    canonical = pd.DataFrame(sweep.generate_random_control_rows())
    valid = analysis.valid_trials(fixture_tables_v002["trial_df"])
    random_arm = valid[valid["search_arm"] == "random_control"].sort_values("proposal_order")
    assert len(random_arm) == len(canonical)
    # canonical's continuous fields are the IEEE-754 .17g STRING serialization
    # used for configuration_id hashing (see sweep.canonical_hyperparameters);
    # the fixture stores them as floats, so compare through float() for those.
    for column in ("learning_rate", "embedding_dropout", "output_dropout"):
        actual = random_arm[column].reset_index(drop=True).astype(float)
        expected = canonical[column].reset_index(drop=True).astype(float)
        assert (actual - expected).abs().max() < 1e-12
    for column in ("hidden_size", "batch_size", "configuration_id"):
        actual = random_arm[column].reset_index(drop=True)
        expected = canonical[column].reset_index(drop=True)
        assert (actual.values == expected.values).all()


def test_fixture_v002_lr_lower_boundary_is_ambiguous_at_checkpoint_12(fixture_tables_v002):
    trial_df = fixture_tables_v002["trial_df"]
    trial_slice = analysis.checkpoint_slice(trial_df, 13, 12)
    boundary_df = analysis.derive_boundary_pressure_table(trial_slice)
    row = boundary_df[(boundary_df["axis"] == "learning_rate") & (boundary_df["boundary_side"] == "lower")].iloc[0]
    assert row["tier"] != "STRONG"
    assert not bool(row["proposal_drift_toward_boundary"])


def test_fixture_v002_lr_lower_boundary_is_convincingly_strong_at_final(fixture_tables_v002):
    trial_df = fixture_tables_v002["trial_df"]
    trial_slice = analysis.checkpoint_slice(trial_df, fixture_script_v002.N_BAYESIAN_VALID, 12)
    boundary_df = analysis.derive_boundary_pressure_table(trial_slice)
    row = boundary_df[(boundary_df["axis"] == "learning_rate") & (boundary_df["boundary_side"] == "lower")].iloc[0]
    assert row["tier"] == "STRONG"
    assert bool(row["proposal_drift_toward_boundary"])
    assert bool(row["neighborhood_supports_direction"])
    assert float(row["top_quartile_near_fraction"]) >= 0.5


def test_fixture_v002_lr_lower_boundary_strengthens_from_checkpoint_24_to_final(fixture_tables_v002):
    trial_df = fixture_tables_v002["trial_df"]
    boundary_24 = analysis.derive_boundary_pressure_table(analysis.checkpoint_slice(trial_df, 25, 12))
    boundary_final = analysis.derive_boundary_pressure_table(
        analysis.checkpoint_slice(trial_df, fixture_script_v002.N_BAYESIAN_VALID, 12))
    evolution_df = analysis.boundary_pressure_evolution(boundary_final, boundary_24)
    row = evolution_df[(evolution_df["axis"] == "learning_rate") & (evolution_df["boundary_side"] == "lower")].iloc[0]
    assert row["direction"] == "strengthening"


def test_render_all_checkpoints_v002_threads_evolution_end_to_end(tmp_path, fixture_tables_v002):
    results = fixture_script_v002.render_all_checkpoints(tmp_path, fixture_tables_v002)
    assert set(results.keys()) == {"checkpoint_12", "checkpoint_24", "final"}
    readme_12 = (tmp_path / "checkpoint_12" / "README.md").read_text(encoding="utf-8")
    readme_24 = (tmp_path / "checkpoint_24" / "README.md").read_text(encoding="utf-8")
    readme_final = (tmp_path / "final" / "README.md").read_text(encoding="utf-8")
    assert "Previous checkpoint available for evolution tracking: no (first checkpoint)." in readme_12
    assert "Previous checkpoint available for evolution tracking: yes." in readme_24
    assert "Previous checkpoint available for evolution tracking: yes." in readme_final


# ------------------------------------------------------------------
# no W&B dependency / no sealed-scope references
# ------------------------------------------------------------------

def test_review_rendering_has_no_wandb_dependency():
    source = Path(rendering.__file__).read_text(encoding="utf-8")
    assert "import wandb" not in source and "from wandb" not in source


def test_review_rendering_never_references_sealed_scopes():
    source = Path(rendering.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("temporal_test", "spatial_holdout", "california"):
        assert forbidden not in lowered


def test_fixture_script_has_no_wandb_dependency():
    source = Path(fixture_script.__file__).read_text(encoding="utf-8")
    assert "import wandb" not in source and "from wandb" not in source


def test_fixture_output_is_clearly_labeled_synthetic_everywhere(tmp_path, fixture_tables):
    results = fixture_script.render_all_checkpoints(tmp_path, fixture_tables)
    for label in results:
        readme = (tmp_path / label / "README.md").read_text(encoding="utf-8")
        assert rendering.SYNTHETIC_BANNER in readme
