from __future__ import annotations

import csv

import numpy as np

from seemc_imaging import Sample, simulate_trajectory
from seemc_imaging.plane_samplers import (
    BSE_ENERGY_FILENAME,
    SE_THETA_FILENAME,
    PlaneSamplerCase,
    cosine_probability_grid,
    deterministic_case_seed,
    export_angle_tables,
    load_case_checkpoint,
    plane_directions,
    save_case_checkpoint,
)
from seemc_imaging.transport import MCConfig

from synthetic_material import write_synthetic_database


def _case():
    case = PlaneSamplerCase(
        incidence_angle_deg=75.0,
        incident_energy_ev=500.0,
        n_primaries=4,
        case_seed=123,
        energy_cutoff_ev=50.0,
        se_energy_ev=np.array([1.0, 10.0, 49.0]),
        bse_energy_ev=np.array([50.0, 450.0]),
        se_theta_deg=np.array([5.0, 70.0, 160.0]),
        bse_theta_deg=np.array([20.0, 150.0]),
        se_primary_id=np.array([0, 0, 3]),
        bse_primary_id=np.array([1, 3]),
        se_counts_per_primary=np.array([2, 0, 0, 1]),
        bse_counts_per_primary=np.array([0, 1, 0, 1]),
    )
    case.validate()
    return case


def test_cosine_probability_grid_and_plane_axes():
    probability = cosine_probability_grid(513)
    assert probability.shape == (513,)
    assert probability[0] == 0.0
    assert probability[-1] == 1.0
    assert np.all(np.diff(probability) > 0.0)
    assert np.allclose(probability + probability[::-1], 1.0)

    vacuum, outward, beam_back = plane_directions(75.0)
    assert np.isclose(np.dot(vacuum, outward), -np.cos(np.deg2rad(75.0)))
    assert np.allclose(np.asarray(beam_back), -np.asarray(vacuum))


def test_case_seed_is_order_independent():
    seed = deterministic_case_seed(20260816, 75.0, 500.0)
    assert seed == deterministic_case_seed(20260816, 75.0, 500.0)
    assert seed != deterministic_case_seed(20260816, 70.0, 500.0)
    assert seed != deterministic_case_seed(20260816, 75.0, 800.0)


def test_checkpoint_round_trip_and_strict_metadata(tmp_path):
    case = _case()
    config = MCConfig()
    path = save_case_checkpoint(
        tmp_path / "case.npz", case, material="Cu", config=config
    )
    loaded = load_case_checkpoint(
        path, material="Cu", config=config, expected_angle_deg=75.0,
        expected_energy_ev=500.0, expected_n_primaries=4,
        expected_case_seed=123,
    )
    assert loaded.sey == 0.75
    assert loaded.bsey == 0.5
    assert np.array_equal(loaded.se_theta_deg, case.se_theta_deg)
    assert np.array_equal(loaded.bse_counts_per_primary,
                          case.bse_counts_per_primary)


def test_export_uses_common_grid_and_support_anchors(tmp_path):
    case = _case()
    probability = cosine_probability_grid(9)
    export_angle_tables(tmp_path, [case], material="Cu",
                        probabilities=probability)

    with open(tmp_path / BSE_ENERGY_FILENAME, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[2] == ["beamE (eV)", "r", "eE (eV)"]
    assert len(rows[3:]) == probability.size
    assert [float(value) for value in rows[3]] == [500.0, 0.0, 50.0]
    assert [float(value) for value in rows[-1]] == [500.0, 1.0, 500.0]

    with open(tmp_path / SE_THETA_FILENAME, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert [float(value) for value in rows[3]] == [500.0, 0.0, 0.0]
    assert [float(value) for value in rows[-1]] == [500.0, 1.0, 165.0]


def test_emissions_keep_primary_id_without_full_history(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)
    result = simulate_trajectory(
        sample, 500.0, 0.35, np.random.default_rng(3),
        history=False, trajectory_id=17,
    )
    assert result.history is None
    assert result.emissions
    assert {emission.root_primary_id for emission in result.emissions} == {17}
