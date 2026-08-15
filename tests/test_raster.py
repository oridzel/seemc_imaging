from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np

from seemc_imaging import (
    DISJOINT_POPULATION_CHANNELS,
    Plane,
    PopulationClassifier,
    RasterConfig,
    RasterDriver,
    Sample,
    TrapezoidalLine,
    sample_beam_reference,
)
from seemc_imaging.transport import TrajectoryResult

from synthetic_material import write_synthetic_database


def _sample(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    return Sample("Synthetic", db_path=db_path)


def _emission(electron_id, energy, is_cascade, generation):
    return SimpleNamespace(
        electron_id=electron_id,
        energy=float(energy),
        is_cascade=bool(is_cascade),
        generation=int(generation),
    )


def _branch_result(two_elastics_before_return=False):
    return_id = 3
    root = SimpleNamespace(
        electron_id=0,
        parent_id=None,
        first_surface_return_event_id=return_id,
    )
    records = [
        root,
        SimpleNamespace(
            electron_id=1, parent_id=0, birth_event_id=2,
            parent_direction_before=(0.0, 0.0, 1.0),
        ),
        SimpleNamespace(
            electron_id=2, parent_id=0, birth_event_id=5,
            parent_direction_before=(0.0, 0.0, -1.0),
        ),
        SimpleNamespace(
            electron_id=3, parent_id=0, birth_event_id=6,
            parent_direction_before=(0.0, 0.0, -1.0),
        ),
    ]
    events = [
        SimpleNamespace(event_id=0, electron_id=0, kind="primary_launch"),
    ]
    if two_elastics_before_return:
        events.append(SimpleNamespace(event_id=1, electron_id=0, kind="elastic"))
    events.append(SimpleNamespace(event_id=3, electron_id=0, kind="elastic"))
    history = SimpleNamespace(
        electrons=records,
        events=events,
        incident_energy=500.0,
        reference_surface_normal=(0.0, 0.0, -1.0),
    )
    emissions = [
        _emission(0, 480.0, False, 0),
        _emission(1, 10.0, True, 1),
        _emission(2, 20.0, True, 2),
        _emission(3, 70.0, True, 1),
    ]
    return TrajectoryResult(tey=4, emissions=emissions, history=history)


def test_causal_lle_classifier_separates_taxonomy_filter_and_diagnostic():
    classifier = PopulationClassifier(50.0)
    counts = classifier.classify(_branch_result())

    assert counts["tey"] == 4
    assert counts["sey_50ev"] == 2
    assert counts["bse_50ev"] == 2
    assert counts["cascade_all"] == 3
    assert counts["primary_all"] == 1
    assert counts["se_cascade_lt50"] == 2
    assert counts["fast_cascade_ge50"] == 1
    assert counts["bse_primary_ge50"] == 1
    assert counts["generation_1"] == 2
    assert counts["generation_2plus"] == 1
    assert counts["se1"] == 1
    assert counts["se2"] == 1
    assert counts["lle_primary"] == 1
    assert counts["non_lle_primary"] == 0
    assert counts["first_event_bse"] == 1
    assert counts["later_return_bse"] == 0

    multiple = classifier.classify(_branch_result(two_elastics_before_return=True))
    assert multiple["lle_primary"] == 1
    assert multiple["first_event_bse"] == 0
    assert multiple["later_return_bse"] == 1


def test_legacy_branch_v1_remains_reproducible():
    classifier = PopulationClassifier(50.0, definition="branch_v1")
    counts = classifier.classify(_branch_result())
    assert counts["se1"] == 1
    assert counts["se2"] == 1
    assert counts["bse1"] == 1
    assert counts["bse2"] == 0


def test_zero_spot_is_exact_and_consumes_no_random_number():
    first = np.random.default_rng(123)
    second = np.random.default_rng(123)
    point = sample_beam_reference(
        12.0, -7.0, 0.0, (0.2, 0.1, 1.0), (0.0, 0.0), first
    )
    assert point == (12.0, -7.0)
    assert first.random() == second.random()


def test_plane_raster_maps_partition_counts_and_export(tmp_path):
    sample = _sample(tmp_path)
    config = RasterConfig(
        energy_ev=500.0,
        x_positions=(-10.0, 10.0),
        y_positions=(-5.0, 5.0),
        primaries_per_pixel=4,
        seed=101,
    )
    result = RasterDriver(sample, Plane(), config).run(
        use_parallel=False, progress=False
    )

    assert result.completed_primaries.shape == (2, 2)
    assert np.all(result.completed_primaries == 4)
    assert np.array_equal(
        result.count_maps["tey"],
        result.count_maps["sey_50ev"] + result.count_maps["bse_50ev"],
    )
    assert np.array_equal(
        result.count_maps["tey"],
        result.count_maps["cascade_all"] + result.count_maps["primary_all"],
    )
    assert np.array_equal(
        result.count_maps["se_cascade_lt50"],
        result.count_maps["se1"] + result.count_maps["se2"],
    )
    assert np.array_equal(
        result.count_maps["primary_all"],
        result.count_maps["lle_primary"] + result.count_maps["non_lle_primary"],
    )
    assert np.array_equal(
        result.count_maps["primary_all"],
        result.count_maps["first_event_bse"]
        + result.count_maps["later_return_bse"],
    )
    assert np.all(result.surface_hit_counts["sample_plane"] == 4)
    assert np.all(result.launch_sem == 0.0)
    assert np.allclose(result.launch_mean[:, :, 0], ((-10.0, 10.0),) * 2)
    assert np.allclose(result.launch_mean[:, :, 1], ((-5.0, -5.0), (5.0, 5.0)))
    assert result.yield_covariance.shape == (2, 2, 17, 17)
    for index, channel in enumerate(result.covariance_channels):
        assert np.allclose(
            result.yield_covariance[:, :, index, index],
            result.sem_maps[channel] ** 2,
        )
    disjoint_covariance = result.covariance(DISJOINT_POPULATION_CHANNELS)
    assert disjoint_covariance.shape == (2, 2, 5, 5)
    assert np.allclose(
        disjoint_covariance,
        np.swapaxes(disjoint_covariance, -1, -2),
    )

    archive = result.save_npz(tmp_path / "raster.npz")
    table = result.save_csv(tmp_path / "raster.csv")
    with np.load(archive, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["format"] == "seemc-imaging-raster-v3"
        assert metadata["classifier_config"]["definition"] == "causal_lle_v2"
        assert metadata["classifier_config"]["lle_max_loss_ev"] == 50.0
        assert data["yield__se1"].shape == (2, 2)
        assert np.array_equal(data["count__tey"], result.count_maps["tey"])
        assert data["yield_covariance"].shape == (2, 2, 17, 17)
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert "yield__non_lle_primary" in rows[0]
    assert "covmean__se1__se2" in rows[0]
    assert "landing_fraction__sample_plane" in rows[0]


def test_beam_spot_changes_launches_not_collision_stream_on_plane(tmp_path):
    sample = _sample(tmp_path)
    common = dict(
        energy_ev=500.0,
        x_positions=(0.0,),
        y_positions=(0.0,),
        primaries_per_pixel=8,
        seed=20260811,
    )
    point = RasterDriver(
        sample, Plane(), RasterConfig(**common, beam_fwhm=0.0)
    ).run(progress=False)
    finite = RasterDriver(
        sample, Plane(), RasterConfig(**common, beam_fwhm=40.0)
    ).run(progress=False)

    for channel in point.classifier.channels:
        assert np.array_equal(point.count_maps[channel], finite.count_maps[channel])
        assert np.array_equal(point.yield_maps[channel], finite.yield_maps[channel])
        assert np.array_equal(point.sem_maps[channel], finite.sem_maps[channel])
    assert np.all(point.launch_sem == 0.0)
    assert np.any(finite.launch_sem > 0.0)


def test_gaussian_spot_mixes_top_and_sidewall_landings(tmp_path):
    sample = _sample(tmp_path)
    line = TrapezoidalLine(top_width=100.0, bottom_width=200.0, height=100.0)
    config = RasterConfig(
        energy_ev=500.0,
        x_positions=(50.0,),
        y_positions=(0.0,),
        primaries_per_pixel=32,
        beam_fwhm=30.0,
        seed=44,
    )
    result = RasterDriver(sample, line, config).run(progress=False)

    assert result.surface_hit_counts["trapezoidal_line.top"][0, 0] > 0
    assert result.surface_hit_counts["trapezoidal_line.right"][0, 0] > 0
    assert sum(values[0, 0] for values in result.surface_hit_counts.values()) == 32
    assert result.local_incidence_mean_rad[0, 0] > 0.0


def test_serial_and_spawn_parallel_rasters_are_identical(tmp_path):
    sample = _sample(tmp_path)
    config = RasterConfig(
        energy_ev=500.0,
        x_positions=(-30.0, 30.0),
        y_positions=(0.0,),
        primaries_per_pixel=3,
        beam_fwhm=10.0,
        seed=77,
    )
    driver = RasterDriver(sample, Plane(), config)
    serial = driver.run(use_parallel=False, progress=False)
    parallel = driver.run(use_parallel=True, workers=2, progress=False)

    for channel in serial.classifier.channels:
        assert np.array_equal(serial.count_maps[channel], parallel.count_maps[channel])
        assert np.array_equal(serial.yield_maps[channel], parallel.yield_maps[channel])
        assert np.array_equal(serial.sem_maps[channel], parallel.sem_maps[channel])
    assert np.array_equal(
        serial.primary_count_covariance,
        parallel.primary_count_covariance,
    )
    assert np.array_equal(serial.yield_covariance, parallel.yield_covariance)
    assert np.array_equal(serial.launch_mean, parallel.launch_mean)
    assert serial.diagnostics == parallel.diagnostics
