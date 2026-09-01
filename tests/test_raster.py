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


def _emission(electron_id, energy, is_cascade, generation,
              surface_normal=(0.0, 0.0, -1.0),
              emission_mechanism="transport_escape"):
    return SimpleNamespace(
        electron_id=electron_id,
        energy=float(energy),
        is_cascade=bool(is_cascade),
        generation=int(generation),
        surface_normal=surface_normal,
        emission_mechanism=emission_mechanism,
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


def _deep_cascade_result():
    """A generation-2 SE whose immediate parent and root leg disagree.

    Electron 2 is born from electron 1 (itself a cascade electron) while that
    parent moves toward vacuum, but electron 1 was seeded by the root primary
    on its *incoming* leg.  ``immediate_parent`` therefore calls electron 2 an
    SE2 while ``root_primary_leg`` calls it an SE1.
    """
    root = SimpleNamespace(
        electron_id=0, parent_id=None, first_surface_return_event_id=3,
    )
    records = [
        root,
        SimpleNamespace(
            electron_id=1, parent_id=0, birth_event_id=2,
            parent_direction_before=(0.0, 0.0, 1.0),   # root going inward
        ),
        SimpleNamespace(
            electron_id=2, parent_id=1, birth_event_id=4,
            parent_direction_before=(0.0, 0.0, -1.0),  # parent SE going out
        ),
    ]
    history = SimpleNamespace(
        electrons=records,
        events=[SimpleNamespace(event_id=3, electron_id=0, kind="elastic")],
        incident_energy=500.0,
        reference_surface_normal=(0.0, 0.0, -1.0),
    )
    emissions = [_emission(2, 10.0, True, 2)]
    return TrajectoryResult(tey=1, emissions=emissions, history=history)


def test_causal_lle_v3_crosses_causal_class_with_the_energy_cut():
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

    # Every cascade electron carries a causal label, including the fast one.
    assert counts["se1_lt50"] == 1
    assert counts["se1_ge50"] == 0
    assert counts["se2_lt50"] == 1
    assert counts["se2_ge50"] == 1
    assert counts["se1"] == 1
    assert counts["se2"] == 2
    assert counts["se1"] + counts["se2"] == counts["cascade_all"]
    assert counts["se1_ge50"] + counts["se2_ge50"] == counts["fast_cascade_ge50"]

    assert counts["lle_primary"] == 1
    assert counts["non_lle_primary"] == 0
    assert counts["first_event_backscatter"] == 1
    assert counts["later_return_primary"] == 0
    assert counts["barrier_reflected_primary"] == 0

    multiple = classifier.classify(_branch_result(two_elastics_before_return=True))
    assert multiple["lle_primary"] == 1
    assert multiple["first_event_backscatter"] == 0
    assert multiple["later_return_primary"] == 1


def test_default_basis_partitions_every_emitted_electron():
    classifier = PopulationClassifier()
    counts = classifier.classify(_branch_result())
    assert sum(counts[name] for name in classifier.disjoint_channels) == counts["tey"]
    assert set(classifier.disjoint_channels) == set(DISJOINT_POPULATION_CHANNELS)


def test_se_parent_rule_changes_which_class_a_deep_cascade_joins():
    result = _deep_cascade_result()

    root_leg = PopulationClassifier(se_parent_rule="root_primary_leg")
    immediate = PopulationClassifier(se_parent_rule="immediate_parent")

    assert root_leg.emission_labels(result)[2] == "se1_lt50"
    assert immediate.emission_labels(result)[2] == "se2_lt50"
    assert root_leg.to_dict()["se_parent_rule"] == "root_primary_leg"
    assert immediate.to_dict()["se_parent_rule"] == "immediate_parent"


def test_default_parent_rule_follows_the_definition():
    assert PopulationClassifier().se_parent_rule == "root_primary_leg"
    assert PopulationClassifier(
        definition="causal_lle_v2"
    ).se_parent_rule == "immediate_parent"


def test_causal_lle_v2_reproduces_the_energy_gated_channels():
    classifier = PopulationClassifier(50.0, definition="causal_lle_v2")
    counts = classifier.classify(_branch_result())

    assert counts["se1"] == 1
    assert counts["se2"] == 1                 # gated: the 70 eV SE is excluded
    assert counts["fast_cascade_ge50"] == 1
    assert counts["se_cascade_lt50"] == counts["se1"] + counts["se2"]
    assert counts["first_event_bse"] == 1
    assert counts["later_return_bse"] == 0
    assert "se1_lt50" not in counts
    assert sum(counts[name] for name in classifier.disjoint_channels) == counts["tey"]


def test_causal_classifier_retains_both_merged_surface_references():
    result = _branch_result()
    result.emissions[1].surface_normal = (0.0, 0.0, 1.0)

    launch = PopulationClassifier(se_reference="launch_surface")
    escape = PopulationClassifier(se_reference="escape_surface")

    assert launch.emission_labels(result)[1] == "se1_lt50"
    assert escape.emission_labels(result)[1] == "se2_lt50"
    assert launch.to_dict()["se_reference"] == "launch_surface"
    assert escape.to_dict()["se_reference"] == "escape_surface"


def test_lle_threshold_uses_strict_less_than_rule():
    result = _branch_result()
    result.emissions[0].energy = 450.0
    counts = PopulationClassifier(lle_max_loss_ev=50.0).classify(result)
    assert counts["lle_primary"] == 0
    assert counts["non_lle_primary"] == 1


def test_fractional_lle_threshold_scales_with_landing_energy():
    result = _branch_result()
    result.emissions[0].energy = 480.0          # 20 eV loss out of 500 eV

    generous = PopulationClassifier(lle_max_loss_frac=0.10)   # 50 eV
    strict = PopulationClassifier(lle_max_loss_frac=0.02)     # 10 eV

    assert generous.classify(result)["lle_primary"] == 1
    assert strict.classify(result)["lle_primary"] == 0
    assert generous.lle_threshold_ev(500.0) == 50.0
    assert strict.lle_threshold_ev(20_000.0) == 400.0
    assert strict.to_dict()["lle_criterion"] == "fractional_energy_loss"
    assert PopulationClassifier().to_dict()["lle_criterion"] == "absolute_energy_loss"


def test_absolute_and_fractional_lle_thresholds_are_mutually_exclusive():
    try:
        PopulationClassifier(lle_max_loss_ev=50.0, lle_max_loss_frac=0.02)
    except ValueError as error:
        assert "not both" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")


def test_non_lle_primary_holds_sub_cutoff_primaries_and_is_not_a_bse_class():
    """The non-LLE class is the complement among *emitted primaries*.

    A primary that crawls out below the 50 eV emission cut is not a BSE by the
    conventional partition, but it is still a non-LLE emitted primary.
    """
    result = _branch_result()
    result.emissions[0].energy = 30.0           # 470 eV loss, below the cut
    counts = PopulationClassifier().classify(result)

    assert counts["slow_primary_lt50"] == 1
    assert counts["bse_primary_ge50"] == 0
    assert counts["non_lle_primary"] == 1
    assert counts["primary_all"] == (
        counts["lle_primary"] + counts["non_lle_primary"]
    )


def test_barrier_reflected_primaries_are_counted_separately_inside_lle():
    result = _branch_result()
    result.emissions[0].energy = 500.0          # never entered, zero loss
    result.emissions[0].emission_mechanism = "incoming_barrier_reflection"
    counts = PopulationClassifier().classify(result)

    assert counts["lle_primary"] == 1
    assert counts["barrier_reflected_primary"] == 1
    # A diagnostic subset, so it must not appear in the disjoint basis.
    assert "barrier_reflected_primary" not in DISJOINT_POPULATION_CHANNELS


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
        result.count_maps["cascade_all"],
        result.count_maps["se1"] + result.count_maps["se2"],
    )
    assert np.array_equal(
        result.count_maps["se_cascade_lt50"],
        result.count_maps["se1_lt50"] + result.count_maps["se2_lt50"],
    )
    assert np.array_equal(
        result.count_maps["fast_cascade_ge50"],
        result.count_maps["se1_ge50"] + result.count_maps["se2_ge50"],
    )
    assert np.array_equal(
        result.count_maps["primary_all"],
        result.count_maps["lle_primary"] + result.count_maps["non_lle_primary"],
    )
    assert np.array_equal(
        result.count_maps["primary_all"],
        result.count_maps["first_event_backscatter"]
        + result.count_maps["later_return_primary"],
    )
    assert np.all(result.surface_hit_counts["sample_plane"] == 4)
    assert np.all(result.launch_sem == 0.0)
    assert np.allclose(result.launch_mean[:, :, 0], ((-10.0, 10.0),) * 2)
    assert np.allclose(result.launch_mean[:, :, 1], ((-5.0, -5.0), (5.0, 5.0)))
    assert result.yield_covariance.shape == (2, 2, 22, 22)
    for index, channel in enumerate(result.covariance_channels):
        assert np.allclose(
            result.yield_covariance[:, :, index, index],
            result.sem_maps[channel] ** 2,
        )
    disjoint_covariance = result.covariance(DISJOINT_POPULATION_CHANNELS)
    assert disjoint_covariance.shape == (2, 2, 6, 6)
    assert np.allclose(
        disjoint_covariance,
        np.swapaxes(disjoint_covariance, -1, -2),
    )

    archive = result.save_npz(tmp_path / "raster.npz")
    table = result.save_csv(tmp_path / "raster.csv")
    with np.load(archive, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["format"] == "seemc-imaging-raster-v3"
        assert metadata["classifier_config"]["definition"] == "causal_lle_v3"
        assert metadata["classifier_config"]["se_parent_rule"] == "root_primary_leg"
        assert metadata["classifier_config"]["lle_criterion"] == "absolute_energy_loss"
        assert metadata["classifier_config"]["lle_max_loss_ev"] == 50.0
        assert data["yield__se1"].shape == (2, 2)
        assert np.array_equal(data["count__tey"], result.count_maps["tey"])
        assert data["yield_covariance"].shape == (2, 2, 22, 22)
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert "yield__non_lle_primary" in rows[0]
    assert "covmean__se1_lt50__se2_lt50" in rows[0]
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
