from __future__ import annotations

import numpy as np

from seemc_imaging import (
    DISJOINT_POPULATION_CHANNELS,
    ProfileFitter,
    ProfileObservation,
    RasterConfig,
    Sample,
    TrapezoidModelLibrary,
    TrapezoidSweepConfig,
    TrapezoidSweepDriver,
    compare_channel_information,
)

from synthetic_material import write_synthetic_database


def _analytic_library():
    x = np.linspace(-500.0, 500.0, 101)
    points = np.asarray([
        (top, bottom, height)
        for top in (450.0, 500.0, 550.0)
        for bottom in (650.0, 700.0, 750.0)
        for height in (450.0, 500.0, 550.0)
    ])
    profiles = []
    for top, bottom, height in points:
        inner = 0.5 * top
        outer = 0.5 * bottom
        angle_factor = height / (0.5 * (bottom - top))
        sidewall = 0.5 * (
            np.tanh((np.abs(x) - inner) / 16.0)
            - np.tanh((np.abs(x) - outer) / 16.0)
        )
        inner_peak = np.exp(-((np.abs(x) - inner) / 12.0) ** 2)
        outer_peak = np.exp(-((np.abs(x) - outer) / 15.0) ** 2)
        profiles.append(np.stack([
            0.12 + 0.15 * inner_peak + 0.003 * angle_factor * sidewall,
            0.70 + 0.25 * sidewall + 0.04 * inner_peak,
            0.025 + 0.01 * sidewall,
            0.18 + 0.22 * outer_peak + 0.02 * sidewall,
            0.32 + 0.18 * sidewall + 0.03 * outer_peak,
        ]))
    yields = np.stack(profiles)
    n_models, n_channels, n_x = yields.shape
    covariance = np.zeros((n_models, n_x, n_channels, n_channels))
    diagonal = np.arange(n_channels)
    covariance[:, :, diagonal, diagonal] = np.asarray(
        [2e-5, 5e-5, 1e-5, 3e-5, 4e-5]
    )
    covariance[:, :, 0, 1] = covariance[:, :, 1, 0] = 6e-6
    completed = np.full((n_models, n_x), 5000, dtype=np.int64)
    return TrapezoidModelLibrary(
        points,
        x,
        DISJOINT_POPULATION_CHANNELS,
        yields,
        covariance,
        completed,
        {"sample_name": "analytic-test"},
    )


def test_joint_fit_recovers_grid_geometry_shift_and_scale():
    library = _analytic_library()
    target_index = library.nearest_model((500.0, 700.0, 500.0))
    shift = 5.0
    scale = 1.17
    target_yields = np.stack([
        np.interp(
            library.x_positions - shift,
            library.x_positions,
            profile,
            left=profile[0],
            right=profile[-1],
        )
        for profile in library.yields[target_index]
    ]) * scale
    observation = ProfileObservation(
        library.x_positions,
        library.channels,
        target_yields,
        library.covariance_of_mean[target_index],
    )
    result = ProfileFitter(library).fit(
        observation,
        shift_values=(-5.0, 0.0, 5.0),
        fit_scale=True,
        include_model_covariance=True,
    )

    assert np.array_equal(result.best_parameters, (500.0, 700.0, 500.0))
    assert result.x_shift == shift
    assert np.isclose(result.scale, scale, rtol=1e-5)
    assert result.reduced_chi_square < 1e-10


def test_information_comparison_uses_joint_covariance():
    library = _analytic_library()
    reports = compare_channel_information(
        library,
        reference_parameters=(500.0, 700.0, 500.0),
        channel_sets={
            "se_pair": ("se1", "se2"),
            "all": DISJOINT_POPULATION_CHANNELS,
        },
        fit_scale=True,
    )

    assert set(reports) == {"se_pair", "all"}
    assert np.all(np.isfinite(reports["all"].parameter_standard_errors))
    assert np.all(
        reports["all"].parameter_standard_errors
        <= reports["se_pair"].parameter_standard_errors
    )


def test_small_transport_sweep_round_trip(tmp_path):
    database = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=database)
    raster = RasterConfig(
        energy_ev=500.0,
        x_positions=(-120.0, 0.0, 120.0),
        y_positions=(0.0,),
        primaries_per_pixel=2,
        beam_fwhm=5.0,
        seed=17,
    )
    sweep = TrapezoidSweepConfig(
        top_widths=(100.0, 120.0),
        bottom_widths=(200.0,),
        heights=(100.0,),
    )
    library = TrapezoidSweepDriver(sample, raster, sweep).run(progress=False)

    assert library.parameters.shape == (2, 3)
    assert library.yields.shape == (2, 17, 3)
    assert library.covariance_of_mean.shape == (2, 3, 17, 17)
    assert library.metadata["classifier_config"]["definition"] == "causal_lle_v2"
    path = library.save_npz(tmp_path / "library.npz")
    loaded = TrapezoidModelLibrary.from_npz(path)
    assert np.array_equal(loaded.parameters, library.parameters)
    assert np.array_equal(loaded.yields, library.yields)
    assert np.array_equal(loaded.covariance_of_mean, library.covariance_of_mean)


def test_information_report_supports_height_only_library():
    library = _analytic_library()
    keep = np.all(library.parameters[:, :2] == (500.0, 700.0), axis=1)
    height_only = TrapezoidModelLibrary(
        library.parameters[keep], library.x_positions, library.channels,
        library.yields[keep], library.covariance_of_mean[keep],
        library.completed_primaries[keep], library.metadata,
    )
    report = compare_channel_information(
        height_only,
        reference_parameters=(500.0, 700.0, 500.0),
        channel_sets={"all": DISJOINT_POPULATION_CHANNELS},
    )["all"]

    assert report.estimable_parameters == ("height",)
    assert np.isnan(report.parameter_standard_errors[:2]).all()
    assert np.isfinite(report.parameter_standard_errors[2])
    payload = report.to_dict()
    assert payload["standard_error_nm"]["top_width"] is None
    assert payload["standard_error_nm"]["bottom_width"] is None
    assert payload["standard_error_nm"]["height"] is not None


def test_fit_rejects_material_and_classifier_mismatch():
    library = _analytic_library()
    observation = ProfileObservation(
        library.x_positions,
        library.channels,
        library.yields[0],
        library.covariance_of_mean[0],
        {
            "sample_name": "different-material",
            "classifier_config": {"definition": "branch_v1"},
        },
    )
    fitter = ProfileFitter(library)
    issues = fitter.compatibility_issues(observation)
    assert any("sample mismatch" in issue for issue in issues)
    assert any("classifier mismatch" in issue for issue in issues)
    try:
        fitter.fit(observation)
    except ValueError as exc:
        assert "incompatible model library and observation" in str(exc)
    else:
        raise AssertionError("incompatible fit was not rejected")


def test_fit_rejects_lle_threshold_mismatch():
    original = _analytic_library()
    metadata = {
        **original.metadata,
        "classifier_config": {
            "definition": "causal_lle_v2",
            "bse_cutoff_ev": 50.0,
            "lle_max_loss_ev": 20.0,
        },
    }
    library = TrapezoidModelLibrary(
        original.parameters, original.x_positions, original.channels,
        original.yields, original.covariance_of_mean,
        original.completed_primaries, metadata,
    )
    observation = ProfileObservation(
        library.x_positions,
        library.channels,
        library.yields[0],
        library.covariance_of_mean[0],
        {
            "sample_name": "analytic-test",
            "classifier_config": {
                "definition": "causal_lle_v2",
                "bse_cutoff_ev": 50.0,
                "lle_max_loss_ev": 50.0,
            },
        },
    )
    issues = ProfileFitter(library).compatibility_issues(observation)
    assert issues == ("LLE maximum loss mismatch: library=20.0, observation=50.0",)
