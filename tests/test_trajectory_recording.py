from __future__ import annotations

import numpy as np
import pytest

from seemc_imaging import (
    RasterConfig,
    RasterDriver,
    RasterTrajectoryArchive,
    Sample,
    TrapezoidalLine,
    animate_trapezoidal_scan,
    simulate_trajectory,
)

from synthetic_material import write_synthetic_database
from seemc_imaging.animation import _trapezoid_surface_height_nm


def _sample(tmp_path):
    database = write_synthetic_database(tmp_path / "synthetic.pkl")
    return Sample("Synthetic", db_path=database)


def _line():
    return TrapezoidalLine(
        top_width=100.0,
        bottom_width=200.0,
        height=100.0,
    )


def _recording_config():
    return RasterConfig(
        energy_ev=500.0,
        x_positions=(-120.0, 0.0, 120.0),
        y_positions=(0.0,),
        primaries_per_pixel=3,
        beam_fwhm=5.0,
        seed=2468,
        record_trajectories=True,
        record_primaries_per_pixel=2,
        trajectory_stride=2,
        trajectory_max_points=40,
    )


def test_nominal_beam_axis_follows_trapezoid_surface_without_launch_jitter():
    values = [
        _trapezoid_surface_height_nm(
            x,
            top_width=10.0,
            bottom_width=20.0,
            height=10.0,
            center_x=0.0,
            substrate_height=0.0,
        )
        for x in (-12.0, -10.0, -7.5, -5.0, 0.0, 5.0, 7.5, 10.0, 12.0)
    ]
    assert values == pytest.approx((0.0, 0.0, 5.0, 10.0, 10.0,
                                    10.0, 5.0, 0.0, 0.0))


def test_transport_tracks_link_electron_ids_and_monotone_physical_time(tmp_path):
    sample = _sample(tmp_path)
    result = simulate_trajectory(
        sample,
        500.0,
        0.35,
        np.random.default_rng(3),
        track=True,
        history=True,
    )
    by_id = {record.electron_id: record for record in result.history.electrons}
    assert len(result.tracks) == len(result.track_times_fs)
    assert len(result.tracks) == len(result.track_electron_ids)
    for electron_id, coordinates, times in zip(
            result.track_electron_ids, result.tracks, result.track_times_fs):
        assert len(coordinates) == len(times)
        assert np.all(np.diff(times) >= 0.0)
        assert times[0] == pytest.approx(by_id[electron_id].birth_time_fs)
        assert times[-1] == pytest.approx(by_id[electron_id].final_time_fs)


def test_recording_is_opt_in_rng_transparent_and_round_trips(tmp_path):
    sample = _sample(tmp_path)
    recorded_config = _recording_config()
    plain_config = RasterConfig(
        energy_ev=recorded_config.energy_ev,
        x_positions=recorded_config.x_positions,
        y_positions=recorded_config.y_positions,
        primaries_per_pixel=recorded_config.primaries_per_pixel,
        beam_fwhm=recorded_config.beam_fwhm,
        seed=recorded_config.seed,
    )
    recorded = RasterDriver(sample, _line(), recorded_config).run(progress=False)
    plain = RasterDriver(sample, _line(), plain_config).run(progress=False)
    for channel in recorded.classifier.channels:
        assert np.array_equal(recorded.count_maps[channel], plain.count_maps[channel])
        assert np.array_equal(recorded.yield_maps[channel], plain.yield_maps[channel])
    assert recorded.has_recorded_trajectories
    assert not plain.has_recorded_trajectories

    path = recorded.save_trajectories_npz(tmp_path / "tracks.npz")
    archive = RasterTrajectoryArchive.load_npz(path)
    assert archive.n_cascades == 3 * 2
    assert archive.metadata["geometry"]["type"] == "TrapezoidalLine"
    assert archive.metadata["config"]["record_primaries_per_pixel"] == 2
    assert archive.points.shape[1] == 5
    assert np.all(np.diff(archive.cascade_electron_offsets) >= 1)
    assert np.all(np.diff(archive.electron_point_offsets) >= 1)
    assert set(archive.electron_population).issubset({
        "se1", "se2", "fast_cascade_ge50",
        "lle_primary", "non_lle_primary",
        "cascade_absorbed", "primary_absorbed",
    })


def test_serial_and_spawn_parallel_trajectory_archives_are_identical(tmp_path):
    sample = _sample(tmp_path)
    driver = RasterDriver(sample, _line(), _recording_config())
    serial = driver.run(use_parallel=False, progress=False).trajectory_archive()
    parallel = driver.run(
        use_parallel=True, workers=2, progress=False
    ).trajectory_archive()
    for field_name in serial.__dataclass_fields__:
        if field_name == "metadata":
            assert serial.metadata == parallel.metadata
        else:
            assert np.array_equal(
                getattr(serial, field_name), getattr(parallel, field_name)
            )


def test_short_gif_animation_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    sample = _sample(tmp_path)
    config = RasterConfig(
        energy_ev=500.0,
        x_positions=(-120.0, 120.0),
        y_positions=(0.0,),
        primaries_per_pixel=1,
        seed=31,
        record_trajectories=True,
    )
    archive = RasterDriver(sample, _line(), config).run(
        progress=False
    ).trajectory_archive()
    output = animate_trapezoidal_scan(
        archive,
        tmp_path / "smoke.gif",
        fps=4,
        frames_per_pixel=2,
        pause_frames=0,
        vacuum_flight_nm=5.0,
        dpi=45,
    )
    assert output.stat().st_size > 1_000


def test_profile_channel_presets_and_validation(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    sample = _sample(tmp_path)
    config = RasterConfig(
        energy_ev=500.0,
        x_positions=(-120.0, 120.0),
        y_positions=(0.0,),
        primaries_per_pixel=1,
        seed=32,
        record_trajectories=True,
    )
    archive = RasterDriver(sample, _line(), config).run(
        progress=False
    ).trajectory_archive()
    conventional = animate_trapezoidal_scan(
        archive,
        tmp_path / "conventional.gif",
        fps=4,
        frames_per_pixel=2,
        pause_frames=0,
        vacuum_flight_nm=5.0,
        dpi=45,
        profile_channels="conventional",
    )
    assert conventional.stat().st_size > 1_000
    renamed = np.asarray([
        "lle_bse" if str(name) == "lle_primary"
        else "non_lle_bse" if str(name) == "non_lle_primary"
        else str(name)
        for name in archive.profile_channels
    ])
    archive.profile_channels = renamed
    v062 = animate_trapezoidal_scan(
        archive,
        tmp_path / "v062.gif",
        fps=4,
        frames_per_pixel=2,
        pause_frames=0,
        vacuum_flight_nm=5.0,
        dpi=45,
        profile_channels="populations",
    )
    assert v062.stat().st_size > 1_000
    with pytest.raises(ValueError, match="unknown profile channels"):
        animate_trapezoidal_scan(
            archive,
            tmp_path / "invalid.gif",
            fps=4,
            frames_per_pixel=2,
            pause_frames=0,
            dpi=45,
            profile_channels="not_a_channel",
        )
