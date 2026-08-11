from __future__ import annotations

import math

import numpy as np
import pytest

from seemc_imaging import (
    Electron,
    MCConfig,
    Plane,
    Sample,
    SOLID_REGION,
    VACUUM_REGION,
    incident_direction,
    simulate_trajectory,
)

from synthetic_material import write_synthetic_database


def test_reference_plane_regions_and_first_hit_contract():
    plane = Plane()

    assert plane.region_at((0.0, 0.0, 1.0)) == SOLID_REGION
    assert plane.region_at((0.0, 0.0, 0.0)) == SOLID_REGION
    assert plane.region_at((0.0, 0.0, -1.0)) == VACUUM_REGION

    hit = plane.first_hit(
        origin=(1.0, 2.0, 5.0),
        direction=(0.0, 0.0, -1.0),
        max_distance=6.0,
        current_region=SOLID_REGION,
    )
    assert hit.distance == 5.0
    assert hit.position == (1.0, 2.0, 0.0)
    assert hit.normal == (0.0, 0.0, -1.0)
    assert hit.surface_id == "sample_plane"
    assert hit.region_from == SOLID_REGION
    assert hit.region_to == VACUUM_REGION
    assert hit.primitive_id == 0

    # The historical transport gives a collision at an exactly coincident
    # free-path endpoint precedence over the boundary.
    assert plane.first_hit(
        (1.0, 2.0, 5.0), (0.0, 0.0, -1.0), 5.0, SOLID_REGION
    ) is None
    assert plane.first_hit(
        (0.0, 0.0, 5.0), (1.0, 0.0, 0.0), 10.0, SOLID_REGION
    ) is None
    assert plane.first_hit(
        (0.0, 0.0, 5.0), (0.0, 0.0, 1.0), 10.0, SOLID_REGION
    ) is None

    surface_hit = plane.first_hit(
        (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 1.0, SOLID_REGION
    )
    assert surface_hit.distance == 0.0


def test_plane_crossing_from_vacuum_reorients_hit_normal():
    plane = Plane()
    hit = plane.first_hit(
        (0.0, 0.0, -3.0), (0.0, 0.0, 1.0), 4.0, VACUUM_REGION
    )
    assert hit.distance == 3.0
    assert hit.position == (0.0, 0.0, 0.0)
    assert hit.normal == (0.0, 0.0, 1.0)
    assert hit.region_from == VACUUM_REGION
    assert hit.region_to == SOLID_REGION


def test_translated_rotated_plane_has_exact_distance_and_normal():
    plane = Plane(
        point=(2.0, -4.0, 7.0),
        outward_normal=(10.0, 0.0, 0.0),
        surface_id="rotated",
    )
    assert plane.region_at((1.0, 99.0, -20.0)) == SOLID_REGION
    assert plane.region_at((3.0, 99.0, -20.0)) == VACUUM_REGION

    hit = plane.first_hit(
        (0.0, 5.0, 9.0), (1.0, 0.0, 0.0), 3.0, SOLID_REGION
    )
    assert hit.distance == 2.0
    assert hit.position == (2.0, 5.0, 9.0)
    assert hit.normal == (1.0, 0.0, 0.0)
    assert hit.surface_id == "rotated"
    assert plane.depth_into_solid((0.5, 100.0, -100.0)) == 1.5
    assert plane.lateral_distance((2.0, 2.0, 15.0), plane.point) == 10.0


def test_plane_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-zero"):
        Plane(outward_normal=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="must differ"):
        Plane(solid_region="same", vacuum_region="same")
    with pytest.raises(ValueError, match="max_distance"):
        Plane().first_hit((0, 0, 1), (0, 0, -1), -1.0, SOLID_REGION)
    with pytest.raises(ValueError, match="current_region"):
        Plane().first_hit((0, 0, 1), (0, 0, -1), 2.0, "unknown")


def test_incident_direction_uses_local_plane_frame(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)
    outward = np.asarray((1.0, 2.0, -3.0), dtype=float)
    outward /= np.linalg.norm(outward)
    angle = 0.7

    solid_energy, direction = incident_direction(
        500.0, sample, angle, surface_normal=outward, azimuth_rad=1.1
    )
    direction = np.asarray(direction)
    refracted_sine = math.sqrt(500.0 / solid_energy) * math.sin(angle)

    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-15)
    assert np.dot(direction, -outward) == pytest.approx(
        math.sqrt(1.0 - refracted_sine ** 2), abs=1e-15
    )


def test_rotated_plane_barrier_reflects_and_transmits_in_local_frame(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    cfg = MCConfig(barrier_model="classical")
    sample = Sample("Synthetic", db_path=db_path, config=cfg)
    plane = Plane(point=(2.0, 0.0, 0.0), outward_normal=(1.0, 0.0, 0.0))

    escaping = Electron(
        sample, sample.Ui + 50.0, plane.point, (1.0, 0.0, 0.0),
        rng=np.random.default_rng(1), geometry=plane,
    )
    assert escaping.escape()
    assert escaping.current_region == VACUUM_REGION
    assert escaping.uvw == [1.0, 0.0, 0.0]
    assert escaping.energy == pytest.approx(50.0, abs=1e-15)

    reflected = Electron(
        sample, 0.5 * sample.Ui, plane.point, (1.0, 0.0, 0.0),
        rng=np.random.default_rng(2), geometry=plane,
    )
    assert not reflected.escape()
    assert reflected.current_region == SOLID_REGION
    assert reflected.uvw == [-1.0, 0.0, 0.0]


def test_complete_transport_uses_translated_rotated_plane(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)
    plane = Plane(
        point=(2.0, -4.0, 7.0),
        outward_normal=(1.0, 0.0, 0.0),
        surface_id="rotated",
    )
    result = simulate_trajectory(
        sample, 500.0, 0.35, np.random.default_rng(3),
        history=True, geometry=plane,
    )

    assert result.history.launch_position == plane.point
    assert result.tey > 0
    for emission in result.emissions:
        assert emission.xyz[0] == 2.0
        assert emission.surface_id == "rotated"
        assert emission.surface_normal == (1.0, 0.0, 0.0)
        assert emission.uz > 0.0
    for event in result.history.events:
        if event.kind in {"emission", "surface_reflection"}:
            assert event.position[0] == 2.0
            assert event.surface_id == "rotated"
