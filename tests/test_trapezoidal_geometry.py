from __future__ import annotations

import math

import numpy as np
import pytest

from seemc_imaging import (
    SOLID_REGION,
    VACUUM_REGION,
    Sample,
    TrapezoidalLine,
    TrapezoidalPrism,
    refract_incident_direction,
    simulate_trajectory,
)

from synthetic_material import write_synthetic_database


def _line(scale=1.0):
    return TrapezoidalLine(
        top_width=100.0 * scale,
        bottom_width=200.0 * scale,
        height=100.0 * scale,
    )


def test_trapezoidal_prism_validates_dimensions():
    with pytest.raises(ValueError, match="positive"):
        TrapezoidalPrism(0.0, 200.0, 100.0)
    with pytest.raises(ValueError, match="undercut"):
        TrapezoidalPrism(200.0, 100.0, 100.0)


def test_vertical_beam_intersects_top_sidewalls_and_substrate_exactly():
    line = _line()
    expected = {
        0.0: ((0.0, 0.0, -100.0), "trapezoidal_line.top"),
        49.0: ((49.0, 0.0, -100.0), "trapezoidal_line.top"),
        75.0: ((75.0, 0.0, -50.0), "trapezoidal_line.right"),
        -75.0: ((-75.0, 0.0, -50.0), "trapezoidal_line.left"),
        110.0: ((110.0, 0.0, 0.0), "trapezoidal_line.substrate"),
    }
    for x, (position, surface_id) in expected.items():
        hit = line.launch_surface(x)
        assert hit.position == pytest.approx(position, abs=1e-13)
        assert hit.surface_id == surface_id
        assert hit.region_from == VACUUM_REGION
        assert hit.region_to == SOLID_REGION

    slope_normal = np.asarray((1.0, 0.0, -0.5), dtype=float)
    slope_normal /= np.linalg.norm(slope_normal)
    assert line.surface_normal_at((75.0, 0.0, -50.0)) == pytest.approx(
        slope_normal, abs=1e-15
    )


def test_vertical_beam_resolves_nearly_vertical_sidewall_boundary_band():
    line = TrapezoidalLine(
        top_width=500.0,
        bottom_width=700.0,
        height=500.0,
    )
    hit = line.launch_surface(260.0)
    assert hit.surface_id == "trapezoidal_line.right"
    assert hit.position == pytest.approx((260.0, 0.0, -450.0), abs=1e-11)


@pytest.mark.parametrize("scale", [1.0, 1.0e3, 1.0e6])
def test_line_intersections_are_scale_invariant(scale):
    line = _line(scale)
    hit = line.launch_surface(75.0 * scale)
    assert np.asarray(hit.position) / scale == pytest.approx(
        (75.0, 0.0, -50.0), abs=1e-11
    )
    assert hit.surface_id.endswith(".right")


def test_scene_union_suppresses_buried_base_and_substrate_seam():
    line = _line()

    # Starting in the substrate beneath the line, z=0 is not a boundary: the
    # prism continues the same solid up to its exposed top at z=-100.
    hit = line.first_hit(
        (0.0, 0.0, 50.0), (0.0, 0.0, -1.0), 200.0, SOLID_REGION
    )
    assert hit.distance == 150.0
    assert hit.position == (0.0, 0.0, -100.0)
    assert hit.surface_id == "trapezoidal_line.top"

    # Outside the base footprint the substrate plane remains exposed.
    hit = line.first_hit(
        (110.0, 0.0, 50.0), (0.0, 0.0, -1.0), 200.0, SOLID_REGION
    )
    assert hit.distance == 50.0
    assert hit.surface_id == "trapezoidal_line.substrate"

    # Preserve the collision-at-the-exact-endpoint precedence rule.
    assert line.first_hit(
        (110.0, 0.0, 50.0), (0.0, 0.0, -1.0), 50.0, SOLID_REGION
    ) is None


def test_surface_start_selects_only_the_outgoing_crossing():
    line = _line()
    point = line.surface_point(75.0)
    outward = line.surface_normal_at(point)
    inward = tuple(-value for value in outward)

    exiting = line.first_hit(point, outward, 500.0, SOLID_REGION)
    assert exiting.distance == pytest.approx(0.0, abs=1e-14)
    assert exiting.surface_id.endswith(".right")

    # The coincident face at t=0 is skipped when moving inward.  Along this
    # normal the ray continues from the line into the semi-infinite substrate,
    # so there is no later exposed crossing within the sampled segment.
    next_hit = line.first_hit(point, inward, 500.0, SOLID_REGION)
    assert next_hit is None


def test_depth_is_measured_to_the_exposed_union_boundary():
    line = _line()
    assert line.depth_into_solid((0.0, 0.0, -90.0)) == pytest.approx(10.0)
    assert line.depth_into_solid((150.0, 0.0, 20.0)) == pytest.approx(20.0)
    assert line.depth_into_solid((0.0, 0.0, -101.0)) == 0.0


def test_fixed_global_beam_refracts_against_sidewall_normal(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)
    line = _line()
    hit = line.launch_surface(75.0, vacuum_direction=(0.0, 0.0, 1.0))
    outward = line.surface_normal_at(
        hit.position, incoming_direction=(0.0, 0.0, 1.0)
    )

    energy, direction = refract_incident_direction(
        500.0, sample, (0.0, 0.0, 1.0), outward
    )
    direction = np.asarray(direction)
    outward = np.asarray(outward)
    vacuum_local_cosine = -float(np.dot((0.0, 0.0, 1.0), outward))
    vacuum_local_sine = math.sqrt(1.0 - vacuum_local_cosine ** 2)
    solid_local_sine = math.sqrt(500.0 / energy) * vacuum_local_sine

    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-15)
    assert np.dot(direction, -outward) == pytest.approx(
        math.sqrt(1.0 - solid_local_sine ** 2), abs=1e-15
    )


@pytest.mark.parametrize("x, expected_angle", [(0.0, 0.0), (75.0, math.atan(2.0))])
def test_complete_transport_on_line_records_exposed_surfaces(
        tmp_path, x, expected_angle):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)
    line = _line()
    launch = line.surface_point(x)
    result = simulate_trajectory(
        sample,
        500.0,
        0.0,
        np.random.default_rng(42),
        history=True,
        geometry=line,
        launch_position=launch,
        vacuum_direction=(0.0, 0.0, 1.0),
    )

    assert result.history.launch_position == launch
    assert result.history.incident_angle == pytest.approx(expected_angle, abs=1e-15)
    assert result.tey > 0
    assert all(not emission.surface_id.endswith(".base")
               for emission in result.emissions)
    assert all(np.dot(emission.uvw, emission.surface_normal) > 0.0
               for emission in result.emissions)
    boundary_events = [
        event for event in result.history.events
        if event.kind in {"emission", "surface_reflection"}
    ]
    assert boundary_events
    assert all(not event.surface_id.endswith(".base") for event in boundary_events)
