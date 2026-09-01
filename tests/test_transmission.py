from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from seemc_imaging import (
    PopulationClassifier,
    RasterConfig,
    RasterDriver,
    Sample,
    Slab,
    SuspendedTrapezoidalLine,
    TransmissionDetector,
)
from seemc_imaging.geometry import SOLID_REGION, VACUUM_REGION
from seemc_imaging.transport import TrajectoryResult

from synthetic_material import write_synthetic_database


def _line(membrane_thickness=100.0):
    return SuspendedTrapezoidalLine(
        top_width=500.0,
        bottom_width=500.0 + 2.0 * 300.0 * math.tan(math.radians(2.0)),
        height=300.0,
        membrane_thickness=membrane_thickness,
    )


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def test_slab_is_bounded_on_both_sides():
    slab = Slab(top_z=0.0, thickness=100.0)
    assert slab.bottom_z == 100.0
    assert slab.region_at((0.0, 0.0, -1.0)) == VACUUM_REGION
    assert slab.region_at((0.0, 0.0, 50.0)) == SOLID_REGION
    assert slab.region_at((0.0, 0.0, 101.0)) == VACUUM_REGION

    # Unlike a semi-infinite Plane, a downward ray inside the solid escapes.
    hit = slab.first_hit((0.0, 0.0, 50.0), (0.0, 0.0, 1.0), 500.0, SOLID_REGION)
    assert hit is not None
    assert hit.surface_id.endswith(".bottom")
    assert hit.region_to == VACUUM_REGION
    assert hit.position[2] == pytest.approx(100.0)


def test_slab_rejects_non_positive_thickness():
    with pytest.raises(ValueError, match="thickness must be positive"):
        Slab(top_z=0.0, thickness=0.0)


def test_suspended_line_exposes_top_sidewall_membrane_and_underside():
    line = _line()
    assert line.total_thickness == pytest.approx(400.0)
    assert line.bottom_z == pytest.approx(100.0)

    landings = {
        0.0: ".top",
        255.0: ".right",
        400.0: ".membrane.top",
    }
    for x, suffix in landings.items():
        assert line.launch_surface(x).surface_id.endswith(suffix)

    # The underside is a real vacuum boundary, which is what allows a
    # transmitted signal to exist at all.
    hit = line.first_hit((0.0, 0.0, 50.0), (0.0, 0.0, 1.0), 5000.0, SOLID_REGION)
    assert hit.surface_id.endswith(".membrane.bottom")
    assert hit.region_to == VACUUM_REGION

    assert line.region_at((0.0, 0.0, -299.0)) == SOLID_REGION   # in the line
    assert line.region_at((0.0, 0.0, 99.0)) == SOLID_REGION     # in the membrane
    assert line.region_at((0.0, 0.0, 101.0)) == VACUUM_REGION   # below it
    assert line.region_at((900.0, 0.0, -50.0)) == VACUUM_REGION  # beside the line


def test_suspended_line_depth_is_measured_to_the_nearer_free_surface():
    line = _line(membrane_thickness=100.0)
    # Just under the line top.
    assert line.depth_into_solid((0.0, 0.0, -295.0)) == pytest.approx(5.0)
    # Mid-membrane away from the line: nearer face is whichever is closer.
    assert line.depth_into_solid((900.0, 0.0, 40.0)) == pytest.approx(40.0)
    assert line.depth_into_solid((900.0, 0.0, 80.0)) == pytest.approx(20.0)
    # Vacuum returns zero.
    assert line.depth_into_solid((900.0, 0.0, -50.0)) == 0.0


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------

def test_transmission_rings_tile_the_forward_hemisphere():
    detector = TransmissionDetector(bf_max_mrad=10.0, adf_max_mrad=50.0,
                                    haadf_max_mrad=200.0)
    assert detector.ring(0.0) == "bf"
    assert detector.ring(9.999) == "bf"
    assert detector.ring(10.0) == "adf"          # boundaries are exclusive
    assert detector.ring(49.9) == "adf"
    assert detector.ring(50.0) == "haadf"
    assert detector.ring(199.9) == "haadf"
    assert detector.ring(200.0) == "beyond_haadf"
    assert detector.ring(1570.0) == "beyond_haadf"


def test_transmission_detector_validates_ring_ordering():
    with pytest.raises(ValueError, match="bf_max_mrad < adf_max_mrad"):
        TransmissionDetector(bf_max_mrad=60.0, adf_max_mrad=50.0)
    with pytest.raises(ValueError, match="90 degrees"):
        TransmissionDetector(haadf_max_mrad=2000.0)


def test_transmission_requires_the_current_definition():
    with pytest.raises(ValueError, match="causal_lle_v3"):
        PopulationClassifier(definition="causal_lle_v2",
                             transmission=TransmissionDetector())


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def _emission(electron_id, energy, is_cascade, uvw, generation=1,
              surface_normal=None):
    """One emission.  The default exit face matches the travel direction:
    forward-going electrons leave through the far side (normal +z), backward
    ones through the entrance face (normal -z)."""
    if surface_normal is None:
        surface_normal = (0.0, 0.0, 1.0) if uvw[2] > 0 else (0.0, 0.0, -1.0)
    return SimpleNamespace(
        electron_id=electron_id,
        energy=float(energy),
        is_cascade=bool(is_cascade),
        generation=int(generation),
        surface_normal=tuple(float(v) for v in surface_normal),
        emission_mechanism="transport_escape",
        uvw=tuple(float(value) for value in uvw),
    )


def _two_sided_result():
    """One backward SE plus three forward electrons at known angles.

    A trajectory has exactly one incident primary, so only electron 0 is a
    root; everything else is a cascade electron born from it.
    """
    root = SimpleNamespace(electron_id=0, parent_id=None,
                           first_surface_return_event_id=3)
    records = [
        root,
        SimpleNamespace(electron_id=1, parent_id=0, birth_event_id=2,
                        parent_direction_before=(0.0, 0.0, 1.0)),
        SimpleNamespace(electron_id=2, parent_id=0, birth_event_id=4,
                        parent_direction_before=(0.0, 0.0, 1.0)),
        SimpleNamespace(electron_id=3, parent_id=0, birth_event_id=5,
                        parent_direction_before=(0.0, 0.0, 1.0)),
    ]
    history = SimpleNamespace(
        electrons=records,
        events=[SimpleNamespace(event_id=3, electron_id=0, kind="elastic")],
        incident_energy=30_000.0,
        incident_direction=(0.0, 0.0, 1.0),
        reference_surface_normal=(0.0, 0.0, -1.0),
    )
    forward = math.radians(1.0)          # 17.45 mrad -> ADF
    emissions = [
        _emission(1, 10.0, True, (0.0, 0.0, -1.0)),      # backward SE
        _emission(2, 10.0, True, (0.0, 0.0, 1.0)),       # forward SE, BF
        _emission(0, 29_990.0, False, (0.0, 0.0, 1.0)),  # forward primary, BF
        _emission(3, 10.0, True,
                  (math.sin(forward), 0.0, math.cos(forward))),  # forward SE, ADF
    ]
    return TrajectoryResult(tey=4, emissions=emissions, history=history)


def test_transmission_basis_partitions_every_emitted_electron():
    classifier = PopulationClassifier(transmission=TransmissionDetector())
    result = _two_sided_result()
    counts = classifier.classify(result)

    assert len(classifier.disjoint_channels) == 11
    assert sum(counts[name] for name in classifier.disjoint_channels) == counts["tey"]
    assert counts["backward_all"] + counts["forward_all"] == counts["tey"]
    assert counts["backward_all"] == 1
    assert counts["forward_all"] == 3


def test_forward_electrons_land_in_the_right_angular_ring():
    classifier = PopulationClassifier(transmission=TransmissionDetector())
    counts = classifier.classify(_two_sided_result())

    assert counts["fwd_bf"] == 2          # the two on-axis emissions
    assert counts["fwd_adf"] == 1         # 17.45 mrad
    assert counts["fwd_haadf"] == 0
    assert counts["fwd_beyond_haadf"] == 0
    assert counts["fwd_lateral_escape"] == 0
    assert counts["transmitted_all"] == 3
    # Of the two BF electrons only one is an original primary; the other is a
    # secondary emitted through the underside, which a real BF detector cannot
    # distinguish but the simulation can.
    assert counts["fwd_bf_primary"] == 1
    assert counts["forward_cascade_all"] == 2
    assert counts["forward_primary_all"] == 1


def test_hemisphere_free_aggregates_keep_their_meaning():
    """se1/se2 and the LLE pair must still count both hemispheres."""
    classifier = PopulationClassifier(transmission=TransmissionDetector())
    counts = classifier.classify(_two_sided_result())

    assert counts["se1"] + counts["se2"] == counts["cascade_all"] == 3
    assert counts["lle_primary"] + counts["non_lle_primary"] == counts["primary_all"]
    # The backward-restricted channels are a strict subset.
    back_cascade = sum(
        counts[f"back_{name}"]
        for name in ("se1_lt50", "se1_ge50", "se2_lt50", "se2_ge50")
    )
    assert back_cascade == 1 < counts["cascade_all"]


def test_ring_widths_change_the_split_but_not_the_total():
    result = _two_sided_result()
    narrow = PopulationClassifier(
        transmission=TransmissionDetector(bf_max_mrad=1.0, adf_max_mrad=5.0,
                                          haadf_max_mrad=10.0)
    ).classify(result)
    wide = PopulationClassifier(
        transmission=TransmissionDetector(bf_max_mrad=100.0, adf_max_mrad=200.0,
                                          haadf_max_mrad=300.0)
    ).classify(result)

    assert narrow["forward_all"] == wide["forward_all"] == 3
    assert narrow["fwd_beyond_haadf"] == 1     # 17.45 mrad now falls outside
    assert wide["fwd_bf"] == 3                 # everything inside the BF disc


def test_sidewall_escape_is_forward_but_not_transmitted():
    """The confound that matters on a topographic specimen.

    A secondary leaving a near-vertical sidewall travels sideways and slightly
    downward.  Its velocity is forward-going, but it escaped into the trench
    beside the line without crossing anything, so it must not be counted as a
    transmitted (STEM) electron.
    """
    result = _two_sided_result()
    sidewall = math.radians(2.0)
    # Outward normal of a 2-degree trapezoid sidewall: almost horizontal,
    # tilted very slightly back toward the source.
    result.emissions[1] = _emission(
        2, 10.0, True,
        (math.cos(math.radians(5.0)), 0.0, math.sin(math.radians(5.0))),
        surface_normal=(math.cos(sidewall), 0.0, -math.sin(sidewall)),
    )
    counts = PopulationClassifier(
        transmission=TransmissionDetector()
    ).classify(result)

    assert counts["forward_all"] == 3            # still forward-going
    assert counts["fwd_lateral_escape"] == 1     # but not transmitted
    assert counts["transmitted_all"] == 2
    assert counts["fwd_bf"] == 1                 # was 2 before the sidewall swap


def test_classifier_without_transmission_is_unchanged():
    plain = PopulationClassifier()
    assert "forward_all" not in plain.channels
    assert plain.disjoint_channels == (
        "se1_lt50", "se1_ge50", "se2_lt50", "se2_ge50",
        "lle_primary", "non_lle_primary",
    )
    assert plain.to_dict()["transmission"] is None


# --------------------------------------------------------------------------
# Raster integration
# --------------------------------------------------------------------------

def test_raster_on_a_membrane_records_both_hemispheres(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)
    # Thin enough that the synthetic material transmits beside the line.
    line = _line(membrane_thickness=40.0)
    config = RasterConfig(
        energy_ev=500.0,
        x_positions=(-400.0, 0.0, 400.0),
        y_positions=(0.0,),
        primaries_per_pixel=8,
        seed=7,
    )
    classifier = PopulationClassifier(transmission=TransmissionDetector())
    result = RasterDriver(sample, line, config, classifier).run(
        use_parallel=False, progress=False
    )

    basis = sum(result.count_maps[name] for name in classifier.disjoint_channels)
    assert np.array_equal(basis, result.count_maps["tey"])
    assert np.array_equal(
        result.count_maps["tey"],
        result.count_maps["backward_all"] + result.count_maps["forward_all"],
    )
    forward_rings = sum(
        result.count_maps[f"fwd_{ring}"]
        for ring in ("bf", "adf", "haadf", "beyond_haadf")
    )
    assert np.array_equal(forward_rings, result.count_maps["transmitted_all"])
    assert np.array_equal(
        forward_rings + result.count_maps["fwd_lateral_escape"],
        result.count_maps["forward_all"],
    )

    # The thick line blocks transmission that the bare membrane allows.
    on_line = result.count_maps["transmitted_all"][0][1]
    off_line = result.count_maps["transmitted_all"][0][0]
    assert on_line == 0
    assert off_line > 0


def test_landing_energy_above_the_table_range_warns(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)
    config = RasterConfig(
        energy_ev=10.0 * sample.Emax,
        x_positions=(-10.0, 10.0),
        y_positions=(0.0,),
        primaries_per_pixel=1,
    )
    with pytest.warns(RuntimeWarning, match="exceeds the .* table range"):
        RasterDriver(sample, _line(), config)
