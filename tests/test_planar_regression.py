from __future__ import annotations

import math

import numpy as np
import pytest

from seemc_imaging import Sample, SurfaceHit, simulate_trajectory
from seemc_imaging.transport import Electron, barrier_transmission

from synthetic_material import write_synthetic_database


# PROVENANCE.
#
# The original snapshot below was generated from the untouched validated optlib
# SEEMC snapshot, before the incoming-barrier transmission step existed.  That
# change made the vacuum->solid barrier a sampled event instead of an implicit
# transmission probability of one, which both alters the physics and shifts
# every downstream RNG draw: seed 42 now diverges from its first step (4/2
# collisions then, 133/281 now).  The old values therefore cannot be met by any
# version of the current transport and are retained only as a record of what
# the fork boundary looked like before that fix.
#
# GOLDEN below was regenerated against seemc_imaging 0.7.4 with numpy 2.4.4 and
# CPython 3.11.  Regenerate it deliberately, never to make a red test pass: a
# diff here means either an intended transport change (record it in this
# comment) or a real regression.
PRE_BARRIER_TRANSMISSION_GOLDEN = {
    3: {
        "counts": (3, 2, 1, 2, 1),
        "diagnostics": {
            "inelastic_events": 26,
            "elastic_events": 46,
            "surface_encounters": 3,
            "escapes": 3,
            "se_created": 14,
        },
    },
    42: {
        "counts": (1, 0, 1, 0, 1),
        "diagnostics": {
            "inelastic_events": 4,
            "elastic_events": 2,
            "surface_encounters": 1,
            "escapes": 1,
            "se_created": 2,
        },
    },
}

GOLDEN = {
    3: {
        "counts": (1, 0, 1, 0, 1),
        "emissions": (
            (402.10451508748577, 0.5267744311071673, False, 0, 0.0),
        ),
        "diagnostics": {
            "inelastic_events": 26,
            "elastic_events": 46,
            "surface_encounters": 2,
            "escapes": 1,
            "internal_reflections": 1,
            "incoming_barrier_encounters": 1,
            "incoming_barrier_reflections": 0,
            "incoming_barrier_transmissions": 1,
            "se_created": 11,
            "se_blocked_pauli": 0,
            "se_pauli_fallback": 0,
            "channel_reclassified": 12,
            "se_below_barrier": 15,
            "omega_cdf_empty": 0,
            "q_window_clipped": 0,
            "q_cdf_empty": 0,
            "step_limit_hit": 0,
            "generation_limit_hit": 0,
        },
    },
    42: {
        "counts": (1, 1, 0, 1, 0),
        "emissions": (
            (3.419013842609088, 0.8803296665367016, True, 1,
             9.925218427135981),
        ),
        "diagnostics": {
            "inelastic_events": 133,
            "elastic_events": 281,
            "surface_encounters": 4,
            "escapes": 1,
            "internal_reflections": 3,
            "incoming_barrier_encounters": 1,
            "incoming_barrier_reflections": 0,
            "incoming_barrier_transmissions": 1,
            "se_created": 67,
            "se_blocked_pauli": 0,
            "se_pauli_fallback": 0,
            "channel_reclassified": 51,
            "se_below_barrier": 66,
            "omega_cdf_empty": 0,
            "q_window_clipped": 0,
            "q_cdf_empty": 0,
            "step_limit_hit": 0,
            "generation_limit_hit": 0,
        },
    },
}


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_matches_validated_planar_snapshot(tmp_path, seed):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)
    result = simulate_trajectory(
        sample, 500.0, 0.35, np.random.default_rng(seed), history=True
    )
    expected = GOLDEN[seed]

    assert (
        result.tey,
        result.sey_cascade,
        result.bse_cascade,
        result.sey_50ev,
        result.bse_50ev,
    ) == expected["counts"]
    assert dict(result.diagnostics) == expected["diagnostics"]

    actual_emissions = [
        (e.energy, e.uz, e.is_cascade, e.generation, e.birth_depth)
        for e in result.emissions
    ]
    assert len(actual_emissions) == len(expected["emissions"])
    for actual, golden in zip(actual_emissions, expected["emissions"]):
        assert actual[2:4] == golden[2:4]
        assert np.allclose(
            (actual[0], actual[1], actual[4]),
            (golden[0], golden[1], golden[4]),
            rtol=0.0,
            atol=1e-12,
        )


def _legacy_travel(self):
    """The pre-geometry hard-coded z=0 free-flight implementation."""
    rate = self.refresh_rates()
    if (not np.isfinite(rate)) or rate <= 0.0:
        self.dead = True
        self.fate = "no_scattering_rate"
        self.last_step_length = 0.0
        return False

    s = -math.log(max(self.rng.random(), 1e-300)) / rate
    hit_surface = False
    if self.uvw[2] < -1e-15:
        s_to_surface = -self.xyz[2] / self.uvw[2]
        if 0.0 <= s_to_surface < s:
            s = s_to_surface
            hit_surface = True

    self.path_length += s
    self.last_step_length = s
    self.xyz[0] += self.uvw[0] * s
    self.xyz[1] += self.uvw[1] * s
    self.xyz[2] += self.uvw[2] * s
    if hit_surface:
        self.xyz[2] = 0.0
        self.last_surface_hit = SurfaceHit(
            distance=s,
            position=tuple(self.xyz),
            normal=(0.0, 0.0, -1.0),
            surface_id="sample_plane",
            region_from="solid",
            region_to="vacuum",
            primitive_id=0,
        )
    else:
        self.last_surface_hit = None
    self._record()
    return hit_surface


def _legacy_escape(self):
    """The pre-geometry hard-coded z=0 barrier implementation."""
    energy_solid = self.energy
    ux, uy, uz = self.uvw
    perpendicular_energy = energy_solid * uz * uz
    if energy_solid <= self.Ui or perpendicular_energy <= self.Ui:
        self._reflect()
        return False

    transmission = barrier_transmission(
        perpendicular_energy, self.Ui, self.cfg
    )
    if transmission < 1.0 and self.rng.random() >= transmission:
        self._reflect()
        return False

    energy_vacuum = energy_solid - self.Ui
    scale = math.sqrt(energy_solid / energy_vacuum)
    ux_out = ux * scale
    uy_out = uy * scale
    uz_out = -math.sqrt(max(1.0 - (ux_out * ux_out + uy_out * uy_out), 0.0))
    self.inside = False
    self.fate = "emitted"
    self.current_region = "vacuum"
    self.uvw = [ux_out, uy_out, uz_out]
    self.energy = energy_vacuum
    self.xyz[2] = 0.0
    self._record()
    return True


def _legacy_reflect(self, hit=None):
    self.uvw[2] = abs(self.uvw[2])
    self.xyz[2] = 0.0
    self._record()


@pytest.mark.parametrize(
    "seed, energy, angle",
    [
        (3, 500.0, 0.35),
        (42, 500.0, 0.35),
        (1234, 100.0, 0.0),       # includes two internal reflections
        (91, 1000.0, 1.4),        # grazing incidence
        (5, 50.0, 0.8),           # includes three internal reflections
    ],
)
def test_plane_backend_is_bitwise_identical_to_hard_coded_surface(
        tmp_path, monkeypatch, seed, energy, angle):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    sample = Sample("Synthetic", db_path=db_path)

    with monkeypatch.context() as legacy_methods:
        legacy_methods.setattr(Electron, "travel", _legacy_travel)
        legacy_methods.setattr(Electron, "escape", _legacy_escape)
        legacy_methods.setattr(Electron, "_reflect", _legacy_reflect)
        legacy = simulate_trajectory(
            sample, energy, angle, np.random.default_rng(seed),
            track=True, history=True, trajectory_id=seed,
        )

    backend = simulate_trajectory(
        sample, energy, angle, np.random.default_rng(seed),
        track=True, history=True, trajectory_id=seed,
    )

    assert backend == legacy
    assert all(event.surface_id == "sample_plane"
               for event in backend.history.events
               if event.kind in {"emission", "surface_reflection"})
    assert all(emission.surface_normal == (0.0, 0.0, -1.0)
               for emission in backend.emissions)
