"""Deterministic, population-resolved SEM raster simulation.

The transport kernel works in Angstrom.  A raster pixel is a nominal parallel
beam ray; each primary is displaced by a two-dimensional Gaussian in the plane
normal to the fixed laboratory beam direction.  Beam-position and transport
random streams are deliberately separate, so changing the spot size does not
change a primary's collision stream after its launch point has been selected.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .transport import Sample, TrajectoryResult, simulate_trajectory
from .trajectory import RasterTrajectoryArchive


FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


BASE_CHANNEL_DEFINITIONS = {
    "tey": "All emitted electrons.",
    "sey_50ev": "Emitted electrons below the configured energy cutoff.",
    "bse_50ev": "Emitted electrons at or above the configured energy cutoff.",
    "cascade_all": "All emitted electrons born in the simulated cascade.",
    "primary_all": "Emitted incident primaries, independent of final energy.",
    "se_cascade_lt50": (
        "Cascade-origin emissions below the configured energy cutoff."
    ),
    "fast_cascade_ge50": (
        "Cascade-origin emissions at or above the configured energy cutoff."
    ),
    "slow_primary_lt50": (
        "Incident-primary emissions below the configured energy cutoff."
    ),
    "bse_primary_ge50": (
        "Incident-primary emissions at or above the configured energy cutoff."
    ),
    "generation_1": "Emitted cascade electrons with generation == 1.",
    "generation_2plus": "Emitted cascade electrons with generation >= 2.",
}

CAUSAL_LLE_V3_CHANNEL_DEFINITIONS = {
    "se1": (
        "causal_lle_v3: cascade emission, any final energy, created while the "
        "classifying parent was directed into the configured reference "
        "surface.  Aggregate of se1_lt50 and se1_ge50."
    ),
    "se2": (
        "causal_lle_v3: cascade emission, any final energy, created while the "
        "classifying parent was directed toward vacuum through the configured "
        "reference surface.  Aggregate of se2_lt50 and se2_ge50."
    ),
    "se1_lt50": (
        "causal_lle_v3: SE1 emitted below the configured energy cutoff."
    ),
    "se1_ge50": (
        "causal_lle_v3: SE1 emitted at or above the configured energy cutoff."
    ),
    "se2_lt50": (
        "causal_lle_v3: SE2 emitted below the configured energy cutoff."
    ),
    "se2_ge50": (
        "causal_lle_v3: SE2 emitted at or above the configured energy cutoff."
    ),
    "lle_primary": (
        "causal_lle_v3: emitted original incident electron with vacuum "
        "energy loss below the configured LLE threshold."
    ),
    "non_lle_primary": (
        "causal_lle_v3: emitted original incident electron with vacuum "
        "energy loss at or above the configured LLE threshold."
    ),
    "first_event_backscatter": (
        "Diagnostic: emitted original incident electron whose first completed "
        "collision was an elastic event that turned it toward the launch "
        "surface.  Scattering-history class; overlaps LLE and is not part of "
        "the disjoint basis."
    ),
    "later_return_primary": (
        "Diagnostic complement of first_event_backscatter: every other "
        "emitted original incident electron, including those that never "
        "entered the solid."
    ),
    "barrier_reflected_primary": (
        "Diagnostic: emitted original incident electron reflected by the "
        "surface barrier on entry.  It never entered the solid, so its vacuum "
        "energy loss is exactly zero and it is an LLE for any positive "
        "threshold.  Subset of lle_primary, not part of the disjoint basis."
    ),
}

CAUSAL_LLE_V2_CHANNEL_DEFINITIONS = {
    "se1": (
        "causal_lle_v2: low-energy cascade emission created while its "
        "immediate energetic parent was directed into the configured "
        "reference surface."
    ),
    "se2": (
        "causal_lle_v2: low-energy cascade emission created while its "
        "immediate energetic parent was directed toward vacuum through the "
        "configured reference surface."
    ),
    "lle_primary": (
        "causal_lle_v2: emitted original incident electron with vacuum "
        "energy loss below the configured LLE threshold."
    ),
    "non_lle_primary": (
        "causal_lle_v2: emitted original incident electron with vacuum "
        "energy loss at or above the configured LLE threshold."
    ),
    "first_event_bse": (
        "Diagnostic: emitted original incident electron whose first completed "
        "collision was an elastic event that turned it toward the launch "
        "surface."
    ),
    "later_return_bse": (
        "Diagnostic complement of first_event_bse: every other emitted "
        "original incident electron."
    ),
}

# Retained name for code written against the 0.7.x default.
CAUSAL_LLE_CHANNEL_DEFINITIONS = CAUSAL_LLE_V2_CHANNEL_DEFINITIONS

LEGACY_BRANCH_V1_CHANNEL_DEFINITIONS = {
    "se1": (
        "legacy branch_v1: low-energy cascade emission born before the "
        "incident primary first turned toward the launch surface."
    ),
    "se2": (
        "legacy branch_v1: low-energy cascade emission born after the "
        "incident primary first turned toward the launch surface."
    ),
    "bse1": (
        "legacy branch_v1: emitted incident primary whose first turn toward "
        "the surface was caused by its first elastic collision."
    ),
    "bse2": (
        "legacy branch_v1: every other emitted incident primary."
    ),
}

# Public defaults describe the current classifier.  The older constants remain
# available through PopulationClassifier(definition=...) so archived
# causal_lle_v2 and 0.6.1-era branch_v1 calculations reproduce exactly.
CHANNEL_DEFINITIONS = {
    **BASE_CHANNEL_DEFINITIONS,
    **CAUSAL_LLE_V3_CHANNEL_DEFINITIONS,
}
V2_CHANNEL_DEFINITIONS = {
    **BASE_CHANNEL_DEFINITIONS,
    **CAUSAL_LLE_V2_CHANNEL_DEFINITIONS,
}
LEGACY_CHANNEL_DEFINITIONS = {
    **BASE_CHANNEL_DEFINITIONS,
    **LEGACY_BRANCH_V1_CHANNEL_DEFINITIONS,
}

POPULATION_CHANNELS = tuple(CHANNEL_DEFINITIONS)
V2_POPULATION_CHANNELS = tuple(V2_CHANNEL_DEFINITIONS)
LEGACY_POPULATION_CHANNELS = tuple(LEGACY_CHANNEL_DEFINITIONS)

# This basis partitions every emitted electron exactly once.  It is therefore
# safe for covariance-aware joint fits, unlike overlapping collections such as
# (TEY, SEY, BSE, SE1, SE2).
#
# The causal SE class and the detector energy cut are separate axes, so the
# default basis crosses them rather than letting the 50 eV convention decide
# which cascade electrons receive a causal label at all.  Aggregates follow:
#   se1 = se1_lt50 + se1_ge50,  se2 = se2_lt50 + se2_ge50,
#   se_cascade_lt50 = se1_lt50 + se2_lt50,
#   fast_cascade_ge50 = se1_ge50 + se2_ge50.
DISJOINT_POPULATION_CHANNELS = (
    "se1_lt50",
    "se1_ge50",
    "se2_lt50",
    "se2_ge50",
    "lle_primary",
    "non_lle_primary",
)

V2_DISJOINT_POPULATION_CHANNELS = (
    "se1",
    "se2",
    "fast_cascade_ge50",
    "lle_primary",
    "non_lle_primary",
)

LEGACY_DISJOINT_POPULATION_CHANNELS = (
    "se1",
    "se2",
    "fast_cascade_ge50",
    "bse1",
    "bse2",
)

# Absolute default LLE threshold.  It is numerically equal to the conventional
# 50 eV SE/BSE emission cut, but the two are unrelated conventions: this one is
# an energy *loss* window on emitted primaries, that one is an emitted-energy
# cut.  They are stored separately in metadata and may be set independently.
DEFAULT_LLE_MAX_LOSS_EV = 50.0

CAUSAL_DEFINITIONS = ("causal_lle_v3", "causal_lle_v2")
SE_PARENT_RULES = ("root_primary_leg", "immediate_parent")

# Exit hemisphere is a third orthogonal axis, alongside the causal class and
# the emitted-energy cut.  It only becomes observable on a geometry that has a
# bottom exit surface -- a free-standing membrane -- where forward-going
# electrons leave the specimen instead of being absorbed in a bulk substrate.
TRANSMISSION_RINGS = ("bf", "adf", "haadf", "beyond_haadf")

TRANSMISSION_CHANNEL_DEFINITIONS = {
    "backward_all": (
        "All emitted electrons leaving with a velocity component back along "
        "the incident beam axis: the reflected (SEM) hemisphere."
    ),
    "forward_all": (
        "All emitted electrons leaving with a velocity component along the "
        "incident beam axis: the transmitted (STEM) hemisphere."
    ),
    "forward_primary_all": (
        "Diagnostic: transmitted original incident electrons."
    ),
    "forward_cascade_all": (
        "Diagnostic: transmitted cascade electrons, i.e. secondaries emitted "
        "through the membrane underside."
    ),
    "fwd_bf": (
        "Transmitted electron inside the bright-field collection angle, "
        "measured as polar angle from the forward beam axis."
    ),
    "fwd_adf": (
        "Transmitted electron in the annular dark-field ring."
    ),
    "fwd_haadf": (
        "Transmitted electron in the high-angle annular dark-field ring."
    ),
    "fwd_beyond_haadf": (
        "Transmitted electron scattered beyond the outer HAADF angle.  It "
        "closes the forward partition and is not a physical detector."
    ),
    "fwd_bf_primary": (
        "Diagnostic: original incident electrons within the bright-field "
        "angle.  Comparing it with fwd_bf shows how much of the BF disc is "
        "secondary-electron contamination."
    ),
    "fwd_adf_primary": (
        "Diagnostic: original incident electrons in the ADF ring."
    ),
    "fwd_haadf_primary": (
        "Diagnostic: original incident electrons in the HAADF ring."
    ),
}


@dataclass(frozen=True)
class TransmissionDetector:
    """Angular segmentation of the transmitted (forward) hemisphere.

    Ring boundaries are polar angles from the forward beam axis in
    milliradians.  They are exclusive upper bounds, so ``bf`` is
    ``theta < bf_max_mrad``.  ``beyond_haadf`` collects everything out to 90
    degrees so that the four rings partition the forward hemisphere exactly.

    These are collection angles, not a detector response: solid-angle
    weighting, gain, and the finite BF-disc/detector-hole geometry of a real
    STEM-in-SEM holder are forward-model steps applied afterwards.
    """

    bf_max_mrad: float = 10.0
    adf_max_mrad: float = 50.0
    haadf_max_mrad: float = 200.0

    def __post_init__(self):
        values = (
            float(self.bf_max_mrad),
            float(self.adf_max_mrad),
            float(self.haadf_max_mrad),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("transmission ring angles must be finite")
        if not 0.0 < values[0] < values[1] < values[2]:
            raise ValueError(
                "transmission ring angles must satisfy "
                "0 < bf_max_mrad < adf_max_mrad < haadf_max_mrad"
            )
        if values[2] > 0.5 * math.pi * 1000.0:
            raise ValueError(
                "haadf_max_mrad must not exceed 90 degrees (1570.8 mrad); "
                "the forward hemisphere ends there"
            )
        object.__setattr__(self, "bf_max_mrad", values[0])
        object.__setattr__(self, "adf_max_mrad", values[1])
        object.__setattr__(self, "haadf_max_mrad", values[2])

    def ring(self, theta_mrad):
        """Ring name for one forward polar angle in milliradians."""
        if theta_mrad < self.bf_max_mrad:
            return "bf"
        if theta_mrad < self.adf_max_mrad:
            return "adf"
        if theta_mrad < self.haadf_max_mrad:
            return "haadf"
        return "beyond_haadf"

    def to_dict(self):
        return {
            "bf_max_mrad": self.bf_max_mrad,
            "adf_max_mrad": self.adf_max_mrad,
            "haadf_max_mrad": self.haadf_max_mrad,
            "angle_reference": "polar_angle_from_forward_beam_axis",
        }


def _vec3(values, name):
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain three finite numbers")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return tuple(value / norm for value in vector)


def _dot(a, b):
    return sum(a[index] * b[index] for index in range(3))


def _beam_basis(direction):
    """Stable right-handed basis (u, v) normal to ``direction``."""
    direction = _vec3(direction, "vacuum_direction")
    reference = (1.0, 0.0, 0.0)
    projection = _dot(reference, direction)
    u = tuple(
        reference[index] - projection * direction[index]
        for index in range(3)
    )
    length = math.sqrt(_dot(u, u))
    if length < 1e-14:
        reference = (0.0, 1.0, 0.0)
        projection = _dot(reference, direction)
        u = tuple(
            reference[index] - projection * direction[index]
            for index in range(3)
        )
        length = math.sqrt(_dot(u, u))
    u = tuple(value / length for value in u)
    v = (
        direction[1] * u[2] - direction[2] * u[1],
        direction[2] * u[0] - direction[0] * u[2],
        direction[0] * u[1] - direction[1] * u[0],
    )
    return u, v


def _pair(values, name, *, nonnegative=False):
    if np.isscalar(values):
        pair = (float(values), float(values))
    else:
        try:
            pair = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number or a pair") from exc
        if len(pair) != 2:
            raise ValueError(f"{name} must be a number or a pair")
    if not all(math.isfinite(value) for value in pair):
        raise ValueError(f"{name} must be finite")
    if nonnegative and any(value < 0.0 for value in pair):
        raise ValueError(f"{name} must be non-negative")
    return pair


@dataclass(frozen=True)
class PopulationClassifier:
    """Post-process one cascade into overlapping physical signal channels.

    The default ``causal_lle_v3`` definition deliberately separates causal
    taxonomy, detector filtering, and image properties:

    * SE1/SE2 are classified from the *direction* of the classifying parent at
      the creating collision, relative to a declared reference surface normal.
      No lateral distance, birth depth, energy loss, or resolution claim enters
      the definition; localization is a consequence to be measured.
    * The causal SE class is crossed with the emitted-energy cut instead of
      being gated by it, so every cascade electron carries a causal label and
      ``se1``/``se2`` mean the whole causal class at any final energy.
    * ``se_parent_rule`` selects whose direction decides.  ``root_primary_leg``
      (default) uses the root incident electron's own leg -- incoming versus
      returning -- at the collision that seeded this electron's lineage, which
      is the conventional literature meaning of SE1/SE2.  ``immediate_parent``
      uses the immediate energetic parent, which may itself be a cascade
      electron; that is a self-consistent generalization but a measurably
      different population, so it is recorded in metadata either way.
    * LLE/non-LLE is an explicit vacuum-energy-loss window on emitted original
      primaries -- an experimentally realizable filter class, not a BSE
      subspecies and not a scattering-history statement.
    * Strict first-event backscatter and barrier-reflected primaries are kept
      as overlapping diagnostics rather than identified with LLE or admitted to
      the default fitting basis.

    ``causal_lle_v2`` reproduces the 0.7.x default (SE classes gated at the
    50 eV cut, immediate-parent rule).  ``branch_v1`` reproduces the 0.6.1-era
    operational labels.  Both are retained so archived libraries refit exactly.
    """

    bse_cutoff_ev: float = 50.0
    lle_max_loss_ev: Optional[float] = None
    definition: str = "causal_lle_v3"
    se_reference: str = "launch_surface"
    se_parent_rule: Optional[str] = None
    lle_max_loss_frac: Optional[float] = None
    transmission: Optional[TransmissionDetector] = None

    def __post_init__(self):
        if self.transmission is not None:
            if self.definition != "causal_lle_v3":
                raise ValueError(
                    "transmission channels require definition='causal_lle_v3'; "
                    "the earlier definitions have no exit-hemisphere axis"
                )
            if not isinstance(self.transmission, TransmissionDetector):
                raise TypeError(
                    "transmission must be a TransmissionDetector or None"
                )
        cutoff = float(self.bse_cutoff_ev)
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("bse_cutoff_ev must be finite and non-negative")
        if self.definition not in set(CAUSAL_DEFINITIONS) | {"branch_v1"}:
            raise ValueError(
                "definition must be one of "
                f"{CAUSAL_DEFINITIONS + ('branch_v1',)}"
            )
        if self.se_reference not in {"launch_surface", "escape_surface"}:
            raise ValueError(
                "se_reference must be 'launch_surface' or 'escape_surface'"
            )

        loss_ev = self.lle_max_loss_ev
        loss_frac = self.lle_max_loss_frac
        if self.definition == "branch_v1":
            if loss_frac is not None:
                raise ValueError(
                    "branch_v1 has no LLE class; lle_max_loss_frac does not "
                    "apply to it"
                )
            loss_ev = (
                DEFAULT_LLE_MAX_LOSS_EV if loss_ev is None else float(loss_ev)
            )
        else:
            if loss_ev is not None and loss_frac is not None:
                raise ValueError(
                    "give lle_max_loss_ev or lle_max_loss_frac, not both; the "
                    "LLE threshold is either absolute or a fraction of E0"
                )
            if loss_ev is None and loss_frac is None:
                loss_ev = DEFAULT_LLE_MAX_LOSS_EV
        if loss_ev is not None:
            loss_ev = float(loss_ev)
            if not math.isfinite(loss_ev) or loss_ev < 0.0:
                raise ValueError(
                    "lle_max_loss_ev must be finite and non-negative"
                )
        if loss_frac is not None:
            loss_frac = float(loss_frac)
            if not math.isfinite(loss_frac) or not 0.0 <= loss_frac <= 1.0:
                raise ValueError(
                    "lle_max_loss_frac must be finite and within [0, 1]"
                )

        rule = self.se_parent_rule
        if rule is None:
            rule = (
                "immediate_parent"
                if self.definition == "causal_lle_v2"
                else "root_primary_leg"
            )
        if rule not in SE_PARENT_RULES:
            raise ValueError(f"se_parent_rule must be one of {SE_PARENT_RULES}")

        object.__setattr__(self, "bse_cutoff_ev", cutoff)
        object.__setattr__(self, "lle_max_loss_ev", loss_ev)
        object.__setattr__(self, "lle_max_loss_frac", loss_frac)
        object.__setattr__(self, "se_parent_rule", rule)

    @property
    def channels(self):
        if self.definition == "branch_v1":
            return LEGACY_POPULATION_CHANNELS
        if self.definition == "causal_lle_v2":
            return V2_POPULATION_CHANNELS
        if self.transmission is None:
            return POPULATION_CHANNELS
        return POPULATION_CHANNELS + tuple(TRANSMISSION_CHANNEL_DEFINITIONS) + tuple(
            f"back_{name}" for name in DISJOINT_POPULATION_CHANNELS
        )

    @property
    def lle_criterion(self):
        """``None`` for branch_v1, else which LLE threshold form is in use."""
        if self.definition == "branch_v1":
            return None
        if self.lle_max_loss_frac is not None:
            return "fractional_energy_loss"
        return "absolute_energy_loss"

    def lle_threshold_ev(self, incident_energy_ev):
        """Absolute LLE threshold in eV for one landing energy."""
        if self.lle_max_loss_frac is not None:
            return float(self.lle_max_loss_frac) * float(incident_energy_ev)
        return float(self.lle_max_loss_ev)

    @property
    def _threshold_text(self):
        if self.lle_max_loss_frac is not None:
            return f"{100.0 * self.lle_max_loss_frac:g}% of E0"
        return f"{self.lle_max_loss_ev:g} eV"

    @property
    def definitions(self):
        if self.definition == "branch_v1":
            return dict(LEGACY_CHANNEL_DEFINITIONS)
        definitions = dict(
            V2_CHANNEL_DEFINITIONS
            if self.definition == "causal_lle_v2"
            else CHANNEL_DEFINITIONS
        )
        if self.transmission is not None:
            definitions.update(TRANSMISSION_CHANNEL_DEFINITIONS)
            rings = self.transmission
            definitions["fwd_bf"] += f" Rule: theta < {rings.bf_max_mrad:g} mrad."
            definitions["fwd_adf"] += (
                f" Rule: {rings.bf_max_mrad:g} <= theta < "
                f"{rings.adf_max_mrad:g} mrad."
            )
            definitions["fwd_haadf"] += (
                f" Rule: {rings.adf_max_mrad:g} <= theta < "
                f"{rings.haadf_max_mrad:g} mrad."
            )
            definitions["fwd_beyond_haadf"] += (
                f" Rule: theta >= {rings.haadf_max_mrad:g} mrad."
            )
            for name in DISJOINT_POPULATION_CHANNELS:
                definitions[f"back_{name}"] = (
                    definitions[name] + " Restricted to the reflected "
                    "hemisphere; the unprefixed name counts both hemispheres."
                )
        threshold = self._threshold_text
        reference = (
            "incident primary's launch surface"
            if self.se_reference == "launch_surface"
            else "emitted electron's actual escape surface"
        )
        rule = (
            "root incident electron's own leg at the collision that seeded "
            "this lineage"
            if self.se_parent_rule == "root_primary_leg"
            else "immediate energetic parent's direction at the creating "
                 "collision"
        )
        for name in ("se1", "se2", "se1_lt50", "se1_ge50",
                     "se2_lt50", "se2_ge50"):
            if name in definitions:
                definitions[name] += (
                    f" Reference: {reference}. Parent rule: {rule}."
                )
        definitions["lle_primary"] += f" Rule: energy loss < {threshold}."
        definitions["non_lle_primary"] += (
            f" Rule: energy loss >= {threshold}."
        )
        return definitions

    @property
    def disjoint_channels(self):
        if self.definition == "branch_v1":
            return LEGACY_DISJOINT_POPULATION_CHANNELS
        if self.definition == "causal_lle_v2":
            return V2_DISJOINT_POPULATION_CHANNELS
        if self.transmission is None:
            return DISJOINT_POPULATION_CHANNELS
        # The backward hemisphere keeps the full causal taxonomy, because that
        # is where the SE mechanism question lives; the forward hemisphere is
        # segmented by collection angle, because that is what a STEM detector
        # actually measures.  Together they still partition TEY exactly.
        return tuple(
            f"back_{name}" for name in DISJOINT_POPULATION_CHANNELS
        ) + tuple(f"fwd_{ring}" for ring in TRANSMISSION_RINGS)

    def to_dict(self):
        causal = self.definition in CAUSAL_DEFINITIONS
        return {
            "definition": self.definition,
            "bse_cutoff_ev": self.bse_cutoff_ev,
            "lle_criterion": self.lle_criterion,
            "lle_max_loss_ev": self.lle_max_loss_ev if causal else None,
            "lle_max_loss_frac": self.lle_max_loss_frac if causal else None,
            "se_reference": (
                self.se_reference
                if causal
                else "root_primary_first_surface_return_event"
            ),
            "se_parent_rule": (
                self.se_parent_rule
                if causal
                else "root_primary_first_surface_return_event"
            ),
            "energy_reference": "vacuum",
            "lle_rule": (
                "energy_loss_strictly_less_than_threshold" if causal else None
            ),
            "transmission": (
                None if self.transmission is None else self.transmission.to_dict()
            ),
        }

    def classify(self, result: TrajectoryResult):
        if result.history is None:
            raise ValueError(
                f"{self.definition} classification requires trajectory history"
            )
        if len(result.emissions) != result.tey:
            raise ValueError(
                "complete emission records are required; use "
                "MCConfig(collect_spectra=True)"
            )

        cutoff = self.bse_cutoff_ev
        counts = {channel: 0 for channel in self.channels}
        counts["tey"] = int(result.tey)
        counts["sey_50ev"] = sum(e.energy < cutoff for e in result.emissions)
        counts["bse_50ev"] = sum(e.energy >= cutoff for e in result.emissions)
        counts["cascade_all"] = sum(e.is_cascade for e in result.emissions)
        counts["primary_all"] = sum(not e.is_cascade for e in result.emissions)
        counts["se_cascade_lt50"] = sum(
            e.is_cascade and e.energy < cutoff for e in result.emissions
        )
        counts["fast_cascade_ge50"] = sum(
            e.is_cascade and e.energy >= cutoff for e in result.emissions
        )
        counts["slow_primary_lt50"] = sum(
            (not e.is_cascade) and e.energy < cutoff for e in result.emissions
        )
        counts["bse_primary_ge50"] = sum(
            (not e.is_cascade) and e.energy >= cutoff for e in result.emissions
        )
        counts["generation_1"] = sum(
            e.is_cascade and e.generation == 1 for e in result.emissions
        )
        counts["generation_2plus"] = sum(
            e.is_cascade and e.generation >= 2 for e in result.emissions
        )

        labels = self.emission_labels(result)
        for label in labels.values():
            if label != "fast_cascade_ge50":
                counts[label] += 1

        if counts["tey"] != counts["sey_50ev"] + counts["bse_50ev"]:
            raise RuntimeError("energy-cut population channels do not partition TEY")
        if counts["tey"] != counts["cascade_all"] + counts["primary_all"]:
            raise RuntimeError("ancestry population channels do not partition TEY")

        # With a transmission detector the disjoint labels carry a hemisphere
        # prefix, so the hemisphere-free aggregates and their invariants are
        # deferred to _add_transmission_counts.
        split_hemispheres = self.transmission is not None
        if self.definition == "causal_lle_v3" and not split_hemispheres:
            counts["se1"] = counts["se1_lt50"] + counts["se1_ge50"]
            counts["se2"] = counts["se2_lt50"] + counts["se2_ge50"]
            if counts["cascade_all"] != counts["se1"] + counts["se2"]:
                raise RuntimeError(
                    "causal SE channels do not cover every cascade emission"
                )
            if counts["se_cascade_lt50"] != (
                    counts["se1_lt50"] + counts["se2_lt50"]):
                raise RuntimeError(
                    "sub-cutoff causal SE channels are incomplete"
                )
            if counts["fast_cascade_ge50"] != (
                    counts["se1_ge50"] + counts["se2_ge50"]):
                raise RuntimeError(
                    "above-cutoff causal SE channels are incomplete"
                )
        elif self.definition != "causal_lle_v3":
            if counts["se_cascade_lt50"] != counts["se1"] + counts["se2"]:
                raise RuntimeError("causal SE channels are incomplete")

        if self.definition == "branch_v1":
            if counts["primary_all"] != counts["bse1"] + counts["bse2"]:
                raise RuntimeError("legacy branch_v1 BSE channels are incomplete")
            return counts

        if not split_hemispheres and counts["primary_all"] != (
                counts["lle_primary"] + counts["non_lle_primary"]):
            raise RuntimeError("LLE channels do not partition emitted primaries")
        first_event, later_return = self._first_event_counts(result)
        if counts["primary_all"] != first_event + later_return:
            raise RuntimeError(
                "first-event diagnostic channels do not partition primaries"
            )
        if self.definition == "causal_lle_v2":
            counts["first_event_bse"] = first_event
            counts["later_return_bse"] = later_return
            return counts

        counts["first_event_backscatter"] = first_event
        counts["later_return_primary"] = later_return
        counts["barrier_reflected_primary"] = (
            self._barrier_reflected_count(result)
        )
        if self.transmission is not None:
            self._add_transmission_counts(result, counts)
        return counts

    def _add_transmission_counts(self, result, counts):
        """Fill the hemisphere aggregates and the angular diagnostics.

        ``classify`` has already counted the disjoint ``back_*``/``fwd_*``
        labels.  The hemisphere-free aggregates (``se1``, ``lle_primary`` and
        friends) are recomputed here over both hemispheres, so those names keep
        the meaning they have without a transmission detector.
        """
        classes = self._causal_classes(result)
        hemispheres = self._exit_hemispheres(result)
        cascade = {"se1_lt50", "se1_ge50", "se2_lt50", "se2_ge50"}
        for emission in result.emissions:
            electron_id = emission.electron_id
            causal_class = classes[electron_id]
            forward, ring = hemispheres[electron_id]
            counts[causal_class] += 1
            counts["forward_all" if forward else "backward_all"] += 1
            if not forward:
                continue
            if causal_class in cascade:
                counts["forward_cascade_all"] += 1
            else:
                counts["forward_primary_all"] += 1
                if ring in ("bf", "adf", "haadf"):
                    counts[f"fwd_{ring}_primary"] += 1

        counts["se1"] = counts["se1_lt50"] + counts["se1_ge50"]
        counts["se2"] = counts["se2_lt50"] + counts["se2_ge50"]

        if counts["tey"] != counts["backward_all"] + counts["forward_all"]:
            raise RuntimeError("exit hemispheres do not partition TEY")
        if counts["cascade_all"] != counts["se1"] + counts["se2"]:
            raise RuntimeError(
                "causal SE channels do not cover every cascade emission"
            )
        if counts["primary_all"] != (
                counts["lle_primary"] + counts["non_lle_primary"]):
            raise RuntimeError("LLE channels do not partition emitted primaries")
        forward_rings = sum(counts[f"fwd_{ring}"] for ring in TRANSMISSION_RINGS)
        if counts["forward_all"] != forward_rings:
            raise RuntimeError(
                "transmission rings do not partition the forward hemisphere"
            )

    def emission_labels(self, result: TrajectoryResult):
        """Return the disjoint population label for each emitted electron."""
        if result.history is None:
            raise ValueError(
                f"{self.definition} classification requires trajectory history"
            )
        if len(result.emissions) != result.tey:
            raise ValueError(
                "complete emission records are required; use "
                "MCConfig(collect_spectra=True)"
            )

        if self.definition == "branch_v1":
            return self._legacy_emission_labels(result)
        return self._causal_lle_emission_labels(result)

    @staticmethod
    def _root_record(history):
        roots = [record for record in history.electrons if record.parent_id is None]
        if len(roots) != 1:
            raise ValueError("population classification expects one incident primary")
        return roots[0]

    def _se_parent_direction(self, records, electron_id):
        """Direction whose sign against the reference normal decides SE1/SE2.

        ``immediate_parent`` returns the immediate energetic parent's direction
        just before the creating collision.  ``root_primary_leg`` walks the
        ancestry up to the generation-1 ancestor and returns the *root incident
        electron's* direction just before the collision that seeded this
        lineage, so the class means "generated on the incoming leg" versus
        "generated on the returning leg" of the beam electron itself.
        """
        record = records[electron_id]
        if self.se_parent_rule == "immediate_parent":
            direction = record.parent_direction_before
            if direction is None:
                raise ValueError(
                    "causal SE classification requires the immediate parent "
                    "direction at the birth collision"
                )
            return direction

        seen = set()
        while True:
            parent_id = record.parent_id
            if parent_id is None:
                raise ValueError(
                    "root-primary-leg SE classification expects a cascade "
                    "electron with a recorded parent"
                )
            parent = records.get(parent_id)
            if parent is None:
                raise ValueError(
                    "root-primary-leg SE classification requires the complete "
                    "ancestry of every emitted cascade electron"
                )
            if parent.parent_id is None:
                direction = record.parent_direction_before
                if direction is None:
                    raise ValueError(
                        "causal SE classification requires the parent "
                        "direction at the birth collision"
                    )
                return direction
            if parent_id in seen:
                raise ValueError("cyclic ancestry in trajectory history")
            seen.add(parent_id)
            record = parent

    def _causal_classes(self, result):
        """Hemisphere-free causal class for each emitted electron.

        This is the classification proper.  Where the electron happens to
        leave -- back toward the source or forward through the specimen -- is a
        separate axis handled by :meth:`_exit_hemispheres`.
        """
        history = result.history
        records = {record.electron_id: record for record in history.electrons}
        launch_normal = tuple(
            float(value) for value in history.reference_surface_normal
        )
        threshold = self.lle_threshold_ev(history.incident_energy)
        split_by_energy = self.definition == "causal_lle_v3"
        classes = {}
        for emission in result.emissions:
            if emission.is_cascade:
                fast = emission.energy >= self.bse_cutoff_ev
                if fast and not split_by_energy:
                    classes[emission.electron_id] = "fast_cascade_ge50"
                    continue
                normal = launch_normal
                if self.se_reference == "escape_surface":
                    if emission.surface_normal is None:
                        raise ValueError(
                            "escape-surface SE classification requires the "
                            "emission surface normal"
                        )
                    normal = tuple(float(value) for value in emission.surface_normal)
                direction = self._se_parent_direction(
                    records, emission.electron_id
                )
                family = "se2" if _dot(direction, normal) > 0.0 else "se1"
                classes[emission.electron_id] = (
                    f"{family}_ge50" if fast else f"{family}_lt50"
                ) if split_by_energy else family
            else:
                energy_loss = max(
                    0.0, float(history.incident_energy) - float(emission.energy)
                )
                classes[emission.electron_id] = (
                    "lle_primary"
                    if energy_loss < threshold
                    else "non_lle_primary"
                )
        return classes

    def _exit_hemispheres(self, result):
        """Map each emission to ``(is_forward, ring_or_None)``.

        Forward means the emitted velocity has a component along the incident
        beam direction, so the electron left through the far side of the
        specimen.  On a bulk substrate this set is empty; it becomes populated
        only when the geometry has a bottom exit surface.
        """
        beam = tuple(float(value) for value in result.history.incident_direction)
        detail = {}
        for emission in result.emissions:
            if emission.uvw is None:
                raise ValueError(
                    "transmission classification requires the emitted "
                    "direction; use MCConfig(collect_spectra=True)"
                )
            projection = _dot(tuple(float(v) for v in emission.uvw), beam)
            if projection <= 0.0:
                detail[emission.electron_id] = (False, None)
                continue
            # Clamp guards against a projection a few ulps above one.
            theta = math.acos(min(1.0, max(-1.0, projection)))
            detail[emission.electron_id] = (
                True, self.transmission.ring(theta * 1000.0)
            )
        return detail

    def _causal_lle_emission_labels(self, result):
        classes = self._causal_classes(result)
        if self.transmission is None:
            return classes
        hemispheres = self._exit_hemispheres(result)
        labels = {}
        for electron_id, causal_class in classes.items():
            forward, ring = hemispheres[electron_id]
            labels[electron_id] = (
                f"fwd_{ring}" if forward else f"back_{causal_class}"
            )
        return labels

    @staticmethod
    def _barrier_reflected_count(result):
        """Emitted primaries turned back by the surface barrier on entry."""
        return sum(
            (not emission.is_cascade)
            and emission.emission_mechanism == "incoming_barrier_reflection"
            for emission in result.emissions
        )

    def _first_event_counts(self, result):
        history = result.history
        root = self._root_record(history)
        primary_count = sum(not emission.is_cascade for emission in result.emissions)
        if primary_count == 0:
            return 0, 0
        collisions = [
            event for event in history.events
            if event.electron_id == root.electron_id
            and event.kind in {"elastic", "inelastic"}
        ]
        first_event = bool(
            collisions
            and root.first_surface_return_event_id == collisions[0].event_id
            and collisions[0].kind == "elastic"
        )
        return (primary_count, 0) if first_event else (0, primary_count)

    def _legacy_emission_labels(self, result):
        history = result.history
        records = {record.electron_id: record for record in history.electrons}
        events = {event.event_id: event for event in history.events}
        root = self._root_record(history)
        return_event = root.first_surface_return_event_id

        labels = {}
        for emission in result.emissions:
            if emission.is_cascade and emission.energy < self.bse_cutoff_ev:
                record = records[emission.electron_id]
                if return_event is not None and return_event < record.birth_event_id:
                    labels[emission.electron_id] = "se2"
                else:
                    labels[emission.electron_id] = "se1"
            elif emission.is_cascade:
                labels[emission.electron_id] = "fast_cascade_ge50"

        primary_emissions = [e for e in result.emissions if not e.is_cascade]
        if primary_emissions:
            elastic_before_return = 0
            if return_event is not None:
                elastic_before_return = sum(
                    event.electron_id == root.electron_id
                    and event.kind == "elastic"
                    and event.event_id <= return_event
                    for event in history.events
                )
            label = (
                "bse1"
                if return_event is not None
                and events[return_event].kind == "elastic"
                and elastic_before_return == 1
                else "bse2"
            )
            for emission in primary_emissions:
                labels[emission.electron_id] = label
        return labels


@dataclass(frozen=True)
class RasterConfig:
    """Raster grid, beam, statistics, and seed configuration.

    Coordinates and ``beam_fwhm`` use the transport length unit (Angstrom).
    The FWHM pair applies along two orthogonal axes normal to the beam.
    """

    energy_ev: float
    x_positions: Sequence[float]
    y_positions: Sequence[float]
    primaries_per_pixel: int
    beam_fwhm: object = 0.0
    vacuum_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    seed: int = 12345
    record_trajectories: bool = False
    record_primaries_per_pixel: Optional[int] = None
    trajectory_stride: int = 1
    trajectory_max_points: Optional[int] = None

    def __post_init__(self):
        energy = float(self.energy_ev)
        if not math.isfinite(energy) or energy < 0.0:
            raise ValueError("energy_ev must be finite and non-negative")
        x = tuple(float(value) for value in self.x_positions)
        y = tuple(float(value) for value in self.y_positions)
        if not x or not y or not all(math.isfinite(value) for value in x + y):
            raise ValueError("x_positions and y_positions must be finite and non-empty")
        n = int(self.primaries_per_pixel)
        if n < 1 or n != self.primaries_per_pixel:
            raise ValueError("primaries_per_pixel must be a positive integer")
        fwhm = _pair(self.beam_fwhm, "beam_fwhm", nonnegative=True)
        direction = _vec3(self.vacuum_direction, "vacuum_direction")
        if direction[2] <= 1e-15:
            raise ValueError(
                "the current raster reference plane requires a beam with "
                "positive global z component"
            )
        seed = int(self.seed)
        record = bool(self.record_trajectories)
        record_n = self.record_primaries_per_pixel
        if record:
            raw_record_n = n if record_n is None else record_n
            record_n = int(raw_record_n)
            if record_n != raw_record_n or record_n < 1 or record_n > n:
                raise ValueError(
                    "record_primaries_per_pixel must be between 1 and "
                    "primaries_per_pixel and must be an integer"
                )
        else:
            if record_n not in (None, 0):
                raise ValueError(
                    "record_primaries_per_pixel requires record_trajectories=True"
                )
            record_n = 0
        stride = int(self.trajectory_stride)
        if stride < 1 or stride != self.trajectory_stride:
            raise ValueError("trajectory_stride must be a positive integer")
        max_points = self.trajectory_max_points
        if max_points is not None:
            max_points = int(max_points)
            if max_points < 2 or max_points != self.trajectory_max_points:
                raise ValueError("trajectory_max_points must be at least 2")
        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(self, "x_positions", x)
        object.__setattr__(self, "y_positions", y)
        object.__setattr__(self, "primaries_per_pixel", n)
        object.__setattr__(self, "beam_fwhm", fwhm)
        object.__setattr__(self, "vacuum_direction", direction)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "record_trajectories", record)
        object.__setattr__(self, "record_primaries_per_pixel", record_n)
        object.__setattr__(self, "trajectory_stride", stride)
        object.__setattr__(self, "trajectory_max_points", max_points)

    @property
    def shape(self):
        return (len(self.y_positions), len(self.x_positions))

    @property
    def beam_sigma(self):
        return tuple(value * FWHM_TO_SIGMA for value in self.beam_fwhm)

    @property
    def n_pixels(self):
        return len(self.x_positions) * len(self.y_positions)

    def to_dict(self):
        return {
            "energy_ev": self.energy_ev,
            "x_positions_angstrom": list(self.x_positions),
            "y_positions_angstrom": list(self.y_positions),
            "primaries_per_pixel": self.primaries_per_pixel,
            "beam_fwhm_angstrom": list(self.beam_fwhm),
            "vacuum_direction": list(self.vacuum_direction),
            "seed": self.seed,
            "record_trajectories": self.record_trajectories,
            "record_primaries_per_pixel": self.record_primaries_per_pixel,
            "trajectory_stride": self.trajectory_stride,
            "trajectory_max_points": self.trajectory_max_points,
        }


def sample_beam_reference(x, y, reference_z, vacuum_direction, beam_sigma, rng):
    """Sample a parallel ray and return its (x, y) at ``reference_z``.

    The Gaussian is defined in the plane normal to the beam, rather than in
    the specimen's global x-y plane.  This gives a physical circular spot at
    oblique incidence; its projection onto the horizontal raster plane is
    correspondingly elongated.
    """
    direction = _vec3(vacuum_direction, "vacuum_direction")
    sigma = _pair(beam_sigma, "beam_sigma", nonnegative=True)
    u, v = _beam_basis(direction)
    du = 0.0 if sigma[0] == 0.0 else float(rng.normal(0.0, sigma[0]))
    dv = 0.0 if sigma[1] == 0.0 else float(rng.normal(0.0, sigma[1]))
    point = (
        float(x) + du * u[0] + dv * v[0],
        float(y) + du * u[1] + dv * v[1],
        float(reference_z) + du * u[2] + dv * v[2],
    )
    along = (float(reference_z) - point[2]) / direction[2]
    projected = tuple(
        point[index] + along * direction[index] for index in range(3)
    )
    return projected[0], projected[1]


def _reference_z(geometry):
    if hasattr(geometry, "top_z"):
        return float(geometry.top_z)
    if hasattr(geometry, "point"):
        return float(geometry.point[2])
    raise TypeError("raster geometry must provide top_z or point")


def _add_diagnostics(target, values):
    for key, value in values.items():
        target[key] = target.get(key, 0) + int(value)


def _decimate_track(coordinates, times_fs, stride, max_points):
    coordinates = np.asarray(coordinates, dtype=float)
    times_fs = np.asarray(times_fs, dtype=float)
    if len(coordinates) != len(times_fs):
        raise ValueError("trajectory coordinates and times must have equal length")
    if len(coordinates) == 0:
        return np.empty((0, 5), dtype=float)
    indices = np.arange(0, len(coordinates), int(stride), dtype=int)
    if indices[-1] != len(coordinates) - 1:
        indices = np.append(indices, len(coordinates) - 1)
    if max_points is not None and len(indices) > max_points:
        selected = np.rint(np.linspace(0, len(indices) - 1, max_points)).astype(int)
        indices = indices[np.unique(selected)]
    return np.column_stack((coordinates[indices], times_fs[indices]))


def _geometry_metadata(geometry):
    metadata = {"type": type(geometry).__name__}
    for name in (
        "top_width", "bottom_width", "height", "center_x", "substrate_z",
        "membrane_thickness", "total_thickness", "bottom_z",
        "surface_id", "solid_region", "vacuum_region",
    ):
        if hasattr(geometry, name):
            value = getattr(geometry, name)
            metadata[name] = float(value) if isinstance(value, (int, float)) else str(value)
    if hasattr(geometry, "point"):
        metadata["point"] = [float(value) for value in geometry.point]
    if hasattr(geometry, "outward_normal"):
        metadata["outward_normal"] = [
            float(value) for value in geometry.outward_normal
        ]
    return metadata


def _simulate_pixel(sample, geometry, config, classifier, task):
    pixel_id, iy, ix, x, y = task
    n = config.primaries_per_pixel
    channels = classifier.channels
    channel_values = np.zeros((n, len(channels)), dtype=np.float64)
    launch_values = np.zeros((n, 4), dtype=np.float64)
    surface_counts = {}
    diagnostics = {}
    trajectory_records = []
    z_reference = _reference_z(geometry)

    for trajectory_id in range(n):
        # Separate deterministic streams make beam changes RNG-transparent to
        # the collision cascade for a fixed pixel and trajectory identifier.
        spot_rng = np.random.default_rng(np.random.SeedSequence(
            [config.seed, int(pixel_id), int(trajectory_id), 0]
        ))
        transport_rng = np.random.default_rng(np.random.SeedSequence(
            [config.seed, int(pixel_id), int(trajectory_id), 1]
        ))
        ray_x, ray_y = sample_beam_reference(
            x, y, z_reference, config.vacuum_direction,
            config.beam_sigma, spot_rng,
        )
        hit = geometry.launch_surface(
            ray_x, ray_y, vacuum_direction=config.vacuum_direction
        )
        outward = tuple(-value for value in hit.normal)
        local_cosine = max(
            -1.0, min(1.0, -_dot(config.vacuum_direction, outward))
        )
        local_angle = math.acos(local_cosine)

        record_track = (
            config.record_trajectories
            and trajectory_id < config.record_primaries_per_pixel
        )
        result = simulate_trajectory(
            sample,
            config.energy_ev,
            local_angle,
            transport_rng,
            track=record_track,
            history=True,
            trajectory_id=trajectory_id,
            geometry=geometry,
            launch_position=hit.position,
            vacuum_direction=config.vacuum_direction,
            surface_normal=outward,
        )
        counts = classifier.classify(result)
        channel_values[trajectory_id] = [counts[name] for name in channels]
        launch_values[trajectory_id] = (
            hit.position[0], hit.position[1], hit.position[2], local_angle,
        )
        surface_counts[hit.surface_id] = surface_counts.get(hit.surface_id, 0) + 1
        _add_diagnostics(diagnostics, result.diagnostics)
        if record_track:
            records_by_id = {
                record.electron_id: record for record in result.history.electrons
            }
            labels = classifier.emission_labels(result)
            electrons = []
            for electron_id, coordinates, times_fs in zip(
                result.track_electron_ids, result.tracks, result.track_times_fs
            ):
                record = records_by_id[electron_id]
                population = labels.get(
                    electron_id,
                    "primary_absorbed" if record.is_primary else "cascade_absorbed",
                )
                electrons.append({
                    "electron_id": int(electron_id),
                    "parent_id": record.parent_id,
                    "generation": int(record.generation),
                    "is_primary": bool(record.is_primary),
                    "birth_energy_ev": float(record.birth_energy),
                    "birth_time_fs": float(record.birth_time_fs),
                    "fate": str(record.fate),
                    "final_energy_ev": float(record.final_energy),
                    "final_direction": tuple(record.final_direction),
                    "population": population,
                    "points": _decimate_track(
                        coordinates,
                        times_fs,
                        config.trajectory_stride,
                        config.trajectory_max_points,
                    ),
                })
            trajectory_records.append({
                "pixel_id": int(pixel_id),
                "iy": int(iy),
                "ix": int(ix),
                "trajectory_id": int(trajectory_id),
                "nominal_xy_angstrom": (float(x), float(y)),
                "launch_xyz_angstrom": tuple(hit.position),
                "local_incidence_rad": float(local_angle),
                "electrons": electrons,
            })

    channel_sum = channel_values.sum(axis=0)
    channel_sum_sq = np.square(channel_values).sum(axis=0)
    channel_cross_sum = channel_values.T @ channel_values
    launch_sum = launch_values.sum(axis=0)
    launch_sum_sq = np.square(launch_values).sum(axis=0)
    return {
        "pixel_id": pixel_id,
        "iy": iy,
        "ix": ix,
        "channel_sum": channel_sum,
        "channel_sum_sq": channel_sum_sq,
        "channel_cross_sum": channel_cross_sum,
        "launch_sum": launch_sum,
        "launch_sum_sq": launch_sum_sq,
        "surface_counts": surface_counts,
        "diagnostics": diagnostics,
        "trajectory_records": trajectory_records,
    }


_RASTER_WORKER = None


def _init_raster_worker(sample_name, db_path, mc_config, geometry,
                        raster_config, classifier):
    global _RASTER_WORKER
    _RASTER_WORKER = SimpleNamespace(
        sample=Sample(sample_name, db_path=db_path, config=mc_config),
        geometry=geometry,
        config=raster_config,
        classifier=classifier,
    )


def _raster_worker_task(task):
    return _simulate_pixel(
        _RASTER_WORKER.sample,
        _RASTER_WORKER.geometry,
        _RASTER_WORKER.config,
        _RASTER_WORKER.classifier,
        task,
    )


@dataclass
class RasterResult:
    """Population maps plus per-primary uncertainty and landing statistics."""

    config: RasterConfig
    sample_name: str
    classifier: PopulationClassifier
    yield_maps: Mapping[str, np.ndarray]
    sem_maps: Mapping[str, np.ndarray]
    count_maps: Mapping[str, np.ndarray]
    primary_count_covariance: np.ndarray
    yield_covariance: np.ndarray
    completed_primaries: np.ndarray
    launch_mean: np.ndarray
    launch_sem: np.ndarray
    local_incidence_mean_rad: np.ndarray
    local_incidence_sem_rad: np.ndarray
    surface_hit_counts: Mapping[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Mapping[str, int] = field(default_factory=dict)
    geometry_metadata: Mapping[str, object] = field(default_factory=dict)
    trajectory_records: Sequence[dict] = field(default_factory=tuple)

    @property
    def x_positions(self):
        return np.asarray(self.config.x_positions, dtype=float)

    @property
    def y_positions(self):
        return np.asarray(self.config.y_positions, dtype=float)

    @property
    def surface_hit_fractions(self):
        return {
            name: values / self.completed_primaries
            for name, values in self.surface_hit_counts.items()
        }

    @property
    def covariance_channels(self):
        return tuple(self.classifier.channels)

    @property
    def has_recorded_trajectories(self):
        return bool(self.trajectory_records)

    def covariance(self, channels=None, *, of_mean=True):
        """Return per-pixel covariance for a selected channel collection.

        The last two axes follow ``channels``.  ``of_mean=True`` returns the
        covariance of the reported pixel yields; otherwise it returns the
        sample covariance of the per-primary emitted counts.
        """
        channels = tuple(channels or self.covariance_channels)
        unknown = [name for name in channels if name not in self.covariance_channels]
        if unknown:
            raise KeyError(f"unknown covariance channels: {unknown}")
        indices = [self.covariance_channels.index(name) for name in channels]
        source = self.yield_covariance if of_mean else self.primary_count_covariance
        return np.take(np.take(source, indices, axis=-2), indices, axis=-1)

    def correlation(self, channels=None):
        """Return per-pixel correlation matrices of per-primary counts."""
        covariance = self.covariance(channels, of_mean=False)
        standard = np.sqrt(np.maximum(np.diagonal(
            covariance, axis1=-2, axis2=-1
        ), 0.0))
        denominator = standard[..., :, None] * standard[..., None, :]
        return np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )

    def metadata(self):
        return {
            "format": "seemc-imaging-raster-v3",
            "sample_name": self.sample_name,
            "length_unit": "angstrom",
            "classifier": self.classifier.definition,
            "classifier_config": self.classifier.to_dict(),
            "bse_cutoff_ev": self.classifier.bse_cutoff_ev,
            "lle_max_loss_ev": (
                self.classifier.lle_max_loss_ev
                if self.classifier.definition == "causal_lle_v2" else None
            ),
            "channel_definitions": self.classifier.definitions,
            "covariance_channels": list(self.covariance_channels),
            "disjoint_population_channels": list(
                self.classifier.disjoint_channels
            ),
            "config": self.config.to_dict(),
            "geometry": dict(self.geometry_metadata),
            "diagnostics": dict(self.diagnostics),
        }

    def save_npz(self, path):
        """Save a compact, self-describing array archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata_json": np.asarray(json.dumps(self.metadata(), sort_keys=True)),
            "x_angstrom": self.x_positions,
            "y_angstrom": self.y_positions,
            "completed_primaries": self.completed_primaries,
            "launch_mean_xyz_angstrom": self.launch_mean,
            "launch_sem_xyz_angstrom": self.launch_sem,
            "local_incidence_mean_rad": self.local_incidence_mean_rad,
            "local_incidence_sem_rad": self.local_incidence_sem_rad,
            "covariance_channels": np.asarray(self.covariance_channels),
            "primary_count_covariance": self.primary_count_covariance,
            "yield_covariance": self.yield_covariance,
        }
        for channel in self.classifier.channels:
            payload[f"yield__{channel}"] = self.yield_maps[channel]
            payload[f"sem__{channel}"] = self.sem_maps[channel]
            payload[f"count__{channel}"] = self.count_maps[channel]
        for surface_id, values in self.surface_hit_counts.items():
            payload[f"landing_count__{surface_id}"] = values
            payload[f"landing_fraction__{surface_id}"] = (
                values / self.completed_primaries
            )
        np.savez_compressed(path, **payload)
        return path

    def trajectory_archive(self):
        if not self.has_recorded_trajectories:
            raise ValueError(
                "this raster has no recorded trajectories; use "
                "RasterConfig(record_trajectories=True)"
            )
        return RasterTrajectoryArchive.from_records(
            self.metadata(),
            self.x_positions,
            self.y_positions,
            self.yield_maps,
            self.sem_maps,
            self.trajectory_records,
        )

    def save_trajectories_npz(self, path):
        """Save recorded cascades to a compact, pickle-free NPZ archive."""
        return self.trajectory_archive().save_npz(path)

    def rows(self):
        """Return one flat, wide record per pixel for CSV/dataframe use."""
        fractions = self.surface_hit_fractions
        surface_ids = sorted(self.surface_hit_counts)
        disjoint = self.classifier.disjoint_channels
        disjoint_covariance = self.covariance(disjoint, of_mean=True)
        rows = []
        nx = len(self.config.x_positions)
        for iy, y in enumerate(self.config.y_positions):
            for ix, x in enumerate(self.config.x_positions):
                row = {
                    "pixel_id": iy * nx + ix,
                    "iy": iy,
                    "ix": ix,
                    "x_angstrom": x,
                    "y_angstrom": y,
                    "completed_primaries": int(self.completed_primaries[iy, ix]),
                    "launch_x_mean_angstrom": self.launch_mean[iy, ix, 0],
                    "launch_y_mean_angstrom": self.launch_mean[iy, ix, 1],
                    "launch_z_mean_angstrom": self.launch_mean[iy, ix, 2],
                    "launch_x_sem_angstrom": self.launch_sem[iy, ix, 0],
                    "launch_y_sem_angstrom": self.launch_sem[iy, ix, 1],
                    "launch_z_sem_angstrom": self.launch_sem[iy, ix, 2],
                    "local_incidence_mean_deg": math.degrees(
                        self.local_incidence_mean_rad[iy, ix]
                    ),
                    "local_incidence_sem_deg": math.degrees(
                        self.local_incidence_sem_rad[iy, ix]
                    ),
                }
                for channel in self.classifier.channels:
                    row[f"count__{channel}"] = self.count_maps[channel][iy, ix]
                    row[f"yield__{channel}"] = self.yield_maps[channel][iy, ix]
                    row[f"sem__{channel}"] = self.sem_maps[channel][iy, ix]
                covariance = disjoint_covariance[iy, ix]
                for first, name_a in enumerate(disjoint):
                    for second in range(first, len(disjoint)):
                        name_b = disjoint[second]
                        row[f"covmean__{name_a}__{name_b}"] = covariance[
                            first, second
                        ]
                for surface_id in surface_ids:
                    row[f"landing_count__{surface_id}"] = int(
                        self.surface_hit_counts[surface_id][iy, ix]
                    )
                    row[f"landing_fraction__{surface_id}"] = (
                        fractions[surface_id][iy, ix]
                    )
                rows.append(row)
        return rows

    def save_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows()
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def plot_channels(self, channels=("sey_50ev", "bse_50ev", "se1", "se2"),
                      quantity="yield", path=None, cmap="gray"):
        """Plot selected maps; matplotlib is an optional dependency."""
        if quantity not in {"yield", "sem", "count"}:
            raise ValueError("quantity must be 'yield', 'sem', or 'count'")
        source = {
            "yield": self.yield_maps,
            "sem": self.sem_maps,
            "count": self.count_maps,
        }[quantity]
        channels = tuple(channels)
        unknown = [channel for channel in channels if channel not in source]
        if unknown:
            raise KeyError(f"unknown channels: {unknown}")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plot_channels requires matplotlib; install seemc-imaging[plot]"
            ) from exc

        ncols = min(3, len(channels))
        nrows = int(math.ceil(len(channels) / ncols))
        figure, axes = plt.subplots(
            nrows, ncols, figsize=(4.5 * ncols, 3.7 * nrows), squeeze=False
        )
        x_nm = self.x_positions / 10.0
        y_nm = self.y_positions / 10.0
        for axis, channel in zip(axes.flat, channels):
            values = source[channel]
            if values.shape[0] == 1:
                axis.plot(x_nm, values[0])
                axis.set_xlabel("x (nm)")
                axis.set_ylabel(quantity)
            else:
                image = axis.imshow(
                    values,
                    origin="lower",
                    aspect="auto",
                    extent=(x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]),
                    cmap=cmap,
                )
                axis.set_xlabel("x (nm)")
                axis.set_ylabel("y (nm)")
                figure.colorbar(image, ax=axis, label=quantity)
            axis.set_title(channel)
        for axis in axes.flat[len(channels):]:
            axis.set_visible(False)
        figure.tight_layout()
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=180, bbox_inches="tight")
        return figure


class RasterDriver:
    """Run independent cascade ensembles over a two-dimensional scan grid."""

    def __init__(self, sample: Sample, geometry, config: RasterConfig,
                 classifier: Optional[PopulationClassifier] = None):
        if not isinstance(sample, Sample):
            raise TypeError("sample must be a seemc_imaging.Sample")
        if not hasattr(geometry, "launch_surface"):
            raise TypeError("geometry must provide launch_surface()")
        if not sample.cfg.collect_spectra:
            raise ValueError(
                "RasterDriver requires MCConfig(collect_spectra=True)"
            )
        # The material tables clamp any energy above their top grid point, so a
        # landing energy outside the tabulated range would otherwise be
        # simulated silently with cross sections taken from the table maximum.
        if float(config.energy_ev) > float(sample.Emax):
            warnings.warn(
                f"landing energy {config.energy_ev:g} eV exceeds the "
                f"{sample.name!r} table range "
                f"[{sample.Emin:g}, {sample.Emax:g}] eV; cross sections are "
                "clamped at the table maximum and the result is not "
                "predictive at this energy",
                RuntimeWarning,
                stacklevel=2,
            )
        self.sample = sample
        self.geometry = geometry
        self.config = config
        self.classifier = classifier or PopulationClassifier(
            bse_cutoff_ev=sample.cfg.bse_cutoff_ev
        )

    def _tasks(self):
        nx = len(self.config.x_positions)
        for iy, y in enumerate(self.config.y_positions):
            for ix, x in enumerate(self.config.x_positions):
                yield (iy * nx + ix, iy, ix, x, y)

    def run(self, use_parallel=False, workers=None, progress=True):
        """Run the raster, parallelizing by pixel when requested.

        Serial and spawn-based parallel runs are deterministic and identical.
        Every worker handles all primaries for one pixel, keeping interprocess
        traffic independent of ``primaries_per_pixel``.
        """
        tasks = list(self._tasks())
        pool = None
        if use_parallel:
            context = mp.get_context("spawn")
            if workers is None:
                workers = min(context.cpu_count(), len(tasks))
            workers = int(workers)
            if workers < 1:
                raise ValueError("workers must be positive")
            pool = context.Pool(
                processes=workers,
                initializer=_init_raster_worker,
                initargs=(
                    self.sample.name,
                    self.sample.db_path,
                    self.sample.cfg,
                    self.geometry,
                    self.config,
                    self.classifier,
                ),
            )
            iterator = pool.imap_unordered(_raster_worker_task, tasks, chunksize=1)
        else:
            iterator = (
                _simulate_pixel(
                    self.sample, self.geometry, self.config,
                    self.classifier, task,
                )
                for task in tasks
            )

        try:
            if progress:
                try:
                    from tqdm import tqdm
                except ImportError:  # pragma: no cover
                    pass
                else:
                    iterator = tqdm(iterator, total=len(tasks), desc="SEM raster")
            payloads = list(iterator)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        return self._assemble(payloads)

    def _assemble(self, payloads):
        shape = self.config.shape
        n = self.config.primaries_per_pixel
        channels = self.classifier.channels
        count_maps = {
            channel: np.zeros(shape, dtype=np.int64) for channel in channels
        }
        yield_maps = {
            channel: np.zeros(shape, dtype=np.float64) for channel in channels
        }
        sem_maps = {
            channel: np.zeros(shape, dtype=np.float64) for channel in channels
        }
        covariance_shape = shape + (len(channels), len(channels))
        primary_count_covariance = np.zeros(covariance_shape, dtype=np.float64)
        yield_covariance = np.zeros(covariance_shape, dtype=np.float64)
        completed = np.full(shape, n, dtype=np.int64)
        launch_mean = np.zeros(shape + (3,), dtype=np.float64)
        launch_sem = np.zeros(shape + (3,), dtype=np.float64)
        incidence_mean = np.zeros(shape, dtype=np.float64)
        incidence_sem = np.zeros(shape, dtype=np.float64)
        surface_maps = {}
        diagnostics = {}
        trajectory_records = []

        for payload in payloads:
            iy, ix = payload["iy"], payload["ix"]
            sums = payload["channel_sum"]
            sums_sq = payload["channel_sum_sq"]
            cross_sums = payload["channel_cross_sum"]
            means = sums / n
            if n > 1:
                variance = np.maximum(
                    (sums_sq - sums * sums / n) / (n - 1), 0.0
                )
                sem = np.sqrt(variance / n)
            else:
                sem = np.zeros_like(means)
                covariance = np.zeros_like(cross_sums)
            if n > 1:
                covariance = (
                    cross_sums - np.outer(sums, sums) / n
                ) / (n - 1)
                covariance = 0.5 * (covariance + covariance.T)
                diagonal = np.maximum(np.diag(covariance), 0.0)
                covariance[np.diag_indices_from(covariance)] = diagonal
            primary_count_covariance[iy, ix] = covariance
            yield_covariance[iy, ix] = covariance / n
            for index, channel in enumerate(channels):
                count_maps[channel][iy, ix] = int(sums[index])
                yield_maps[channel][iy, ix] = means[index]
                sem_maps[channel][iy, ix] = sem[index]

            launch_sums = payload["launch_sum"]
            launch_sums_sq = payload["launch_sum_sq"]
            launch_means = launch_sums / n
            if n > 1:
                launch_variance = np.maximum(
                    (launch_sums_sq - launch_sums * launch_sums / n) / (n - 1),
                    0.0,
                )
                launch_sems = np.sqrt(launch_variance / n)
            else:
                launch_sems = np.zeros_like(launch_means)
            launch_mean[iy, ix] = launch_means[:3]
            launch_sem[iy, ix] = launch_sems[:3]
            incidence_mean[iy, ix] = launch_means[3]
            incidence_sem[iy, ix] = launch_sems[3]

            for surface_id, value in payload["surface_counts"].items():
                if surface_id not in surface_maps:
                    surface_maps[surface_id] = np.zeros(shape, dtype=np.int64)
                surface_maps[surface_id][iy, ix] = int(value)
            _add_diagnostics(diagnostics, payload["diagnostics"])
            trajectory_records.extend(payload["trajectory_records"])

        if len(payloads) != self.config.n_pixels:
            warnings.warn(
                f"requested {self.config.n_pixels} pixels but received "
                f"{len(payloads)} worker results",
                RuntimeWarning,
                stacklevel=2,
            )
        return RasterResult(
            config=self.config,
            sample_name=self.sample.name,
            classifier=self.classifier,
            yield_maps=yield_maps,
            sem_maps=sem_maps,
            count_maps=count_maps,
            primary_count_covariance=primary_count_covariance,
            yield_covariance=yield_covariance,
            completed_primaries=completed,
            launch_mean=launch_mean,
            launch_sem=launch_sem,
            local_incidence_mean_rad=incidence_mean,
            local_incidence_sem_rad=incidence_sem,
            surface_hit_counts=surface_maps,
            diagnostics=diagnostics,
            geometry_metadata=_geometry_metadata(self.geometry),
            trajectory_records=tuple(sorted(
                trajectory_records,
                key=lambda item: (item["pixel_id"], item["trajectory_id"]),
            )),
        )


__all__ = [
    "CHANNEL_DEFINITIONS",
    "DISJOINT_POPULATION_CHANNELS",
    "LEGACY_CHANNEL_DEFINITIONS",
    "LEGACY_DISJOINT_POPULATION_CHANNELS",
    "LEGACY_POPULATION_CHANNELS",
    "POPULATION_CHANNELS",
    "PopulationClassifier",
    "RasterConfig",
    "RasterDriver",
    "RasterResult",
    "RasterTrajectoryArchive",
    "sample_beam_reference",
]
