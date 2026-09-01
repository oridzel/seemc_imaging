"""Covariance-aware model libraries and profile metrology for trapezoidal lines.

The transport outputs many overlapping signals.  Formal joint fits should use
mutually exclusive channels so that one emitted electron is not counted as two
independent measurements.  The default basis is defined in ``raster.py`` and
partitions TEY into causal SE1 and SE2 -- each split by the emitted-energy cut
-- plus low-loss and non-low-loss emitted primaries.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .geometry import TrapezoidalLine
from .raster import (
    DISJOINT_POPULATION_CHANNELS,
    TRANSMISSION_RINGS,
    LEGACY_DISJOINT_POPULATION_CHANNELS,
    V2_DISJOINT_POPULATION_CHANNELS,
    PopulationClassifier,
    RasterConfig,
    RasterDriver,
    RasterResult,
)
from .transport import Sample


PARAMETER_NAMES = ("top_width", "bottom_width", "height")

DEFAULT_CHANNEL_SETS = {
    "se1": ("se1_lt50", "se1_ge50"),
    "se2": ("se2_lt50", "se2_ge50"),
    "se1_lt50": ("se1_lt50",),
    "se2_lt50": ("se2_lt50",),
    "lle_primary": ("lle_primary",),
    "non_lle_primary": ("non_lle_primary",),
    "low_energy_pair": ("se1_lt50", "se2_lt50"),
    "causal_se_quartet": ("se1_lt50", "se1_ge50", "se2_lt50", "se2_ge50"),
    "energy_loss_pair": ("lle_primary", "non_lle_primary"),
    "primary_pair": ("lle_primary", "non_lle_primary"),
    "energy_cut_pair": ("sey_50ev", "bse_50ev"),
    "all_disjoint": DISJOINT_POPULATION_CHANNELS,
    "causal_lle_v3": DISJOINT_POPULATION_CHANNELS,
}

TRANSMISSION_DISJOINT_POPULATION_CHANNELS = tuple(
    f"back_{name}" for name in DISJOINT_POPULATION_CHANNELS
) + tuple(f"fwd_{ring}" for ring in TRANSMISSION_RINGS)

TRANSMISSION_CHANNEL_SETS = {
    "se1": ("se1_lt50", "se1_ge50"),
    "se2": ("se2_lt50", "se2_ge50"),
    "lle_primary": ("lle_primary",),
    "non_lle_primary": ("non_lle_primary",),
    # Reflected (SEM) side.
    "back_low_energy_pair": ("back_se1_lt50", "back_se2_lt50"),
    "back_primary_pair": ("back_lle_primary", "back_non_lle_primary"),
    "backward_causal": tuple(
        f"back_{name}" for name in DISJOINT_POPULATION_CHANNELS
    ),
    # Transmitted (STEM) side.
    "bf": ("fwd_bf",),
    "adf": ("fwd_adf",),
    "haadf": ("fwd_haadf",),
    "stem_rings": ("fwd_bf", "fwd_adf", "fwd_haadf"),
    "forward_rings": tuple(f"fwd_{ring}" for ring in TRANSMISSION_RINGS),
    "bf_and_haadf": ("fwd_bf", "fwd_haadf"),
    # Both sides.
    "hemisphere_pair": ("backward_all", "forward_all"),
    "energy_cut_pair": ("sey_50ev", "bse_50ev"),
    "all_disjoint": TRANSMISSION_DISJOINT_POPULATION_CHANNELS,
    "causal_lle_v3_transmission": TRANSMISSION_DISJOINT_POPULATION_CHANNELS,
}

V2_CHANNEL_SETS = {
    "se1": ("se1",),
    "se2": ("se2",),
    "lle_primary": ("lle_primary",),
    "non_lle_primary": ("non_lle_primary",),
    "low_energy_pair": ("se1", "se2"),
    "energy_loss_pair": ("lle_primary", "non_lle_primary"),
    "primary_pair": ("lle_primary", "non_lle_primary"),
    "energy_cut_pair": ("sey_50ev", "bse_50ev"),
    "all_disjoint": V2_DISJOINT_POPULATION_CHANNELS,
    "causal_lle_v2": V2_DISJOINT_POPULATION_CHANNELS,
}

V062_DISJOINT_POPULATION_CHANNELS = (
    "se1",
    "se2",
    "fast_cascade_ge50",
    "lle_bse",
    "non_lle_bse",
)

V062_CHANNEL_SETS = {
    "se1": ("se1",),
    "se2": ("se2",),
    "lle_bse": ("lle_bse",),
    "non_lle_bse": ("non_lle_bse",),
    "low_energy_pair": ("se1", "se2"),
    "energy_loss_pair": ("lle_bse", "non_lle_bse"),
    "primary_pair": ("lle_bse", "non_lle_bse"),
    "energy_cut_pair": ("sey_50ev", "bse_50ev"),
    "all_disjoint": V062_DISJOINT_POPULATION_CHANNELS,
    "v062_causal_lle": V062_DISJOINT_POPULATION_CHANNELS,
}

LEGACY_CHANNEL_SETS = {
    "se1": ("se1",),
    "se2": ("se2",),
    "bse1": ("bse1",),
    "bse2": ("bse2",),
    "low_energy_pair": ("se1", "se2"),
    "primary_pair": ("bse1", "bse2"),
    "energy_cut_pair": ("sey_50ev", "bse_50ev"),
    "all_disjoint": LEGACY_DISJOINT_POPULATION_CHANNELS,
    "legacy_branch_v1": LEGACY_DISJOINT_POPULATION_CHANNELS,
}


def _positive_grid(values, name):
    try:
        result = tuple(sorted(set(float(value) for value in values)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite positive numbers") from exc
    if not result or not all(math.isfinite(value) and value > 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive numbers")
    return result


def _channel_indices(available, requested):
    requested = tuple(requested)
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise KeyError(f"unknown channels: {unknown}")
    return requested, [available.index(name) for name in requested]


def _select_covariance(values, indices):
    return np.take(np.take(values, indices, axis=-2), indices, axis=-1)


def _interpolate_last_axis(x_new, x_old, values):
    """Interpolate arrays whose first axis corresponds to ``x_old``."""
    values = np.asarray(values, dtype=float)
    flat = values.reshape(len(x_old), -1)
    output = np.empty((len(x_new), flat.shape[1]), dtype=float)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(
            x_new, x_old, flat[:, column],
            left=flat[0, column], right=flat[-1, column],
        )
    return output.reshape((len(x_new),) + values.shape[1:])


def _precision(covariance, relative_floor=1e-6, absolute_floor=1e-12):
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(eigenvalues)), float(np.max(np.diag(covariance))), 0.0)
    floor = max(float(absolute_floor), float(relative_floor) * scale)
    clipped = np.maximum(eigenvalues, floor)
    return (eigenvectors / clipped) @ eigenvectors.T


@dataclass(frozen=True)
class TrapezoidSweepConfig:
    """Cartesian parameter grid in the transport length unit (Angstrom)."""

    top_widths: Sequence[float]
    bottom_widths: Sequence[float]
    heights: Sequence[float]
    center_x: float = 0.0
    substrate_z: float = 0.0

    def __post_init__(self):
        top = _positive_grid(self.top_widths, "top_widths")
        bottom = _positive_grid(self.bottom_widths, "bottom_widths")
        height = _positive_grid(self.heights, "heights")
        center = float(self.center_x)
        substrate = float(self.substrate_z)
        if not math.isfinite(center) or not math.isfinite(substrate):
            raise ValueError("center_x and substrate_z must be finite")
        valid = [point for point in itertools.product(top, bottom, height)
                 if point[1] >= point[0]]
        if not valid:
            raise ValueError("the grid contains no bottom_width >= top_width models")
        object.__setattr__(self, "top_widths", top)
        object.__setattr__(self, "bottom_widths", bottom)
        object.__setattr__(self, "heights", height)
        object.__setattr__(self, "center_x", center)
        object.__setattr__(self, "substrate_z", substrate)

    @property
    def parameter_points(self):
        return tuple(
            point for point in itertools.product(
                self.top_widths, self.bottom_widths, self.heights
            ) if point[1] >= point[0]
        )

    def to_dict(self):
        return {
            "top_widths_angstrom": list(self.top_widths),
            "bottom_widths_angstrom": list(self.bottom_widths),
            "heights_angstrom": list(self.heights),
            "center_x_angstrom": self.center_x,
            "substrate_z_angstrom": self.substrate_z,
        }


@dataclass(frozen=True)
class ProfileObservation:
    """One-dimensional population profile and covariance of its mean."""

    x_positions: np.ndarray
    channels: tuple
    yields: np.ndarray
    covariance_of_mean: np.ndarray
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        x = np.asarray(self.x_positions, dtype=float)
        channels = tuple(str(name) for name in self.channels)
        yields = np.asarray(self.yields, dtype=float)
        covariance = np.asarray(self.covariance_of_mean, dtype=float)
        if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
            raise ValueError("x_positions must be a finite one-dimensional grid")
        if len(set(channels)) != len(channels) or not channels:
            raise ValueError("channels must be unique and non-empty")
        if yields.shape != (len(channels), len(x)):
            raise ValueError("yields must have shape (channel, x)")
        if covariance.shape != (len(x), len(channels), len(channels)):
            raise ValueError("covariance_of_mean must have shape (x, channel, channel)")
        if not np.all(np.isfinite(yields)) or not np.all(np.isfinite(covariance)):
            raise ValueError("profile arrays must be finite")
        object.__setattr__(self, "x_positions", x)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "yields", yields)
        object.__setattr__(self, "covariance_of_mean", covariance)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_raster_result(cls, result: RasterResult, row=0):
        if result.config.shape[0] <= row:
            raise IndexError("raster row is out of range")
        channels = tuple(result.classifier.channels)
        yields = np.stack([result.yield_maps[name][row] for name in channels])
        return cls(
            result.x_positions,
            channels,
            yields,
            result.yield_covariance[row],
            result.metadata(),
        )

    @classmethod
    def from_npz(cls, path, row=0):
        """Load raster v2/v3, with diagonal-SEM fallback for v1 archives."""
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if "covariance_channels" in data:
                channels = tuple(str(value) for value in data["covariance_channels"])
            else:
                channels = tuple(metadata["channel_definitions"])
            yields = np.stack([data[f"yield__{name}"][row] for name in channels])
            if "yield_covariance" in data:
                covariance = np.asarray(data["yield_covariance"][row], dtype=float)
            else:
                sem = np.stack([data[f"sem__{name}"][row] for name in channels], axis=-1)
                covariance = np.zeros((sem.shape[0], len(channels), len(channels)))
                diagonal = np.arange(len(channels))
                covariance[:, diagonal, diagonal] = sem * sem
            x = np.asarray(data["x_angstrom"], dtype=float)
        return cls(x, channels, yields, covariance, metadata)

    def select(self, channels):
        channels, indices = _channel_indices(self.channels, channels)
        return ProfileObservation(
            self.x_positions,
            channels,
            self.yields[indices],
            _select_covariance(self.covariance_of_mean, indices),
            self.metadata,
        )


@dataclass(frozen=True)
class TrapezoidModelLibrary:
    """A covariance-bearing forward-model library on a parameter grid."""

    parameters: np.ndarray
    x_positions: np.ndarray
    channels: tuple
    yields: np.ndarray
    covariance_of_mean: np.ndarray
    completed_primaries: np.ndarray
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        parameters = np.asarray(self.parameters, dtype=float)
        x = np.asarray(self.x_positions, dtype=float)
        channels = tuple(str(name) for name in self.channels)
        yields = np.asarray(self.yields, dtype=float)
        covariance = np.asarray(self.covariance_of_mean, dtype=float)
        completed = np.asarray(self.completed_primaries, dtype=np.int64)
        n_models = len(parameters)
        expected_yields = (n_models, len(channels), len(x))
        expected_cov = (n_models, len(x), len(channels), len(channels))
        if parameters.ndim != 2 or parameters.shape[1] != len(PARAMETER_NAMES):
            raise ValueError("parameters must have shape (model, 3)")
        if yields.shape != expected_yields or covariance.shape != expected_cov:
            raise ValueError("model-library yield or covariance shape is invalid")
        if completed.shape != (n_models, len(x)):
            raise ValueError("completed_primaries must have shape (model, x)")
        if len(set(channels)) != len(channels) or not channels:
            raise ValueError("channels must be unique and non-empty")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "x_positions", x)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "yields", yields)
        object.__setattr__(self, "covariance_of_mean", covariance)
        object.__setattr__(self, "completed_primaries", completed)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_results(cls, parameter_points, results, metadata=None):
        points = np.asarray(parameter_points, dtype=float)
        results = tuple(results)
        if len(points) != len(results) or not results:
            raise ValueError("one raster result is required per parameter point")
        if any(result.config.shape[0] != 1 for result in results):
            raise ValueError("trapezoid model libraries currently require ny=1")
        channels = tuple(results[0].classifier.channels)
        x = results[0].x_positions
        for result in results[1:]:
            if tuple(result.classifier.channels) != channels:
                raise ValueError("all model results must use the same channels")
            if not np.array_equal(result.x_positions, x):
                raise ValueError("all model results must use the same x grid")
        yields = np.stack([
            np.stack([result.yield_maps[name][0] for name in channels])
            for result in results
        ])
        covariance = np.stack([result.yield_covariance[0] for result in results])
        completed = np.stack([result.completed_primaries[0] for result in results])
        payload_metadata = dict(metadata or {})
        payload_metadata.setdefault("sample_name", results[0].sample_name)
        payload_metadata.setdefault("raster_config", results[0].config.to_dict())
        payload_metadata.setdefault(
            "classifier_config", results[0].classifier.to_dict()
        )
        payload_metadata.setdefault(
            "channel_definitions", results[0].classifier.definitions
        )
        payload_metadata.setdefault(
            "disjoint_population_channels",
            list(results[0].classifier.disjoint_channels),
        )
        return cls(points, x, channels, yields, covariance, completed, payload_metadata)

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "seemc-imaging-trapezoid-library-v1",
            "length_unit": "angstrom",
            "parameter_names": list(PARAMETER_NAMES),
            "channels": list(self.channels),
            **dict(self.metadata),
        }
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            parameter_names=np.asarray(PARAMETER_NAMES),
            parameters_angstrom=self.parameters,
            x_angstrom=self.x_positions,
            channels=np.asarray(self.channels),
            yields=self.yields,
            covariance_of_mean=self.covariance_of_mean,
            completed_primaries=self.completed_primaries,
        )
        return path

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            return cls(
                data["parameters_angstrom"],
                data["x_angstrom"],
                tuple(str(value) for value in data["channels"]),
                data["yields"],
                data["covariance_of_mean"],
                data["completed_primaries"],
                metadata,
            )

    def observation(self, model_index):
        index = int(model_index)
        return ProfileObservation(
            self.x_positions,
            self.channels,
            self.yields[index],
            self.covariance_of_mean[index],
            {"parameters_angstrom": self.parameters[index].tolist(), **self.metadata},
        )

    def nearest_model(self, parameters):
        target = np.asarray(parameters, dtype=float)
        if target.shape != (3,):
            raise ValueError("parameters must contain top width, bottom width, height")
        scales = np.maximum(np.ptp(self.parameters, axis=0), 1.0)
        return int(np.argmin(np.sum(((self.parameters - target) / scales) ** 2, axis=1)))


class TrapezoidSweepDriver:
    """Run a common-random-number raster for every valid grid point."""

    def __init__(self, sample: Sample, raster_config: RasterConfig,
                 sweep_config: TrapezoidSweepConfig,
                 classifier: Optional[PopulationClassifier] = None):
        if not isinstance(sample, Sample):
            raise TypeError("sample must be a seemc_imaging.Sample")
        if raster_config.shape[0] != 1:
            raise ValueError("parameter sweeps currently require y_positions of length 1")
        self.sample = sample
        self.raster_config = raster_config
        self.sweep_config = sweep_config
        self.classifier = classifier

    def run(self, use_parallel=False, workers=None, progress=True):
        points = self.sweep_config.parameter_points
        iterator = points
        if progress:
            try:
                from tqdm import tqdm
            except ImportError:  # pragma: no cover
                pass
            else:
                iterator = tqdm(points, desc="Trapezoid models")
        results = []
        for top_width, bottom_width, height in iterator:
            geometry = TrapezoidalLine(
                top_width=top_width,
                bottom_width=bottom_width,
                height=height,
                center_x=self.sweep_config.center_x,
                substrate_z=self.sweep_config.substrate_z,
            )
            result = RasterDriver(
                self.sample, geometry, self.raster_config, self.classifier
            ).run(
                use_parallel=use_parallel,
                workers=workers,
                progress=False,
            )
            results.append(result)
        metadata = {
            "sweep_config": self.sweep_config.to_dict(),
            "common_random_numbers": True,
            "seed_note": (
                "Every geometry reuses the same pixel/trajectory transport and "
                "beam-spot seed keys to reduce Monte Carlo noise in model differences."
            ),
        }
        return TrapezoidModelLibrary.from_results(points, results, metadata)


@dataclass(frozen=True)
class ProfileFitResult:
    channels: tuple
    best_model_index: int
    best_parameters: np.ndarray
    x_shift: float
    scale: float
    channel_offsets: np.ndarray
    chi_square: float
    degrees_of_freedom: int
    model_scores: np.ndarray
    model_shifts: np.ndarray

    @property
    def reduced_chi_square(self):
        return self.chi_square / self.degrees_of_freedom

    def to_dict(self):
        return {
            "channels": list(self.channels),
            "best_model_index": self.best_model_index,
            "best_parameters_angstrom": {
                name: float(value)
                for name, value in zip(PARAMETER_NAMES, self.best_parameters)
            },
            "best_parameters_nm": {
                name: float(value) / 10.0
                for name, value in zip(PARAMETER_NAMES, self.best_parameters)
            },
            "x_shift_angstrom": self.x_shift,
            "x_shift_nm": self.x_shift / 10.0,
            "scale": self.scale,
            "channel_offsets": {
                name: float(value)
                for name, value in zip(self.channels, self.channel_offsets)
            },
            "chi_square": self.chi_square,
            "degrees_of_freedom": self.degrees_of_freedom,
            "reduced_chi_square": self.reduced_chi_square,
        }

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


class ProfileFitter:
    """Fit a measured/simulated line profile to a discrete model library."""

    def __init__(self, library: TrapezoidModelLibrary):
        self.library = library

    @staticmethod
    def _canonical_se_reference(value):
        """Map historical SE-reference descriptions to stable identifiers."""
        aliases = {
            "immediate_parent_direction_vs_launch_surface_normal":
                "launch_surface",
            "immediate_parent_direction_vs_escape_surface_normal":
                "escape_surface",
        }
        return aliases.get(value, value)

    @staticmethod
    def _default_se_parent_rule(definition):
        """The parent rule an archive used when it recorded none.

        Only ``causal_lle_v3`` ever wrote the field, so an archive without it
        predates the option and used the immediate-parent rule.
        """
        if definition in (None, "branch_v1"):
            return None
        return "immediate_parent"

    @staticmethod
    def _classifier_config(metadata, channels):
        config = metadata.get("classifier_config")
        if isinstance(config, Mapping):
            config = dict(config)
            config["se_reference"] = ProfileFitter._canonical_se_reference(
                config.get("se_reference")
            )
            config.setdefault("se_parent_rule", None)
            if config["se_parent_rule"] is None:
                config["se_parent_rule"] = ProfileFitter._default_se_parent_rule(
                    config.get("definition")
                )
            config.setdefault("lle_max_loss_frac", None)
            return config
        definition = metadata.get("classifier")
        channel_set = set(channels)
        if definition is None:
            if {"bse1", "bse2"}.issubset(channel_set):
                definition = "branch_v1"
            elif {"fwd_bf", "back_se1_lt50"}.issubset(channel_set):
                definition = "causal_lle_v3"
            elif {"se1_lt50", "se2_lt50"}.issubset(channel_set):
                definition = "causal_lle_v3"
            elif {"lle_primary", "non_lle_primary"}.issubset(channel_set):
                definition = "causal_lle_v2"
            elif {"lle_bse", "non_lle_bse"}.issubset(channel_set):
                definition = "causal_lle_v2"
        if definition is None:
            return {}
        is_v062 = {"lle_bse", "non_lle_bse"}.issubset(channel_set)
        config = {
            "definition": definition,
            "bse_cutoff_ev": metadata.get("bse_cutoff_ev"),
            "lle_max_loss_ev": metadata.get(
                "lle_max_loss_ev", metadata.get("low_loss_max_ev")
            ),
            "lle_max_loss_frac": metadata.get("lle_max_loss_frac"),
            "se_parent_rule": (
                metadata.get("se_parent_rule")
                or ProfileFitter._default_se_parent_rule(definition)
            ),
            "se_reference": (
                metadata.get("se_reference")
                or ("escape_surface" if is_v062 else None)
            ),
        }
        config["se_reference"] = ProfileFitter._canonical_se_reference(
            config["se_reference"]
        )
        return config

    def compatibility_issues(self, observation: ProfileObservation):
        """Return scientific metadata mismatches that invalidate a fit."""
        library_meta = self.library.metadata
        observation_meta = observation.metadata
        issues = []

        library_sample = library_meta.get("sample_name")
        observation_sample = observation_meta.get("sample_name")
        if (library_sample is not None and observation_sample is not None
                and str(library_sample) != str(observation_sample)):
            issues.append(
                f"sample mismatch: library={library_sample!r}, "
                f"observation={observation_sample!r}"
            )

        library_raster = library_meta.get("raster_config", {})
        observation_raster = observation_meta.get("config", {})
        for key, label in (
            ("energy_ev", "landing energy"),
            ("beam_fwhm_angstrom", "beam FWHM"),
        ):
            first = library_raster.get(key)
            second = observation_raster.get(key)
            compatible = True
            if first is not None and second is not None:
                try:
                    compatible = bool(np.allclose(
                        np.asarray(first, dtype=float),
                        np.asarray(second, dtype=float),
                        rtol=0.0, atol=1e-12,
                    ))
                except ValueError:
                    compatible = False
            if first is not None and second is not None and not compatible:
                issues.append(
                    f"{label} mismatch: library={first!r}, observation={second!r}"
                )

        library_classifier = self._classifier_config(
            library_meta, self.library.channels
        )
        observation_classifier = self._classifier_config(
            observation_meta, observation.channels
        )
        first_definition = library_classifier.get("definition")
        second_definition = observation_classifier.get("definition")
        if (first_definition is not None and second_definition is not None
                and first_definition != second_definition):
            issues.append(
                "classifier mismatch: "
                f"library={first_definition!r}, observation={second_definition!r}"
            )
        if first_definition == second_definition:
            for key, label in (
                ("se_reference", "SE reference"),
                ("se_parent_rule", "SE parent rule"),
                ("lle_criterion", "LLE criterion"),
                ("transmission", "transmission detector"),
            ):
                first_value = library_classifier.get(key)
                second_value = observation_classifier.get(key)
                if (first_value is not None and second_value is not None
                        and first_value != second_value):
                    issues.append(
                        f"{label} mismatch: "
                        f"library={first_value!r}, "
                        f"observation={second_value!r}"
                    )
            for key, label in (
                ("bse_cutoff_ev", "SE/BSE cutoff"),
                ("lle_max_loss_ev", "LLE maximum loss"),
                ("lle_max_loss_frac", "LLE maximum loss fraction"),
            ):
                first = library_classifier.get(key)
                second = observation_classifier.get(key)
                if (first is not None and second is not None
                        and not math.isclose(
                            float(first), float(second), rel_tol=0.0, abs_tol=1e-12
                        )):
                    issues.append(
                        f"{label} mismatch: library={first!r}, "
                        f"observation={second!r}"
                    )
        return tuple(issues)

    def _score(self, observation, model_index, channels, shift,
               fit_scale, fit_channel_offsets, include_model_covariance,
               relative_floor, absolute_floor):
        channels, model_indices = _channel_indices(self.library.channels, channels)
        _, observation_indices = _channel_indices(observation.channels, channels)
        target = observation.yields[observation_indices].T
        target_covariance = _select_covariance(
            observation.covariance_of_mean, observation_indices
        )
        sample_x = observation.x_positions - float(shift)
        model = _interpolate_last_axis(
            sample_x, self.library.x_positions,
            self.library.yields[model_index, model_indices].T,
        )
        model_covariance = _interpolate_last_axis(
            sample_x, self.library.x_positions,
            _select_covariance(
                self.library.covariance_of_mean[model_index], model_indices
            ),
        )

        n_channels = len(channels)
        scale = 1.0
        offsets = np.zeros(n_channels)
        for _ in range(3):
            covariance = target_covariance.copy()
            if include_model_covariance:
                covariance = covariance + scale * scale * model_covariance
            precisions = [
                _precision(value, relative_floor, absolute_floor)
                for value in covariance
            ]
            n_columns = int(fit_scale) + (n_channels if fit_channel_offsets else 0)
            if n_columns:
                normal = np.zeros((n_columns, n_columns))
                right = np.zeros(n_columns)
                for pixel, precision in enumerate(precisions):
                    columns = []
                    if fit_scale:
                        columns.append(model[pixel])
                    if fit_channel_offsets:
                        columns.extend(np.eye(n_channels))
                    design = np.stack(columns, axis=1)
                    baseline = np.zeros(n_channels) if fit_scale else model[pixel]
                    normal += design.T @ precision @ design
                    right += design.T @ precision @ (target[pixel] - baseline)
                coefficients = np.linalg.pinv(normal, rcond=1e-12) @ right
                cursor = 0
                scale = float(coefficients[cursor]) if fit_scale else 1.0
                cursor += int(fit_scale)
                offsets = (
                    coefficients[cursor:cursor + n_channels]
                    if fit_channel_offsets else np.zeros(n_channels)
                )
            prediction = scale * model + offsets

        covariance = target_covariance.copy()
        if include_model_covariance:
            covariance = covariance + scale * scale * model_covariance
        precisions = [
            _precision(value, relative_floor, absolute_floor)
            for value in covariance
        ]
        residual = target - prediction
        chi_square = sum(
            float(value @ precision @ value)
            for value, precision in zip(residual, precisions)
        )
        nuisance = int(fit_scale) + (n_channels if fit_channel_offsets else 0)
        varying_geometry = int(np.sum(np.ptp(self.library.parameters, axis=0) > 0.0))
        dof = max(target.size - nuisance - varying_geometry, 1)
        return chi_square, dof, scale, np.asarray(offsets, dtype=float)

    def fit(self, observation: ProfileObservation,
            channels=None, shift_values=(0.0,),
            fit_scale=True, fit_channel_offsets=False,
            include_model_covariance=True,
            relative_covariance_floor=1e-6,
            absolute_covariance_floor=1e-12,
            allow_incompatible=False):
        issues = self.compatibility_issues(observation)
        if issues and not allow_incompatible:
            raise ValueError(
                "incompatible model library and observation:\n- "
                + "\n- ".join(issues)
                + "\nUse allow_incompatible=True only for an intentional "
                  "cross-condition comparison."
            )
        if channels is None:
            available = set(self.library.channels)
            metadata_channels = self.library.metadata.get(
                "disjoint_population_channels"
            )
            if metadata_channels and set(metadata_channels).issubset(available):
                channels = tuple(metadata_channels)
            elif set(DISJOINT_POPULATION_CHANNELS).issubset(available):
                channels = DISJOINT_POPULATION_CHANNELS
            elif set(LEGACY_DISJOINT_POPULATION_CHANNELS).issubset(available):
                channels = LEGACY_DISJOINT_POPULATION_CHANNELS
            else:
                raise ValueError("model library has no recognized disjoint basis")
        channels = tuple(channels)
        shifts = tuple(float(value) for value in shift_values)
        if not shifts or not all(math.isfinite(value) for value in shifts):
            raise ValueError("shift_values must contain finite values")
        scores = np.full(len(self.library.parameters), np.inf)
        best_shifts = np.zeros(len(scores))
        best_payloads = [None] * len(scores)
        for model_index in range(len(scores)):
            for shift in shifts:
                payload = self._score(
                    observation, model_index, channels, shift,
                    fit_scale, fit_channel_offsets, include_model_covariance,
                    relative_covariance_floor, absolute_covariance_floor,
                )
                if payload[0] < scores[model_index]:
                    scores[model_index] = payload[0]
                    best_shifts[model_index] = shift
                    best_payloads[model_index] = payload
        best = int(np.argmin(scores))
        chi_square, dof, scale, offsets = best_payloads[best]
        return ProfileFitResult(
            channels,
            best,
            self.library.parameters[best].copy(),
            float(best_shifts[best]),
            float(scale),
            offsets,
            float(chi_square),
            int(dof),
            scores,
            best_shifts,
        )

    def predict(self, observation: ProfileObservation, result: ProfileFitResult):
        """Evaluate the best-fit profiles on the observation's x grid."""
        _, indices = _channel_indices(self.library.channels, result.channels)
        sample_x = observation.x_positions - result.x_shift
        model = _interpolate_last_axis(
            sample_x,
            self.library.x_positions,
            self.library.yields[result.best_model_index, indices].T,
        ).T
        return result.scale * model + result.channel_offsets[:, None]


@dataclass(frozen=True)
class InformationResult:
    name: str
    channels: tuple
    reference_parameters: np.ndarray
    parameter_covariance: np.ndarray
    fisher_information: np.ndarray
    rank: int
    condition_number: float
    estimable_parameters: tuple = PARAMETER_NAMES

    @property
    def parameter_standard_errors(self):
        return np.sqrt(np.maximum(np.diag(self.parameter_covariance), 0.0))

    @property
    def parameter_correlation(self):
        standard = self.parameter_standard_errors
        denominator = standard[:, None] * standard[None, :]
        valid = (
            (denominator > 0.0)
            & np.isfinite(denominator)
            & np.isfinite(self.parameter_covariance)
        )
        return np.divide(
            self.parameter_covariance,
            denominator,
            out=np.full_like(self.parameter_covariance, np.nan),
            where=valid,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "channels": list(self.channels),
            "reference_parameters_nm": {
                key: float(value) / 10.0
                for key, value in zip(PARAMETER_NAMES, self.reference_parameters)
            },
            "standard_error_nm": {
                key: (float(value) / 10.0 if np.isfinite(value) else None)
                for key, value in zip(PARAMETER_NAMES, self.parameter_standard_errors)
            },
            "estimable_parameters": list(self.estimable_parameters),
            "rank": self.rank,
            "condition_number": self.condition_number,
            "parameter_correlation": [
                [float(value) if np.isfinite(value) else None for value in row]
                for row in self.parameter_correlation
            ],
        }


def _finite_difference(library, reference_index, parameter_index):
    reference = library.parameters[reference_index]
    other = [index for index in range(3) if index != parameter_index]
    matches = np.all(np.isclose(
        library.parameters[:, other], reference[other], rtol=0.0, atol=1e-12
    ), axis=1)
    values = library.parameters[:, parameter_index]
    lower = np.where(matches & (values < reference[parameter_index]))[0]
    upper = np.where(matches & (values > reference[parameter_index]))[0]
    if len(lower):
        low = lower[np.argmax(values[lower])]
    else:
        low = reference_index
    if len(upper):
        high = upper[np.argmin(values[upper])]
    else:
        high = reference_index
    if low == high:
        raise ValueError(
            f"parameter grid cannot estimate a derivative for {PARAMETER_NAMES[parameter_index]}"
        )
    return (library.yields[high] - library.yields[low]) / (values[high] - values[low])


def compare_channel_information(
        library: TrapezoidModelLibrary,
        reference_parameters=None,
        channel_sets: Optional[Mapping[str, Sequence[str]]] = None,
        fit_scale=True,
        fit_channel_offsets=False,
        covariance_multiplier=1.0,
        relative_covariance_floor=1e-6,
        absolute_covariance_floor=1e-12):
    """Compute local Fisher bounds for several measurable channel sets.

    Finite differences use adjacent models that share the other two geometry
    parameters.  Common random numbers in ``TrapezoidSweepDriver`` suppress
    Monte Carlo noise in these derivatives.
    """
    if reference_parameters is None:
        unique = [np.unique(library.parameters[:, index]) for index in range(3)]
        reference_parameters = [values[len(values) // 2] for values in unique]
    reference_index = library.nearest_model(reference_parameters)
    reference = library.parameters[reference_index]
    estimable_indices = tuple(
        index for index in range(3)
        if np.ptp(library.parameters[:, index]) > 0.0
    )
    if not estimable_indices:
        raise ValueError("the model library varies no geometry parameters")
    derivatives = np.stack([
        _finite_difference(library, reference_index, index)
        for index in estimable_indices
    ], axis=-1)  # (channel, x, estimable parameter)
    if channel_sets is None:
        available = set(library.channels)
        # Order matters: a causal_lle_v3 library also contains every
        # causal_lle_v2 channel name, so the more specific basis is tested
        # first and the v2 basis only matches a genuinely v2 archive.
        if set(TRANSMISSION_DISJOINT_POPULATION_CHANNELS).issubset(available):
            channel_sets = TRANSMISSION_CHANNEL_SETS
        elif set(DISJOINT_POPULATION_CHANNELS).issubset(available):
            channel_sets = DEFAULT_CHANNEL_SETS
        elif set(V2_DISJOINT_POPULATION_CHANNELS).issubset(available):
            channel_sets = V2_CHANNEL_SETS
        elif set(V062_DISJOINT_POPULATION_CHANNELS).issubset(available):
            channel_sets = V062_CHANNEL_SETS
        elif set(LEGACY_DISJOINT_POPULATION_CHANNELS).issubset(available):
            channel_sets = LEGACY_CHANNEL_SETS
        else:
            raise ValueError(
                "the library contains neither the causal/LLE nor legacy "
                "disjoint population basis"
            )
    channel_sets = {
        name: tuple(channels)
        for name, channels in dict(channel_sets).items()
        if set(channels).issubset(set(library.channels))
    }
    if not channel_sets:
        raise ValueError("none of the requested channel sets is available")
    output = {}
    for name, channels in channel_sets.items():
        channels, indices = _channel_indices(library.channels, channels)
        n_channels = len(channels)
        geometry_derivatives = derivatives[indices].transpose(1, 0, 2)
        reference_yields = library.yields[reference_index, indices].T
        covariance = _select_covariance(
            library.covariance_of_mean[reference_index], indices
        ) * float(covariance_multiplier)
        nuisance = int(fit_scale) + (n_channels if fit_channel_offsets else 0)
        n_geometry = len(estimable_indices)
        fisher = np.zeros((n_geometry + nuisance, n_geometry + nuisance))
        for pixel in range(len(library.x_positions)):
            columns = [geometry_derivatives[pixel]]
            if fit_scale:
                columns.append(reference_yields[pixel][:, None])
            if fit_channel_offsets:
                columns.append(np.eye(n_channels))
            design = np.concatenate(columns, axis=1)
            precision = _precision(
                covariance[pixel],
                relative_covariance_floor,
                absolute_covariance_floor,
            )
            fisher += design.T @ precision @ design
        eigenvalues = np.linalg.eigvalsh(0.5 * (fisher + fisher.T))
        tolerance = max(float(np.max(eigenvalues)), 1.0) * 1e-10
        rank = int(np.sum(eigenvalues > tolerance))
        positive = eigenvalues[eigenvalues > tolerance]
        condition = (
            float(np.max(positive) / np.min(positive))
            if len(positive) else math.inf
        )
        full_covariance = np.linalg.pinv(fisher, rcond=1e-10)
        active_covariance = full_covariance[:n_geometry, :n_geometry]
        if rank < fisher.shape[0]:
            # A rank-deficient full problem has an unconstrained combination;
            # report infinity instead of an optimistically small pseudo-bound.
            active_covariance = active_covariance.copy()
            active_covariance[np.diag_indices(n_geometry)] = math.inf
        parameter_covariance = np.full((3, 3), np.nan)
        fisher_geometry = np.full((3, 3), np.nan)
        parameter_covariance[np.ix_(estimable_indices, estimable_indices)] = active_covariance
        fisher_geometry[np.ix_(estimable_indices, estimable_indices)] = fisher[
            :n_geometry, :n_geometry
        ]
        output[name] = InformationResult(
            name,
            channels,
            reference.copy(),
            parameter_covariance,
            fisher_geometry,
            rank,
            condition,
            tuple(PARAMETER_NAMES[index] for index in estimable_indices),
        )
    return output


__all__ = [
    "DEFAULT_CHANNEL_SETS",
    "LEGACY_CHANNEL_SETS",
    "TRANSMISSION_CHANNEL_SETS",
    "TRANSMISSION_DISJOINT_POPULATION_CHANNELS",
    "V2_CHANNEL_SETS",
    "V062_CHANNEL_SETS",
    "V062_DISJOINT_POPULATION_CHANNELS",
    "InformationResult",
    "PARAMETER_NAMES",
    "ProfileFitResult",
    "ProfileFitter",
    "ProfileObservation",
    "TrapezoidModelLibrary",
    "TrapezoidSweepConfig",
    "TrapezoidSweepDriver",
    "compare_channel_information",
]
