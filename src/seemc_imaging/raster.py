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


FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


CHANNEL_DEFINITIONS = {
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
    "se1": (
        "branch_v1: low-energy cascade emission born before the incident "
        "primary first turned toward the launch surface."
    ),
    "se2": (
        "branch_v1: low-energy cascade emission born after the incident "
        "primary first turned toward the launch surface."
    ),
    "bse1": (
        "branch_v1: emitted incident primary whose first turn toward the "
        "surface was caused by its first elastic collision."
    ),
    "bse2": (
        "branch_v1: every other emitted incident primary, including "
        "multiple-scattering returns."
    ),
}

POPULATION_CHANNELS = tuple(CHANNEL_DEFINITIONS)


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

    ``branch_v1`` is an operational definition, not a claim that SE1/SE2 or
    BSE1/BSE2 have one universal definition in the SEM literature.  The
    energy-cut and ancestry channels remain available alongside it.
    """

    bse_cutoff_ev: float = 50.0
    definition: str = "branch_v1"

    def __post_init__(self):
        cutoff = float(self.bse_cutoff_ev)
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("bse_cutoff_ev must be finite and non-negative")
        if self.definition != "branch_v1":
            raise ValueError("only definition='branch_v1' is currently supported")
        object.__setattr__(self, "bse_cutoff_ev", cutoff)

    @property
    def channels(self):
        return POPULATION_CHANNELS

    @property
    def definitions(self):
        return dict(CHANNEL_DEFINITIONS)

    def classify(self, result: TrajectoryResult):
        if result.history is None:
            raise ValueError("branch_v1 classification requires trajectory history")
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

        history = result.history
        records = {record.electron_id: record for record in history.electrons}
        events = {event.event_id: event for event in history.events}
        roots = [record for record in history.electrons if record.parent_id is None]
        if len(roots) != 1:
            raise ValueError("branch_v1 expects exactly one incident primary")
        root = roots[0]
        return_event = root.first_surface_return_event_id

        for emission in result.emissions:
            if emission.is_cascade and emission.energy < cutoff:
                record = records[emission.electron_id]
                if return_event is not None and return_event < record.birth_event_id:
                    counts["se2"] += 1
                else:
                    counts["se1"] += 1

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
            if (
                return_event is not None
                and events[return_event].kind == "elastic"
                and elastic_before_return == 1
            ):
                counts["bse1"] = len(primary_emissions)
            else:
                counts["bse2"] = len(primary_emissions)

        if counts["tey"] != counts["sey_50ev"] + counts["bse_50ev"]:
            raise RuntimeError("energy-cut population channels do not partition TEY")
        if counts["tey"] != counts["cascade_all"] + counts["primary_all"]:
            raise RuntimeError("ancestry population channels do not partition TEY")
        if counts["se_cascade_lt50"] != counts["se1"] + counts["se2"]:
            raise RuntimeError("branch_v1 SE channels are incomplete")
        if counts["primary_all"] != counts["bse1"] + counts["bse2"]:
            raise RuntimeError("branch_v1 BSE channels are incomplete")
        return counts


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
        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(self, "x_positions", x)
        object.__setattr__(self, "y_positions", y)
        object.__setattr__(self, "primaries_per_pixel", n)
        object.__setattr__(self, "beam_fwhm", fwhm)
        object.__setattr__(self, "vacuum_direction", direction)
        object.__setattr__(self, "seed", seed)

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


def _simulate_pixel(sample, geometry, config, classifier, task):
    pixel_id, iy, ix, x, y = task
    n = config.primaries_per_pixel
    channels = classifier.channels
    channel_values = np.zeros((n, len(channels)), dtype=np.float64)
    launch_values = np.zeros((n, 4), dtype=np.float64)
    surface_counts = {}
    diagnostics = {}
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

        result = simulate_trajectory(
            sample,
            config.energy_ev,
            local_angle,
            transport_rng,
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

    channel_sum = channel_values.sum(axis=0)
    channel_sum_sq = np.square(channel_values).sum(axis=0)
    launch_sum = launch_values.sum(axis=0)
    launch_sum_sq = np.square(launch_values).sum(axis=0)
    return {
        "pixel_id": pixel_id,
        "iy": iy,
        "ix": ix,
        "channel_sum": channel_sum,
        "channel_sum_sq": channel_sum_sq,
        "launch_sum": launch_sum,
        "launch_sum_sq": launch_sum_sq,
        "surface_counts": surface_counts,
        "diagnostics": diagnostics,
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
    completed_primaries: np.ndarray
    launch_mean: np.ndarray
    launch_sem: np.ndarray
    local_incidence_mean_rad: np.ndarray
    local_incidence_sem_rad: np.ndarray
    surface_hit_counts: Mapping[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Mapping[str, int] = field(default_factory=dict)

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

    def metadata(self):
        return {
            "format": "seemc-imaging-raster-v1",
            "sample_name": self.sample_name,
            "length_unit": "angstrom",
            "classifier": self.classifier.definition,
            "bse_cutoff_ev": self.classifier.bse_cutoff_ev,
            "channel_definitions": self.classifier.definitions,
            "config": self.config.to_dict(),
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

    def rows(self):
        """Return one flat, wide record per pixel for CSV/dataframe use."""
        fractions = self.surface_hit_fractions
        surface_ids = sorted(self.surface_hit_counts)
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
        completed = np.full(shape, n, dtype=np.int64)
        launch_mean = np.zeros(shape + (3,), dtype=np.float64)
        launch_sem = np.zeros(shape + (3,), dtype=np.float64)
        incidence_mean = np.zeros(shape, dtype=np.float64)
        incidence_sem = np.zeros(shape, dtype=np.float64)
        surface_maps = {}
        diagnostics = {}

        for payload in payloads:
            iy, ix = payload["iy"], payload["ix"]
            sums = payload["channel_sum"]
            sums_sq = payload["channel_sum_sq"]
            means = sums / n
            if n > 1:
                variance = np.maximum(
                    (sums_sq - sums * sums / n) / (n - 1), 0.0
                )
                sem = np.sqrt(variance / n)
            else:
                sem = np.zeros_like(means)
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
            completed_primaries=completed,
            launch_mean=launch_mean,
            launch_sem=launch_sem,
            local_incidence_mean_rad=incidence_mean,
            local_incidence_sem_rad=incidence_sem,
            surface_hit_counts=surface_maps,
            diagnostics=diagnostics,
        )


__all__ = [
    "CHANNEL_DEFINITIONS",
    "POPULATION_CHANNELS",
    "PopulationClassifier",
    "RasterConfig",
    "RasterDriver",
    "RasterResult",
    "sample_beam_reference",
]
