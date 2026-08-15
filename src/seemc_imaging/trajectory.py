"""Portable, pickle-free storage for trajectories recorded during a raster."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRAJECTORY_FORMAT = "seemc-imaging-raster-trajectories-v1"
POINT_COLUMNS = ("x_angstrom", "y_angstrom", "z_angstrom", "energy_ev", "time_fs")


def _as_unicode(values):
    values = [str(value) for value in values]
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"U{width}")


@dataclass
class RasterTrajectoryArchive:
    """Flat ragged-array representation of recorded raster cascades.

    ``cascade_electron_offsets`` maps cascades to electrons and
    ``electron_point_offsets`` maps electrons to rows of ``points``.  This
    keeps the NPZ loadable with ``allow_pickle=False`` while retaining the
    complete parent/generation/fate information needed by an animator.
    """

    metadata: dict
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    profile_channels: np.ndarray
    profile_yields: np.ndarray
    profile_sems: np.ndarray
    cascade_pixel_id: np.ndarray
    cascade_iy: np.ndarray
    cascade_ix: np.ndarray
    cascade_trajectory_id: np.ndarray
    cascade_nominal_xy_angstrom: np.ndarray
    cascade_launch_xyz_angstrom: np.ndarray
    cascade_local_incidence_rad: np.ndarray
    cascade_electron_offsets: np.ndarray
    electron_id: np.ndarray
    electron_parent_id: np.ndarray
    electron_generation: np.ndarray
    electron_is_primary: np.ndarray
    electron_birth_energy_ev: np.ndarray
    electron_birth_time_fs: np.ndarray
    electron_fate: np.ndarray
    electron_final_energy_ev: np.ndarray
    electron_final_direction: np.ndarray
    electron_population: np.ndarray
    electron_point_offsets: np.ndarray
    points: np.ndarray

    @property
    def n_cascades(self):
        return int(len(self.cascade_pixel_id))

    @property
    def n_electrons(self):
        return int(len(self.electron_id))

    @property
    def n_points(self):
        return int(len(self.points))

    @property
    def shape(self):
        return (len(self.y_angstrom), len(self.x_angstrom))

    def cascade_electron_indices(self, cascade_index):
        cascade_index = int(cascade_index)
        start = int(self.cascade_electron_offsets[cascade_index])
        stop = int(self.cascade_electron_offsets[cascade_index + 1])
        return range(start, stop)

    def electron_points(self, electron_index):
        electron_index = int(electron_index)
        start = int(self.electron_point_offsets[electron_index])
        stop = int(self.electron_point_offsets[electron_index + 1])
        return self.points[start:stop]

    def cascades_at_pixel(self, pixel_id):
        return np.flatnonzero(self.cascade_pixel_id == int(pixel_id))

    def validate(self):
        if self.metadata.get("format") != TRAJECTORY_FORMAT:
            raise ValueError("unsupported trajectory archive format")
        if self.points.ndim != 2 or self.points.shape[1] != len(POINT_COLUMNS):
            raise ValueError("points must have columns x, y, z, energy, time")
        if len(self.cascade_electron_offsets) != self.n_cascades + 1:
            raise ValueError("invalid cascade_electron_offsets")
        if len(self.electron_point_offsets) != self.n_electrons + 1:
            raise ValueError("invalid electron_point_offsets")
        if int(self.cascade_electron_offsets[-1]) != self.n_electrons:
            raise ValueError("cascade offsets do not close on electron count")
        if int(self.electron_point_offsets[-1]) != self.n_points:
            raise ValueError("electron offsets do not close on point count")
        if np.any(np.diff(self.cascade_electron_offsets) < 0):
            raise ValueError("cascade offsets must be monotone")
        if np.any(np.diff(self.electron_point_offsets) < 0):
            raise ValueError("electron offsets must be monotone")
        if not np.all(np.isfinite(self.points)):
            raise ValueError("trajectory points must be finite")
        for electron_index in range(self.n_electrons):
            times = self.electron_points(electron_index)[:, 4]
            if np.any(np.diff(times) < -1e-12):
                raise ValueError("electron times must be nondecreasing")
        return self

    @classmethod
    def from_records(cls, metadata, x_angstrom, y_angstrom, profile_maps,
                     sem_maps, records):
        records = sorted(
            records,
            key=lambda item: (item["pixel_id"], item["trajectory_id"]),
        )
        cascade_electron_offsets = [0]
        electron_point_offsets = [0]
        electron_rows = []
        point_rows = []
        for cascade in records:
            electrons = sorted(cascade["electrons"], key=lambda item: item["electron_id"])
            for electron in electrons:
                points = np.asarray(electron["points"], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != len(POINT_COLUMNS):
                    raise ValueError("recorded track must have five columns")
                electron_rows.append(electron)
                point_rows.append(points)
                electron_point_offsets.append(
                    electron_point_offsets[-1] + len(points)
                )
            cascade_electron_offsets.append(
                cascade_electron_offsets[-1] + len(electrons)
            )

        channels = tuple(profile_maps)
        profile_shape = (0, len(y_angstrom), len(x_angstrom))
        profile_yields = (
            np.stack([profile_maps[name] for name in channels])
            if channels else np.empty(profile_shape, dtype=float)
        )
        profile_sems = (
            np.stack([sem_maps[name] for name in channels])
            if channels else np.empty(profile_shape, dtype=float)
        )
        points = (
            np.concatenate(point_rows, axis=0)
            if point_rows else np.empty((0, len(POINT_COLUMNS)), dtype=float)
        )

        archive_metadata = dict(metadata)
        archive_metadata.update({
            "format": TRAJECTORY_FORMAT,
            "point_columns": list(POINT_COLUMNS),
            "time_unit": "fs",
            "trajectory_count": len(records),
            "electron_count": len(electron_rows),
            "point_count": len(points),
        })

        def cascade_array(name, dtype):
            return np.asarray([row[name] for row in records], dtype=dtype)

        def electron_array(name, dtype):
            return np.asarray([row[name] for row in electron_rows], dtype=dtype)

        parent_ids = [
            -1 if row["parent_id"] is None else row["parent_id"]
            for row in electron_rows
        ]
        result = cls(
            metadata=archive_metadata,
            x_angstrom=np.asarray(x_angstrom, dtype=float),
            y_angstrom=np.asarray(y_angstrom, dtype=float),
            profile_channels=_as_unicode(channels),
            profile_yields=np.asarray(profile_yields, dtype=float),
            profile_sems=np.asarray(profile_sems, dtype=float),
            cascade_pixel_id=cascade_array("pixel_id", np.int64),
            cascade_iy=cascade_array("iy", np.int64),
            cascade_ix=cascade_array("ix", np.int64),
            cascade_trajectory_id=cascade_array("trajectory_id", np.int64),
            cascade_nominal_xy_angstrom=np.asarray(
                [row["nominal_xy_angstrom"] for row in records], dtype=float
            ).reshape((-1, 2)),
            cascade_launch_xyz_angstrom=np.asarray(
                [row["launch_xyz_angstrom"] for row in records], dtype=float
            ).reshape((-1, 3)),
            cascade_local_incidence_rad=cascade_array(
                "local_incidence_rad", float
            ),
            cascade_electron_offsets=np.asarray(
                cascade_electron_offsets, dtype=np.int64
            ),
            electron_id=electron_array("electron_id", np.int64),
            electron_parent_id=np.asarray(parent_ids, dtype=np.int64),
            electron_generation=electron_array("generation", np.int64),
            electron_is_primary=electron_array("is_primary", bool),
            electron_birth_energy_ev=electron_array("birth_energy_ev", float),
            electron_birth_time_fs=electron_array("birth_time_fs", float),
            electron_fate=_as_unicode(row["fate"] for row in electron_rows),
            electron_final_energy_ev=electron_array("final_energy_ev", float),
            electron_final_direction=np.asarray(
                [row["final_direction"] for row in electron_rows], dtype=float
            ).reshape((-1, 3)),
            electron_population=_as_unicode(
                row["population"] for row in electron_rows
            ),
            electron_point_offsets=np.asarray(
                electron_point_offsets, dtype=np.int64
            ),
            points=points,
        )
        return result.validate()

    def save_npz(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for name in self.__dataclass_fields__:
            if name != "metadata":
                payload[name] = getattr(self, name)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            kwargs = {
                name: np.array(data[name], copy=True)
                for name in cls.__dataclass_fields__
                if name != "metadata"
            }
        return cls(metadata=metadata, **kwargs).validate()


__all__ = ["POINT_COLUMNS", "RasterTrajectoryArchive", "TRAJECTORY_FORMAT"]
