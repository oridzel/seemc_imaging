"""Geometry contracts and the analytic single-plane backend.

Coordinates and distances use the same unit as the transport kernel
(currently Angstrom).  A geometry backend never samples randomness: it only
answers deterministic region and nearest-interface questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable


Vec3 = Tuple[float, float, float]
SOLID_REGION = "solid"
VACUUM_REGION = "vacuum"


def _vec3(values, name: str) -> Vec3:
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain three finite numbers")
    return values


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass(frozen=True)
class SurfaceHit:
    """Nearest crossing of one interface along a finite ray segment.

    ``normal`` is a unit vector pointing from ``region_from`` into
    ``region_to``.  This crossing-oriented convention lets the transport use
    the same reflection/refraction equations on either side of an interface.
    """

    distance: float
    position: Vec3
    normal: Vec3
    surface_id: str
    region_from: str
    region_to: str
    primitive_id: Optional[int] = None

    def __post_init__(self):
        distance = float(self.distance)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("SurfaceHit.distance must be finite and non-negative")
        position = _vec3(self.position, "position")
        normal = _vec3(self.normal, "normal")
        norm = math.sqrt(_dot(normal, normal))
        if norm == 0.0:
            raise ValueError("SurfaceHit.normal must be non-zero")
        normal = tuple(value / norm for value in normal)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "surface_id", str(self.surface_id))
        object.__setattr__(self, "region_from", str(self.region_from))
        object.__setattr__(self, "region_to", str(self.region_to))


@runtime_checkable
class Geometry(Protocol):
    """Minimal interface seen by the continuous-flight transport kernel."""

    def first_hit(self, origin, direction, max_distance,
                  current_region) -> Optional[SurfaceHit]:
        """Return the nearest interface strictly before ``max_distance``."""

    def region_at(self, point) -> str:
        """Return the region containing ``point``."""


@dataclass(frozen=True)
class Plane:
    """An infinite analytic interface separating one solid from vacuum.

    ``outward_normal`` points from the solid into vacuum.  Points exactly on
    the interface belong to the solid, matching the historical SEEMC launch at
    ``z = 0``.  The default is therefore bit-for-bit compatible with a solid
    occupying ``z > 0`` and vacuum occupying ``z < 0``.
    """

    point: Vec3 = (0.0, 0.0, 0.0)
    outward_normal: Vec3 = (0.0, 0.0, -1.0)
    surface_id: str = "sample_plane"
    solid_region: str = SOLID_REGION
    vacuum_region: str = VACUUM_REGION
    direction_epsilon: float = 1e-15

    def __post_init__(self):
        point = _vec3(self.point, "point")
        normal = _vec3(self.outward_normal, "outward_normal")
        norm = math.sqrt(_dot(normal, normal))
        if norm == 0.0:
            raise ValueError("outward_normal must be non-zero")
        normal = tuple(value / norm for value in normal)
        epsilon = float(self.direction_epsilon)
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("direction_epsilon must be finite and non-negative")
        if str(self.solid_region) == str(self.vacuum_region):
            raise ValueError("solid_region and vacuum_region must differ")

        object.__setattr__(self, "point", point)
        object.__setattr__(self, "outward_normal", normal)
        object.__setattr__(self, "surface_id", str(self.surface_id))
        object.__setattr__(self, "solid_region", str(self.solid_region))
        object.__setattr__(self, "vacuum_region", str(self.vacuum_region))
        object.__setattr__(self, "direction_epsilon", epsilon)

    @property
    def is_reference_plane(self) -> bool:
        """Whether this is the historical ``z=0`` SEEMC interface."""
        return (
            self.point == (0.0, 0.0, 0.0)
            and self.outward_normal == (0.0, 0.0, -1.0)
            and self.solid_region == SOLID_REGION
            and self.vacuum_region == VACUUM_REGION
        )

    def signed_distance(self, point) -> float:
        """Signed normal distance; positive values lie in vacuum."""
        point = _vec3(point, "point")
        delta = (
            point[0] - self.point[0],
            point[1] - self.point[1],
            point[2] - self.point[2],
        )
        return _dot(delta, self.outward_normal)

    def region_at(self, point) -> str:
        return self.vacuum_region if self.signed_distance(point) > 0.0 \
            else self.solid_region

    def depth_into_solid(self, point) -> float:
        """Normal depth below the interface, clamped to zero in vacuum."""
        if self.is_reference_plane:
            return max(float(point[2]), 0.0)
        return max(-self.signed_distance(point), 0.0)

    def lateral_distance(self, point, reference) -> float:
        """Distance parallel to the plane from ``reference`` to ``point``."""
        point = _vec3(point, "point")
        reference = _vec3(reference, "reference")
        dx = point[0] - reference[0]
        dy = point[1] - reference[1]
        dz = point[2] - reference[2]
        if self.is_reference_plane:
            return math.hypot(dx, dy)
        normal_component = _dot((dx, dy, dz), self.outward_normal)
        tx = dx - normal_component * self.outward_normal[0]
        ty = dy - normal_component * self.outward_normal[1]
        tz = dz - normal_component * self.outward_normal[2]
        return math.sqrt(max(tx * tx + ty * ty + tz * tz, 0.0))

    def first_hit(self, origin, direction, max_distance,
                  current_region) -> Optional[SurfaceHit]:
        origin = _vec3(origin, "origin")
        direction = _vec3(direction, "direction")
        max_distance = float(max_distance)
        if math.isnan(max_distance) or max_distance < 0.0:
            raise ValueError("max_distance must be non-negative and not NaN")

        current_region = str(current_region)
        denominator = _dot(direction, self.outward_normal)
        if current_region == self.solid_region:
            if denominator <= self.direction_epsilon:
                return None
            crossing_normal = self.outward_normal
            next_region = self.vacuum_region
        elif current_region == self.vacuum_region:
            if denominator >= -self.direction_epsilon:
                return None
            crossing_normal = tuple(-value for value in self.outward_normal)
            next_region = self.solid_region
        else:
            raise ValueError(
                f"Plane does not contain current_region={current_region!r}"
            )

        # Preserve the exact arithmetic of the validated planar kernel.  This
        # is still a geometry-backend query; the special branch only prevents
        # an avoidable last-bit change in the regression reference case.
        if self.is_reference_plane and current_region == self.solid_region:
            distance = -origin[2] / direction[2]
        else:
            distance = -self.signed_distance(origin) / denominator

        # A collision at exactly max_distance retains the historical rule:
        # complete the sampled path and collide rather than process a boundary.
        if distance < 0.0 or not distance < max_distance:
            return None

        if self.is_reference_plane:
            position = (
                origin[0] + direction[0] * distance,
                origin[1] + direction[1] * distance,
                0.0,
            )
        else:
            raw = (
                origin[0] + direction[0] * distance,
                origin[1] + direction[1] * distance,
                origin[2] + direction[2] * distance,
            )
            residual = self.signed_distance(raw)
            position = tuple(
                raw[index] - residual * self.outward_normal[index]
                for index in range(3)
            )

        return SurfaceHit(
            distance=distance,
            position=position,
            normal=crossing_normal,
            surface_id=self.surface_id,
            region_from=current_region,
            region_to=next_region,
            primitive_id=0,
        )


__all__ = [
    "Geometry",
    "Plane",
    "SOLID_REGION",
    "SurfaceHit",
    "VACUUM_REGION",
]
