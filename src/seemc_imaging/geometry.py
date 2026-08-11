"""Geometry contracts and analytic surface backends.

Coordinates and distances use the same unit as the transport kernel
(currently Angstrom).  A geometry backend never samples randomness: it only
answers deterministic region and nearest-interface questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, Tuple, runtime_checkable


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


def _add_scaled(origin, direction, distance) -> Vec3:
    return tuple(
        origin[index] + direction[index] * distance for index in range(3)
    )


def _unit(values, name: str) -> Vec3:
    vector = _vec3(values, name)
    norm = math.sqrt(_dot(vector, vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return tuple(value / norm for value in vector)


@dataclass(frozen=True)
class _SurfaceCandidate:
    """A primitive boundary hit with a canonical solid-to-vacuum normal."""

    distance: float
    position: Vec3
    outward_normal: Vec3
    surface_id: str
    primitive_id: Optional[int]


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

    def launch_surface(self, x, y=0.0, vacuum_direction=(0.0, 0.0, 1.0),
                       clearance=1.0):
        """Return the vacuum-to-solid hit for one parallel raster ray.

        ``x`` and ``y`` locate the ray on the global horizontal plane through
        ``self.point``.  This matches :meth:`TrapezoidalLine.launch_surface`
        and lets the same raster driver validate a featureless plane before it
        is used on structured specimens.
        """
        direction = _unit(vacuum_direction, "vacuum_direction")
        if _dot(direction, self.outward_normal) >= -self.direction_epsilon:
            raise ValueError("vacuum_direction must point into the plane")
        clearance = float(clearance)
        if not math.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("clearance must be finite and positive")

        reference = (float(x), float(y), self.point[2])
        denominator = _dot(direction, self.outward_normal)
        along_ray = -self.signed_distance(reference) / denominator
        position = _add_scaled(reference, direction, along_ray)
        residual = self.signed_distance(position)
        position = tuple(
            position[index] - residual * self.outward_normal[index]
            for index in range(3)
        )
        origin = tuple(
            position[index] - clearance * direction[index]
            for index in range(3)
        )
        hit = self.first_hit(
            origin, direction, 2.0 * clearance, self.vacuum_region
        )
        if hit is None:
            raise RuntimeError("beam ray did not intersect the plane")
        return hit

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

    def candidate_hits(self, origin, direction, max_distance):
        """Return the plane crossing without assuming a composite region.

        ``Scene`` uses this lower-level query to suppress boundaries buried
        inside a union of solids.  The normal is always the primitive's
        canonical solid-to-vacuum normal.
        """
        origin = _vec3(origin, "origin")
        direction = _vec3(direction, "direction")
        max_distance = float(max_distance)
        denominator = _dot(direction, self.outward_normal)
        if abs(denominator) <= self.direction_epsilon:
            return []
        distance = -self.signed_distance(origin) / denominator
        if distance < 0.0 or not distance < max_distance:
            return []
        raw = _add_scaled(origin, direction, distance)
        residual = self.signed_distance(raw)
        position = tuple(
            raw[index] - residual * self.outward_normal[index]
            for index in range(3)
        )
        return [_SurfaceCandidate(
            distance=distance,
            position=position,
            outward_normal=self.outward_normal,
            surface_id=self.surface_id,
            primitive_id=0,
        )]

    def surface_candidates_at(self, point, tolerance=1e-9):
        if abs(self.signed_distance(point)) <= tolerance:
            return [_SurfaceCandidate(
                distance=0.0,
                position=_vec3(point, "point"),
                outward_normal=self.outward_normal,
                surface_id=self.surface_id,
                primitive_id=0,
            )]
        return []


@dataclass(frozen=True)
class TrapezoidalPrism:
    """An infinite analytic trapezoidal prism extruded along global ``y``.

    The top is at ``substrate_z - height`` and the buried base is at
    ``substrate_z``.  Width changes linearly from ``top_width`` to
    ``bottom_width``.  The primitive is a closed convex volume; use
    :class:`TrapezoidalLine` for the physically exposed union with a
    semi-infinite substrate.
    """

    top_width: float
    bottom_width: float
    height: float
    center_x: float = 0.0
    substrate_z: float = 0.0
    surface_id: str = "line"
    solid_region: str = SOLID_REGION
    vacuum_region: str = VACUUM_REGION
    direction_epsilon: float = 1e-15
    boundary_tolerance: float = 1e-9
    _faces: tuple = field(init=False, repr=False)

    def __post_init__(self):
        top_width = float(self.top_width)
        bottom_width = float(self.bottom_width)
        height = float(self.height)
        center_x = float(self.center_x)
        substrate_z = float(self.substrate_z)
        epsilon = float(self.direction_epsilon)
        tolerance = float(self.boundary_tolerance)
        values = (top_width, bottom_width, height, center_x, substrate_z,
                  epsilon, tolerance)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("trapezoid parameters must be finite")
        if top_width <= 0.0 or bottom_width <= 0.0 or height <= 0.0:
            raise ValueError("top_width, bottom_width, and height must be positive")
        if bottom_width < top_width:
            raise ValueError(
                "bottom_width must be at least top_width; undercut lines are "
                "not supported by this first analytic backend"
            )
        if epsilon < 0.0 or tolerance <= 0.0:
            raise ValueError("geometry tolerances must be positive")
        if str(self.solid_region) == str(self.vacuum_region):
            raise ValueError("solid_region and vacuum_region must differ")

        object.__setattr__(self, "top_width", top_width)
        object.__setattr__(self, "bottom_width", bottom_width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "substrate_z", substrate_z)
        object.__setattr__(self, "surface_id", str(self.surface_id))
        object.__setattr__(self, "solid_region", str(self.solid_region))
        object.__setattr__(self, "vacuum_region", str(self.vacuum_region))
        object.__setattr__(self, "direction_epsilon", epsilon)
        object.__setattr__(self, "boundary_tolerance", tolerance)

        top_z = substrate_z - height
        half_top = 0.5 * top_width
        slope = 0.5 * (bottom_width - top_width) / height
        faces = (
            # (point on face, canonical outward normal, label, local face id)
            ((center_x, 0.0, top_z), (0.0, 0.0, -1.0), "top", 0),
            ((center_x, 0.0, substrate_z), (0.0, 0.0, 1.0), "base", 1),
            ((center_x + half_top, 0.0, top_z),
             _unit((1.0, 0.0, -slope), "right normal"), "right", 2),
            ((center_x - half_top, 0.0, top_z),
             _unit((-1.0, 0.0, -slope), "left normal"), "left", 3),
        )
        object.__setattr__(self, "_faces", faces)

    @property
    def top_z(self):
        return self.substrate_z - self.height

    @property
    def half_top_width(self):
        return 0.5 * self.top_width

    @property
    def half_bottom_width(self):
        return 0.5 * self.bottom_width

    def _face_value(self, point, face):
        point_on_face, normal, _, _ = face
        return _dot(tuple(point[i] - point_on_face[i] for i in range(3)), normal)

    def region_at(self, point) -> str:
        point = _vec3(point, "point")
        tolerance = self.boundary_tolerance
        inside = all(
            self._face_value(point, face) <= tolerance for face in self._faces
        )
        return self.solid_region if inside else self.vacuum_region

    def candidate_hits(self, origin, direction, max_distance):
        origin = _vec3(origin, "origin")
        direction = _vec3(direction, "direction")
        max_distance = float(max_distance)
        if math.isnan(max_distance) or max_distance < 0.0:
            raise ValueError("max_distance must be non-negative and not NaN")

        hits = []
        tolerance = self.boundary_tolerance
        for face in self._faces:
            point_on_face, normal, label, face_id = face
            denominator = _dot(direction, normal)
            if abs(denominator) <= self.direction_epsilon:
                continue
            numerator = _dot(
                tuple(point_on_face[i] - origin[i] for i in range(3)), normal
            )
            distance = numerator / denominator
            if distance < 0.0 or not distance < max_distance:
                continue
            raw = _add_scaled(origin, direction, distance)
            residual = self._face_value(raw, face)
            position = tuple(
                raw[index] - residual * normal[index] for index in range(3)
            )
            if all(
                self._face_value(position, other) <= tolerance
                for other in self._faces
            ):
                hits.append(_SurfaceCandidate(
                    distance=distance,
                    position=position,
                    outward_normal=normal,
                    surface_id=f"{self.surface_id}.{label}",
                    primitive_id=face_id,
                ))
        return hits

    def surface_candidates_at(self, point, tolerance=None):
        point = _vec3(point, "point")
        tolerance = self.boundary_tolerance if tolerance is None else float(tolerance)
        if self.region_at(point) != self.solid_region:
            # A caller may be a few ulps outside an exact face.  Check all
            # half-spaces with the caller's tolerance before rejecting it.
            if any(self._face_value(point, face) > tolerance for face in self._faces):
                return []
        hits = []
        for face in self._faces:
            if abs(self._face_value(point, face)) <= tolerance:
                _, normal, label, face_id = face
                hits.append(_SurfaceCandidate(
                    distance=0.0,
                    position=point,
                    outward_normal=normal,
                    surface_id=f"{self.surface_id}.{label}",
                    primitive_id=face_id,
                ))
        return hits

    def first_hit(self, origin, direction, max_distance,
                  current_region) -> Optional[SurfaceHit]:
        """Standalone convex-prism crossing, mainly useful for unit tests."""
        current_region = str(current_region)
        if current_region not in (self.solid_region, self.vacuum_region):
            raise ValueError(
                f"TrapezoidalPrism does not contain current_region={current_region!r}"
            )
        direction = _vec3(direction, "direction")
        candidates = sorted(
            self.candidate_hits(origin, direction, max_distance),
            key=lambda hit: hit.distance,
        )
        for candidate in candidates:
            projection = _dot(direction, candidate.outward_normal)
            if current_region == self.solid_region and projection > self.direction_epsilon:
                normal = candidate.outward_normal
                next_region = self.vacuum_region
            elif (current_region == self.vacuum_region
                  and projection < -self.direction_epsilon):
                normal = tuple(-value for value in candidate.outward_normal)
                next_region = self.solid_region
            else:
                continue
            return SurfaceHit(
                candidate.distance, candidate.position, normal,
                candidate.surface_id, current_region, next_region,
                candidate.primitive_id,
            )
        return None


@dataclass(frozen=True)
class Scene:
    """Nearest-hit union of analytic solid primitives.

    Candidate interfaces are filtered against the union before being returned.
    Consequently an overlapping or coincident face buried inside another solid
    is not a transport boundary.
    """

    primitives: tuple
    solid_region: str = SOLID_REGION
    vacuum_region: str = VACUUM_REGION
    position_epsilon: float = 1e-9
    direction_epsilon: float = 1e-15

    def __init__(self, primitives: Iterable, solid_region=SOLID_REGION,
                 vacuum_region=VACUUM_REGION, position_epsilon=1e-9,
                 direction_epsilon=1e-15):
        primitives = tuple(primitives)
        if not primitives:
            raise ValueError("Scene requires at least one primitive")
        for primitive in primitives:
            if not hasattr(primitive, "candidate_hits"):
                raise TypeError(
                    "Scene primitives must provide candidate_hits()"
                )
            if not hasattr(primitive, "region_at"):
                raise TypeError("Scene primitives must provide region_at()")
        if str(solid_region) == str(vacuum_region):
            raise ValueError("solid_region and vacuum_region must differ")
        if position_epsilon <= 0.0 or direction_epsilon < 0.0:
            raise ValueError("geometry tolerances must be positive")
        object.__setattr__(self, "primitives", primitives)
        object.__setattr__(self, "solid_region", str(solid_region))
        object.__setattr__(self, "vacuum_region", str(vacuum_region))
        object.__setattr__(self, "position_epsilon", float(position_epsilon))
        object.__setattr__(self, "direction_epsilon", float(direction_epsilon))

    def _primitive_is_solid(self, primitive, point):
        primitive_solid = getattr(primitive, "solid_region", self.solid_region)
        return primitive.region_at(point) == primitive_solid

    def region_at(self, point) -> str:
        point = _vec3(point, "point")
        return self.solid_region if any(
            self._primitive_is_solid(primitive, point)
            for primitive in self.primitives
        ) else self.vacuum_region

    def _scale_tolerance(self, point):
        scale = max(1.0, *(abs(value) for value in point))
        return max(self.position_epsilon, 64.0 * math.ulp(scale))

    def _sampling_offset(self, point):
        # Primitive ``region_at`` methods deliberately classify points within
        # their boundary tolerance as solid.  Sample far enough to leave that
        # closed boundary band when deciding whether a face is exposed.
        return 4.0 * self._scale_tolerance(point)

    def _candidate_groups(self, origin, direction, max_distance):
        candidates = []
        for object_id, primitive in enumerate(self.primitives):
            for candidate in primitive.candidate_hits(
                    origin, direction, max_distance):
                candidates.append((candidate.distance, object_id, candidate))
        candidates.sort(key=lambda item: (item[0], item[1],
                                          -1 if item[2].primitive_id is None
                                          else item[2].primitive_id))
        groups = []
        for item in candidates:
            tolerance = self._scale_tolerance(item[2].position)
            if groups and abs(item[0] - groups[-1][0][0]) <= tolerance:
                groups[-1].append(item)
            else:
                groups.append([item])
        return groups

    def first_hit(self, origin, direction, max_distance,
                  current_region) -> Optional[SurfaceHit]:
        origin = _vec3(origin, "origin")
        direction = _vec3(direction, "direction")
        max_distance = float(max_distance)
        if math.isnan(max_distance) or max_distance < 0.0:
            raise ValueError("max_distance must be non-negative and not NaN")
        current_region = str(current_region)
        if current_region not in (self.solid_region, self.vacuum_region):
            raise ValueError(f"Scene does not contain current_region={current_region!r}")

        groups = self._candidate_groups(origin, direction, max_distance)
        for index, group in enumerate(groups):
            distance = group[0][0]
            position = group[0][2].position
            delta = self._sampling_offset(position)

            # ``region_at`` deliberately treats a tolerance-wide band around
            # each primitive face as solid.  At a shallow ray/face angle, a
            # fixed displacement along the ray can remain inside that band on
            # both sides and make a real crossing look buried.  Scale the
            # probe distance by the strongest crossing projection in this
            # coincident group so the before/after samples leave the band.
            crossing_projections = []
            for _, _, candidate in group:
                projection = _dot(direction, candidate.outward_normal)
                if current_region == self.solid_region:
                    valid = projection > self.direction_epsilon
                else:
                    valid = projection < -self.direction_epsilon
                if valid:
                    crossing_projections.append(abs(projection))
            if crossing_projections:
                delta = max(
                    delta,
                    self._sampling_offset(position) / max(crossing_projections),
                )
            if distance > 0.0:
                delta = min(delta, 0.25 * distance)
            if index + 1 < len(groups):
                gap = groups[index + 1][0][0] - distance
                if gap > 0.0:
                    delta = min(delta, 0.25 * gap)
            delta = max(delta, 64.0 * math.ulp(max(1.0, abs(distance))))

            before = self.region_at(_add_scaled(origin, direction, distance - delta))
            after = self.region_at(_add_scaled(origin, direction, distance + delta))
            if before != current_region or after == current_region:
                continue

            matching = []
            for _, object_id, candidate in group:
                projection = _dot(direction, candidate.outward_normal)
                if current_region == self.solid_region:
                    valid = projection > self.direction_epsilon
                else:
                    valid = projection < -self.direction_epsilon
                if valid:
                    matching.append((abs(projection), -object_id, candidate))
            if not matching:
                continue
            candidate = max(matching, key=lambda item: (item[0], item[1]))[2]
            if current_region == self.solid_region:
                normal = candidate.outward_normal
                next_region = self.vacuum_region
            else:
                normal = tuple(-value for value in candidate.outward_normal)
                next_region = self.solid_region
            return SurfaceHit(
                distance=distance,
                position=candidate.position,
                normal=normal,
                surface_id=candidate.surface_id,
                region_from=current_region,
                region_to=next_region,
                primitive_id=candidate.primitive_id,
            )
        return None

    def surface_candidate_at(self, point, incoming_direction=None):
        point = _vec3(point, "point")
        tolerance = self._scale_tolerance(point)
        offset = self._sampling_offset(point)
        candidates = []
        for object_id, primitive in enumerate(self.primitives):
            if not hasattr(primitive, "surface_candidates_at"):
                continue
            for candidate in primitive.surface_candidates_at(point, tolerance):
                outward = candidate.outward_normal
                inside_point = tuple(
                    point[i] - offset * outward[i] for i in range(3)
                )
                outside_point = tuple(
                    point[i] + offset * outward[i] for i in range(3)
                )
                if (self.region_at(inside_point) == self.solid_region
                        and self.region_at(outside_point) == self.vacuum_region):
                    candidates.append((object_id, candidate))
        if not candidates:
            raise ValueError("point is not on an exposed Scene surface")

        if incoming_direction is not None:
            incoming = _unit(incoming_direction, "incoming_direction")
            illuminated = [
                (-_dot(incoming, candidate.outward_normal), -object_id, candidate)
                for object_id, candidate in candidates
                if _dot(incoming, candidate.outward_normal) < -self.direction_epsilon
            ]
            if illuminated:
                return max(illuminated, key=lambda item: (item[0], item[1]))[2]

        # Deterministic edge convention: prefer the most upward-facing surface.
        return min(
            candidates,
            key=lambda item: (item[1].outward_normal[2], item[0],
                              -1 if item[1].primitive_id is None
                              else item[1].primitive_id),
        )[1]

    def surface_normal_at(self, point, incoming_direction=None):
        return self.surface_candidate_at(point, incoming_direction).outward_normal


@dataclass(frozen=True)
class TrapezoidalLine:
    """A raised trapezoidal line united with a semi-infinite substrate.

    Coordinates use the historical SEEMC convention: vacuum is toward
    negative ``z`` and the substrate occupies ``z >= substrate_z``.  The line
    is infinite along ``y``.  All dimensions are in the transport length unit
    (currently Angstrom).
    """

    top_width: float
    bottom_width: float
    height: float
    center_x: float = 0.0
    substrate_z: float = 0.0
    surface_id: str = "trapezoidal_line"
    solid_region: str = SOLID_REGION
    vacuum_region: str = VACUUM_REGION
    position_epsilon: float = 1e-9
    prism: TrapezoidalPrism = field(init=False, repr=False)
    substrate: Plane = field(init=False, repr=False)
    scene: Scene = field(init=False, repr=False)

    def __post_init__(self):
        prism = TrapezoidalPrism(
            top_width=self.top_width,
            bottom_width=self.bottom_width,
            height=self.height,
            center_x=self.center_x,
            substrate_z=self.substrate_z,
            surface_id=self.surface_id,
            solid_region=self.solid_region,
            vacuum_region=self.vacuum_region,
            boundary_tolerance=self.position_epsilon,
        )
        substrate = Plane(
            point=(0.0, 0.0, float(self.substrate_z)),
            outward_normal=(0.0, 0.0, -1.0),
            surface_id=f"{self.surface_id}.substrate",
            solid_region=self.solid_region,
            vacuum_region=self.vacuum_region,
        )
        scene = Scene(
            (substrate, prism),
            solid_region=self.solid_region,
            vacuum_region=self.vacuum_region,
            position_epsilon=self.position_epsilon,
        )
        object.__setattr__(self, "top_width", prism.top_width)
        object.__setattr__(self, "bottom_width", prism.bottom_width)
        object.__setattr__(self, "height", prism.height)
        object.__setattr__(self, "center_x", prism.center_x)
        object.__setattr__(self, "substrate_z", prism.substrate_z)
        object.__setattr__(self, "surface_id", prism.surface_id)
        object.__setattr__(self, "solid_region", prism.solid_region)
        object.__setattr__(self, "vacuum_region", prism.vacuum_region)
        object.__setattr__(self, "position_epsilon", float(self.position_epsilon))
        object.__setattr__(self, "prism", prism)
        object.__setattr__(self, "substrate", substrate)
        object.__setattr__(self, "scene", scene)

    @property
    def top_z(self):
        return self.prism.top_z

    @property
    def point(self):
        """Default launch point: the centre of the line top."""
        return (self.center_x, 0.0, self.top_z)

    @property
    def outward_normal(self):
        """Default launch normal at the centre of the line top."""
        return (0.0, 0.0, -1.0)

    def region_at(self, point):
        return self.scene.region_at(point)

    def first_hit(self, origin, direction, max_distance, current_region):
        return self.scene.first_hit(origin, direction, max_distance, current_region)

    def surface_normal_at(self, point, incoming_direction=None):
        return self.scene.surface_normal_at(point, incoming_direction)

    def launch_surface(self, x, y=0.0, vacuum_direction=(0.0, 0.0, 1.0),
                       clearance=None):
        """Intersect a beam ray with the exposed line/substrate surface.

        ``x`` and ``y`` locate the ray on the horizontal plane through the line
        top.  ``vacuum_direction`` points from the electron source toward the
        specimen.  The returned hit is oriented vacuum-to-solid, so its
        ``normal`` points into the solid; negate it to obtain the local outward
        normal.
        """
        direction = _unit(vacuum_direction, "vacuum_direction")
        if direction[2] <= self.scene.direction_epsilon:
            raise ValueError("vacuum_direction must point toward increasing z")
        if clearance is None:
            clearance = max(
                4.0 * self.height, self.bottom_width, self.top_width, 1.0
            ) / direction[2]
        clearance = float(clearance)
        if not math.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("clearance must be finite and positive")
        reference = (float(x), float(y), self.top_z)
        origin = tuple(
            reference[index] - clearance * direction[index]
            for index in range(3)
        )
        if self.region_at(origin) != self.vacuum_region:
            raise RuntimeError("computed beam origin is not in vacuum")
        max_distance = clearance + (self.height + self.bottom_width + 1.0) \
            / direction[2]
        hit = self.first_hit(origin, direction, max_distance, self.vacuum_region)
        if hit is None:
            raise RuntimeError("beam ray did not intersect the specimen")
        return hit

    def surface_point(self, x, y=0.0, vacuum_direction=(0.0, 0.0, 1.0)):
        return self.launch_surface(x, y, vacuum_direction).position

    def depth_into_solid(self, point):
        """Shortest cross-sectional distance to the exposed boundary."""
        point = _vec3(point, "point")
        if self.region_at(point) != self.solid_region:
            return 0.0
        x = point[0] - self.center_x
        z = point[2] - self.substrate_z
        a = 0.5 * self.top_width
        b = 0.5 * self.bottom_width
        h = self.height

        def segment_distance(px, pz, ax, az, bx, bz):
            vx, vz = bx - ax, bz - az
            wx, wz = px - ax, pz - az
            vv = vx * vx + vz * vz
            t = 0.0 if vv == 0.0 else max(0.0, min(1.0, (wx * vx + wz * vz) / vv))
            dx = px - (ax + t * vx)
            dz = pz - (az + t * vz)
            return math.hypot(dx, dz)

        distances = [
            segment_distance(x, z, -a, -h, a, -h),
            segment_distance(x, z, -b, 0.0, -a, -h),
            segment_distance(x, z, a, -h, b, 0.0),
        ]
        # Exposed substrate consists of two horizontal rays outside the base.
        if x <= -b:
            distances.append(abs(z))
        else:
            distances.append(math.hypot(x + b, z))
        if x >= b:
            distances.append(abs(z))
        else:
            distances.append(math.hypot(x - b, z))
        return min(distances)

    def lateral_distance(self, point, reference):
        point = _vec3(point, "point")
        reference = _vec3(reference, "reference")
        return math.hypot(point[0] - reference[0], point[1] - reference[1])


__all__ = [
    "Geometry",
    "Plane",
    "Scene",
    "SOLID_REGION",
    "SurfaceHit",
    "TrapezoidalLine",
    "TrapezoidalPrism",
    "VACUUM_REGION",
]
