"""Standalone SEEMC imaging transport and provenance package."""

from .geometry import (
    Geometry,
    Plane,
    SOLID_REGION,
    SurfaceHit,
    VACUUM_REGION,
)

from .transport import (
    Electron,
    ElectronRecord,
    Emission,
    HistoryEvent,
    MCConfig,
    SEEMC,
    Sample,
    Secondary,
    TrajectoryHistory,
    TrajectoryResult,
    incident_direction,
    simulate_trajectory,
)

__all__ = [
    "Electron",
    "ElectronRecord",
    "Emission",
    "Geometry",
    "HistoryEvent",
    "MCConfig",
    "Plane",
    "SEEMC",
    "SOLID_REGION",
    "Sample",
    "Secondary",
    "SurfaceHit",
    "TrajectoryHistory",
    "TrajectoryResult",
    "VACUUM_REGION",
    "incident_direction",
    "simulate_trajectory",
]

__version__ = "0.2.0"
