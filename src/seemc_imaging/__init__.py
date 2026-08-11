"""Standalone SEEMC imaging transport and provenance package."""

from .geometry import (
    Geometry,
    Plane,
    Scene,
    SOLID_REGION,
    SurfaceHit,
    TrapezoidalLine,
    TrapezoidalPrism,
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
    refract_incident_direction,
    simulate_trajectory,
)

from .raster import (
    CHANNEL_DEFINITIONS,
    POPULATION_CHANNELS,
    PopulationClassifier,
    RasterConfig,
    RasterDriver,
    RasterResult,
    sample_beam_reference,
)

__all__ = [
    "Electron",
    "ElectronRecord",
    "Emission",
    "Geometry",
    "HistoryEvent",
    "MCConfig",
    "CHANNEL_DEFINITIONS",
    "POPULATION_CHANNELS",
    "Plane",
    "PopulationClassifier",
    "RasterConfig",
    "RasterDriver",
    "RasterResult",
    "Scene",
    "SEEMC",
    "SOLID_REGION",
    "Sample",
    "Secondary",
    "SurfaceHit",
    "TrapezoidalLine",
    "TrapezoidalPrism",
    "TrajectoryHistory",
    "TrajectoryResult",
    "VACUUM_REGION",
    "incident_direction",
    "refract_incident_direction",
    "simulate_trajectory",
    "sample_beam_reference",
]

__version__ = "0.4.0"
