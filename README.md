# seemc-imaging

`seemc-imaging` is a standalone fork of the validated planar SEEMC transport
kernel. Its first purpose is to retain the raw ancestry and scattering history
needed for population-resolved SEM signals without changing the random draws or
the established scattering physics.

The package currently contains:

- the validated semi-infinite transport model behind a geometry interface;
- an exact analytic `Plane` backend with translated/rotated planes, local
  normals, and solid/vacuum region IDs;
- an analytic `TrapezoidalLine` united with a semi-infinite substrate, with
  exact top, sidewall, and exposed-substrate intersections;
- a nearest-hit `Scene` that suppresses faces buried inside unions of solids;
- fixed laboratory-frame beam directions with local surface refraction;
- original-primary and cascade yield bookkeeping;
- the conventional 50 eV SE/BSE split;
- opt-in collision, ancestry, boundary, and fate history;
- deterministic serial and multiprocessing seeds;
- a deterministic two-dimensional raster driver with Gaussian beam-spot
  sampling in the plane normal to the beam;
- per-pixel count, yield, and standard-error maps for energy, ancestry,
  generation, causal SE1/SE2, and LLE/non-LLE channels;
- the complete per-pixel cross-channel covariance of per-primary counts and
  of the mean yield;
- common-random-number trapezoid parameter sweeps and compressed model
  libraries;
- covariance-aware joint profile fitting, position/gain nuisance parameters,
  and channel-by-channel Fisher-information comparisons;
- actual landing-position, local-incidence, and per-surface landing-fraction
  maps;
- opt-in raster trajectory capture with electron identity, ancestry,
  population/fate, energy, and physical femtosecond flight time;
- a dark-theme trapezoid scan animator with a moving beam, energy- or
  population-colored cascade tails, ballistic vacuum continuation, and an
  accumulating SE/BSE/TEY profile;
- compressed NPZ, wide CSV, and optional figure export;
- regression tests against the pre-fork planar snapshot.

Detector response is deliberately not hard-coded yet. The event history is the
raw evidence from which alternative population definitions can be compared.
Version 0.7.3 separates causal taxonomy from expected spatial resolution and
experimental filtering. Its default `causal_lle_v2` classifier uses the
immediate energetic parent's direction for SE1/SE2 and an explicit vacuum
energy-loss threshold for LLE/non-LLE emitted primaries. It merges the 0.6.2
escape-surface SE reference as an optional, metadata-tracked classifier while
retaining the 0.7.0 launch-surface default. Strict first-event BSE
is an overlapping diagnostic, not a synonym for LLE. The former `branch_v1`
classifier remains available solely to reproduce 0.6.1-era results. Version
0.6.2 is reproduced with `--se-reference escape_surface`.

## Installation

```bash
python -m pip install -e .
```

The material database is not bundled. Point `db_path` to a database produced
with the corrected optlib table conventions.

### Earlier-version migration

Do not discard an earlier raster or model library. Version 0.7.3 recognizes
both historical branches:

- 0.6.1-era `branch_v1` archives retain SE1/SE2/BSE1/BSE2. Reproduce them with
  `--population-definition branch_v1`.
- 0.6.2 archives retain `lle_bse`/`non_lle_bse` and the escape-surface SE
  reference. Reproduce that classifier with `--se-reference escape_surface`.

The fitter resolves `--channels all_disjoint` to the basis actually stored in
the library. The animation `populations` preset also adapts to current, 0.6.2,
or branch-v1 trajectory archives.

The fitter also canonicalizes the 0.7.0 metadata description
`immediate_parent_direction_vs_launch_surface_normal` to `launch_surface`.
Those terms encode the same SE1/SE2 reference and can therefore be compared
without `--allow-incompatible`.

New simulations default to `causal_lle_v2`. A new observation and its model
library must use the same classifier, SE reference, and LLE threshold. The fitter rejects
old/new classifier mixtures instead of silently comparing identically named
SE channels with different operational definitions.

## Record one cascade

```python
import json
import numpy as np

from seemc_imaging import Sample, simulate_trajectory

sample = Sample("Cu", db_path="MaterialDatabase.pkl")
result = simulate_trajectory(
    sample,
    E0=1000.0,
    angle_rad=np.deg2rad(30.0),
    rng=np.random.default_rng(20260811),
    history=True,
    trajectory_id=0,
)

history = result.history
print(history.electrons[0])
print(history.events_for(0)[:3])

with open("trajectory_history.json", "w", encoding="utf-8") as stream:
    json.dump(history.to_dict(), stream, indent=2)
```

The default geometry is the historical plane at `z=0`, with solid at `z>0`.
It can also be supplied explicitly or translated/rotated:

```python
import numpy as np

from seemc_imaging import Plane, Sample, simulate_trajectory

sample = Sample("Cu", db_path="MaterialDatabase.pkl")
surface = Plane(
    point=(100.0, 0.0, 0.0),          # Angstrom
    outward_normal=(1.0, 0.0, 0.0),
    surface_id="sample_plane",
)
result = simulate_trajectory(
    sample,
    E0=1000.0,
    angle_rad=np.deg2rad(30.0),
    azimuth_rad=np.deg2rad(15.0),
    rng=np.random.default_rng(12345),
    geometry=surface,
    history=True,
)
```

Every boundary event and emitted-electron record carries the `surface_id`,
local crossing normal, source/target regions, and backend primitive ID. Emission
`uz` is now the cosine relative to the local outward normal; for the default
plane it is exactly the former `abs(uvw[2])` value.

## Trapezoidal line geometry

The first structured specimen is a single-material line, infinite along `y`,
on a semi-infinite substrate. Vacuum is toward negative `z`, the substrate
surface is at `z=0`, and dimensions are currently specified in Angstrom:

```python
import numpy as np

from seemc_imaging import Sample, TrapezoidalLine, simulate_trajectory

sample = Sample("Cu", db_path="MaterialDatabase.pkl")
line = TrapezoidalLine(
    top_width=500.0,       # 50 nm
    bottom_width=700.0,    # 70 nm
    height=500.0,          # 50 nm
)

# x locates a vertical beam ray on the horizontal plane through the line top.
beam = (0.0, 0.0, 1.0)
launch_hit = line.launch_surface(x=300.0, vacuum_direction=beam)
result = simulate_trajectory(
    sample,
    E0=1000.0,
    angle_rad=0.0,         # ignored when vacuum_direction is supplied
    rng=np.random.default_rng(12345),
    geometry=line,
    launch_position=launch_hit.position,
    vacuum_direction=beam,
    history=True,
)
```

The global beam stays vertical across the scan. On a sidewall, its local angle
is computed from the sidewall normal and the incident direction is refracted
through the surface barrier. Surface IDs distinguish `.top`, `.left`, `.right`,
and `.substrate`. The buried `.base` face is never returned as a physical
crossing.

Run the included noisy one-dimensional scan with:

```bash
python examples/trapezoidal_line_scan.py MaterialDatabase.pkl \
  --material Cu --energy-ev 1000 --pixels 101 --trajectories 100 \
  --top-width-nm 50 --bottom-width-nm 70 --height-nm 50 \
  --field-width-nm 150 --output line_scan.csv
```

The CSV contains the surface profile, local incidence angle, TEY, cascade-origin
SE/BSE yields, the conventional 50 eV split, and the TEY standard error at each
raster position. It is a raw emission-yield profile, not yet a detector-specific
SEM intensity.

## Population-resolved raster

The production raster samples a finite Gaussian beam in the plane normal to
the laboratory beam direction. Each pixel contains independent primary
cascades; pixels can run serially or in parallel with identical seeded results:

```python
import numpy as np

from seemc_imaging import (
    RasterConfig, RasterDriver, Sample, TrapezoidalLine,
)

sample = Sample("Cu", db_path="MaterialDatabase.pkl")
line = TrapezoidalLine(
    top_width=500.0, bottom_width=700.0, height=500.0,
)
config = RasterConfig(
    energy_ev=1000.0,
    x_positions=np.linspace(-750.0, 750.0, 101),
    y_positions=np.linspace(-250.0, 250.0, 21),
    primaries_per_pixel=100,
    beam_fwhm=20.0,       # 2 nm; scalar or (u, v), in Angstrom
    seed=20260811,
)
result = RasterDriver(sample, line, config).run(use_parallel=True)
result.save_npz("trapezoidal_raster.npz")
result.save_csv("trapezoidal_raster.csv")
```

Every population has `count_maps`, `yield_maps`, and `sem_maps`. For example,
`result.yield_maps["se1"]` is the causal incoming-parent SE1 yield image, while
`result.yield_maps["lle_primary"]` is the energy-filterable low-loss original-
primary image and
`result.yield_maps["sey_50ev"]` retains the conventional energy-cut SE image.
The result also reports actual landing coordinates, local incidence, and the
fraction of primaries landing on each named surface.

Run the full command-line example with:

```bash
python examples/trapezoidal_raster.py MaterialDatabase.pkl \
  --material Cu --energy-ev 1000 --nx 101 --ny 21 \
  --primaries-per-pixel 100 --beam-fwhm-nm 2 --parallel \
  --lle-max-loss-ev 50 \
  --se-reference launch_surface \
  --output-prefix trapezoidal_raster
```

This writes a self-describing compressed NPZ and a wide per-pixel CSV. Add
`--plot` after installing `seemc-imaging[plot]` to render six population maps.
See [raster-driver.md](docs/raster-driver.md) for the exact channel definitions,
RNG scheme, uncertainty calculation, and output fields.

## Record and animate a one-row scan

Trajectory storage is off by default. A small movie run needs only a few
primaries at each beam position. This 51-position, three-primary scan transports
and records just 153 independent primaries:

```bash
python examples/trapezoidal_raster.py MaterialDatabase.pkl \
  --material Cu --energy-ev 1000 \
  --top-width-nm 50 --bottom-width-nm 70 --height-nm 50 \
  --field-width-nm 100 --nx 51 --ny 1 \
  --primaries-per-pixel 3 --beam-fwhm-nm 2 \
  --record-trajectories --trajectory-max-points 500 \
  --parallel --output-prefix trapezoid_movie
```

In addition to the ordinary raster NPZ and CSV, this writes
`trapezoid_movie.trajectories.npz`. The trajectory archive is a compressed,
pickle-free ragged array. It preserves the beam pixel and actual landing point,
primary/cascade and parent IDs, generation, operational emitted-population
label, fate, position, energy, and physical free-flight time for every retained
path point.

If a scan uses more primaries for a less noisy profile, record only a small
subset with `--record-primaries-per-pixel N`. `--trajectory-stride N` and
`--trajectory-max-points N` reduce storage while always preserving the first
and final path points.

Render an MP4 with:

```bash
python examples/animate_trapezoidal_scan.py \
  trapezoid_movie.trajectories.npz \
  --output trapezoid_movie.mp4 \
  --fps 30 --frames-per-pixel 16 --pause-frames 4 \
  --color-by energy --vacuum-flight-nm 35 \
  --profile-channels populations
```

Use `--color-by population` for fixed SE1/SE2/fast-cascade/LLE/non-LLE colors,
or choose a `.gif` output when ffmpeg is unavailable. The lower panel defaults
to the four population-resolved yields SE1, SE2, LLE, and non-LLE. Select only
conventional energy-cut signals with `--profile-channels conventional`, restore
the original three curves with `--profile-channels tey_se_bse`, or provide an
explicit comma-separated list such as
`--profile-channels se1,se2,lle_primary,non_lle_primary`. Legacy trajectory
archives automatically use their former SE1/SE2/BSE1/BSE2 preset.
The MP4 writer automatically pads odd image dimensions by one pixel for H.264
compatibility. The renderer preserves
the relative femtosecond timing within each independently simulated primary
cascade. Several primaries at one pixel are intentionally overlaid as a visual
ensemble; they do not share an experimental absolute time origin. The straight
vacuum continuation after emission is added only for visualization and does
not alter transport or yield results. Install the optional renderer with
`python -m pip install -e '.[animation]'`.

## Trapezoid metrology sweep and joint fit

Build a small (3\times3\times3) forward-model library around the nominal
50/70/50 nm line:

```bash
python examples/trapezoidal_parameter_sweep.py MaterialDatabase.pkl \
  --material Cu --energy-ev 1000 \
  --top-widths-nm 48,50,52 \
  --bottom-widths-nm 68,70,72 \
  --heights-nm 45,50,55 \
  --field-width-nm 100 --nx 201 \
  --primaries-per-pixel 1000 --beam-fwhm-nm 2 \
  --lle-max-loss-ev 50 \
  --se-reference launch_surface \
  --parallel --output trapezoid_model_library.npz
```

Every geometry reuses the same beam-spot and transport seed keys. This common
random-number design reduces Monte Carlo noise in finite differences without
changing the marginal distribution of any model.

Fit a raster NPZ and compare the dimensional information in several channel
sets:

```bash
python examples/fit_trapezoidal_profile.py \
  trapezoid_model_library.npz trapezoid_profile_0p5nm_5000.npz \
  --channels all_disjoint \
  --shift-range-nm 2 --shift-step-nm 0.1 \
  --plot --output-prefix trapezoid_fit
```

The default joint basis is `se1`, `se2`, `fast_cascade_ge50`, `lle_primary`,
and `non_lle_primary`. These five populations are mutually exclusive and
partition TEY, so
their full covariance can be used without treating overlapping channels as
independent measurements. The fitter searches the discrete geometry library,
profiles a global yield scale and lateral shift, and can optionally fit one
constant background per channel. The information report estimates local
Cramér--Rao bounds for individual and combined channel sets. See
[signal-taxonomy.md](docs/signal-taxonomy.md) for the definitions and
[metrology.md](docs/metrology.md) for assumptions and interpretation. Fits now
reject material, beam, and classifier mismatches unless explicitly overridden.

## Run a small ensemble

```python
from seemc_imaging import SEEMC

model = SEEMC(
    energy_array=[200.0, 500.0, 1000.0],
    sample_name="Cu",
    angle=0.0,
    n_traj=100,
    db_path="MaterialDatabase.pkl",
    seed=12345,
    history=True,
).run_simulation(use_parallel=False)

first_history = model.histories[0][0]
```

History can be much larger than yield results. Use it for development and
population studies; leave `history=False` for high-statistics production runs
that need only yields or spectra.

## Build incidence-angle plane samplers

Version 0.7.4 includes a resumable workflow that exports the six planar SE/BSE
tables used by the correction-factor model. Its default energy nodes match the
existing JMONSEL sampler library, while its angle grid becomes denser near
grazing incidence:

```bash
python examples/generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu --angles-deg 75 --primaries 20000 \
  --quantiles 513 --workers 12 --seed 20260816 --resume \
  --output sampler_library/Cu_SEEMC
```

The emitted-energy split is SE below 50 eV and BSE at or above 50 eV. Polar
angle is measured from the beam-back direction; downstream azimuth sampling
must still enforce the outward sample half-space. See
[plane-sampler-library.md](docs/plane-sampler-library.md) for the grids,
coordinate equation, full-run command, checkpoints, and validation workflow.

## Testing

```bash
python -m pip install -e '.[dev]'
pytest
```

The test suite uses a small synthetic material database. It verifies that:

- history collection consumes no random numbers and changes no physical result;
- fixed seeds reproduce golden trajectories from the untouched optlib snapshot;
- the `Plane` backend is bitwise identical to the former hard-coded travel,
  barrier, refraction, and reflection equations, including grazing incidence
  and repeated internal reflections;
- translated and rotated planes report exact first hits, local normals,
  regions, depth, and lateral distance;
- trapezoidal top/side/substrate intersections agree with closed-form results
  over six orders of length scale;
- the line/substrate union suppresses its buried base and coincident seam;
- a fixed global beam acquires the correct local sidewall incidence and
  refraction;
- a zero-width beam consumes no spot-sampling random number;
- changing the beam FWHM on a homogeneous plane changes landing coordinates
  but not the assigned collision streams or population counts;
- a Gaussian spot at an edge mixes top and sidewall landings correctly;
- every population partition identity holds per trajectory and per pixel;
- causal SE labels follow the immediate parent's direction without using birth
  location or presumed resolution as membership criteria;
- LLE/non-LLE and strict first-event/later-return BSE each partition emitted
  original primaries, while remaining distinct classifications;
- legacy `branch_v1` outputs remain reproducible;
- serial and spawn-parallel raster maps are exactly identical;
- optional trajectory recording leaves every physical raster result unchanged,
  retains monotone physical time, and is identical in serial and spawn-parallel
  runs;
- trajectory archives round-trip without pickle and the animation renderer
  produces a valid movie;
- covariance diagonals equal the squared channel SEMs and serial/parallel
  covariance matrices are exactly identical;
- a synthetic unknown profile recovers its known trapezoid, lateral shift,
  and gain in the joint fitter;
- ancestry links, collision counts, emission links, and terminal fates are
  internally consistent;
- the ensemble driver retains histories by energy and trajectory ID.

See [history-schema.md](docs/history-schema.md) for the recorded data and
[geometry-roadmap.md](docs/geometry-roadmap.md) for the proposed hybrid geometry
architecture.
