# seemc-imaging

`seemc-imaging` is a standalone fork of the validated planar SEEMC transport
kernel. Its first purpose is to retain the raw ancestry and scattering history
needed for population-resolved SEM signals without changing the random draws or
the established scattering physics.

The package currently contains:

- the validated semi-infinite transport model behind a geometry interface;
- an exact analytic `Plane` backend with translated/rotated planes, local
  normals, and solid/vacuum region IDs;
- original-primary and cascade yield bookkeeping;
- the conventional 50 eV SE/BSE split;
- opt-in collision, ancestry, boundary, and fate history;
- deterministic serial and multiprocessing seeds;
- regression tests against the pre-fork planar snapshot.

Structured surface geometry, rastering, detector response, and
SE1/SE2/BSE1/BSE2 classifiers are deliberately not hard-coded yet. The event
history is the raw evidence from which alternative population definitions can
be compared.

## Installation

```bash
python -m pip install -e .
```

The material database is not bundled. Point `db_path` to a database produced
with the corrected optlib table conventions.

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
- ancestry links, collision counts, emission links, and terminal fates are
  internally consistent;
- the ensemble driver retains histories by energy and trajectory ID.

See [history-schema.md](docs/history-schema.md) for the recorded data and
[geometry-roadmap.md](docs/geometry-roadmap.md) for the proposed hybrid geometry
architecture.
