# Provenance and event history

The history layer stores observations, not population labels. In particular,
SE1, SE2, LLE/non-LLE, and first-event BSE are not assigned during transport. This keeps the
validated physics kernel independent of definitions that we may want to refine
or compare later.

## Electron identity

Every requested electron receives a deterministic `electron_id`, including a
secondary that is created but intentionally not transported because it is below
the barrier or exceeds a configured cascade limit.

Each `ElectronRecord` stores:

- `electron_id`, `parent_id`, and `root_primary_id`;
- generation and original-primary status;
- complete birth position, energy, and direction;
- the launch-surface outward normal used as the branch reference;
- physical birth and final time in femtoseconds when trajectory timing is
  requested;
- parent energy and direction immediately before and after the birth collision;
- sampled table channel and physical creation mechanism;
- elastic/inelastic event counts and surface encounters;
- first reversal relative to the incident beam direction;
- first turn toward the launch surface;
- maximum normal depth below the launch plane, maximum in-plane lateral
  distance, and total path length;
- final position, direction, energy, and fate.

The two reversal measures are intentionally separate. A direction can enter the
backward beam hemisphere before it actually points toward the emitting surface,
especially at oblique incidence.

## Event types

| `kind` | Meaning |
| --- | --- |
| `primary_launch` | Primary just inside the surface after refraction |
| `secondary_birth` | Cascade electron created in an inelastic event |
| `elastic` | Completed elastic collision |
| `inelastic` | Completed energy-loss event, with channel and mechanism |
| `surface_reflection` | Barrier encounter followed by internal reflection |
| `emission` | Successful solid-to-vacuum crossing |
| `termination` | Absorption, configured limit, invalid state, or no rate |

Each collision stores the position, free-flight length, energy and direction
before and after, elapsed femtoseconds, polar scattering angle, and azimuth.
Inelastic events also
store energy loss, momentum transfer, sampled channel, physical mechanism, and
the created child ID when applicable.

Every `surface_reflection` and `emission` event additionally stores:

- `surface_id`;
- crossing-oriented local `surface_normal`;
- `region_from` and attempted `region_to`;
- the backend `primitive_id`.

The same fields are attached to each `Emission`. Its `uz` value is the cosine
with the local outward normal, while full global `xyz` and `uvw` remain
available. For an internal reflection, `region_to` describes the other side of
the encountered interface even though the electron remains in `region_from`.

Creation mechanisms currently include:

- `binary`;
- `plasmon`;
- `binary_pauli_fallback`;
- `binary_dropped` when the configured model discards a Pauli-blocked request.

## Information available to later classifiers

A post-processing classifier can determine, without rerunning transport:

- whether a secondary's parent was the original primary or a cascade electron;
- whether that parent was still on its incident branch or had already reversed;
- which collision created the secondary and by which mechanism;
- whether an emitted primary returned on its first completed collision or a
  later scattering history;
- emission energy, full direction, birth depth, maximum depth, lateral spread,
  and path length;
- vacuum energy loss of the emitted original primary;
- alternative energy-cut definitions alongside ancestry definitions.

That is sufficient to compare causal SE1/SE2, LLE/non-LLE, strict first-event
BSE, scattering-order, and legacy definitions while
preserving the raw data used by every definition.
