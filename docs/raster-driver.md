# Raster driver and population channels

## Pixel simulation

For each nominal pixel center \((x_i,y_j)\), the driver transports
`primaries_per_pixel` independent incident primaries and their complete
cascades. The yield in channel \(k\) is

\[
Y_k(x_i,y_j)=\frac{1}{N}\sum_{p=1}^{N}n_{k,p},
\]

where \(n_{k,p}\) is the emitted-electron count in that channel for primary
\(p\). The reported uncertainty is the standard error of the mean calculated
from the distribution of the per-primary counts. This remains valid when a
cascade produces correlated multiple emissions; it does not assume that the
total counts are Poisson.

The output retains three maps for every channel:

- `count`: total emitted-electron count in the pixel;
- `yield`: count per incident primary;
- `sem`: standard error of the yield.

Version 0.5.0 also retains, for every pixel, the full sample covariance of the
per-primary count vector and the covariance of the mean yield:

\[
\widehat{\operatorname{Cov}}(\bar{\mathbf Y})
=\frac{1}{N}\widehat{\operatorname{Cov}}(\mathbf n_p).
\]

Its diagonal is exactly the square of the reported channel SEM. The NPZ stores
the full covariance matrices (17-by-17 for the default classifier) as
`primary_count_covariance` and `yield_covariance`. The CSV includes the upper
triangle for the five-channel disjoint fitting basis.

## Beam-spot sampling

`beam_fwhm` is a scalar or pair in Angstrom. It specifies a Gaussian full width
at half maximum along two orthogonal axes in the plane normal to the fixed
laboratory beam direction. At oblique incidence, a circular beam therefore has
the expected elongated projection onto the global raster plane.

Each primary has two deterministic random streams:

1. beam displacement, seeded by `(seed, pixel_id, trajectory_id, 0)`;
2. transport, seeded by `(seed, pixel_id, trajectory_id, 1)`.

Changing the beam diameter consequently changes the surface location but not
the collision random stream assigned to that primary. Serial and spawn-based
multiprocessing runs are identical for the same configuration.

The result also stores mean and SEM of the actual landing coordinates, local
incidence angle, and the landing count/fraction for every surface ID. Surface
fractions are particularly helpful at an edge, where one nominal pixel's beam
spot can illuminate the top, sidewall, and substrate.

## Population channels

The driver deliberately stores overlapping decompositions rather than forcing
one ambiguous SE/BSE definition:

| Channel | Definition |
| --- | --- |
| `tey` | All emitted electrons |
| `sey_50ev`, `bse_50ev` | Emission energy below / at or above the configured cutoff (50 eV by default) |
| `cascade_all`, `primary_all` | Cascade origin / original incident primary |
| `se_cascade_lt50` | Low-energy cascade emission |
| `fast_cascade_ge50` | Fast cascade or delta electron |
| `slow_primary_lt50` | Strongly slowed emitted incident primary |
| `bse_primary_ge50` | Original-primary BSE also above the energy cutoff |
| `generation_1`, `generation_2plus` | Emitted cascade generation |

The following exact identities are checked for every trajectory:

\[
\mathrm{TEY}=\mathrm{SE}_{<50\,eV}+\mathrm{BSE}_{\geq50\,eV}
=\mathrm{cascade}_{all}+\mathrm{primary}_{all}.
\]

## Default `causal_lle_v2` labels

The default post-processor separates causal SE taxonomy, energy filtering, and
scattering-history diagnostics.

- `se1`: a low-energy emitted cascade electron created while its immediate
  energetic parent was directed into the configured reference surface;
- `se2`: a low-energy emitted cascade electron created while its immediate
  energetic parent was directed toward vacuum through that reference surface;
- `lle_primary`: an emitted original incident electron with
  \(E_0-E_{exit}<\Delta E_c\);
- `non_lle_primary`: an emitted original incident electron with
  \(E_0-E_{exit}\geq\Delta E_c\).

Both \(E_0\) and \(E_{exit}\) use the vacuum kinetic-energy reference. The
default \(\Delta E_c\) is 50 eV and is configurable with
`PopulationClassifier(lle_max_loss_ev=...)` or `--lle-max-loss-ev`. The value is
stored in the NPZ metadata and compatibility-checked during fitting.

No location, spatial resolution, or energy loss is included in SE1/SE2
membership. Those are properties to calculate after classification. The exact
direction test uses the immediate parent's direction immediately before the
creating inelastic collision. By default, the reference normal is the outward
normal of the primary's launch surface. `se_reference="escape_surface"` (CLI:
`--se-reference escape_surface`) instead uses the emitted child's actual exit
surface and reproduces the 0.6.2 classifier. A zero normal component is assigned
to the incoming class so the pair is exhaustive. The reference choice is stored
in metadata and compatibility-checked during fitting.

The five covariance-safe channels are

\[
\mathrm{TEY}=\mathrm{SE1}+\mathrm{SE2}
+\mathrm{fast\_cascade}_{\geq50\,eV}
+\mathrm{LLE}_{primary}+\mathrm{nonLLE}_{primary}.
\]

This is the default basis for joint metrology. LLE is not identified with a
single- or first-event backscatter.

## Scattering-history diagnostic

Two additional, overlapping channels partition emitted original primaries:

- `first_event_bse`: the first completed collision was elastic and that same
  event first turned the incident electron toward the launch surface;
- `later_return_bse`: every other emitted original incident electron.

These channels measure how rare the strict first-event route is. They are not
part of the default joint basis because they overlap the LLE/non-LLE pair.

## Legacy `branch_v1`

`PopulationClassifier(definition="branch_v1")` exactly reproduces the 0.6.1-era
SE1/SE2/BSE1/BSE2 rules and their five-channel covariance basis. New work
should use `causal_lle_v2`; the legacy definition exists so completed model
libraries and long simulations remain reproducible. Version 0.6.2 is reproduced
with `causal_lle_v2` and `se_reference="escape_surface"`.

## Output formats

`RasterResult.save_npz()` writes a compressed, self-describing raster-v3 archive with
the axes, metadata JSON, all maps, landing statistics, surface fractions, and
diagnostics. `RasterResult.save_csv()` writes one wide row per pixel for easy
inspection and dataframe analysis.

The current maps are raw emitted-electron yields. No detector solid angle,
energy response, electrostatic collection, or detector efficiency is applied
yet.

## Optional trajectory archive

Set `RasterConfig(record_trajectories=True)` or pass
`--record-trajectories` to the raster example to retain electron paths.
Ordinary rasters do not allocate or save track arrays. The optional controls
are:

- `record_primaries_per_pixel`: record the first deterministic subset of
  primaries while still using every primary for yield statistics;
- `trajectory_stride`: retain every Nth path point and both endpoints;
- `trajectory_max_points`: cap the retained points per transported electron.

`RasterResult.save_trajectories_npz()` writes the
`seemc-imaging-raster-trajectories-v1` format. Cascades, electrons, and points
are stored as flat arrays connected by two monotone offset arrays, so the NPZ
loads with `allow_pickle=False`. Each point contains global `(x,y,z)`, kinetic
energy, and elapsed femtoseconds. Electron arrays retain ID, parent ID,
generation, birth time/energy, final direction/energy, fate, and the disjoint
emitted-population label when applicable. The archive also embeds the raster
profiles and trapezoid metadata used by the animator.

Recording consumes no random numbers. Free-flight time is calculated from the
sampled path length and relativistic speed, but never feeds back into collision
sampling, scattering, barrier transmission, or geometry. Serial and spawn
parallel trajectory archives are therefore identical for a fixed seed and
configuration.

The animation lower panel can display any one to six stored yield channels.
Its `populations` preset shows SE1, SE2, LLE, and non-LLE; `conventional` shows
only the energy-cut SE and BSE signals; and `tey_se_bse` restores TEY plus
conventional SE/BSE. Version 0.6.2 archives automatically use their
SE1/SE2/LLE-BSE/non-LLE-BSE channels, while branch-v1 archives fall back to
SE1/SE2/BSE1/BSE2. These curves always use every primary simulated at each pixel,
even when only a small subset is retained as trajectories.
