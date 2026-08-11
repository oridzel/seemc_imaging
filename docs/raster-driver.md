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

## Operational `branch_v1` labels

SE1, SE2, BSE1, and BSE2 are produced by a named post-processing definition so
they can be revised without changing or rerunning the transport kernel:

- `se1`: a low-energy emitted cascade electron born before the incident
  primary first turned toward the launch surface;
- `se2`: a low-energy emitted cascade electron born after that turn;
- `bse1`: an emitted incident primary whose first turn toward the surface was
  caused by its first elastic collision;
- `bse2`: every other emitted incident primary, including returns formed by
  multiple elastic scattering.

The surface-return event is used instead of reversal relative to the beam.
Those conditions differ at oblique incidence, and surface return is the one
directly connected to escape and image formation.

These labels obey

\[
\mathrm{SE1}+\mathrm{SE2}=\mathrm{se\_cascade\_lt50},\qquad
\mathrm{BSE1}+\mathrm{BSE2}=\mathrm{primary\_all}.
\]

The raw history and the unambiguous energy/ancestry maps remain the scientific
reference. `branch_v1` is an explicit hypothesis that can be compared with
alternative definitions later.

## Output formats

`RasterResult.save_npz()` writes a compressed, self-describing archive with
the axes, metadata JSON, all maps, landing statistics, surface fractions, and
diagnostics. `RasterResult.save_csv()` writes one wide row per pixel for easy
inspection and dataframe analysis.

The current maps are raw emitted-electron yields. No detector solid angle,
energy response, electrostatic collection, or detector efficiency is applied
yet.
