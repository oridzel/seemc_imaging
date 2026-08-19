# Planar sampler library workflow

`examples/generate_plane_sampler_grid.py` builds incidence-angle-specific
SE/BSE tables for the plane sampler. It runs one independent SEEMC ensemble
per angle and primary energy, writes a restartable raw checkpoint, and then
exports the same six CSV roles used by the existing JMONSEL library.

## Recommended grids

The default primary-energy nodes exactly match the supplied JMONSEL sampler
library:

```text
75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500,
800, 1000, 1500, 2000, 3500, 5000, 8000, 10000 eV
```

The default incidence-angle grid is deliberately denser near grazing
incidence, where holder interception and therefore the correction factor
changes most rapidly:

```text
0, 15, 30, 45, 60, 65, 70, 75, 80, 83, 85, 87, 89 degrees
```

Use the exact energy grid even if the correction-factor scan later uses other
beam energies. The sampler can interpolate in energy while remaining
compatible with John's table layout.

## Pilot at 75 degrees

Install the source package and first run the one-angle pilot:

```bash
python -m pip install -e .

python examples/generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --energies-ev 75,100,125,150,175,200,250,300,350,400,500,800,1000,1500,2000,3500,5000,8000,10000 \
  --angles-deg 75 \
  --primaries 20000 \
  --quantiles 513 \
  --workers 12 \
  --seed 20260816 \
  --resume \
  --output sampler_library/Cu_SEEMC
```

Every completed angle-energy case is immediately saved under
`alpha_75deg/raw/`. If the run is interrupted, repeat the same command with
`--resume`; a checkpoint is reused only when its angle, energy, primary count,
seed, material, full `MCConfig`, and material-database SHA-256 all match.

After the 75-degree tables have passed the existing correction-factor test,
remove `--angles-deg 75` to run the complete default angle grid. You may also
stage the expensive calculation, for example:

```bash
# Coarse angles first
python examples/generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --angles-deg 0,30,45,60,70,75,80,85,89 \
  --primaries 20000 --workers 12 --resume \
  --output sampler_library/Cu_SEEMC

# Fill the refinement angles into the same library
python examples/generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --angles-deg 15,65,83,87 \
  --primaries 20000 --workers 12 --resume \
  --output sampler_library/Cu_SEEMC
```

Each invocation rewrites the top-level manifest for the angles selected in
that invocation. Preserve or merge manifests if the grid is generated in
separate batches; the angle directories and raw checkpoints themselves are
not removed.

## Definitions and coordinate convention

For compatibility with the existing six tables, the populations use the
emitted vacuum-energy cut:

- SE: `emission_energy < 50 eV`
- BSE: `emission_energy >= 50 eV`

This is intentionally not the causal cascade/primary classification that
SEEMC also reports.

The tabulated polar angle `theta` is measured from the beam-back axis, i.e.
opposite the incident vacuum direction. It is not measured from the tilted
sample normal. At incidence angle `alpha`, the maximum outward polar angle is
`90 + alpha` degrees, so the 75-degree table has support from 0 to 165 degrees.

Polar angle alone does not determine whether a direction is outside the
sample. Azimuth must be conditioned on the outward-half-space constraint

```text
dot(emission_direction, sample_outward_normal) > 0.
```

For the coordinate basis used by the exporter, let `b` be the beam-back axis,
`e1` lie in the incidence plane toward the outward-normal side, and `e2` be
perpendicular to that plane. With

```text
d = cos(theta) b + sin(theta) [cos(phi) e1 + sin(phi) e2],
```

the allowed azimuths obey

```text
cos(theta) cos(alpha) + sin(theta) sin(alpha) cos(phi) > 0.
```

Sampling uniformly only over this allowed interval is preferable to clamping
an invalid direction to the boundary. Rejection from a uniform azimuth is
also correct for that conditional-uniform rule. The exported `readme.txt` in
every angle directory repeats the convention so it travels with the tables.

## Output and quality checks

Each angle directory contains:

- `SEYFromPlane_SEVaccum_t0nmCuFPA.csv`
- `BSEYFromPlane_SEVaccum_t0nmCuFPA.csv`
- `SEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv`
- `BSEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv`
- `SEThetaFromPlaneSampler_uncoatedCuFPA.csv`
- `BSEThetaFromPlaneSampler_uncoatedCuFPA.csv`
- `readme.txt`
- one compressed raw `.npz` checkpoint per primary energy

The inverse CDFs use one common 513-point cosine-clustered probability grid,
which resolves both tails and makes interpolation across energy and angle
well-defined. The first and last entries explicitly anchor the physical
support: SE energy 0–50 eV, BSE energy 50–beam energy, and polar angle
0–`90 + alpha` degrees.

At the library root:

- `sampler_manifest.csv` gives emission counts, yields, and per-primary
  standard errors for every completed case.
- `sampler_generation.json` records grids, seeds, angle convention,
  classification, material-database hash, and every transport option.

Before using a new angle in the correction-factor calculation, check:

1. `SEY + BSEY = TEY` and the manifest uncertainty is acceptable.
2. Every raw emitted direction has positive dot product with the outward
   sample normal (the exporter enforces this).
3. The empirical beam-relative angle histogram reconstructed from the raw
   checkpoint agrees with samples drawn from the exported inverse CDF.
4. The downstream azimuth sampler produces no inward directions.
5. The transport-model settings match the settings John used for the reference
   tables. SEEMC and JMONSEL will not agree solely because their grids match.

The CLI exposes the low-energy elastic model and surface-barrier model because
they can materially change low-energy yield. Do not relabel a default SEEMC
run as JMONSEL-equivalent without matching those physics choices; use
`sampler_generation.json` as the comparison record.

