# Covariance-aware trapezoid metrology

## Model-library construction

`TrapezoidSweepDriver` evaluates every valid Cartesian combination of top
width, bottom width, and height. Undercut combinations with bottom width less
than top width are excluded because the current analytic geometry does not
support them.

All models use the same raster seed and therefore the same deterministic
`(pixel_id, trajectory_id, stream_id)` seed keys. This common-random-number
design is intentional: adjacent geometries see corresponding beam-spot and
transport variates, reducing Monte Carlo noise in model differences and finite
derivatives. It does not force trajectories to remain identical after their
geometry-dependent states diverge.

The model-library NPZ contains:

- `parameters_angstrom`: top width, bottom width, and height for every model;
- `x_angstrom`: the common line-scan grid;
- `channels`: population order;
- `yields`: array with axes `(model, channel, x)`;
- `covariance_of_mean`: array with axes `(model, x, channel, channel)`;
- `completed_primaries`: per-model, per-pixel primary counts;
- JSON metadata recording the raster and sweep configuration.

## Joint profile fit

`ProfileFitter` compares an observed one-dimensional raster with every model.
For lateral shift (s), scale (a), and optional channel offsets
\(\mathbf b\), the residual at pixel (i) is

\[
\mathbf r_i=\mathbf y_i-
\left[a\,\mathbf m(x_i-s;\mathbf p)+\mathbf b\right].
\]

The score is

\[
\chi^2(\mathbf p,s)=
\sum_i \mathbf r_i^T
\left(\Sigma_{y,i}+a^2\Sigma_{m,i}\right)^{-1}
\mathbf r_i.
\]

The scale and offsets are solved by generalized least squares for each
geometry/shift candidate. The geometry result is currently the best discrete
grid point; the package does not claim sub-grid parameter interpolation yet.

Raster-v2 and raster-v3 observations use the exact cross-channel covariance.
Older v1 NPZ files remain readable, but only their diagonal SEM variances are
available, so the fit cannot recover correlations that were never saved.

Before scoring, the fitter compares material, landing energy, beam FWHM,
classifier definition, SE reference surface, SE/BSE cutoff, and LLE threshold
whenever those fields are present. Incompatible files are rejected by default.
Historical descriptive SE-reference strings are canonicalized before this
check, so the 0.7.0
`immediate_parent_direction_vs_launch_surface_normal` value is equivalent to
the current `launch_surface` identifier.
The override is meant for intentional cross-condition diagnostics, not ordinary
metrology.

## Channel basis

The default joint basis is:

1. `se1_lt50`
2. `se1_ge50`
3. `se2_lt50`
4. `se2_ge50`
5. `lle_primary`
6. `non_lle_primary`

These categories are mutually exclusive and partition TEY. The signal
concepts are explicitly operational: SE1/SE2 use a parent direction at
creation, selected by `se_parent_rule`, while LLE/non-LLE use a configured
vacuum energy-loss threshold, absolute or fractional. The fit does not assume
that any class is intrinsically localized or high resolution.

The information report also compares individual channels, the sub-cutoff
SE1/SE2 pair, the full causal SE quartet, the LLE/non-LLE pair, the
conventional energy-cut SE/BSE pair, and all six disjoint populations.

0.7.x `causal_lle_v2` libraries retain their energy-gated five-channel basis,
legacy 0.6.1-era model libraries their `branch_v1` five-channel basis, and
0.6.2 libraries their `lle_bse`/`non_lle_bse` basis. All can still be
fitted. The CLI resolves `--channels all_disjoint` to the basis actually
stored in the library.

## Local information estimate

At a selected library point, adjacent grid models provide finite-difference
derivatives \(\partial\mathbf m/\partial\mathbf p\). For a channel set, the
Fisher matrix is

\[
F=\sum_i J_i^T\Sigma_i^{-1}J_i.
\]

Gain and optional background derivatives are included as nuisance columns, so
the reported geometry covariance is marginalized over them. The square roots
of its diagonal are local Cramér--Rao bounds, not a complete measurement
uncertainty budget. They depend on the simulated material, beam energy, spot,
primary count, grid spacing, detector assumptions, and accuracy of the
`causal_lle_v3` population model.

At least two grid values are required along each parameter at the reference
geometry. Three values are preferable because they provide central finite
differences. A first coarse library should identify the useful region; a
second denser library can then refine it.

A refinement library may intentionally hold one or two geometry parameters
fixed. In that case the information report estimates derivatives and standard
errors only for parameters that vary. Fixed parameters are recorded as not
estimated rather than being assigned a zero or causing the report to fail.
