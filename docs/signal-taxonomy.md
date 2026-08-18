# Electron-signal taxonomy

## Design rule

SEEMC keeps three questions separate:

1. **Causal taxonomy:** which trajectory generated the emitted electron?
2. **Observable filter:** which emitted energies and angles can a detector
   accept?
3. **Image property:** what spatial information does the selected signal carry?

Localization and resolution are calculated consequences. They are not class
membership criteria.

## Causal SE1 and SE2

For an emitted cascade electron below the configured SE/BSE energy cutoff, the
creating inelastic collision retains the immediate energetic parent's incoming
direction. Let \(\hat{n}\) be the selected outward reference normal and
\(\mathbf{u}_p\) the immediate parent's direction just before that collision.

\[
\begin{aligned}
\mathrm{SE1}:&\quad \mathbf{u}_p\!\cdot\!\hat{n}\leq0,\\
\mathrm{SE2}:&\quad \mathbf{u}_p\!\cdot\!\hat{n}>0.
\end{aligned}
\]

Thus SE1 means parent directed into the specimen and SE2 means parent directed
toward vacuum. The definition contains no distance from the beam axis, birth
depth, energy-loss, or resolution requirement. It also works for later cascade
generations because it follows the immediate energetic parent rather than only
the root primary's global history.

The default `launch_surface` reference uses the surface struck by the incident
primary. It is causal and does not condition membership on the child's later
fate. The merged `escape_surface` option uses the surface through which the
emitted SE actually leaves and reproduces the 0.6.2 classifier. Select it with
`PopulationClassifier(se_reference="escape_surface")` or
`--se-reference escape_surface`. The selected reference is stored in metadata
and compatibility-checked during fitting.

## Low-loss emitted primaries

For an emitted original incident electron, define vacuum energy loss

\[
\Delta E=\max(0,E_0-E_{exit}),
\]

where both energies use the vacuum kinetic-energy reference. Given a declared
threshold \(\Delta E_c\),

\[
\begin{aligned}
\mathrm{LLE}:&\quad \Delta E<\Delta E_c,\\
\mathrm{nonLLE}:&\quad \Delta E\geq\Delta E_c.
\end{aligned}
\]

The code names are `lle_primary` and `non_lle_primary`. The threshold is a
classifier parameter and is stored in every raster and model library. LLE is
an experimentally filterable energy class; it is not called BSE1.

## First-event BSE diagnostic

`first_event_bse` requires that the original incident electron's first
completed collision is elastic and that this same event first turns it toward
the launch surface. `later_return_bse` is its complement among emitted original
primaries.

This is a scattering-history classification, independent of energy loss. It
is retained to measure the population and test hypotheses about single-event
backscatter, channeling, or localization. It overlaps LLE/non-LLE and therefore
is not included with them in one covariance-safe joint basis.

## Covariance-safe default basis

Every emitted electron belongs to exactly one of:

1. `se1`
2. `se2`
3. `fast_cascade_ge50`
4. `lle_primary`
5. `non_lle_primary`

Consequently,

\[
\mathrm{TEY}=\mathrm{SE1}+\mathrm{SE2}
+\mathrm{fast\ cascade}+\mathrm{LLE}+\mathrm{nonLLE}.
\]

This is a mathematically disjoint intrinsic basis. It is useful for mechanism
analysis and as an ideal-information bound. A real SEM fit must forward-model
the detector's energy, angle, solid-angle, and gain response to construct
experimentally measurable channels.

## Legacy reproduction

`PopulationClassifier(definition="branch_v1")` reproduces the 0.6.1-era
classification. Its SE classes use the root primary's first surface-return
time, and its BSE1/BSE2 labels use the former operational elastic-return rule.
The legacy mode is not the default and should be identified explicitly in any
continued analysis of those old archives. Version 0.6.2 instead corresponds to
the current causal/LLE classifier with `se_reference="escape_surface"`.
