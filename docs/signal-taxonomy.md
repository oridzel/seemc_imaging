# Electron-signal taxonomy

## Design rule

SEEMC keeps three questions separate:

1. **Causal taxonomy:** which trajectory generated the emitted electron?
2. **Observable filter:** which emitted energies and angles can a detector
   accept?
3. **Image property:** what spatial information does the selected signal carry?

Localization and resolution are calculated consequences. They are not class
membership criteria. Neither is an emitted-energy cut: the 50 eV convention is
a filter, so it is *crossed with* the causal SE class rather than deciding
which cascade electrons receive a causal label at all.

## Causal SE1 and SE2

For an emitted cascade electron, let \(\hat{n}\) be the selected outward
reference normal and \(\mathbf{u}_p\) the classifying parent's direction just
before the inelastic collision that created it.

\[
\begin{aligned}
\mathrm{SE1}:&\quad \mathbf{u}_p\!\cdot\!\hat{n}\leq0,\\
\mathrm{SE2}:&\quad \mathbf{u}_p\!\cdot\!\hat{n}>0.
\end{aligned}
\]

SE1 means the parent was directed into the specimen; SE2 means it was directed
toward vacuum. The definition contains no distance from the beam axis, birth
depth, energy loss, or resolution requirement. The narrow SE1 and broad SE2
spatial responses are consequences of the short escape depth and of transport
statistics under the usual bulk-specimen geometry, and are to be measured, not
assumed.

### Which parent — `se_parent_rule`

`root_primary_leg` (default) walks the ancestry up to the generation-1 ancestor
and uses the **root incident electron's own direction** just before the
collision that seeded this lineage. SE1 then means "generated on the incoming
leg of the beam electron" and SE2 means "generated on its returning leg", which
is the conventional literature meaning (Drescher, Reimer & Seidel 1970;
Peters 1982; Silvis-Cividjian, Hagen & Kruit 2005).

`immediate_parent` uses the immediate energetic parent, which for a
second-or-later generation electron is itself a cascade electron. It is
self-consistent for deep cascades and reproduces `causal_lle_v2`, but it is a
measurably different population: in a 500 eV synthetic-material ensemble about
a third of emitted SEs have a cascade parent, and the two rules disagree on
roughly one SE2 in nine. Whichever rule is selected is stored in metadata and
compatibility-checked during fitting.

### Which surface — `se_reference`

The default `launch_surface` reference uses the surface struck by the incident
primary. It is causal and does not condition membership on the child's later
fate. The merged `escape_surface` option uses the surface through which the
emitted SE actually leaves and reproduces the 0.6.2 classifier.

**Non-planar geometry.** `launch_surface` is resolved *per primary*, so on a
trapezoidal line the reference normal is the top facet for some pixels and a
sidewall for others. The classification stays causal everywhere, but an SE1
line profile across an edge therefore splices two different reference frames.
That is intended — each pixel is referred to the surface its own beam struck —
but it must be stated when such a profile is interpreted or fitted, and it is
the reason the reference is recorded in metadata rather than assumed.

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

The code names are `lle_primary` and `non_lle_primary`. LLE is an
experimentally filterable energy class; it is not called BSE1, and it is not a
scattering-history statement.

**These are emitted primaries, not "BSEs".** The non-LLE class is the
complement of LLE among *all* emitted original incident electrons, so it also
contains primaries emitted below the 50 eV cut, which the conventional
partition counts as secondaries. Plot legends and prose say "primary"
accordingly.

**Threshold form.** `lle_max_loss_ev` sets \(\Delta E_c\) absolutely and
defaults to 50 eV. `lle_max_loss_frac` sets it as a fraction of \(E_0\)
instead — `0.02` reproduces the common \(E/E_0>0.98\) low-loss window — and is
the right choice when sweeping beam energy, since a fixed 50 eV is 10% of a
500 eV beam and 0.25% of a 20 keV beam. The two are mutually exclusive; which
one was used is recorded as `lle_criterion` and compatibility-checked.

The default absolute threshold is numerically equal to the SE/BSE emission cut,
but the two are unrelated conventions — one is an energy-*loss* window on
emitted primaries, the other an emitted-*energy* cut — and they are stored and
may be set independently.

## Scattering-history and mechanism diagnostics

`first_event_backscatter` requires that the original incident electron's first
completed collision is elastic and that this same event first turns it toward
the launch surface. `later_return_primary` is its complement among emitted
original primaries.

This is a scattering-history classification, independent of energy loss. It is
retained to measure the population and test hypotheses about single-event
backscatter, channeling, or localization. It overlaps LLE/non-LLE and therefore
is not included with them in one covariance-safe joint basis. It is also a
small population — Winkelmann et al. find first-event backscatters to be of
order 0.17% of emitted BSEs at 20 keV — so it cannot by itself carry a
high-resolution BSE signal, and is reported as a diagnostic rather than named
as a species.

`barrier_reflected_primary` counts emitted primaries turned back by the surface
barrier on entry. They never entered the solid, so \(\Delta E=0\) exactly and
they fall in `lle_primary` for any positive threshold. This is correct for an
energy-window class — a filter would pass them — but it means the LLE yield at
low \(E_0\) carries a pure specular-reflection component, so the count is
exposed separately. It is a subset of `lle_primary`, not a basis channel.

## Exit hemisphere and the transmitted signal

Exit hemisphere is a third orthogonal axis, alongside the causal class and the
emitted-energy cut. An emitted electron is **forward** (transmitted) when its
emitted velocity has a positive component along the incident beam direction,
and **backward** (reflected) otherwise. The test is on the electron's own
direction, not on which face it crossed, so a forward-scattered electron
leaving through a sidewall counts as transmitted.

On a bulk substrate the forward set is always empty, because there is no far
surface to leave through. It becomes populated only on a geometry with a
bottom exit -- `SuspendedTrapezoidalLine`, whose support is a finite `Slab`
rather than a semi-infinite `Plane`. That is the whole difference between the
SEM and STEM-in-SEM cases in this code.

### Angular segmentation

`TransmissionDetector` splits the forward hemisphere by polar angle from the
forward beam axis, in milliradians, with exclusive upper bounds:

| Channel | Rule (default) |
| --- | --- |
| `fwd_bf` | theta < 10 mrad |
| `fwd_adf` | 10 <= theta < 50 mrad |
| `fwd_haadf` | 50 <= theta < 200 mrad |
| `fwd_beyond_haadf` | theta >= 200 mrad |

`fwd_beyond_haadf` exists to close the partition out to 90 degrees; it is not a
physical detector.

**Forward is not the same as transmitted.** Being forward-going is a property
of the emitted velocity; being *transmitted* additionally requires leaving
through a surface that faces away from the source, which is what "crossed the
specimen" means. On a flat foil the two coincide. On a topographic specimen
they do not: a secondary leaving a near-vertical sidewall travels sideways and
slightly downward, so it is forward-going but escaped into the trench beside
the feature without crossing anything. Those land in `fwd_lateral_escape`; the
angular rings and `transmitted_all` contain only far-side exits. The
distinction is not academic -- on a 50 nm line at 0.9 kV, where the specimen is
opaque and true transmission is exactly zero, the entire forward hemisphere is
sidewall escape, peaking at the sidewall and vanishing over flat membrane. These are **collection angles, not a detector response**:
solid-angle weighting, gain, the BF-disc/detector-hole geometry of a real
holder, and any post-specimen optics are forward-model steps applied
afterwards, in the same spirit as the note that a real SEM fit must
forward-model the backscatter detector.

`fwd_bf_primary`, `fwd_adf_primary` and `fwd_haadf_primary` are diagnostics
counting only original incident electrons in each ring. Comparing `fwd_bf`
with `fwd_bf_primary` shows how much of the bright-field disc is secondary
electrons emitted through the underside -- a distinction a real detector cannot
make and the simulation can.

## Covariance-safe default basis

Every emitted electron belongs to exactly one of:

1. `se1_lt50`
2. `se1_ge50`
3. `se2_lt50`
4. `se2_ge50`
5. `lle_primary`
6. `non_lle_primary`

Consequently,

\[
\mathrm{TEY}=\mathrm{SE1}+\mathrm{SE2}+\mathrm{LLE}+\mathrm{nonLLE},
\]

with the aggregates

\[
\begin{aligned}
\mathrm{se1}&=\mathrm{se1\_lt50}+\mathrm{se1\_ge50},\\
\mathrm{se2}&=\mathrm{se2\_lt50}+\mathrm{se2\_ge50},\\
\mathrm{se\_cascade\_lt50}&=\mathrm{se1\_lt50}+\mathrm{se2\_lt50},\\
\mathrm{fast\_cascade\_ge50}&=\mathrm{se1\_ge50}+\mathrm{se2\_ge50}.
\end{aligned}
\]

Note that `se1` and `se2` now denote the whole causal class at any final
energy. Under `causal_lle_v2` they denoted only the sub-cutoff part; the
sub-cutoff channels are `se1_lt50` and `se2_lt50`.

This is a mathematically disjoint intrinsic basis. It is useful for mechanism
analysis and as an ideal-information bound. A real SEM fit must forward-model
the detector's energy, angle, solid-angle, and gain response to construct
experimentally measurable channels.

### With a transmission detector

When a `TransmissionDetector` is supplied the basis becomes ten channels:

1. `back_se1_lt50`
2. `back_se1_ge50`
3. `back_se2_lt50`
4. `back_se2_ge50`
5. `back_lle_primary`
6. `back_non_lle_primary`
7. `fwd_bf`
8. `fwd_adf`
9. `fwd_haadf`
10. `fwd_beyond_haadf`

The reflected hemisphere keeps the full causal taxonomy, because that is where
the SE mechanism question lives; the forward hemisphere is segmented by
collection angle, because that is what a STEM detector actually measures.
Together they still partition TEY exactly.

Crossing all three axes at once would give twenty-four channels and answer a
question nobody is asking, so the two halves use the decomposition appropriate
to each. The unprefixed aggregates `se1`, `se2`, `se1_lt50`, `lle_primary` and
so on keep counting **both** hemispheres, so those names mean the same thing
whether or not a transmission detector is attached; `backward_all` and
`forward_all` give the hemisphere totals.

## Reproducing earlier classifications

`PopulationClassifier(definition="causal_lle_v2")` reproduces the 0.7.x
default: SE classes gated at the 50 eV cut, so cascade electrons at or above it
land in `fast_cascade_ge50` without a causal label, and the immediate-parent
rule. Its diagnostics keep their old names `first_event_bse` and
`later_return_bse`. Version 0.6.2 corresponds to that classifier with
`se_reference="escape_surface"`.

`PopulationClassifier(definition="branch_v1")` reproduces the 0.6.1-era
classification. Its SE classes use the root primary's first surface-return
time, and its BSE1/BSE2 labels use the former operational elastic-return rule.

Neither is the default. Both should be identified explicitly in any continued
analysis of those archives; the fitter refuses to mix classifications, SE
references, parent rules, or LLE criteria across a library and an observation.
