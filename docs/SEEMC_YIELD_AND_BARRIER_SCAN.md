# Running SEEMC Yield Curves and Surface-Barrier Scans

This section describes how to calculate Cu **TEY, SEY, and BSEY** with SEEMC at a chosen incidence angle and number of primary trajectories, and how to explore the surface-barrier parameters.

## 1. Run yields at a chosen angle and trajectory count

Use `generate_plane_sampler_grid.py`.

Example: **normal incidence**, 100,000 primaries per energy:

```bash
python3 generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --angles-deg 0 \
  --primaries 100000 \
  --elastic-low-energy-model browning \
  --elastic-cutoff-ev 50 \
  --barrier-model abrupt \
  --workers 18 \
  --output yield_tests/Cu_alpha0_abrupt_100k
```

Example: **45° incidence**, 50,000 primaries per energy:

```bash
python3 generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --angles-deg 45 \
  --primaries 50000 \
  --elastic-low-energy-model browning \
  --elastic-cutoff-ev 50 \
  --barrier-model abrupt \
  --workers 18 \
  --output yield_tests/Cu_alpha45_abrupt_50k
```

If `--energies-ev` is omitted, the standard sampler energy grid is used.

For a custom energy grid concentrated around the TEY maximum:

```bash
python3 generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --angles-deg 0 \
  --energies-ev 50 75 100 125 150 175 200 250 300 400 500 700 1000 1500 2000 3000 5000 \
  --primaries 50000 \
  --elastic-low-energy-model browning \
  --elastic-cutoff-ev 50 \
  --barrier-model abrupt \
  --workers 18 \
  --output yield_tests/Cu_alpha0_abrupt_dense
```

## 2. Output yields

For each incidence angle, the output directory contains:

```text
alpha_0deg/
    SEYFromPlane_SEVaccum_t0nmCuFPA.csv
    BSEYFromPlane_SEVaccum_t0nmCuFPA.csv
```

The total electron yield is

\[
\mathrm{TEY} = \mathrm{SEY} + \mathrm{BSEY}.
\]

A simple plotting example:

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

d = Path("yield_tests/Cu_alpha0_abrupt_100k/alpha_0deg")

sey = pd.read_csv(
    d / "SEYFromPlane_SEVaccum_t0nmCuFPA.csv",
    skiprows=1,
)

bsey = pd.read_csv(
    d / "BSEYFromPlane_SEVaccum_t0nmCuFPA.csv",
    skiprows=1,
)

E = sey.iloc[:, 0].to_numpy()
SEY = sey.iloc[:, 1].to_numpy()
BSEY = bsey.iloc[:, 1].to_numpy()
TEY = SEY + BSEY

plt.plot(E, TEY, "o-", label="TEY")
plt.plot(E, SEY, "o-", label="SEY")
plt.plot(E, BSEY, "o-", label="BSEY")

plt.xscale("log")
plt.xlabel("Incident energy (eV)")
plt.ylabel("Yield")
plt.legend()
plt.grid()
plt.show()

print("TEY max =", TEY.max(), "at", E[TEY.argmax()], "eV")
```

## 3. Surface-barrier models

The current SEEMC transport supports:

```text
abrupt
expqm
classical
```

For the finite-width quantum barrier, use:

```text
--barrier-model expqm
--barrier-width-angstrom <width>
```

Example: 0.5 Å barrier width:

```bash
python3 generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --angles-deg 0 \
  --primaries 20000 \
  --barrier-model expqm \
  --barrier-width-angstrom 0.5 \
  --elastic-low-energy-model browning \
  --elastic-cutoff-ev 50 \
  --workers 18 \
  --output barrier_scan/expqm_0p5A
```

Example: 1 Å:

```bash
python3 generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --angles-deg 0 \
  --primaries 20000 \
  --barrier-model expqm \
  --barrier-width-angstrom 1.0 \
  --elastic-low-energy-model browning \
  --elastic-cutoff-ev 50 \
  --workers 18 \
  --output barrier_scan/expqm_1A
```

Useful exploratory widths include:

```text
0.25 Å
0.5 Å
1 Å
2 Å
5 Å
```

For an exploratory parameter scan, 20,000–50,000 primaries per energy are usually sufficient. Once a promising parameter set is identified, rerun it with 100,000 or more primaries.

## 4. Important interpretation of barrier width

For the implemented models, the **abrupt barrier is the low-transmission limit**.

The finite-width `expqm` model approaches the abrupt limit as the width becomes small and approaches the classical transmission limit as the barrier becomes broader.

Therefore, if the current **abrupt-barrier** SEEMC TEY is already larger than experiment, increasing the barrier width is generally not expected to reduce TEY. It tends to increase electron transmission through the surface barrier.

For example, if the measured Cu TEY maximum is approximately

\[
\mathrm{TEY}_{\max,\mathrm{exp}} \approx 1.4,
\]

while the abrupt-barrier SEEMC curve is higher, barrier width alone is unlikely to solve the discrepancy.

## 5. Inner potential / work-function scan

The outgoing surface barrier is controlled by the Cu inner potential

\[
U_i = E_F + \Phi,
\]

where \(E_F\) is the Fermi energy and \(\Phi\) is the work function.

Increasing \(U_i\) makes escape more difficult for low-energy secondaries and can therefore reduce SEY and TEY.

A useful exploratory scan would be approximately:

```text
Ui = 12, 13, 13.4, 14, 15, 16 eV
```

or, preferably, vary the work function over a physically reasonable range while keeping the bulk Fermi energy fixed.

`generate_plane_sampler_grid.py` now supports a direct work-function override:

```text
--work-function-ev <value>
```

This overrides only the surface work function `Phi`. The material-database Fermi energy `E_F` is left unchanged, so the barrier used by the transport is

\[
U_i = E_F + \Phi.
\]

For example, to test `Phi = 4.8 eV`:

```bash
python3 generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --angles-deg 0 \
  --primaries 50000 \
  --elastic-low-energy-model browning \
  --elastic-cutoff-ev 50 \
  --barrier-model abrupt \
  --work-function-ev 4.8 \
  --workers 18 \
  --output work_function_scan/Cu_phi_4p8eV
```

If `--work-function-ev` is omitted, the original `MaterialDatabase.pkl` work function is used. Because the override is part of `MCConfig`, resumable sampler checkpoints with a different work function are rejected rather than silently reused.

## 6. Recommended fitting strategy

For comparing SEEMC with measured Cu yields:

1. Start with **normal incidence**.
2. Use the validated Browning low-energy elastic model:
   ```text
   --elastic-low-energy-model browning
   --elastic-cutoff-ev 50
   ```
3. Start from the `abrupt` barrier model.
4. Compare **TEY, SEY, and BSEY separately**, not TEY alone.
5. If TEY is too high, determine whether the excess comes primarily from:
   - SEY,
   - BSEY,
   - or both.
6. Explore the physically defensible work-function / inner-potential range.
7. Once the normal-incidence yield curve is understood, test whether the same parameters reproduce the measured angular dependence.
8. Use higher statistics only for the final candidate parameter sets.

This avoids fitting an angular-transport problem with a surface-barrier parameter and helps separate bulk transport, escape-barrier physics, and RFA transport effects.
