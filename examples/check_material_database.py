"""Inspect a MaterialDatabase pickle and report structural problems.

Run this when ``Sample(...)`` fails while building splines.  SciPy reports
"x/y must be strictly increasing" from inside the spline constructor without
saying which table is at fault; this script checks every interpolation grid and
array shape and says exactly what is wrong.

    python examples/check_material_database.py MaterialDatabase.pkl
    python examples/check_material_database.py MaterialDatabase.pkl --material Si

``--fix`` writes a repaired copy.  It only ever sorts a grid into ascending
order and drops exactly-duplicated grid points, applying the same permutation
to the dependent arrays.  It never interpolates, rescales, or invents data, and
it refuses to touch anything it cannot repair unambiguously.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# Grids that must be strictly increasing, and the arrays indexed by each.
#   key -> (axis label, {dependent array name: axis index})
GRID_DEPENDENCIES = {
    "energy": ("energy", {
        "imfp": 0, "emfp": 0, "inv_imfp_se": 0, "inv_imfp_pl": 0,
        "decs": 1, "diimfp_se": 2, "diimfp_pl": 2,
    }),
    "omega": ("omega", {"elf_se": 0, "elf_pl": 0}),
    "q": ("q", {"elf_se": 1, "elf_pl": 1}),
    "decs_theta": ("decs_theta", {"decs": 0}),
}


def describe(values):
    values = np.asarray(values, dtype=float)
    return (f"n={values.size}, range [{values.min():.6g}, {values.max():.6g}]"
            if values.size else "empty")


def grid_report(name, values):
    """Return (problems, detail) for one grid."""
    problems = []
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        problems.append(f"expected a 1-D grid, got shape {values.shape}")
        return problems, describe(values)
    if values.size < 2:
        problems.append(f"only {values.size} point(s); need at least 2")
        return problems, describe(values)
    if not np.all(np.isfinite(values)):
        bad = np.flatnonzero(~np.isfinite(values))
        problems.append(
            f"{bad.size} non-finite value(s), first at index {int(bad[0])}"
        )
    steps = np.diff(values)
    flat = np.flatnonzero(steps == 0.0)
    down = np.flatnonzero(steps < 0.0)
    if flat.size:
        i = int(flat[0])
        problems.append(
            f"{flat.size} repeated value(s); first at indices {i},{i + 1} "
            f"(both {values[i]:.10g})"
        )
    if down.size:
        i = int(down[0])
        problems.append(
            f"{down.size} decreasing step(s); first at index {i}: "
            f"{values[i]:.10g} -> {values[i + 1]:.10g}"
        )
    if name == "q" and values.size and values.min() <= 0.0:
        problems.append(
            "q must be strictly positive (the sampler works in log q); "
            f"minimum is {values.min():.6g}"
        )
    return problems, describe(values)


def check(material, *, verbose=True):
    """Return a list of problem strings for one material dict."""
    findings = []
    name = material.get("name", "<unnamed>")
    if verbose:
        print(f"Material {name!r}")

    for key, (label, dependents) in GRID_DEPENDENCIES.items():
        if key not in material:
            findings.append(f"{key}: missing from the database")
            continue
        problems, detail = grid_report(key, material[key])
        status = "OK " if not problems else "BAD"
        if verbose:
            print(f"  [{status}] {label:<12s} {detail}")
        for problem in problems:
            findings.append(f"{key}: {problem}")
            if verbose:
                print(f"          - {problem}")

        size = np.asarray(material[key]).size
        for array_name, axis in dependents.items():
            if array_name not in material:
                continue
            array = np.asarray(material[array_name])
            if axis >= array.ndim:
                findings.append(
                    f"{array_name}: has {array.ndim} axes, expected axis "
                    f"{axis} to index {key}"
                )
                continue
            if array.shape[axis] != size:
                # A transposed ELF is legitimate and handled by the loader.
                if (array_name.startswith("elf")
                        and size in array.shape):
                    continue
                findings.append(
                    f"{array_name}: axis {axis} has length "
                    f"{array.shape[axis]} but {key} has {size} points"
                )
                if verbose:
                    print(f"          - {findings[-1]}")

    energy = np.asarray(material.get("energy", []), dtype=float)
    e_fermi = float(material.get("e_fermi", 0.0))
    for array_name in ("elf_se", "elf_pl", "decs", "diimfp_se", "diimfp_pl",
                       "imfp", "emfp"):
        if array_name not in material:
            continue
        array = np.asarray(material[array_name], dtype=float)
        if np.all(np.isfinite(array)):
            continue
        bad = ~np.isfinite(array)
        count = int(np.count_nonzero(bad))
        # There is no inelastic channel below the Fermi level, so a
        # non-finite IMFP there is how the table says "no scattering".  The
        # loader turns it into a zero inverse MFP, which is correct.
        if (array_name == "imfp" and energy.size == array.size
                and np.all(energy[bad] <= e_fermi)):
            if verbose:
                print(f"  [ok ] imfp: {count} non-finite entries, all at or "
                      f"below E_F = {e_fermi:g} eV (expected: no inelastic "
                      "channel there)")
            continue
        findings.append(f"{array_name}: {count} non-finite entries")
        if verbose:
            print(f"  [BAD] {array_name}: {count} non-finite entries")

    if verbose and "energy" in material:
        energy = np.asarray(material["energy"], dtype=float)
        if energy.size:
            print(f"  energy range: {energy.min():.6g} to {energy.max():.6g} eV"
                  f"  ({'covers' if energy.max() >= 30_000.0 else 'DOES NOT cover'}"
                  " 30 kV)")
    return findings


# Screened-Rutherford total elastic cross section (Newbury/Goldstein form),
# used only as an order-of-magnitude yardstick for the tabulated EMFP.
def _rutherford_emfp_nm(Z, energy_kev, number_density_cm3):
    alpha = 3.4e-3 * Z ** 0.67 / energy_kev
    sigma = (5.21e-21 * (Z ** 2 / energy_kev ** 2)
             * (4.0 * np.pi / (alpha * (1.0 + alpha)))
             * ((energy_kev + 511.0) / (energy_kev + 1022.0)) ** 2)
    mfp_cm = 1.0 / (number_density_cm3 * sigma)
    return mfp_cm * 1.0e7


def mfp_report(material, *, verbose=True, number_density_cm3=4.99e22,
               atomic_number=None):
    """Compare the tabulated mean free paths against a coarse expectation.

    This does not validate the physics -- it catches the gross case where a
    rebuilt table is off by orders of magnitude or has the wrong energy
    dependence.  A mean free path near the interatomic spacing, or one that
    fails to grow with energy, is not a plausible table.
    """
    findings = []
    if "energy" not in material or "emfp" not in material:
        return findings
    energy = np.asarray(material["energy"], dtype=float)
    emfp = np.asarray(material["emfp"], dtype=float)
    imfp = np.asarray(material.get("imfp", []), dtype=float)
    Z = atomic_number or material.get("atomic_number")

    if verbose:
        print("  mean free paths (table units, normally Angstrom):")
        header = f"    {'E (eV)':>10s} {'EMFP':>10s} {'IMFP':>10s}"
        if Z:
            header += f" {'EMFP/Rutherford':>17s}"
        print(header)
        if not Z:
            print("    (no atomic_number in the database, so the "
                  "EMFP/Rutherford sanity column is omitted;")
            print("     pass --atomic-number to enable it)")
    worst_ratio, worst_energy = None, None
    if Z:
        for e_val, m_val in zip(energy, emfp):
            if e_val <= 100.0:
                continue
            expected = _rutherford_emfp_nm(Z, e_val / 1000.0, number_density_cm3)
            ratio = (m_val / 10.0) / expected
            if worst_ratio is None or abs(np.log10(max(ratio, 1e-12))) > abs(
                    np.log10(max(worst_ratio, 1e-12))):
                worst_ratio, worst_energy = ratio, e_val

    probes = [0, len(energy) // 4, len(energy) // 2,
              3 * len(energy) // 4, len(energy) - 1]
    for i in sorted(set(probes)):
        row = f"    {energy[i]:10.4g} {emfp[i]:10.4g}"
        row += f" {imfp[i]:10.4g}" if imfp.size == energy.size else f" {'-':>10s}"
        if Z and energy[i] > 100.0:
            expected_nm = _rutherford_emfp_nm(Z, energy[i] / 1000.0,
                                              number_density_cm3)
            ratio = (emfp[i] / 10.0) / expected_nm     # table assumed Angstrom
            row += f" {ratio:17.3g}"
        if verbose:
            print(row)

    if np.any(emfp <= 0):
        findings.append("emfp: contains non-positive entries")
    if worst_ratio is not None and not 0.1 <= worst_ratio <= 10.0:
        findings.append(
            f"emfp: at E = {worst_energy:.4g} eV it is {worst_ratio:.3g}x the "
            "screened-Rutherford estimate.  That yardstick is only good to a "
            "factor of a few, so a value this far off means wrong units or a "
            "wrong energy dependence, not model disagreement"
        )
    # An elastic mean free path at or below the interatomic spacing (~2.35 A
    # in Si) is unphysical at any energy above a few tens of eV.
    high = energy > 1000.0
    if np.any(high) and np.any(emfp[high] < 2.0):
        worst = energy[high][np.argmin(emfp[high])]
        findings.append(
            f"emfp: falls below 2 Angstrom (interatomic spacing) at "
            f"E = {worst:.4g} eV -- about 100x too much elastic scattering; "
            "check the units and the energy dependence of the rebuilt table"
        )
    # Above ~1 keV the elastic MFP must grow with energy, and roughly as fast
    # as the cross section falls.  A table that grows far too slowly is the
    # signature of a high-energy extension that saturated instead of
    # extrapolating.
    if np.count_nonzero(high) >= 2:
        e_hi, m_hi = energy[high], emfp[high]
        if m_hi[-1] <= m_hi[0]:
            findings.append(
                f"emfp: does not increase between {e_hi[0]:.4g} and "
                f"{e_hi[-1]:.4g} eV ({m_hi[0]:.4g} -> {m_hi[-1]:.4g}); the "
                "elastic cross section must fall with energy"
            )
        elif Z:
            grew = m_hi[-1] / m_hi[0]
            should = (_rutherford_emfp_nm(Z, e_hi[-1] / 1000.0, number_density_cm3)
                      / _rutherford_emfp_nm(Z, e_hi[0] / 1000.0, number_density_cm3))
            if grew < 0.3 * should:
                findings.append(
                    f"emfp: grows only {grew:.2f}x between {e_hi[0]:.4g} and "
                    f"{e_hi[-1]:.4g} eV where it should grow about "
                    f"{should:.1f}x.  A high-energy extension that flattens "
                    "like this leaves the beam over-scattered at the top of "
                    "the range even if the low-energy end looks right"
            )
    if verbose:
        for problem in findings:
            print(f"  [BAD] {problem}")
    return findings


def decs_report(material, *, verbose=True):
    """Summarise the differential elastic cross section's angular width.

    The mean free path sets *how often* an electron scatters; the DECS sets
    *how far it turns* each time.  Both must be right for a transmitted
    angular distribution to be meaningful, so report the median deflection at
    a few energies: at keV energies elastic scattering in a light element is
    strongly forward-peaked and the median should be small and falling with
    energy.
    """
    findings = []
    if "decs" not in material or "decs_theta" not in material:
        return findings
    theta = np.asarray(material["decs_theta"], dtype=float)
    decs = np.asarray(material["decs"], dtype=float)
    energy = np.asarray(material.get("energy", []), dtype=float)
    if decs.ndim != 2 or theta.size != decs.shape[0]:
        return findings

    # Solid-angle weight: dsigma/dOmega * 2 pi sin(theta) dtheta.
    weight = decs * np.sin(theta)[:, None]
    cumulative = np.cumsum(weight, axis=0)
    totals = cumulative[-1]
    medians = np.full(decs.shape[1], np.nan)
    for j in range(decs.shape[1]):
        if totals[j] > 0:
            medians[j] = np.interp(0.5 * totals[j], cumulative[:, j], theta)

    if verbose:
        print("  elastic angular distribution (median deflection per event):")
        print(f"    {'E (eV)':>10s} {'median (deg)':>13s} {'median (mrad)':>14s}")
    probes = sorted(set([0, decs.shape[1] // 2, decs.shape[1] - 1]))
    for j in probes:
        e_val = energy[j] if energy.size == decs.shape[1] else float("nan")
        med = medians[j]
        if verbose:
            print(f"    {e_val:10.4g} {np.degrees(med):13.2f} {1000 * med:14.1f}")

    if energy.size == decs.shape[1]:
        high = energy > 5000.0
        if np.any(high) and np.nanmedian(medians[high]) > np.radians(20.0):
            findings.append(
                f"decs: median deflection above 5 keV is "
                f"{np.degrees(np.nanmedian(medians[high])):.0f} degrees; "
                "elastic scattering in a light element at these energies "
                "should be strongly forward-peaked, so this angular table "
                "will broaden a transmitted beam even with a correct emfp"
            )
    if verbose:
        for problem in findings:
            print(f"  [BAD] {problem}")
    return findings


def repair(material, *, verbose=True):
    """Sort and de-duplicate grids, permuting dependent arrays to match."""
    repaired = dict(material)
    actions = []
    for key, (_, dependents) in GRID_DEPENDENCIES.items():
        if key not in repaired:
            continue
        values = np.asarray(repaired[key], dtype=float)
        if values.ndim != 1 or values.size < 2:
            continue
        if np.all(np.diff(values) > 0.0):
            continue
        order = np.argsort(values, kind="stable")
        ordered = values[order]
        keep_mask = np.ones(ordered.size, dtype=bool)
        keep_mask[1:] = np.diff(ordered) > 0.0
        keep = order[keep_mask]
        dropped = values.size - keep.size
        was_sorted = np.array_equal(order, np.arange(values.size))
        repaired[key] = values[keep]
        for array_name, axis in dependents.items():
            if array_name not in repaired:
                continue
            array = np.asarray(repaired[array_name])
            if axis >= array.ndim or array.shape[axis] != values.size:
                actions.append(
                    f"  ! {array_name}: axis {axis} length "
                    f"{array.shape[axis] if axis < array.ndim else 'n/a'} does "
                    f"not match {key} ({values.size}); left untouched, so the "
                    "result will still be inconsistent"
                )
                continue
            repaired[array_name] = np.take(array, keep, axis=axis)
        detail = []
        if not was_sorted:
            detail.append("sorted ascending")
        if dropped:
            detail.append(f"dropped {dropped} duplicate point(s)")
        actions.append(f"  {key}: " + " and ".join(detail)
                       + f" -> {keep.size} points")
    if verbose:
        for line in actions:
            print(line)
    return repaired, actions


def load(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database", type=Path)
    parser.add_argument("--material", default=None,
                        help="check only this material (default: all)")
    parser.add_argument("--skip-mfp", action="store_true",
                        help="skip the mean-free-path plausibility report")
    parser.add_argument("--number-density", type=float, default=4.99e22,
                        help="atoms per cm^3 for the MFP yardstick "
                             "(default: silicon)")
    parser.add_argument("--atomic-number", type=int, default=None,
                        help="override the atomic number used by the yardstick")
    parser.add_argument("--fix", type=Path, default=None,
                        metavar="OUTPUT",
                        help="write a repaired copy to OUTPUT (sorting and "
                             "de-duplicating grids only)")
    args = parser.parse_args()

    data = load(args.database)
    if isinstance(data, dict):
        materials = [data]
    elif isinstance(data, list):
        materials = data
    else:
        parser.error(f"unrecognized database format: {type(data).__name__}")

    if args.material is not None:
        materials = [m for m in materials if m.get("name") == args.material]
        if not materials:
            parser.error(f"no material named {args.material!r} in the database")

    total = 0
    repaired_all = []
    for material in materials:
        findings = check(material)
        if not args.skip_mfp:
            findings = findings + mfp_report(
                material, number_density_cm3=args.number_density,
                atomic_number=args.atomic_number)
            findings = findings + decs_report(material)
        total += len(findings)
        if args.fix is not None:
            if findings:
                print("  repairing:")
                fixed, _ = repair(material)
                repaired_all.append(fixed)
            else:
                repaired_all.append(material)
        print()

    if total == 0:
        print("No structural problems found.")
    else:
        print(f"{total} problem(s) found.")

    if args.fix is not None:
        payload = repaired_all[0] if isinstance(data, dict) else repaired_all
        args.fix.parent.mkdir(parents=True, exist_ok=True)
        with open(args.fix, "wb") as stream:
            pickle.dump(payload, stream)
        print(f"Wrote repaired database to {args.fix}")
        print("Re-run this checker on it before using it for physics.")

    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
