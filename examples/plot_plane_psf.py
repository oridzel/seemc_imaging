#!/usr/bin/env python3
"""Plot a surface-emission PSF from a PSF-enabled SEEMC plane-sampler checkpoint."""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("plane_psf"))
    p.add_argument("--map-bins", type=int, default=301)
    p.add_argument("--radial-bins", type=int, default=250)
    p.add_argument(
        "--length-unit", default="simulation length units",
        help="axis label only; set to nm after confirming transport coordinate units",
    )
    return p


def main():
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.checkpoint, allow_pickle=False) as z:
        required = ("se_emission_xyz", "bse_emission_xyz")
        missing = [k for k in required if k not in z]
        if missing:
            raise KeyError(
                f"checkpoint lacks {missing}; use a PSF-enabled plane_samplers.py"
            )
        se_xyz = z["se_emission_xyz"].astype(float)
        bse_xyz = z["bse_emission_xyz"].astype(float)
        n_primary = int(z["n_primaries"].item())
        energy_ev = float(z["incident_energy_ev"].item())
        cutoff = float(z["energy_cutoff_ev"].item())

    if se_xyz.ndim != 2 or se_xyz.shape[1] != 3:
        raise ValueError("se_emission_xyz must have shape (N,3)")
    if bse_xyz.ndim != 2 or bse_xyz.shape[1] != 3:
        raise ValueError("bse_emission_xyz must have shape (N,3)")

    se_r = np.hypot(se_xyz[:, 0], se_xyz[:, 1])
    bse_r = np.hypot(bse_xyz[:, 0], bse_xyz[:, 1])

    def quantiles(r):
        return np.quantile(r, [0.50, 0.80, 0.90, 0.95, 0.99])

    print(f"E0 = {energy_ev:g} eV, Nprimary = {n_primary}")
    for name, r in (("SE", se_r), ("BSE", bse_r)):
        q = quantiles(r)
        print(
            f"{name}: N={r.size}, yield={r.size/n_primary:.6g}, "
            f"r50={q[0]:.6g}, r80={q[1]:.6g}, r90={q[2]:.6g}, "
            f"r95={q[3]:.6g}, r99={q[4]:.6g} {args.length_unit}"
        )

    # Common map window: 99.9% of |x| and |y| values.
    map_half = max(
        np.quantile(np.abs(se_xyz[:, :2]), 0.999) if se_xyz.size else 0.0,
        np.quantile(np.abs(bse_xyz[:, :2]), 0.999) if bse_xyz.size else 0.0,
    )
    map_half = max(map_half, 1.0)

    def psf_map(xyz, label, filename):
        H, xe, ye = np.histogram2d(
            xyz[:, 0], xyz[:, 1], bins=args.map_bins,
            range=[[-map_half, map_half], [-map_half, map_half]],
        )
        area = (xe[1] - xe[0]) * (ye[1] - ye[0])
        d = H.T / (n_primary * area)

        fig, ax = plt.subplots(figsize=(7, 6))
        pos = d[d > 0]
        norm = None
        if pos.size:
            norm = LogNorm(vmin=max(pos.min(), pos.max()*1e-5), vmax=pos.max())
        im = ax.imshow(
            d, origin="lower",
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
            aspect="equal", norm=norm,
        )
        ax.set_xlabel(f"x - beam position ({args.length_unit})")
        ax.set_ylabel(f"y - beam position ({args.length_unit})")
        ax.set_title(f"{label} surface-emission PSF, E0={energy_ev:g} eV")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(f"electrons / primary / {args.length_unit}²")
        fig.tight_layout()
        fig.savefig(args.output / filename, dpi=220)
        plt.close(fig)

    psf_map(se_xyz, f"SE (< {cutoff:g} eV)", "psf_SE_2D.png")
    psf_map(bse_xyz, f"BSE (>= {cutoff:g} eV)", "psf_BSE_2D.png")

    # Radial PSF.
    rmax = max(se_r.max(initial=0.0), bse_r.max(initial=0.0))
    edges = np.linspace(0.0, rmax, args.radial_bins + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    annulus = np.pi * (edges[1:]**2 - edges[:-1]**2)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in (("SE", se_r), ("BSE", bse_r)):
        counts, _ = np.histogram(r, bins=edges)
        d = counts / (n_primary * annulus)
        m = d > 0
        ax.plot(rc[m], d[m], label=name)
    ax.set_yscale("log")
    ax.set_xlabel(f"Radius ({args.length_unit})")
    ax.set_ylabel(f"Radial PSF (electrons / primary / {args.length_unit}²)")
    ax.set_title(f"Radially averaged PSF, E0={energy_ev:g} eV")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "psf_radial.png", dpi=220)
    plt.close(fig)

    # Encircled signal.
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in (("SE", se_r), ("BSE", bse_r)):
        rs = np.sort(r)
        f = np.arange(1, rs.size + 1) / rs.size
        ax.plot(rs, f, label=name)
    for q in (0.5, 0.8, 0.9, 0.95):
        ax.axhline(q, linewidth=0.8, alpha=0.35)
    ax.set_xlabel(f"Radius ({args.length_unit})")
    ax.set_ylabel("Fraction of emitted signal within radius")
    ax.set_ylim(0, 1.01)
    ax.set_title(f"Encircled signal, E0={energy_ev:g} eV")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "psf_encircled.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
