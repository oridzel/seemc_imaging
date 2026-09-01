"""Overlay the reflected and transmitted line-scan signals from two energies.

Reads the NPZ archives written by ``suspended_line_stem_linescan.py`` and draws
three panels on a shared position axis:

1. Total SE yield at each landing energy, normalised to the bare membrane.
2. The transmitted angular channels, normalised to the bare membrane.
3. The same signals min-max scaled, so edge position and width can be compared
   directly regardless of absolute level.

Normalising to the bare membrane (rather than to each curve's peak) keeps the
sign of the contrast: above 1.0 means the line raises that signal, below 1.0
means it lowers it.

    python examples/plot_linescan_overlay.py \\
        si_line_0p9kV.npz si_line_30kV.npz --output si_line_overlay.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated palette (see the dataviz reference palette).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"
SERIES_1 = "#2a78d6"   # categorical slot 1, blue
SERIES_2 = "#eb6834"   # categorical slot 2, orange
# Ordinal ramp for the angular rings: one hue, light -> dark with the angle.
RING_COLORS = {
    "fwd_bf": "#86b6ef",
    "fwd_adf": "#3987e5",
    "fwd_haadf": "#1c5cab",
    "fwd_beyond_haadf": "#0d366b",
}
RING_LABELS = {
    "fwd_bf": "BF",
    "fwd_adf": "ADF",
    "fwd_haadf": "HAADF",
    "fwd_beyond_haadf": "Beyond HAADF",
}

NM = 10.0


def load(path):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    # The raster archive stores the scan settings under "config"; older
    # metrology libraries use "raster_config".
    metadata["_config"] = metadata.get("config",
                                       metadata.get("raster_config", {}))
    x = archive["x_angstrom"] / NM
    return archive, metadata, x


def yields(archive, channel):
    return archive[f"yield__{channel}"][0]


def errors(archive, channel):
    key = f"sem__{channel}"
    return archive[key][0] if key in archive.files else np.zeros_like(
        yields(archive, channel))


def membrane_level(x, values, edge_nm):
    """Mean over the bare-membrane region on both sides of the line."""
    mask = np.abs(x) > edge_nm
    level = values[mask].mean()
    return level if level > 0 else np.nan


def draw_geometry(axis, geometry, *, label=False):
    half_top = 0.5 * geometry["top_width"] / NM
    half_base = 0.5 * geometry["bottom_width"] / NM
    axis.axvspan(-half_base, half_base, color=INK, alpha=0.045, lw=0, zorder=0)
    for edge in (half_top, half_base):
        for sign in (-1.0, 1.0):
            axis.axvline(sign * edge, color=GRID, lw=0.9, ls="--", zorder=1)
    if label:
        axis.annotate(
            "line footprint", xy=(0.0, 0.97), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=8, color=INK_SOFT,
        )


def style(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, alpha=0.55, lw=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SOFT, labelsize=9)


def plot_series(axis, x, values, error, color, label, *, normalise, band=True):
    scaled = values / normalise
    axis.plot(x, scaled, color=color, lw=1.8, label=label, zorder=3,
              solid_capstyle="round")
    if band and np.any(error > 0):
        lo = (values - error) / normalise
        hi = (values + error) / normalise
        axis.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("low_energy", type=Path, help="e.g. si_line_0p9kV.npz")
    parser.add_argument("high_energy", type=Path, help="e.g. si_line_30kV.npz")
    parser.add_argument("--output", type=Path, default=Path("linescan_overlay.png"))
    parser.add_argument("--membrane-edge-nm", type=float, default=40.0,
                        help="|x| beyond which the scan is bare membrane")
    parser.add_argument("--se-channel", default="sey_50ev",
                        help="channel used as 'total SE' (default sey_50ev)")
    parser.add_argument("--edge-window-nm", type=float, nargs=2,
                        default=(12.0, 40.0), metavar=("INNER", "OUTER"),
                        help="radial window for the folded edge panel")
    args = parser.parse_args()

    low, low_meta, x = load(args.low_energy)
    high, high_meta, x_high = load(args.high_energy)
    if not np.allclose(x, x_high):
        raise SystemExit("the two scans use different position grids")
    geometry = low_meta["geometry"]

    kv_low = low_meta["_config"]["energy_ev"] / 1000.0
    kv_high = high_meta["_config"]["energy_ev"] / 1000.0

    figure, axes = plt.subplots(3, 1, figsize=(9.6, 10.6), facecolor=SURFACE)
    figure.subplots_adjust(hspace=0.34)
    axes[0].sharex(axes[1])

    # ---- Panel 1: total SE at both energies -----------------------------
    axis = axes[0]
    style(axis)
    draw_geometry(axis, geometry, label=True)
    for archive, kv, color in ((low, kv_low, SERIES_1), (high, kv_high, SERIES_2)):
        values = yields(archive, args.se_channel)
        level = membrane_level(x, values, args.membrane_edge_nm)
        plot_series(axis, x, values, errors(archive, args.se_channel), color,
                    f"{kv:g} kV  (membrane {level:.3f}/primary)",
                    normalise=level)
    axis.axhline(1.0, color=INK_SOFT, lw=0.8, ls=":", zorder=1)
    axis.set_ylabel("SE yield / bare-membrane level", color=INK, fontsize=10)
    axis.set_title(
        "Total secondary-electron signal across the line",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper right")

    # ---- Panel 2: transmitted rings at the high energy ------------------
    axis = axes[1]
    style(axis)
    draw_geometry(axis, geometry)
    present = [c for c in RING_COLORS if f"yield__{c}" in high.files]
    for channel in present:
        values = yields(high, channel)
        level = membrane_level(x, values, args.membrane_edge_nm)
        if not np.isfinite(level):
            continue
        plot_series(axis, x, values, errors(high, channel),
                    RING_COLORS[channel],
                    f"{RING_LABELS[channel]}  ({level:.3f}/primary)",
                    normalise=level)
    axis.axhline(1.0, color=INK_SOFT, lw=0.8, ls=":", zorder=1)
    axis.set_ylabel("Transmitted / bare-membrane level", color=INK, fontsize=10)
    axis.set_xlabel("Beam position (nm)", color=INK, fontsize=10)
    axis.set_title(
        f"Transmitted signal by collection angle, {kv_high:g} kV",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=2,
                loc="upper right")

    # ---- Panel 3: folded edge profile -----------------------------------
    # Min-max scaling the full scan is unreadable: the weak channels are
    # dominated by counting noise far from the line.  Folding the two edges
    # together and zooming on the transition halves the noise and shows the
    # thing that actually matters -- how far the edge signal reaches inward.
    axis = axes[2]
    style(axis)
    half_top = 0.5 * geometry["top_width"] / NM
    half_base = 0.5 * geometry["bottom_width"] / NM
    for edge in (half_top, half_base):
        axis.axvline(edge, color=GRID, lw=0.9, ls="--", zorder=1)
    axis.axvspan(0.0, half_base, color=INK, alpha=0.045, lw=0, zorder=0)

    window = (np.abs(x) >= args.edge_window_nm[0]) & (
        np.abs(x) <= args.edge_window_nm[1])
    # Only the two SE curves here: the transmitted channels are already in
    # panel 2, and at this scaling their counting noise buries the shape.
    for archive, channel, color, label in (
        (low, args.se_channel, SERIES_1, f"SE, {kv_low:g} kV"),
        (high, args.se_channel, SERIES_2, f"SE, {kv_high:g} kV"),
    ):
        if f"yield__{channel}" not in archive.files:
            continue
        values = yields(archive, channel).astype(float)
        folded = 0.5 * (values + values[::-1])      # x grid is symmetric
        radius = np.abs(x)[window]
        signal = folded[window]
        order = np.argsort(radius)
        radius, signal = radius[order], signal[order]
        span = signal.max() - signal.min()
        if span <= 0:
            continue
        axis.plot(radius, (signal - signal.min()) / span, color=color, lw=1.8,
                  label=label, zorder=3, solid_capstyle="round")

    # Probe FWHM as a scale bar, so the edge width can be judged against it.
    probe_nm = low_meta["_config"]["beam_fwhm_angstrom"][0] / NM
    bar_centre = half_top
    axis.plot([bar_centre - 0.5 * probe_nm, bar_centre + 0.5 * probe_nm],
              [0.06, 0.06], color=INK, lw=2.4, solid_capstyle="butt", zorder=4)
    axis.annotate(f"probe FWHM {probe_nm:g} nm",
                  xy=(bar_centre, 0.10), ha="center", va="bottom",
                  fontsize=8, color=INK)

    axis.set_xlim(args.edge_window_nm[0], args.edge_window_nm[1])
    axis.set_ylim(-0.05, 1.12)
    axis.set_ylabel("Folded edge profile, scaled", color=INK, fontsize=10)
    axis.set_xlabel("Distance from line centre (nm)", color=INK, fontsize=10)
    axis.set_title(
        "Edge transition: both edges folded together, each scaled to its own peak",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=2,
                loc="upper left")

    subtitle = (
        f"{geometry['top_width'] / NM:g} nm line, "
        f"{geometry['bottom_width'] / NM:.1f} nm base, "
        f"{geometry['height'] / NM:g} nm high on a "
        f"{geometry['membrane_thickness'] / NM:g} nm membrane "
        f"({geometry['total_thickness'] / NM:g} nm total); "
        f"{low_meta['_config']['beam_fwhm_angstrom'][0] / NM:g} nm FWHM probe, "
        f"{low_meta['_config']['primaries_per_pixel']} primaries/pixel. "
        "Bands are +/-1 SEM.  Dashed lines mark the top and base edges."
    )
    figure.text(0.01, 0.005, subtitle, fontsize=8, color=INK_SOFT, ha="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190, bbox_inches="tight",
                   facecolor=SURFACE)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
