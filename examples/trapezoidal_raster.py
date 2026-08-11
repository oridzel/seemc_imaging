"""Generate population-resolved SEM raster maps of a trapezoidal line.

The geometry is infinite along y, so the expected signal is a one-dimensional
profile repeated along y.  Independent Monte Carlo noise remains two-
dimensional, making this a useful first raster and noise-validation example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from seemc_imaging import (
    RasterConfig,
    RasterDriver,
    Sample,
    TrapezoidalLine,
)


def _with_extension(prefix, extension):
    return prefix.parent / f"{prefix.name}{extension}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--material", default="Cu")
    parser.add_argument("--energy-ev", type=float, default=1000.0)
    parser.add_argument("--top-width-nm", type=float, default=50.0)
    parser.add_argument("--bottom-width-nm", type=float, default=70.0)
    parser.add_argument("--height-nm", type=float, default=50.0)
    parser.add_argument("--field-width-nm", type=float, default=150.0)
    parser.add_argument("--field-height-nm", type=float, default=50.0)
    parser.add_argument("--nx", type=int, default=101)
    parser.add_argument("--ny", type=int, default=21)
    parser.add_argument("--primaries-per-pixel", type=int, default=100)
    parser.add_argument(
        "--beam-fwhm-nm",
        type=float,
        default=2.0,
        help="Gaussian beam FWHM in the plane normal to the beam",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("trapezoidal_raster")
    )
    args = parser.parse_args()

    positive = {
        "--top-width-nm": args.top_width_nm,
        "--bottom-width-nm": args.bottom_width_nm,
        "--height-nm": args.height_nm,
        "--field-width-nm": args.field_width_nm,
        "--field-height-nm": args.field_height_nm,
    }
    for name, value in positive.items():
        if value <= 0.0:
            parser.error(f"{name} must be positive")
    if args.nx < 2 or args.ny < 1 or args.primaries_per_pixel < 1:
        parser.error("--nx must be >=2; --ny and --primaries-per-pixel must be >=1")
    if args.beam_fwhm_nm < 0.0:
        parser.error("--beam-fwhm-nm must be non-negative")

    # The transport uses Angstrom internally; user-facing geometry is in nm.
    line = TrapezoidalLine(
        top_width=10.0 * args.top_width_nm,
        bottom_width=10.0 * args.bottom_width_nm,
        height=10.0 * args.height_nm,
    )
    x = np.linspace(
        -5.0 * args.field_width_nm,
        5.0 * args.field_width_nm,
        args.nx,
    )
    if args.ny == 1:
        y = np.asarray([0.0])
    else:
        y = np.linspace(
            -5.0 * args.field_height_nm,
            5.0 * args.field_height_nm,
            args.ny,
        )
    config = RasterConfig(
        energy_ev=args.energy_ev,
        x_positions=x,
        y_positions=y,
        primaries_per_pixel=args.primaries_per_pixel,
        beam_fwhm=10.0 * args.beam_fwhm_nm,
        seed=args.seed,
    )
    sample = Sample(args.material, db_path=args.database)
    result = RasterDriver(sample, line, config).run(
        use_parallel=args.parallel,
        workers=args.workers,
        progress=True,
    )

    archive = result.save_npz(_with_extension(args.output_prefix, ".npz"))
    table = result.save_csv(_with_extension(args.output_prefix, ".csv"))
    print(f"Wrote {archive}")
    print(f"Wrote {table}")
    if args.plot:
        figure_path = _with_extension(args.output_prefix, ".png")
        result.plot_channels(
            channels=("sey_50ev", "bse_50ev", "se1", "se2", "bse1", "bse2"),
            path=figure_path,
        )
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
