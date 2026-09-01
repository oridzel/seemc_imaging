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
    PopulationClassifier,
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
    parser.add_argument(
        "--lle-max-loss-ev", "--low-loss-max-ev",
        dest="lle_max_loss_ev", type=float, default=None,
        help=(
            "absolute maximum vacuum energy loss for the LLE primary channel "
            "(default 50 eV; mutually exclusive with --lle-max-loss-frac)"
        ),
    )
    parser.add_argument(
        "--lle-max-loss-frac",
        dest="lle_max_loss_frac", type=float, default=None,
        help=(
            "LLE threshold as a fraction of E0, e.g. 0.02 for E/E0 > 0.98; "
            "use this instead of --lle-max-loss-ev when sweeping beam energy"
        ),
    )
    parser.add_argument(
        "--population-definition",
        choices=("causal_lle_v3", "causal_lle_v2", "branch_v1"),
        default="causal_lle_v3",
        help=(
            "causal_lle_v2 and branch_v1 reproduce archived 0.7.x and "
            "0.6.1-era outputs; new work should use causal_lle_v3"
        ),
    )
    parser.add_argument(
        "--se-reference",
        choices=("launch_surface", "escape_surface"),
        default="launch_surface",
        help=(
            "surface normal used for causal SE1/SE2; escape_surface "
            "reproduces the 0.6.2 classifier"
        ),
    )
    parser.add_argument(
        "--se-parent-rule",
        choices=("root_primary_leg", "immediate_parent"),
        default=None,
        help=(
            "whose direction decides SE1/SE2: the root incident electron's "
            "own leg (default for causal_lle_v3, and the conventional "
            "literature meaning) or the immediate energetic parent, which may "
            "itself be a cascade electron (default for causal_lle_v2)"
        ),
    )
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
        "--record-trajectories",
        action="store_true",
        help="record electron paths for animation (off by default)",
    )
    parser.add_argument(
        "--record-primaries-per-pixel",
        type=int,
        help="record only the first N primaries at each pixel (default: all)",
    )
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=1,
        help="retain every Nth trajectory point, always preserving endpoints",
    )
    parser.add_argument(
        "--trajectory-max-points",
        type=int,
        help="maximum retained points per electron after striding",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        help="trajectory NPZ path (default: PREFIX.trajectories.npz)",
    )
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
    if args.lle_max_loss_ev is not None and args.lle_max_loss_frac is not None:
        parser.error(
            "give --lle-max-loss-ev or --lle-max-loss-frac, not both"
        )
    if args.lle_max_loss_ev is not None and args.lle_max_loss_ev < 0.0:
        parser.error("--lle-max-loss-ev must be non-negative")
    if args.lle_max_loss_frac is not None and not 0.0 <= args.lle_max_loss_frac <= 1.0:
        parser.error("--lle-max-loss-frac must lie within [0, 1]")
    if args.record_primaries_per_pixel is not None:
        if not args.record_trajectories:
            parser.error(
                "--record-primaries-per-pixel requires --record-trajectories"
            )
        if not 1 <= args.record_primaries_per_pixel <= args.primaries_per_pixel:
            parser.error(
                "--record-primaries-per-pixel must be between 1 and "
                "--primaries-per-pixel"
            )
    if args.trajectory_stride < 1:
        parser.error("--trajectory-stride must be positive")
    if args.trajectory_max_points is not None and args.trajectory_max_points < 2:
        parser.error("--trajectory-max-points must be at least 2")

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
        record_trajectories=args.record_trajectories,
        record_primaries_per_pixel=args.record_primaries_per_pixel,
        trajectory_stride=args.trajectory_stride,
        trajectory_max_points=args.trajectory_max_points,
    )
    sample = Sample(args.material, db_path=args.database)
    classifier = PopulationClassifier(
        bse_cutoff_ev=sample.cfg.bse_cutoff_ev,
        lle_max_loss_ev=args.lle_max_loss_ev,
        lle_max_loss_frac=args.lle_max_loss_frac,
        definition=args.population_definition,
        se_reference=args.se_reference,
        se_parent_rule=args.se_parent_rule,
    )
    result = RasterDriver(sample, line, config, classifier).run(
        use_parallel=args.parallel,
        workers=args.workers,
        progress=True,
    )

    archive = result.save_npz(_with_extension(args.output_prefix, ".npz"))
    table = result.save_csv(_with_extension(args.output_prefix, ".csv"))
    print(f"Wrote {archive}")
    print(f"Wrote {table}")
    if args.record_trajectories:
        trajectory_path = args.trajectory_output or _with_extension(
            args.output_prefix, ".trajectories.npz"
        )
        result.save_trajectories_npz(trajectory_path)
        print(f"Wrote {trajectory_path}")
    if args.plot:
        figure_path = _with_extension(args.output_prefix, ".png")
        channels = (
            ("sey_50ev", "bse_50ev", "se1", "se2", "bse1", "bse2")
            if args.population_definition == "branch_v1"
            else (
                "sey_50ev", "bse_50ev", "se1", "se2",
                "lle_primary", "non_lle_primary",
            )
        )
        result.plot_channels(
            channels=channels,
            path=figure_path,
        )
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
