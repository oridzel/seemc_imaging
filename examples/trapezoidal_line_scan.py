"""Generate an animation-ready one-row SEEMC scan across one or more trapezoidal lines.

The scan remains strictly one-dimensional (ny=1).  In addition to the usual
CSV and compact raster NPZ, trajectory recording is enabled by default so the
same run produces PREFIX.trajectories.npz for animate_trapezoidal_scan.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from seemc_imaging import PopulationClassifier, RasterConfig, RasterDriver, Sample
from seemc_imaging.geometry import TrapezoidalLineArray


def _derived_outputs(csv_path: Path):
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_suffix(".csv")
    stem = csv_path.with_suffix("")
    return (
        csv_path,
        Path(f"{stem}.npz"),
        Path(f"{stem}.trajectories.npz"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--material", default="Cu")
    parser.add_argument("--energy-ev", type=float, default=1000.0)

    parser.add_argument("--top-width-nm", type=float, default=50.0)
    parser.add_argument("--bottom-width-nm", type=float, default=70.0)
    parser.add_argument("--height-nm", type=float, default=50.0)
    parser.add_argument("--n-lines", type=int, default=1)
    parser.add_argument(
        "--pitch-nm", type=float, default=100.0,
        help="centre-to-centre pitch of adjacent lines",
    )
    parser.add_argument(
        "--field-width-nm", type=float, default=None,
        help=(
            "total x scan width; default covers the whole line array plus "
            "40 nm of substrate on each side"
        ),
    )

    # Keep the old line-scan naming for compatibility.
    parser.add_argument("--pixels", type=int, default=101)
    parser.add_argument(
        "--trajectories", type=int, default=100,
        help="Monte Carlo primaries per scan pixel",
    )
    parser.add_argument(
        "--beam-fwhm-nm", type=float, default=0.0,
        help="Gaussian beam FWHM; default 0 keeps the historical point beam",
    )
    parser.add_argument("--seed", type=int, default=20260811)

    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int)

    # The line-scan example is intended to be animation-ready, so recording is
    # ON by default.  Only a subset of primaries needs to be retained for a
    # useful movie; all primaries still contribute to the yield statistics.
    parser.add_argument(
        "--record-primaries-per-pixel",
        type=int,
        default=10,
        help=(
            "number of primaries per pixel retained in the trajectory archive "
            "(default 10; all primaries still contribute to yields)"
        ),
    )
    parser.add_argument(
        "--record-all-trajectories",
        action="store_true",
        help="retain every simulated primary in the trajectory archive",
    )
    parser.add_argument(
        "--no-record-trajectories",
        action="store_true",
        help="skip the animation trajectory archive",
    )
    parser.add_argument(
        "--trajectory-stride", type=int, default=1,
        help="retain every Nth trajectory point, preserving endpoints",
    )
    parser.add_argument(
        "--trajectory-max-points", type=int,
        help="maximum retained points per electron after striding",
    )

    parser.add_argument(
        "--output", type=Path, default=Path("line_scan.csv"),
        help=(
            "CSV output path; companion .npz and .trajectories.npz paths are "
            "derived automatically"
        ),
    )
    parser.add_argument("--raster-output", type=Path)
    parser.add_argument("--trajectory-output", type=Path)
    args = parser.parse_args()

    if args.pixels < 2:
        parser.error("--pixels must be >=2")
    if args.trajectories < 1:
        parser.error("--trajectories must be >=1")
    if args.n_lines < 1:
        parser.error("--n-lines must be >=1")
    if args.top_width_nm <= 0 or args.bottom_width_nm <= 0 or args.height_nm <= 0:
        parser.error("trapezoid dimensions must be positive")
    if args.n_lines > 1 and args.pitch_nm < args.bottom_width_nm:
        parser.error(
            "--pitch-nm must be >= --bottom-width-nm to avoid overlapping lines"
        )
    if args.beam_fwhm_nm < 0:
        parser.error("--beam-fwhm-nm must be non-negative")
    if args.trajectory_stride < 1:
        parser.error("--trajectory-stride must be positive")
    if args.trajectory_max_points is not None and args.trajectory_max_points < 2:
        parser.error("--trajectory-max-points must be at least 2")
    if args.record_primaries_per_pixel < 1:
        parser.error("--record-primaries-per-pixel must be positive")

    # User-facing dimensions are nm; SEEMC geometry/raster coordinates are Å.
    geometry = TrapezoidalLineArray(
        top_width=10.0 * args.top_width_nm,
        bottom_width=10.0 * args.bottom_width_nm,
        height=10.0 * args.height_nm,
        n_lines=args.n_lines,
        pitch=10.0 * args.pitch_nm,
    )

    array_span_nm = geometry.span / 10.0
    field_width_nm = (
        float(args.field_width_nm)
        if args.field_width_nm is not None
        else array_span_nm + 80.0
    )
    if field_width_nm <= 0:
        parser.error("--field-width-nm must be positive")
    if field_width_nm < array_span_nm:
        parser.error(
            f"--field-width-nm={field_width_nm:g} does not cover the full "
            f"line-array span of {array_span_nm:g} nm"
        )

    x = np.linspace(
        -5.0 * field_width_nm,
        +5.0 * field_width_nm,
        args.pixels,
    )
    y = np.asarray([0.0])

    record_trajectories = not args.no_record_trajectories
    if record_trajectories:
        if args.record_all_trajectories:
            record_n = args.trajectories
        else:
            record_n = min(args.record_primaries_per_pixel, args.trajectories)
    else:
        record_n = None

    config = RasterConfig(
        energy_ev=args.energy_ev,
        x_positions=x,
        y_positions=y,
        primaries_per_pixel=args.trajectories,
        beam_fwhm=10.0 * args.beam_fwhm_nm,
        seed=args.seed,
        record_trajectories=record_trajectories,
        record_primaries_per_pixel=record_n,
        trajectory_stride=args.trajectory_stride,
        trajectory_max_points=args.trajectory_max_points,
    )

    sample = Sample(args.material, db_path=args.database)
    classifier = PopulationClassifier(
        bse_cutoff_ev=sample.cfg.bse_cutoff_ev,
        definition="causal_lle_v3",
        se_reference="launch_surface",
    )

    result = RasterDriver(sample, geometry, config, classifier).run(
        use_parallel=args.parallel,
        workers=args.workers,
        progress=True,
    )

    csv_path, default_raster_npz, default_trajectory_npz = _derived_outputs(
        args.output
    )
    raster_path = args.raster_output or default_raster_npz
    trajectory_path = args.trajectory_output or default_trajectory_npz

    result.save_csv(csv_path)
    result.save_npz(raster_path)

    print(f"Wrote CSV:              {csv_path}")
    print(f"Wrote raster NPZ:       {raster_path}")

    if record_trajectories:
        result.save_trajectories_npz(trajectory_path)
        print(f"Wrote animation NPZ:    {trajectory_path}")
        print(
            f"Recorded {record_n} of {args.trajectories} primaries/pixel "
            "for animation; all primaries contributed to yields."
        )

    centers_nm = [center / 10.0 for center in geometry.line_centers]
    print(
        f"Geometry: {args.n_lines} line(s), pitch={args.pitch_nm:g} nm, "
        f"centers={centers_nm} nm"
    )
    print(f"Scan: ny=1, nx={args.pixels}, field width={field_width_nm:g} nm")
    print(
        "Physical total-signal channels in the saved archive: "
        "cascade_all=full SE, primary_all=full BSE, tey=SE+BSE."
    )


if __name__ == "__main__":
    main()
