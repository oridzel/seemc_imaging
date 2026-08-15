"""Animate a recorded one-row SEEMC raster over a trapezoidal line."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from seemc_imaging import RasterTrajectoryArchive
from seemc_imaging.animation import animate_trapezoidal_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--output", type=Path, default=Path("trapezoid_scan.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames-per-pixel", type=int, default=16)
    parser.add_argument("--pause-frames", type=int, default=4)
    parser.add_argument(
        "--pixel-stride", type=int, default=1,
        help="animate every Nth recorded beam position",
    )
    parser.add_argument(
        "--color-by", choices=("energy", "population"), default="energy"
    )
    parser.add_argument("--tail-fraction", type=float, default=0.45)
    parser.add_argument("--vacuum-flight-nm", type=float, default=35.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title")
    parser.add_argument(
        "--profile-channels",
        default="populations",
        help=(
            "lower-panel channels: populations (default), conventional, "
            "tey_se_bse, or a comma-separated channel list"
        ),
    )
    args = parser.parse_args()

    if args.fps < 1:
        parser.error("--fps must be positive")
    if args.frames_per_pixel < 2:
        parser.error("--frames-per-pixel must be at least 2")
    if args.pause_frames < 0:
        parser.error("--pause-frames must be non-negative")
    if args.pixel_stride < 1:
        parser.error("--pixel-stride must be positive")
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must lie in (0, 1]")
    if args.vacuum_flight_nm < 0.0:
        parser.error("--vacuum-flight-nm must be non-negative")

    archive = RasterTrajectoryArchive.load_npz(args.trajectories)
    output = animate_trapezoidal_scan(
        archive,
        args.output,
        fps=args.fps,
        frames_per_pixel=args.frames_per_pixel,
        pause_frames=args.pause_frames,
        pixel_stride=args.pixel_stride,
        color_by=args.color_by,
        tail_fraction=args.tail_fraction,
        vacuum_flight_nm=args.vacuum_flight_nm,
        dpi=args.dpi,
        title=args.title,
        profile_channels=args.profile_channels,
    )
    duration = (
        len(np.unique(archive.cascade_pixel_id)[::args.pixel_stride])
        * (args.frames_per_pixel + args.pause_frames)
        / args.fps
    )
    print(f"Wrote {output} ({duration:.1f} s)")


if __name__ == "__main__":
    main()
