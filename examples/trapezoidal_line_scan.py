"""Generate a noisy SEM line scan across one or more analytic trapezoidal lines.

The script keeps the beam direction fixed in the laboratory frame and traces
``n_trajectories`` independent primaries at each raster position.  It writes
yield per incident primary; detector acceptance and SE1/SE2 classifiers are
intentionally deferred to later package phases.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from seemc_imaging import Sample, simulate_trajectory
from seemc_imaging.geometry import TrapezoidalLineArray


def run_line_scan(sample, geometry, energy_ev, x_positions_angstrom,
                  n_trajectories, seed, vacuum_direction=(0.0, 0.0, 1.0)):
    rows = []
    for pixel_id, x in enumerate(x_positions_angstrom):
        launch_hit = geometry.launch_surface(
            float(x), vacuum_direction=vacuum_direction
        )
        outward = tuple(-value for value in launch_hit.normal)
        local_angle = math.acos(max(
            -1.0,
            min(1.0, -float(np.dot(vacuum_direction, outward))),
        ))
        counts = np.zeros((n_trajectories, 5), dtype=float)
        for trajectory_id in range(n_trajectories):
            rng = np.random.default_rng(np.random.SeedSequence(
                [int(seed), int(pixel_id), int(trajectory_id)]
            ))
            result = simulate_trajectory(
                sample,
                float(energy_ev),
                0.0,
                rng,
                geometry=geometry,
                launch_position=launch_hit.position,
                vacuum_direction=vacuum_direction,
                trajectory_id=trajectory_id,
            )
            counts[trajectory_id] = (
                result.tey,
                result.sey_cascade,
                result.bse_cascade,
                result.sey_50ev,
                result.bse_50ev,
            )
        mean = counts.mean(axis=0)
        sem = counts.std(axis=0) / math.sqrt(n_trajectories)
        rows.append({
            "pixel_id": pixel_id,
            "x_nm": float(x) / 10.0,
            "surface_z_nm": launch_hit.position[2] / 10.0,
            "surface_id": launch_hit.surface_id,
            "local_incidence_deg": math.degrees(local_angle),
            "n_trajectories": n_trajectories,
            "tey": mean[0],
            "tey_sem": sem[0],
            "sey_cascade": mean[1],
            "bse_cascade": mean[2],
            "sey_50ev": mean[3],
            "bse_50ev": mean[4],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--material", default="Cu")
    parser.add_argument("--energy-ev", type=float, default=1000.0)
    parser.add_argument("--top-width-nm", type=float, default=50.0)
    parser.add_argument("--bottom-width-nm", type=float, default=70.0)
    parser.add_argument("--height-nm", type=float, default=50.0)
    parser.add_argument("--n-lines", type=int, default=1,
                        help="number of adjacent trapezoidal lines")
    parser.add_argument("--pitch-nm", type=float, default=100.0,
                        help="centre-to-centre line pitch in nm")
    parser.add_argument(
        "--field-width-nm", type=float, default=None,
        help=("total scan width in nm; default automatically covers the "
              "full line array plus 40 nm substrate margin on each side"),
    )
    parser.add_argument("--pixels", type=int, default=101)
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, default=Path("line_scan.csv"))
    args = parser.parse_args()

    if args.pixels < 2 or args.trajectories < 1:
        parser.error("--pixels must be >=2 and --trajectories must be >=1")
    if args.n_lines < 1:
        parser.error("--n-lines must be >=1")
    if args.n_lines > 1 and args.pitch_nm < args.bottom_width_nm:
        parser.error("--pitch-nm must be >= --bottom-width-nm to avoid overlapping lines")

    # SEEMC currently uses Angstrom internally; CLI dimensions are friendlier
    # in nanometres for SEM test structures.
    geometry = TrapezoidalLineArray(
        top_width=10.0 * args.top_width_nm,
        bottom_width=10.0 * args.bottom_width_nm,
        height=10.0 * args.height_nm,
        n_lines=args.n_lines,
        pitch=10.0 * args.pitch_nm,
    )
    sample = Sample(args.material, db_path=args.database)
    array_span_nm = geometry.span / 10.0
    field_width_nm = (
        args.field_width_nm
        if args.field_width_nm is not None
        else array_span_nm + 80.0
    )
    if field_width_nm <= 0.0:
        parser.error("--field-width-nm must be positive")
    if field_width_nm < array_span_nm:
        parser.error(
            f"--field-width-nm={field_width_nm:g} does not cover the full "
            f"line array span of {array_span_nm:g} nm"
        )
    x = np.linspace(
        -5.0 * field_width_nm,
        5.0 * field_width_nm,
        args.pixels,
    )
    rows = run_line_scan(
        sample,
        geometry,
        args.energy_ev,
        x,
        args.trajectories,
        args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    centers_nm = [center / 10.0 for center in geometry.line_centers]
    print(f"Wrote {len(rows)} pixels to {args.output}")
    print(f"Geometry: {args.n_lines} line(s), pitch={args.pitch_nm:g} nm, "
          f"centers_nm={centers_nm}")
    print(f"Scan field: {field_width_nm:g} nm total width")


if __name__ == "__main__":
    main()

