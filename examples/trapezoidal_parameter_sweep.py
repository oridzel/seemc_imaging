"""Build a covariance-bearing SEEMC model library for trapezoidal metrology."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from seemc_imaging import (
    PopulationClassifier,
    RasterConfig,
    Sample,
    TrapezoidSweepConfig,
    TrapezoidSweepDriver,
)


def _nm_grid(text):
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated numbers in nm") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("grid values must be finite and positive")
    return values


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
            "0.6.1-era libraries; new work should use causal_lle_v3"
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
    parser.add_argument("--top-widths-nm", type=_nm_grid, default=(48.0, 50.0, 52.0))
    parser.add_argument("--bottom-widths-nm", type=_nm_grid, default=(68.0, 70.0, 72.0))
    parser.add_argument("--heights-nm", type=_nm_grid, default=(45.0, 50.0, 55.0))
    parser.add_argument("--field-width-nm", type=float, default=100.0)
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--primaries-per-pixel", type=int, default=1000)
    parser.add_argument("--beam-fwhm-nm", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("trapezoid_model_library.npz")
    )
    args = parser.parse_args()

    if args.nx < 3 or args.primaries_per_pixel < 2:
        parser.error("--nx must be >=3 and --primaries-per-pixel must be >=2")
    if args.field_width_nm <= 0.0 or args.beam_fwhm_nm < 0.0:
        parser.error("field width must be positive and beam FWHM non-negative")
    if args.lle_max_loss_ev is not None and args.lle_max_loss_frac is not None:
        parser.error(
            "give --lle-max-loss-ev or --lle-max-loss-frac, not both"
        )
    if args.lle_max_loss_ev is not None and args.lle_max_loss_ev < 0.0:
        parser.error("--lle-max-loss-ev must be non-negative")
    if args.lle_max_loss_frac is not None and not 0.0 <= args.lle_max_loss_frac <= 1.0:
        parser.error("--lle-max-loss-frac must lie within [0, 1]")

    x = np.linspace(
        -5.0 * args.field_width_nm,
        5.0 * args.field_width_nm,
        args.nx,
    )
    raster = RasterConfig(
        energy_ev=args.energy_ev,
        x_positions=x,
        y_positions=(0.0,),
        primaries_per_pixel=args.primaries_per_pixel,
        beam_fwhm=10.0 * args.beam_fwhm_nm,
        seed=args.seed,
    )
    sweep = TrapezoidSweepConfig(
        top_widths=tuple(10.0 * value for value in args.top_widths_nm),
        bottom_widths=tuple(10.0 * value for value in args.bottom_widths_nm),
        heights=tuple(10.0 * value for value in args.heights_nm),
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
    library = TrapezoidSweepDriver(
        sample, raster, sweep, classifier
    ).run(
        use_parallel=args.parallel,
        workers=args.workers,
        progress=True,
    )
    output = library.save_npz(args.output)

    summary = output.with_suffix(".models.csv")
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "model_index", "top_width_nm", "bottom_width_nm", "height_nm",
            "sidewall_angle_deg", "completed_primaries",
        ))
        writer.writeheader()
        for index, (top, bottom, height) in enumerate(library.parameters):
            run = 0.5 * (bottom - top)
            angle = 90.0 if run == 0.0 else math.degrees(math.atan2(height, run))
            writer.writerow({
                "model_index": index,
                "top_width_nm": top / 10.0,
                "bottom_width_nm": bottom / 10.0,
                "height_nm": height / 10.0,
                "sidewall_angle_deg": angle,
                "completed_primaries": int(library.completed_primaries[index].sum()),
            })
    print(f"Wrote {len(library.parameters)} models to {output}")
    print(f"Wrote model table to {summary}")


if __name__ == "__main__":
    main()
