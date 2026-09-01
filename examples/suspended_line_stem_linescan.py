"""Line scan across a trapezoidal Si line on a suspended membrane.

Records the reflected (SEM) and transmitted (STEM) signals in one pass, at one
or more landing energies, and writes a raster archive plus a comparison figure
per energy.

Default geometry follows the measured specimen: a 50 nm-wide line with a
near-vertical sidewall on a free-standing Si membrane, 130 nm from the line top
to the membrane underside.

Example
-------
    python examples/suspended_line_stem_linescan.py MaterialDatabase.pkl \\
      --material Si --energies-kv 0.9,30 \\
      --top-width-nm 50 --sidewall-deg 2 \\
      --line-height-nm 30 --membrane-thickness-nm 100 \\
      --beam-fwhm-nm 1.6 --field-width-nm 120 --nx 241 \\
      --primaries-per-pixel 400 --parallel \\
      --output-prefix si_line
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from seemc_imaging import (
    PopulationClassifier,
    RasterConfig,
    RasterDriver,
    Sample,
    SuspendedTrapezoidalLine,
    TransmissionDetector,
)

NM = 10.0  # Angstrom per nanometre


def _csv_floats(text):
    values = tuple(float(item) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated number list")
    return values


def bottom_width_nm(top_width_nm, line_height_nm, sidewall_deg):
    """Base width of a line whose sidewalls lean out by ``sidewall_deg``.

    The angle is measured from vertical, so zero gives a rectangular line and
    the base grows by ``2 * height * tan(angle)``.
    """
    return top_width_nm + 2.0 * line_height_nm * math.tan(math.radians(sidewall_deg))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database", type=Path)
    parser.add_argument("--material", default="Si")
    parser.add_argument(
        "--energies-kv", type=_csv_floats, default=(0.9, 30.0),
        help="landing energies in kV, comma separated (default 0.9,30)",
    )

    geometry = parser.add_argument_group("specimen")
    geometry.add_argument("--top-width-nm", type=float, default=50.0)
    geometry.add_argument(
        "--sidewall-deg", type=float, default=2.0,
        help="sidewall angle from vertical; 0 gives a rectangular line",
    )
    geometry.add_argument("--line-height-nm", type=float, default=30.0)
    geometry.add_argument(
        "--membrane-thickness-nm", type=float, default=100.0,
        help="free-standing support below the line base",
    )

    beam = parser.add_argument_group("beam and scan")
    beam.add_argument(
        "--beam-fwhm-nm", type=float, default=1.6,
        help="probe FWHM; the fitted 5 nm sigma is an effective edge spread, "
             "not a probe size, so do not put it here",
    )
    beam.add_argument("--field-width-nm", type=float, default=120.0)
    beam.add_argument("--nx", type=int, default=241)
    beam.add_argument("--primaries-per-pixel", type=int, default=400)
    beam.add_argument("--seed", type=int, default=20250831)

    detector = parser.add_argument_group("transmitted detector")
    detector.add_argument("--bf-max-mrad", type=float, default=10.0)
    detector.add_argument("--adf-max-mrad", type=float, default=50.0)
    detector.add_argument("--haadf-max-mrad", type=float, default=200.0)

    run = parser.add_argument_group("run")
    run.add_argument("--lle-max-loss-frac", type=float, default=0.02,
                     help="low-loss window as a fraction of E0 (default 2%%, "
                          "i.e. E/E0 > 0.98); scales correctly across energies")
    run.add_argument("--parallel", action="store_true")
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--output-prefix", type=Path, default=Path("suspended_line"))
    run.add_argument("--no-plot", action="store_true")
    return parser


def validate(parser, args):
    for name in ("top_width_nm", "line_height_nm", "membrane_thickness_nm",
                 "field_width_nm"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.beam_fwhm_nm < 0.0:
        parser.error("--beam-fwhm-nm must be non-negative")
    if not 0.0 <= args.sidewall_deg < 90.0:
        parser.error("--sidewall-deg must lie in [0, 90)")
    if args.nx < 2 or args.primaries_per_pixel < 1:
        parser.error("--nx must be >= 2 and --primaries-per-pixel >= 1")
    if any(value <= 0.0 for value in args.energies_kv):
        parser.error("--energies-kv must be positive")
    bottom = bottom_width_nm(args.top_width_nm, args.line_height_nm,
                             args.sidewall_deg)
    if bottom >= args.field_width_nm:
        parser.error(
            f"the line base is {bottom:.1f} nm wide but the field is only "
            f"{args.field_width_nm:.1f} nm; widen the field so the scan "
            "reaches bare membrane on both sides"
        )


def run_one_energy(sample, geometry, args, energy_ev):
    x_half = 0.5 * args.field_width_nm * NM
    config = RasterConfig(
        energy_ev=energy_ev,
        x_positions=np.linspace(-x_half, x_half, args.nx),
        y_positions=(0.0,),
        primaries_per_pixel=args.primaries_per_pixel,
        beam_fwhm=args.beam_fwhm_nm * NM,
        seed=args.seed,
    )
    classifier = PopulationClassifier(
        bse_cutoff_ev=sample.cfg.bse_cutoff_ev,
        lle_max_loss_frac=args.lle_max_loss_frac,
        transmission=TransmissionDetector(
            bf_max_mrad=args.bf_max_mrad,
            adf_max_mrad=args.adf_max_mrad,
            haadf_max_mrad=args.haadf_max_mrad,
        ),
    )
    driver = RasterDriver(sample, geometry, config, classifier)
    return driver.run(use_parallel=args.parallel, workers=args.workers,
                      progress=True)


def plot_energy(result, geometry, path, energy_ev, beam_fwhm_nm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_nm = np.asarray(result.x_positions, dtype=float) / NM
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.0, 7.2), sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0)},
    )

    for channel, label in (
        ("se1_lt50", "SE1 (E < 50 eV)"),
        ("se2_lt50", "SE2 (E < 50 eV)"),
        ("back_lle_primary", "Low-loss primary, reflected"),
        ("back_non_lle_primary", "Non-LLE primary, reflected"),
    ):
        top.plot(x_nm, result.yield_maps[channel][0], linewidth=1.5, label=label)
    top.set_ylabel("Yield per primary")
    top.set_title(
        f"{energy_ev / 1000.0:g} kV, {beam_fwhm_nm:g} nm FWHM probe  "
        f"({geometry.top_width / NM:g} nm line, "
        f"{geometry.total_thickness / NM:g} nm total thickness)"
    )
    top.legend(fontsize=8, ncol=2)
    top.grid(alpha=0.25)

    for channel, label in (
        ("fwd_bf", "BF"),
        ("fwd_adf", "ADF"),
        ("fwd_haadf", "HAADF"),
        ("forward_all", "All transmitted"),
    ):
        bottom.plot(x_nm, result.yield_maps[channel][0], linewidth=1.5, label=label)
    bottom.set_ylabel("Yield per primary")
    bottom.set_xlabel("Beam position (nm)")
    bottom.legend(fontsize=8, ncol=2)
    bottom.grid(alpha=0.25)

    # Mark the nominal line edges at the top and base.
    for axis in (top, bottom):
        for edge in (0.5 * geometry.top_width / NM, 0.5 * geometry.bottom_width / NM):
            axis.axvline(edge, color="0.6", linewidth=0.8, linestyle="--")
            axis.axvline(-edge, color="0.6", linewidth=0.8, linestyle="--")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)

    bottom_nm = bottom_width_nm(args.top_width_nm, args.line_height_nm,
                                args.sidewall_deg)
    geometry = SuspendedTrapezoidalLine(
        top_width=args.top_width_nm * NM,
        bottom_width=bottom_nm * NM,
        height=args.line_height_nm * NM,
        membrane_thickness=args.membrane_thickness_nm * NM,
    )
    sample = Sample(args.material, db_path=args.database)

    print(f"Material {sample.name!r}: tabulated "
          f"{sample.Emin:g}-{sample.Emax:g} eV")
    print(f"Line: {args.top_width_nm:g} nm top, {bottom_nm:.2f} nm base "
          f"({args.sidewall_deg:g} deg sidewall), {args.line_height_nm:g} nm high")
    print(f"Membrane: {args.membrane_thickness_nm:g} nm; total thickness "
          f"{geometry.total_thickness / NM:g} nm")
    print(f"Scan: {args.field_width_nm:g} nm field, {args.nx} points "
          f"({args.field_width_nm / (args.nx - 1):.3f} nm step), "
          f"{args.primaries_per_pixel} primaries per pixel")

    for energy_kv in args.energies_kv:
        energy_ev = energy_kv * 1000.0
        tag = f"{energy_kv:g}kV".replace(".", "p")
        print(f"\n=== {energy_kv:g} kV ===")
        result = run_one_energy(sample, geometry, args, energy_ev)
        prefix = Path(f"{args.output_prefix}_{tag}")
        archive = result.save_npz(prefix.with_suffix(".npz"))
        table = result.save_csv(prefix.with_suffix(".csv"))
        print(f"Wrote {archive}")
        print(f"Wrote {table}")

        centre = args.nx // 2
        transmitted = float(result.yield_maps["forward_all"][0][centre])
        reflected = float(result.yield_maps["backward_all"][0][centre])
        print(f"  on-line yields: reflected {reflected:.4f}, "
              f"transmitted {transmitted:.4f} per primary")
        if transmitted == 0.0:
            print("  no transmitted signal at this energy: the beam does not "
                  "penetrate the full thickness, which is the expected result "
                  "at low kV and is why the STEM channels are informative only "
                  "at high kV.")
        if not args.no_plot:
            figure_path = prefix.with_suffix(".png")
            plot_energy(result, geometry, figure_path, energy_ev,
                        args.beam_fwhm_nm)
            print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
