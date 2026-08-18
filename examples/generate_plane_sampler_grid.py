#!/usr/bin/env python3
"""Build angle- and energy-resolved planar SE/BSE sampler tables with SEEMC."""

from __future__ import annotations

import argparse
from pathlib import Path

from seemc_imaging import MCConfig
from seemc_imaging.plane_samplers import (
    DEFAULT_INCIDENCE_ANGLES_DEG,
    JMONSEL_ENERGIES_EV,
    generate_plane_sampler_library,
)


def _grid(values):
    parsed = []
    for value in values:
        parsed.extend(float(item) for item in str(value).split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("grid must contain at least one number")
    return parsed


def _comma(values):
    return ",".join(format(value, "g") for value in values)


def build_parser() -> argparse.ArgumentParser:
    defaults = MCConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run independent planar SEEMC cases and export the six legacy CSV "
            "tables plus event-level joint E/theta/phi direction data in the "
            "raw NPZ checkpoints."
        )
    )
    parser.add_argument("database", type=Path, help="SEEMC MaterialDatabase.pkl")
    parser.add_argument("--material", default="Cu", help="database material name")
    parser.add_argument(
        "--energies-ev", nargs="+", default=[_comma(JMONSEL_ENERGIES_EV)],
        metavar="E", help="comma- or space-separated primary energies",
    )
    parser.add_argument(
        "--angles-deg", nargs="+",
        default=[_comma(DEFAULT_INCIDENCE_ANGLES_DEG)], metavar="A",
        help="comma- or space-separated incidence angles in [0,90)",
    )
    parser.add_argument("--primaries", type=int, default=20_000,
                        help="independent primaries per angle-energy case")
    parser.add_argument("--quantiles", type=int, default=513,
                        help="common cosine-clustered inverse-CDF grid size")
    parser.add_argument("--workers", type=int, default=1,
                        help="spawned workers within each case")
    parser.add_argument("--seed", type=int, default=20260816,
                        help="base seed used to derive stable case seeds")
    parser.add_argument("--output", type=Path,
                        default=Path("sampler_library/Cu_SEEMC"))
    parser.add_argument("--resume", action="store_true",
                        help="reuse matching raw .npz checkpoints")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing case checkpoints")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable per-trajectory progress bars")

    physics = parser.add_argument_group("transport model")
    physics.add_argument(
        "--elastic-low-energy-model",
        choices=("elsepa", "browning", "linear"),
        default=defaults.elastic_low_energy_model,
    )
    physics.add_argument(
        "--elastic-cutoff-ev", type=float,
        default=defaults.elastic_cutoff_energy,
    )
    physics.add_argument(
        "--barrier-model", choices=("abrupt", "classical", "expqm"),
        default=defaults.barrier_model,
    )
    physics.add_argument(
        "--barrier-width-angstrom", type=float, default=defaults.barrier_width,
        help="required only for --barrier-model expqm",
    )
    physics.add_argument(
        "--no-incoming-barrier-reflection", action="store_true",
        help=(
            "disable vacuum->solid quantum/specular reflection of incident "
            "primaries; useful only for comparison with the historical SEEMC path"
        ),
    )
    physics.add_argument(
        "--se-channel-rule", choices=("mao", "table"),
        default=defaults.se_channel_rule,
    )
    physics.add_argument(
        "--feg-fermi-energy-ev", type=float,
        default=defaults.feg_fermi_energy,
    )
    physics.add_argument(
        "--on-pauli-block", choices=("fallback", "drop"),
        default=defaults.on_pauli_block,
    )
    physics.add_argument(
        "--imfp-energy-ref", choices=("vb_bottom", "fermi"),
        default=defaults.imfp_energy_ref,
    )
    physics.add_argument(
        "--emfp-energy-ref", choices=("vb_bottom", "vacuum"),
        default=defaults.emfp_energy_ref,
    )
    physics.add_argument(
        "--se-bse-cutoff-ev", type=float, default=defaults.bse_cutoff_ev,
        help="compatibility split; leave at 50 eV for the six legacy tables",
    )
    return parser


def _build_joint_exports_from_v2_cases(output_dir: Path) -> int:
    """Build RFA-ready joint files from v2 raw checkpoints when available.

    The actual per-case checkpoint is written inside ``plane_samplers.py``.
    This finalizer deliberately refuses to invent phi from a legacy v1 file.
    """
    import numpy as np
    try:
        from seemc_imaging.plane_sampler_joint_export import write_joint_angle_samplers
    except ImportError:
        print(
            "NOTE: plane_sampler_joint_export.py is not installed; legacy CSVs "
            "were written, but joint RFA NPZ aggregation was skipped."
        )
        return 0

    raw = sorted(Path(output_dir).rglob("E_*eV.npz"))
    if not raw:
        return 0

    groups = {}
    legacy = []
    for path in raw:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "se_phi_deg" not in data.files or "bse_phi_deg" not in data.files:
                    legacy.append(path)
                    continue
                angle = float(data["incidence_angle_deg"])
        except Exception:
            continue
        groups.setdefault(angle, []).append(path)

    if legacy:
        print(
            f"WARNING: {len(legacy)} raw checkpoint(s) are still v1 and contain "
            "no phi/direction. Update seemc_imaging/plane_samplers.py to call "
            "plane_sampler_joint_export.save_case_v2(); phi cannot be recovered "
            "from those old files."
        )

    n_written = 0
    for angle, paths in sorted(groups.items()):
        # Put the joint files beside the raw cases. If a project uses a nested
        # raw/ subdirectory, place them one level above it so they sit with the
        # six legacy CSVs.
        parent = paths[0].parent
        target = parent.parent if parent.name.lower() in {"raw", "cases"} else parent
        out = write_joint_angle_samplers(paths, target)
        print(
            f"Joint sampler {angle:g} deg: "
            f"SE={out['SE'].name}, BSE={out['BSE'].name}"
        )
        n_written += 2
    return n_written


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    energies = _grid(args.energies_ev)
    angles = _grid(args.angles_deg)
    config = MCConfig(
        imfp_energy_ref=args.imfp_energy_ref,
        emfp_energy_ref=args.emfp_energy_ref,
        elastic_low_energy_model=args.elastic_low_energy_model,
        elastic_cutoff_energy=args.elastic_cutoff_ev,
        barrier_model=args.barrier_model,
        barrier_width=args.barrier_width_angstrom,
        incoming_barrier_reflection=not args.no_incoming_barrier_reflection,
        se_channel_rule=args.se_channel_rule,
        feg_fermi_energy=args.feg_fermi_energy_ev,
        on_pauli_block=args.on_pauli_block,
        bse_cutoff_ev=args.se_bse_cutoff_ev,
        collect_spectra=True,
    )
    config.validate()
    cases = generate_plane_sampler_library(
        args.database,
        args.output,
        material=args.material,
        energies_ev=energies,
        incidence_angles_deg=angles,
        n_primaries=args.primaries,
        probability_count=args.quantiles,
        config=config,
        base_seed=args.seed,
        workers=args.workers,
        resume=args.resume,
        overwrite=args.overwrite,
        progress=not args.no_progress,
    )
    n_joint = _build_joint_exports_from_v2_cases(args.output)
    print(
        f"Wrote {len(cases)} cases, {len(angles)} angle directories, and "
        f"a manifest under {args.output}; joint sampler files written: {n_joint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
