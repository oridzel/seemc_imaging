#!/usr/bin/env python3
"""Calculate TEY, SEY, and BSEY versus primary energy with SEEMC.

This is a command-line front end for ``seemc_imaging.transport.SEEMC``.  It
keeps the material database unchanged: ``--work-function`` is a runtime
surface-barrier override, not a database edit.

Examples
--------
Quick Si check at selected energies::

    python3 examples/yield_vs_energy.py ../MaterialDatabase.pkl \
        --material Si \
        --energies 100 200 500 1000 5000 10000 30000 \
        --work-function 4.8 \
        --barrier-model abrupt \
        --n-trajectories 500 \
        --workers 8 \
        --output Si_yield_quick.csv

Production curve using the JMONSEL-style exponential barrier::

    python3 examples/yield_vs_energy.py ../MaterialDatabase.pkl \
        --material Si \
        --energy-min 50 --energy-max 30000 \
        --energy-points 100 --energy-spacing log \
        --work-function 4.8 \
        --barrier-model expqm --barrier-width 1.0 \
        --elastic-model browning --elastic-cutoff 50 \
        --n-trajectories 10000 \
        --workers 8 \
        --output Si_yield_phi4p8_expqm1A.csv \
        --plot

The incidence angle is measured from the surface normal.  The CSV contains
both the conventional emitted-energy split (SE < cutoff, BSE >= cutoff) and
the cascade/primary ancestry split used internally by SEEMC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import shlex
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TEY, SEY, and BSEY versus energy using the planar "
            "SEEMC transport kernel."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "database",
        type=Path,
        help="MaterialDatabase.pkl path",
    )
    parser.add_argument(
        "--material",
        help="material name exactly as stored in the database",
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="list database materials and exit (does not require --material)",
    )

    energy = parser.add_argument_group("primary-energy grid")
    energy.add_argument(
        "--energies",
        nargs="+",
        type=_positive_float,
        metavar="EV",
        help="explicit primary energies in eV; overrides the generated grid",
    )
    energy.add_argument("--energy-min", type=_positive_float, default=50.0)
    energy.add_argument("--energy-max", type=_positive_float, default=30_000.0)
    energy.add_argument("--energy-points", type=_positive_int, default=80)
    energy.add_argument(
        "--energy-spacing",
        choices=("log", "linear"),
        default="log",
        help="spacing used when --energies is not supplied",
    )

    run = parser.add_argument_group("run control")
    run.add_argument(
        "--angle",
        type=_finite_float,
        default=0.0,
        help="incidence angle in degrees from the surface normal",
    )
    run.add_argument(
        "--n-trajectories",
        "--n-traj",
        "--n-primary",
        dest="n_trajectories",
        type=_positive_int,
        default=2_000,
        help="primary trajectories per energy",
    )
    run.add_argument(
        "--workers",
        type=_nonnegative_int,
        default=1,
        help="worker processes; 1 is serial and 0 uses all detected CPU cores",
    )
    run.add_argument("--seed", type=int, default=12_345)
    run.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show per-energy progress bars",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="load/validate the material and print the resolved setup without transport",
    )

    surface = parser.add_argument_group("surface barrier")
    surface.add_argument(
        "--work-function",
        type=_positive_float,
        default=None,
        metavar="EV",
        help="runtime work-function override; omit to use the database value",
    )
    surface.add_argument(
        "--barrier-model",
        choices=("abrupt", "classical", "expqm"),
        default="abrupt",
    )
    surface.add_argument(
        "--barrier-width",
        "--barrier-width-angstrom",
        type=float,
        default=0.0,
        metavar="ANGSTROM",
        help="expqm barrier width in Angstrom; must be >0 for expqm",
    )
    surface.add_argument(
        "--incoming-barrier-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply reciprocal barrier reflection to incident primaries",
    )

    physics = parser.add_argument_group("transport physics")
    physics.add_argument(
        "--q-unit",
        choices=("auto", "A^-1", "a0^-1"),
        default="auto",
        help=(
            "unit of database q; auto reads metadata and otherwise uses A^-1, "
            "the optlib database convention"
        ),
    )
    physics.add_argument(
        "--elastic-model",
        choices=("elsepa", "browning", "linear"),
        default="elsepa",
        help="low-energy elastic cross-section treatment",
    )
    physics.add_argument(
        "--elastic-cutoff",
        type=_positive_float,
        default=50.0,
        metavar="EV",
    )
    physics.add_argument(
        "--atomic-number",
        type=_positive_float,
        default=None,
        help="effective Z; only needed for Browning when absent from the DB",
    )
    physics.add_argument(
        "--bse-cutoff",
        type=_positive_float,
        default=50.0,
        metavar="EV",
        help="emission-energy boundary between conventional SE and BSE",
    )
    physics.add_argument(
        "--se-channel-rule",
        choices=("mao", "table"),
        default="mao",
    )
    physics.add_argument(
        "--feg-fermi-energy",
        type=_positive_float,
        default=None,
        metavar="EV",
        help="optional Fermi energy used only by binary-encounter SE kinematics",
    )
    physics.add_argument(
        "--on-pauli-block",
        choices=("fallback", "drop"),
        default="fallback",
    )
    physics.add_argument(
        "--imfp-energy-ref",
        choices=("vb_bottom", "fermi"),
        default="vb_bottom",
    )
    physics.add_argument(
        "--emfp-energy-ref",
        choices=("vb_bottom", "vacuum"),
        default="vb_bottom",
    )
    physics.add_argument(
        "--se-direction-model",
        choices=("momentum", "isotropic"),
        default="momentum",
    )
    physics.add_argument(
        "--plasmon-se-direction",
        choices=("momentum", "isotropic"),
        default="isotropic",
    )
    physics.add_argument("--n-q-sample", type=_positive_int, default=64)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output CSV; default is <material>_yield_vs_energy.csv",
    )
    output.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        default=None,
        metavar="PNG",
        help="save a yield plot; omit PNG to use the CSV stem",
    )
    output.add_argument(
        "--show",
        action="store_true",
        help="display the plot interactively (also enables --plot)",
    )
    output.add_argument(
        "--xscale",
        choices=("auto", "linear", "log"),
        default="auto",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of existing CSV/JSON/PNG outputs",
    )
    return parser


def load_database_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        records = data
    else:
        raise ValueError("unrecognized MaterialDatabase.pkl format")
    if not all("name" in item for item in records):
        raise ValueError("every database material must contain a 'name' key")
    return records


def select_record(records: Iterable[dict[str, Any]], name: str) -> dict[str, Any]:
    records = list(records)
    matches = [record for record in records if record.get("name") == name]
    if not matches:
        allowed = ", ".join(str(record.get("name")) for record in records)
        raise ValueError(f"material {name!r} is not in the database; choose from: {allowed}")
    if len(matches) > 1:
        raise ValueError(f"database contains duplicate material name {name!r}")
    return matches[0]


def list_materials(records: Iterable[dict[str, Any]]) -> None:
    print(f"{'material':<18} {'work function (eV)':>20} {'energy table (eV)':>28} {'q unit':>10}")
    print("-" * 82)
    for record in records:
        energy = np.asarray(record.get("energy", []), dtype=float)
        if energy.size:
            energy_range = f"{energy.min():g} .. {energy.max():g}"
        else:
            energy_range = "unknown"
        phi = record.get("work_function")
        phi_text = "not stored" if phi is None else f"{float(phi):g}"
        print(
            f"{str(record['name']):<18} {phi_text:>20} "
            f"{energy_range:>28} {str(record.get('q_unit', 'not stored')):>10}"
        )


def normalize_q_unit(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower().replace("angstrom", "a").replace("å", "a")
    token = token.replace("**", "^").replace(" ", "")
    if token in {"a^-1", "a-1", "1/a", "a^{-1}"}:
        return "A^-1"
    if token in {"a0^-1", "a0-1", "1/a0", "bohr^-1", "bohr-1"}:
        return "a0^-1"
    return None


def resolve_q_unit(requested: str, record: dict[str, Any]) -> tuple[str, str]:
    if requested != "auto":
        return requested, "command line"
    stored = normalize_q_unit(record.get("q_unit"))
    if stored is not None:
        return stored, "database metadata"
    warnings.warn(
        "database has no recognized q_unit metadata; using 'A^-1', the optlib "
        "MaterialDatabase convention. Pass --q-unit a0^-1 if this custom "
        "database stores inverse Bohr.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "A^-1", "optlib fallback"


def build_energy_grid(args: argparse.Namespace) -> np.ndarray:
    if args.energies:
        energies = np.asarray(args.energies, dtype=float)
    else:
        if args.energy_max < args.energy_min:
            raise ValueError("--energy-max must be >= --energy-min")
        if args.energy_points == 1:
            if args.energy_min != args.energy_max:
                raise ValueError(
                    "--energy-points 1 requires equal --energy-min and --energy-max"
                )
            energies = np.asarray([args.energy_min], dtype=float)
        elif args.energy_spacing == "log":
            energies = np.geomspace(args.energy_min, args.energy_max, args.energy_points)
        else:
            energies = np.linspace(args.energy_min, args.energy_max, args.energy_points)
    if not np.all(np.isfinite(energies)) or np.any(energies <= 0.0):
        raise ValueError("all primary energies must be finite and positive")
    # Sorting and de-duplication make the output deterministic for explicit input.
    return np.unique(energies)


def safe_stem(material: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", material.strip()).strip("._")
    return stem or "material"


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    csv_path = args.output or Path(f"{safe_stem(args.material)}_yield_vs_energy.csv")
    csv_path = csv_path.expanduser()
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_suffix(".csv")
    json_path = csv_path.with_suffix(".metadata.json")

    plot_requested = args.plot is not None or args.show
    if not plot_requested:
        png_path = None
    elif args.plot in (None, "auto"):
        png_path = csv_path.with_suffix(".png")
    else:
        png_path = Path(args.plot).expanduser()
        if png_path.suffix.lower() != ".png":
            png_path = png_path.with_suffix(".png")
    return csv_path, json_path, png_path


def ensure_outputs_available(paths: Iterable[Path | None], overwrite: bool) -> None:
    existing = [path for path in paths if path is not None and path.exists()]
    if existing and not overwrite:
        joined = ", ".join(os.fspath(path) for path in existing)
        raise FileExistsError(f"output already exists: {joined}; pass --overwrite to replace")
    for path in paths:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, mc: Any) -> None:
    columns = (
        "energy_eV",
        "n_completed",
        "TEY",
        "TEY_sem",
        "SEY_below_cutoff",
        "SEY_below_cutoff_sem",
        "BSEY_above_cutoff",
        "BSEY_above_cutoff_sem",
        "SEY_cascade",
        "SEY_cascade_sem",
        "BSEY_primary",
        "BSEY_primary_sem",
    )
    rows = zip(
        mc.energy_array,
        mc.n_completed,
        mc.tey,
        mc.tey_err,
        mc.sey_50ev,
        mc.sey_50ev_err,
        mc.bse_50ev,
        mc.bse_50ev_err,
        mc.sey,
        mc.sey_err,
        mc.bse,
        mc.bse_err,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [int(value) if i == 1 else f"{float(value):.12g}" for i, value in enumerate(row)]
            )


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    mc: Any,
    elapsed_seconds: float,
    q_unit_source: str,
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "database": os.fspath(args.database.resolve()),
        "material": mc.sample.name,
        "angle_deg_from_normal": args.angle,
        "n_trajectories_requested_per_energy": args.n_trajectories,
        "seed": args.seed,
        "workers": args.resolved_workers,
        "elapsed_seconds": elapsed_seconds,
        "q_unit_source": q_unit_source,
        "material_parameters": {
            "work_function_database_eV": mc.sample.work_function_db,
            "work_function_used_eV": mc.sample.work_function,
            "fermi_energy_eV": mc.sample.e_fermi,
            "inner_potential_eV": mc.sample.Ui,
        },
        "mc_config": asdict(mc.cfg),
        "energies_eV": mc.energy_array,
        "n_completed": mc.n_completed,
        "diagnostics": dict(mc.diagnostics),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def make_plot(path: Path, mc: Any, args: argparse.Namespace) -> None:
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.errorbar(
        mc.energy_array,
        mc.tey,
        yerr=mc.tey_err,
        marker="o",
        markersize=4,
        capsize=2,
        linewidth=1.4,
        label="TEY",
    )
    ax.errorbar(
        mc.energy_array,
        mc.sey_50ev,
        yerr=mc.sey_50ev_err,
        marker="s",
        markersize=3.5,
        capsize=2,
        linewidth=1.2,
        label=f"SEY (<{mc.cfg.bse_cutoff_ev:g} eV)",
    )
    ax.errorbar(
        mc.energy_array,
        mc.bse_50ev,
        yerr=mc.bse_50ev_err,
        marker="^",
        markersize=3.5,
        capsize=2,
        linewidth=1.2,
        label=f"BSEY (≥{mc.cfg.bse_cutoff_ev:g} eV)",
    )
    use_log = args.xscale == "log" or (
        args.xscale == "auto"
        and mc.energy_array.size > 1
        and mc.energy_array.max() / mc.energy_array.min() >= 20.0
    )
    ax.set_xscale("log" if use_log else "linear")
    ax.set_xlabel("Primary energy (eV)")
    ax.set_ylabel("Yield (electrons / primary)")
    ax.set_title(
        f"{mc.sample.name}, {args.angle:g}°; "
        f"φ={mc.sample.work_function:g} eV, {mc.cfg.barrier_model} barrier"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    if args.show:
        plt.show()
    plt.close(fig)


def print_resolved_setup(args: argparse.Namespace, energies: np.ndarray, mc: Any, source: str) -> None:
    print("Resolved yield calculation")
    print(f"  database:       {args.database.resolve()}")
    print(f"  material:       {mc.sample.name}")
    print(f"  energies:       {energies.size} point(s), {energies.min():g} .. {energies.max():g} eV")
    print(f"  angle:          {args.angle:g} deg from normal")
    print(f"  trajectories:   {args.n_trajectories:,} per energy")
    print(f"  workers:        {args.resolved_workers}")
    print(
        f"  work function:  {mc.sample.work_function:g} eV "
        f"({'CLI override' if args.work_function is not None else 'database'})"
    )
    width = f", width={mc.cfg.barrier_width:g} A" if mc.cfg.barrier_model == "expqm" else ""
    print(f"  barrier:        {mc.cfg.barrier_model}{width}")
    print(f"  elastic model:  {mc.cfg.elastic_low_energy_model} below {mc.cfg.elastic_cutoff_energy:g} eV")
    print(f"  q unit:         {mc.cfg.q_unit} ({source})")
    print(f"  SE/BSE cutoff:  {mc.cfg.bse_cutoff_ev:g} eV")
    print()
    print(mc.sample.consistency_report())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        records = load_database_records(args.database)
        if args.list_materials:
            list_materials(records)
            return 0
        if not args.material:
            parser.error("--material is required unless --list-materials is used")
        if not (0.0 <= args.angle < 90.0):
            parser.error("--angle must satisfy 0 <= angle < 90 degrees")
        if args.barrier_model == "expqm" and args.barrier_width <= 0.0:
            parser.error("--barrier-model expqm requires --barrier-width > 0 Angstrom")
        if args.barrier_model != "expqm" and args.barrier_width < 0.0:
            parser.error("--barrier-width cannot be negative")

        record = select_record(records, args.material)
        q_unit, q_unit_source = resolve_q_unit(args.q_unit, record)
        energies = build_energy_grid(args)

        if args.workers == 0:
            args.resolved_workers = os.cpu_count() or 1
        else:
            args.resolved_workers = args.workers

        # Import after parsing so --help and --list-materials stay lightweight.
        from seemc_imaging.transport import MCConfig, SEEMC

        config = MCConfig(
            imfp_energy_ref=args.imfp_energy_ref,
            emfp_energy_ref=args.emfp_energy_ref,
            q_unit=q_unit,
            elastic_low_energy_model=args.elastic_model,
            elastic_cutoff_energy=args.elastic_cutoff,
            atomic_number=args.atomic_number,
            barrier_model=args.barrier_model,
            barrier_width=args.barrier_width,
            work_function_ev=args.work_function,
            incoming_barrier_reflection=args.incoming_barrier_reflection,
            se_channel_rule=args.se_channel_rule,
            feg_fermi_energy=args.feg_fermi_energy,
            on_pauli_block=args.on_pauli_block,
            se_direction_model=args.se_direction_model,
            plasmon_se_direction=args.plasmon_se_direction,
            bse_cutoff_ev=args.bse_cutoff,
            n_q_sample=args.n_q_sample,
            collect_spectra=False,
        )
        mc = SEEMC(
            energy_array=energies,
            sample_name=args.material,
            angle=math.radians(args.angle),
            n_traj=args.n_trajectories,
            db_path=args.database,
            config=config,
            seed=args.seed,
        )
        if not math.isfinite(mc.sample.work_function) or mc.sample.work_function <= 0.0:
            raise ValueError(
                f"material {args.material!r} has no positive work function in the "
                "database; provide --work-function EV"
            )
        print_resolved_setup(args, energies, mc, q_unit_source)
        if args.dry_run:
            print("Dry run complete; transport was not started.")
            return 0

        csv_path, json_path, png_path = output_paths(args)
        ensure_outputs_available((csv_path, json_path, png_path), args.overwrite)

        started = time.perf_counter()
        mc.run_simulation(
            use_parallel=args.resolved_workers > 1,
            workers=args.resolved_workers,
            progress=args.progress,
            verbose=True,
        )
        elapsed = time.perf_counter() - started

        print()
        print(mc.summary())
        write_csv(csv_path, mc)
        write_metadata(json_path, args, mc, elapsed, q_unit_source)
        if png_path is not None:
            make_plot(png_path, mc, args)

        print()
        print(f"CSV:      {csv_path.resolve()}")
        print(f"Metadata: {json_path.resolve()}")
        if png_path is not None:
            print(f"Plot:     {png_path.resolve()}")
        return 0
    except (FileNotFoundError, FileExistsError, ImportError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
