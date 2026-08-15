"""Fit a one-dimensional raster NPZ to a SEEMC trapezoid model library."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from seemc_imaging import (
    DEFAULT_CHANNEL_SETS,
    LEGACY_CHANNEL_SETS,
    PARAMETER_NAMES,
    ProfileFitter,
    ProfileObservation,
    TrapezoidModelLibrary,
    compare_channel_information,
)


def _channels(text, available):
    available = set(available)
    for choices in (DEFAULT_CHANNEL_SETS, LEGACY_CHANNEL_SETS):
        if text in choices and set(choices[text]).issubset(available):
            return choices[text]
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if values and set(values).issubset(available):
        return values
    raise ValueError(
        f"channel set {text!r} is unavailable; archive channels are "
        f"{sorted(available)}"
    )


def _prefix_path(prefix, suffix):
    return prefix.parent / f"{prefix.name}{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("observation", type=Path)
    parser.add_argument(
        "--channels", default="all_disjoint",
        help="named set (all_disjoint, low_energy_pair, energy_loss_pair, "
             "energy_cut_pair, or legacy_branch_v1) or comma-separated channels",
    )
    parser.add_argument("--shift-range-nm", type=float, default=2.0)
    parser.add_argument("--shift-step-nm", type=float, default=0.1)
    parser.add_argument("--no-fit-scale", action="store_true")
    parser.add_argument("--fit-channel-offsets", action="store_true")
    parser.add_argument("--ignore-model-covariance", action="store_true")
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="override material, beam, or classifier compatibility errors",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=Path("trapezoid_fit"))
    args = parser.parse_args()

    if args.shift_range_nm < 0.0 or args.shift_step_nm <= 0.0:
        parser.error("shift range must be non-negative and shift step positive")
    count = int(round(2.0 * args.shift_range_nm / args.shift_step_nm)) + 1
    shifts = np.linspace(-10.0 * args.shift_range_nm, 10.0 * args.shift_range_nm, count)

    library = TrapezoidModelLibrary.from_npz(args.library)
    observation = ProfileObservation.from_npz(args.observation)
    try:
        channels = _channels(args.channels, library.channels)
    except ValueError as exc:
        parser.error(str(exc))
    fitter = ProfileFitter(library)
    fit = fitter.fit(
        observation,
        channels=channels,
        shift_values=shifts,
        fit_scale=not args.no_fit_scale,
        fit_channel_offsets=args.fit_channel_offsets,
        include_model_covariance=not args.ignore_model_covariance,
        allow_incompatible=args.allow_incompatible,
    )
    fit_path = fit.save_json(_prefix_path(args.output_prefix, ".fit.json"))

    score_path = _prefix_path(args.output_prefix, ".scores.csv")
    with score_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("model_index", "top_width_nm", "bottom_width_nm", "height_nm",
                  "best_shift_nm", "chi_square")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, parameters in enumerate(library.parameters):
            writer.writerow({
                "model_index": index,
                "top_width_nm": parameters[0] / 10.0,
                "bottom_width_nm": parameters[1] / 10.0,
                "height_nm": parameters[2] / 10.0,
                "best_shift_nm": fit.model_shifts[index] / 10.0,
                "chi_square": fit.model_scores[index],
            })

    reports = compare_channel_information(
        library,
        reference_parameters=fit.best_parameters,
        fit_scale=not args.no_fit_scale,
        fit_channel_offsets=args.fit_channel_offsets,
        covariance_multiplier=2.0 if not args.ignore_model_covariance else 1.0,
    )
    information_path = _prefix_path(args.output_prefix, ".information.json")
    information_path.write_text(
        json.dumps({name: report.to_dict() for name, report in reports.items()}, indent=2),
        encoding="utf-8",
    )
    information_csv = _prefix_path(args.output_prefix, ".information.csv")
    with information_csv.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "channel_set", "channels", "top_width_sem_nm",
            "bottom_width_sem_nm", "height_sem_nm", "rank", "condition_number",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, report in reports.items():
            errors = report.parameter_standard_errors / 10.0
            def available(value):
                return float(value) if np.isfinite(value) else ""
            writer.writerow({
                "channel_set": name,
                "channels": ",".join(report.channels),
                "top_width_sem_nm": available(errors[0]),
                "bottom_width_sem_nm": available(errors[1]),
                "height_sem_nm": available(errors[2]),
                "rank": report.rank,
                "condition_number": report.condition_number,
            })

    print(json.dumps(fit.to_dict(), indent=2))
    print(f"Wrote {fit_path}")
    print(f"Wrote {score_path}")
    print(f"Wrote {information_path}")
    print(f"Wrote {information_csv}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("--plot requires matplotlib") from exc
        selected = observation.select(fit.channels)
        prediction = fitter.predict(observation, fit)
        figure, axes = plt.subplots(
            len(fit.channels), 1, figsize=(8.0, 2.3 * len(fit.channels)),
            sharex=True, squeeze=False,
        )
        for axis, channel, measured, modeled in zip(
                axes[:, 0], fit.channels, selected.yields, prediction):
            axis.plot(observation.x_positions / 10.0, measured, label="observation")
            axis.plot(observation.x_positions / 10.0, modeled, label="best model")
            axis.set_ylabel(channel)
            axis.grid(alpha=0.2)
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel("x (nm)")
        title = ", ".join(
            f"{name}={value / 10.0:g} nm"
            for name, value in zip(PARAMETER_NAMES, fit.best_parameters)
        )
        figure.suptitle(title)
        figure.tight_layout()
        plot_path = _prefix_path(args.output_prefix, ".png")
        figure.savefig(plot_path, dpi=180, bbox_inches="tight")
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
