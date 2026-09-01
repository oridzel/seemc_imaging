"""Animation of a recorded one-dimensional SEM scan over a trapezoidal line."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .trajectory import RasterTrajectoryArchive


_REST_ENERGY_EV = 510_998.95069
_C_ANGSTROM_PER_FS = 2_997.92458

POPULATION_COLORS = {
    "se1": "#39d98a",
    "se2": "#f9c74f",
    "se1_lt50": "#39d98a",
    "se1_ge50": "#1f9d63",
    "se2_lt50": "#f9c74f",
    "se2_ge50": "#f9844a",
    "fast_cascade_ge50": "#f9844a",
    "lle_primary": "#43aaef",
    "non_lle_primary": "#c77dff",
    "lle_bse": "#43aaef",
    "non_lle_bse": "#c77dff",
    "bse1": "#43aaef",
    "bse2": "#c77dff",
    "cascade_absorbed": "#98a2b3",
    "primary_absorbed": "#e5e7eb",
    # Reflected hemisphere keeps the causal colours.
    "back_se1_lt50": "#39d98a",
    "back_se1_ge50": "#1f9d63",
    "back_se2_lt50": "#f9c74f",
    "back_se2_ge50": "#f9844a",
    "back_lle_primary": "#43aaef",
    "back_non_lle_primary": "#c77dff",
    # Transmitted hemisphere: a cool ramp from the BF disc outwards.
    "fwd_bf": "#e0f2fe",
    "fwd_adf": "#7dd3fc",
    "fwd_haadf": "#0ea5e9",
    "fwd_beyond_haadf": "#1e3a8a",
}

# Emitted primaries are labelled "primary", not "BSE".  The non-LLE class is
# the complement of LLE among *all* emitted original incident electrons, so it
# also holds primaries emitted below the 50 eV cut, which the conventional
# partition counts as SEs rather than BSEs.
PROFILE_STYLES = {
    "tey": ("TEY", "#f8fafc"),
    "sey_50ev": ("SE, E < 50 eV", "#39d98a"),
    "bse_50ev": ("BSE, E ≥ 50 eV", "#43aaef"),
    "se1": ("SE1 (all E)", "#39d98a"),
    "se2": ("SE2 (all E)", "#f9c74f"),
    "se1_lt50": ("SE1, E < 50 eV", "#39d98a"),
    "se1_ge50": ("SE1, E ≥ 50 eV", "#1f9d63"),
    "se2_lt50": ("SE2, E < 50 eV", "#f9c74f"),
    "se2_ge50": ("SE2, E ≥ 50 eV", "#f9844a"),
    "fast_cascade_ge50": ("Fast cascade", "#f9844a"),
    "lle_primary": ("Low-loss primary", "#43aaef"),
    "non_lle_primary": ("Non-LLE primary", "#c77dff"),
    "first_event_backscatter": ("First-event backscatter", "#7dd3fc"),
    "later_return_primary": ("Later-return primary", "#c77dff"),
    "barrier_reflected_primary": ("Barrier-reflected primary", "#94a3b8"),
    "lle_bse": ("Low-loss primary", "#43aaef"),
    "non_lle_bse": ("Non-LLE primary", "#c77dff"),
    "bse1": ("BSE1", "#43aaef"),
    "bse2": ("BSE2", "#c77dff"),
    "backward_all": ("Reflected (all)", "#39d98a"),
    "forward_all": ("Transmitted (all)", "#0ea5e9"),
    "forward_primary_all": ("Transmitted primaries", "#0369a1"),
    "forward_cascade_all": ("Transmitted SE", "#7dd3fc"),
    "back_se1_lt50": ("SE1, E < 50 eV (refl.)", "#39d98a"),
    "back_se1_ge50": ("SE1, E \u2265 50 eV (refl.)", "#1f9d63"),
    "back_se2_lt50": ("SE2, E < 50 eV (refl.)", "#f9c74f"),
    "back_se2_ge50": ("SE2, E \u2265 50 eV (refl.)", "#f9844a"),
    "back_lle_primary": ("Low-loss primary (refl.)", "#43aaef"),
    "back_non_lle_primary": ("Non-LLE primary (refl.)", "#c77dff"),
    "fwd_bf": ("BF", "#e0f2fe"),
    "fwd_adf": ("ADF", "#7dd3fc"),
    "fwd_haadf": ("HAADF", "#0ea5e9"),
    "fwd_beyond_haadf": ("Beyond HAADF", "#1e3a8a"),
    "fwd_bf_primary": ("BF primaries", "#0369a1"),
    "fwd_adf_primary": ("ADF primaries", "#0284c7"),
    "fwd_haadf_primary": ("HAADF primaries", "#075985"),
}

PROFILE_PRESETS = {
    "populations": ("se1", "se2", "lle_primary", "non_lle_primary"),
    "populations_disjoint": (
        "se1_lt50", "se1_ge50", "se2_lt50", "se2_ge50",
        "lle_primary", "non_lle_primary",
    ),
    "v062_populations": ("se1", "se2", "lle_bse", "non_lle_bse"),
    "legacy_populations": ("se1", "se2", "bse1", "bse2"),
    "conventional": ("sey_50ev", "bse_50ev"),
    "tey_se_bse": ("tey", "sey_50ev", "bse_50ev"),
    "stem": ("fwd_bf", "fwd_adf", "fwd_haadf", "fwd_beyond_haadf"),
    "stem_rings": ("fwd_bf", "fwd_adf", "fwd_haadf"),
    "hemispheres": ("backward_all", "forward_all"),
    "se_and_stem": (
        "se1", "se2", "fwd_bf", "fwd_adf", "fwd_haadf",
    ),
}


def _flight_time_fs(distance_angstrom, energy_ev):
    if distance_angstrom <= 0.0 or energy_ev <= 0.0:
        return 0.0
    gamma = 1.0 + energy_ev / _REST_ENERGY_EV
    beta_squared = max(1.0 - 1.0 / (gamma * gamma), 0.0)
    if beta_squared == 0.0:
        return 0.0
    return distance_angstrom / (_C_ANGSTROM_PER_FS * math.sqrt(beta_squared))


def _collapse_equal_times(points):
    """Keep the last state at an instantaneous collision/barrier event."""
    if len(points) < 2:
        return points
    reverse_unique = np.unique(points[::-1, 4], return_index=True)[1]
    keep = np.sort(len(points) - 1 - reverse_unique)
    return points[keep]


def _display_track(archive, electron_index, vacuum_flight_nm):
    points = _collapse_equal_times(archive.electron_points(electron_index))
    if len(points) == 0:
        return points
    points = np.array(points, dtype=float, copy=True)
    if str(archive.electron_fate[electron_index]) == "emitted":
        direction = archive.electron_final_direction[electron_index]
        distance = 10.0 * float(vacuum_flight_nm)
        terminal = points[-1].copy()
        terminal[:3] += distance * direction
        terminal[3] = archive.electron_final_energy_ev[electron_index]
        terminal[4] += _flight_time_fs(distance, terminal[3])
        points = np.vstack((points, terminal))
    return points


_ANIMATABLE_GEOMETRIES = ("TrapezoidalLine", "SuspendedTrapezoidalLine")


def _geometry_values(archive):
    geometry = archive.metadata.get("geometry", {})
    if geometry.get("type") not in _ANIMATABLE_GEOMETRIES:
        raise ValueError(
            "trapezoidal scan animation requires one of "
            f"{_ANIMATABLE_GEOMETRIES} in the archive geometry metadata"
        )
    required = ("top_width", "bottom_width", "height", "center_x", "substrate_z")
    missing = [name for name in required if name not in geometry]
    if missing:
        raise ValueError(f"trajectory archive is missing geometry values: {missing}")
    values = {name: float(geometry[name]) / 10.0 for name in required}
    # A suspended membrane has a finite underside; a bulk substrate does not.
    thickness = geometry.get("membrane_thickness")
    values["membrane_thickness"] = (
        None if thickness is None else float(thickness) / 10.0
    )
    return values


def _trapezoid_surface_height_nm(
        x_nm, *, top_width, bottom_width, height, center_x,
        substrate_height):
    """Return the nominal normal-incidence beam intersection height."""
    distance = abs(float(x_nm) - float(center_x))
    half_top = 0.5 * float(top_width)
    half_bottom = 0.5 * float(bottom_width)
    if distance <= half_top:
        return float(substrate_height) + float(height)
    if distance >= half_bottom or half_bottom <= half_top:
        return float(substrate_height)
    side_fraction = (half_bottom - distance) / (half_bottom - half_top)
    return float(substrate_height) + float(height) * side_fraction


def animate_trapezoidal_scan(
        archive, output, *, fps=30, frames_per_pixel=16, pause_frames=4,
        pixel_stride=1, color_by="energy", tail_fraction=0.45,
        vacuum_flight_nm=35.0, dpi=150, title=None,
        profile_channels="populations"):
    """Render an MP4 or GIF from a :class:`RasterTrajectoryArchive`.

    Physical femtosecond timing is preserved within each independently
    simulated primary cascade.  Multiple primaries recorded at one raster
    position are displayed together as a visualization ensemble; their
    absolute start times are not an experimental beam-current timebase.
    """
    if not isinstance(archive, RasterTrajectoryArchive):
        archive = RasterTrajectoryArchive.load_npz(archive)
    archive.validate()
    if len(archive.y_angstrom) != 1:
        raise ValueError("animation currently requires a one-row raster (--ny 1)")
    if archive.n_cascades == 0:
        raise ValueError("trajectory archive contains no recorded cascades")
    frames_per_pixel = int(frames_per_pixel)
    pause_frames = int(pause_frames)
    pixel_stride = int(pixel_stride)
    fps = int(fps)
    if frames_per_pixel < 2 or pause_frames < 0 or pixel_stride < 1 or fps < 1:
        raise ValueError("invalid frame, pause, pixel-stride, or fps setting")
    if color_by not in {"energy", "population"}:
        raise ValueError("color_by must be 'energy' or 'population'")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if isinstance(profile_channels, str):
        if profile_channels in PROFILE_PRESETS:
            requested_preset = profile_channels
            profile_channels = PROFILE_PRESETS[requested_preset]
            available = {str(value) for value in archive.profile_channels}
            if (requested_preset == "populations"
                    and not set(profile_channels).issubset(available)
                    and set(PROFILE_PRESETS["v062_populations"]).issubset(
                        available
                    )):
                profile_channels = PROFILE_PRESETS["v062_populations"]
            if (requested_preset == "populations"
                    and not set(profile_channels).issubset(available)
                    and set(PROFILE_PRESETS["legacy_populations"]).issubset(
                        available
                    )):
                profile_channels = PROFILE_PRESETS["legacy_populations"]
        else:
            profile_channels = tuple(
                value.strip() for value in profile_channels.split(",")
                if value.strip()
            )
    else:
        profile_channels = tuple(str(value) for value in profile_channels)
    if not profile_channels:
        raise ValueError("profile_channels must contain at least one channel")
    if len(profile_channels) > 6:
        raise ValueError("profile_channels supports at most six channels")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LogNorm, to_rgba
        from matplotlib.cm import ScalarMappable
        from matplotlib.patches import Polygon, Rectangle
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "animation requires matplotlib; install seemc-imaging[animation]"
        ) from exc

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        raise ValueError("animation output must end in .mp4 or .gif")

    geometry = _geometry_values(archive)
    top_width = geometry["top_width"]
    bottom_width = geometry["bottom_width"]
    height = geometry["height"]
    center_x = geometry["center_x"]
    substrate_height = -geometry["substrate_z"]
    x_nm = archive.x_angstrom / 10.0

    available_pixels = np.unique(archive.cascade_pixel_id)[::pixel_stride]
    cascades_by_pixel = {
        int(pixel_id): archive.cascades_at_pixel(pixel_id)
        for pixel_id in available_pixels
    }
    tracks = {}
    pixel_duration = {}
    for pixel_id, cascade_indices in cascades_by_pixel.items():
        duration = 0.0
        for cascade_index in cascade_indices:
            for electron_index in archive.cascade_electron_indices(cascade_index):
                track = _display_track(archive, electron_index, vacuum_flight_nm)
                tracks[electron_index] = track
                if len(track):
                    duration = max(duration, float(track[-1, 4]))
        pixel_duration[pixel_id] = max(duration, 1e-12)

    max_energy = max(float(archive.metadata["config"]["energy_ev"]), 1.0)
    positive_energy = archive.points[:, 3][archive.points[:, 3] > 0.0]
    min_energy = max(
        min(float(np.percentile(positive_energy, 1.0)), 1.0)
        if len(positive_energy) else 0.1,
        0.1,
    )
    energy_norm = LogNorm(vmin=min_energy, vmax=max(max_energy, min_energy * 1.01))
    energy_cmap = plt.get_cmap("turbo")

    figure = plt.figure(figsize=(11.5, 7.4), facecolor="#070b12")
    grid = figure.add_gridspec(4, 1, height_ratios=(1, 1, 1, 0.82), hspace=0.08)
    axis = figure.add_subplot(grid[:3, 0])
    profile_axis = figure.add_subplot(grid[3, 0], sharex=axis)
    for current_axis in (axis, profile_axis):
        current_axis.set_facecolor("#070b12")
        current_axis.tick_params(colors="#cbd5e1")
        for spine in current_axis.spines.values():
            spine.set_color("#344054")

    membrane_thickness = geometry.get("membrane_thickness")
    x_margin = max(0.08 * (x_nm[-1] - x_nm[0]), 8.0)
    axis.set_xlim(x_nm[0] - x_margin, x_nm[-1] + x_margin)
    if membrane_thickness is None:
        lower_limit = -max(0.35 * height, 12.0)
    else:
        # Leave room below the membrane so transmitted tracks stay visible.
        lower_limit = -(membrane_thickness + max(0.5 * vacuum_flight_nm, 10.0))
    axis.set_ylim(lower_limit, height + vacuum_flight_nm)
    axis.set_ylabel("Height above line base (nm)", color="#e5e7eb")
    axis.tick_params(labelbottom=False)
    axis.grid(color="#334155", alpha=0.18, linewidth=0.7)

    if membrane_thickness is None:
        support = Rectangle(
            (x_nm[0] - 2.0 * x_margin, axis.get_ylim()[0]),
            x_nm[-1] - x_nm[0] + 4.0 * x_margin,
            substrate_height - axis.get_ylim()[0],
            facecolor="#263548", edgecolor="none", zorder=0,
        )
    else:
        support = Rectangle(
            (x_nm[0] - 2.0 * x_margin, substrate_height - membrane_thickness),
            x_nm[-1] - x_nm[0] + 4.0 * x_margin,
            membrane_thickness,
            facecolor="#263548", edgecolor="#9fb3c8", linewidth=1.0, zorder=0,
        )
    axis.add_patch(support)
    trapezoid = Polygon(
        [
            (center_x - bottom_width / 2.0, substrate_height),
            (center_x - top_width / 2.0, substrate_height + height),
            (center_x + top_width / 2.0, substrate_height + height),
            (center_x + bottom_width / 2.0, substrate_height),
        ],
        closed=True, facecolor="#344b66", edgecolor="#9fb3c8",
        linewidth=1.4, zorder=1,
    )
    axis.add_patch(trapezoid)
    axis.axhline(substrate_height, color="#9fb3c8", linewidth=1.0, zorder=1)

    beam_line, = axis.plot([], [], color="#59e1ff", linewidth=2.1,
                           alpha=0.9, zorder=4)
    beam_halo, = axis.plot([], [], color="#59e1ff", linewidth=7.0,
                           alpha=0.13, zorder=3)
    beam_spot = axis.scatter([], [], s=62, facecolor="#d8fbff",
                             edgecolor="#59e1ff", linewidth=1.4, zorder=8)
    status = axis.text(
        0.015, 0.965, "", transform=axis.transAxes, va="top", ha="left",
        color="#f8fafc", fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#101827",
              "edgecolor": "#334155", "alpha": 0.88},
    )
    sample_name = archive.metadata.get("sample_name", "sample")
    energy_ev = float(archive.metadata["config"]["energy_ev"])
    main_title = title or (
        f"SEEMC electron-cascade scan • {sample_name} • {energy_ev / 1000.0:g} keV"
    )
    axis.set_title(main_title, color="#f8fafc", fontsize=14, pad=12)

    profile_axis.set_xlabel("Beam position x (nm)", color="#e5e7eb")
    profile_axis.set_ylabel("Yield", color="#e5e7eb")
    profile_axis.grid(color="#334155", alpha=0.22, linewidth=0.7)
    channel_names = [str(value) for value in archive.profile_channels]
    unknown_channels = [
        channel for channel in profile_channels if channel not in channel_names
    ]
    if unknown_channels:
        raise ValueError(
            f"unknown profile channels {unknown_channels}; available channels: "
            f"{channel_names}"
        )
    default_colors = ("#f8fafc", "#39d98a", "#f9c74f", "#43aaef",
                      "#c77dff", "#f9844a")
    profile_specs = []
    for index, channel in enumerate(profile_channels):
        label, color = PROFILE_STYLES.get(
            channel,
            (channel.replace("_", " "), default_colors[index]),
        )
        profile_specs.append((channel, label, color))
    profile_lines = []
    profile_values = []
    for channel, label, color in profile_specs:
        if channel not in channel_names:
            continue
        values = archive.profile_yields[channel_names.index(channel), 0]
        line, = profile_axis.plot([], [], color=color, linewidth=1.8, label=label)
        profile_lines.append(line)
        profile_values.append(values)
        profile_axis.plot(x_nm, values, color=color, linewidth=0.8, alpha=0.14)
    if profile_values:
        all_values = np.concatenate(profile_values)
        upper = max(float(np.max(all_values)) * 1.12, 0.1)
        profile_axis.set_ylim(0.0, upper)
        profile_axis.legend(
            loc="upper right", ncol=len(profile_lines), frameon=False,
            fontsize=8.5, labelcolor="#e5e7eb",
        )
    profile_cursor = profile_axis.axvline(
        x_nm[0], color="#59e1ff", linewidth=1.2, alpha=0.85
    )

    if color_by == "energy":
        mapper = ScalarMappable(norm=energy_norm, cmap=energy_cmap)
        colorbar = figure.colorbar(mapper, ax=axis, pad=0.012, fraction=0.035)
        colorbar.set_label("Instantaneous energy (eV)", color="#e5e7eb")
        colorbar.ax.tick_params(colors="#cbd5e1")
        colorbar.outline.set_edgecolor("#344054")
    else:
        handles = [
            plt.Line2D([0], [0], color=color, lw=2.4, label=name)
            for name, color in POPULATION_COLORS.items()
            if np.any(archive.electron_population == name)
        ]
        if handles:
            axis.legend(
                handles=handles, loc="upper right", frameon=False,
                fontsize=8, labelcolor="#e5e7eb", ncol=2,
            )

    dynamic_artists = []
    frames_each = frames_per_pixel + pause_frames
    total_frames = len(available_pixels) * frames_each

    def clear_dynamic():
        while dynamic_artists:
            dynamic_artists.pop().remove()

    def interpolate_head(track, time_fs):
        if time_fs < track[0, 4]:
            return None
        if time_fs >= track[-1, 4]:
            return track[-1, :4]
        upper = int(np.searchsorted(track[:, 4], time_fs, side="right"))
        lower = max(upper - 1, 0)
        t0, t1 = track[lower, 4], track[upper, 4]
        fraction = 1.0 if t1 <= t0 else (time_fs - t0) / (t1 - t0)
        return track[lower, :4] + fraction * (track[upper, :4] - track[lower, :4])

    def update(frame_index):
        clear_dynamic()
        pixel_sequence_index = min(frame_index // frames_each, len(available_pixels) - 1)
        local_frame = frame_index % frames_each
        pixel_id = int(available_pixels[pixel_sequence_index])
        progress = min(local_frame / max(frames_per_pixel - 1, 1), 1.0)
        time_fs = progress * pixel_duration[pixel_id]
        cascade_indices = cascades_by_pixel[pixel_id]
        ix = int(archive.cascade_ix[cascade_indices[0]])
        nominal_x = float(x_nm[ix])
        nominal_height = _trapezoid_surface_height_nm(
            nominal_x,
            top_width=top_width,
            bottom_width=bottom_width,
            height=height,
            center_x=center_x,
            substrate_height=substrate_height,
        )
        beam_top = axis.get_ylim()[1]
        # The cyan glyph represents the commanded beam axis.  Recorded launch
        # points include Gaussian beam-spot sampling and must not drive it:
        # their small random mean offset can make a monotone raster look as if
        # it steps backward or changes incidence angle.  The actual cascade
        # tracks below retain their sampled launch coordinates.
        beam_line.set_data(
            [nominal_x, nominal_x], [beam_top, nominal_height]
        )
        beam_halo.set_data(
            [nominal_x, nominal_x], [beam_top, nominal_height]
        )
        beam_spot.set_offsets(np.asarray([[nominal_x, nominal_height]]))

        for cascade_index in cascade_indices:
            for electron_index in archive.cascade_electron_indices(cascade_index):
                track = tracks[electron_index]
                if len(track) == 0 or time_fs < track[0, 4]:
                    continue
                head = interpolate_head(track, time_fs)
                if head is None:
                    continue
                visible = track[track[:, 4] <= time_fs]
                if len(visible) == 0 or not np.allclose(visible[-1, :4], head):
                    visible = np.vstack((visible, np.r_[head, time_fs]))
                tail_start = time_fs - tail_fraction * pixel_duration[pixel_id]
                visible = visible[visible[:, 4] >= tail_start]
                if len(visible) == 1:
                    visible = np.vstack((visible, visible))
                xy = np.column_stack((visible[:, 0] / 10.0, -visible[:, 2] / 10.0))
                segments = np.stack((xy[:-1], xy[1:]), axis=1)
                if color_by == "energy":
                    colors = energy_cmap(energy_norm(np.maximum(visible[1:, 3], min_energy)))
                else:
                    color = POPULATION_COLORS.get(
                        str(archive.electron_population[electron_index]), "#e5e7eb"
                    )
                    colors = np.tile(np.asarray(to_rgba(color)),
                                     (len(segments), 1))
                if len(colors):
                    colors[:, 3] *= np.linspace(0.12, 0.92, len(colors))
                    collection = LineCollection(
                        segments, colors=colors, linewidths=1.45, zorder=5
                    )
                    axis.add_collection(collection)
                    dynamic_artists.append(collection)
                head_energy = max(float(head[3]), min_energy)
                if color_by == "energy":
                    head_color = energy_cmap(energy_norm(head_energy))
                else:
                    head_color = POPULATION_COLORS.get(
                        str(archive.electron_population[electron_index]), "#e5e7eb"
                    )
                marker = axis.scatter(
                    [head[0] / 10.0], [-head[2] / 10.0], s=16,
                    facecolor=head_color, edgecolor="white", linewidth=0.25,
                    alpha=0.95, zorder=7,
                )
                dynamic_artists.append(marker)

        for line, values in zip(profile_lines, profile_values):
            line.set_data(x_nm[:ix + 1], values[:ix + 1])
        profile_cursor.set_xdata([nominal_x, nominal_x])
        status.set_text(
            f"pixel {pixel_sequence_index + 1}/{len(available_pixels)}   "
            f"x = {nominal_x:+.1f} nm   "
            f"t = {time_fs:.3f} fs   "
            f"primaries = {len(cascade_indices)}"
        )
        return [beam_line, beam_halo, beam_spot, profile_cursor, status,
                *profile_lines, *dynamic_artists]

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000.0 / fps,
        blit=False, repeat=False,
    )
    if suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            plt.close(figure)
            raise RuntimeError(
                "ffmpeg is unavailable; install ffmpeg or choose a .gif output"
            )
        writer = FFMpegWriter(
            fps=fps, bitrate=3200,
            metadata={"title": main_title, "artist": "SEEMC imaging"},
            extra_args=[
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt", "yuv420p",
            ],
        )
    else:
        writer = PillowWriter(fps=fps)
    animation.save(output, writer=writer, dpi=dpi)
    plt.close(figure)
    return output


__all__ = [
    "POPULATION_COLORS",
    "PROFILE_PRESETS",
    "PROFILE_STYLES",
    "animate_trapezoidal_scan",
]
