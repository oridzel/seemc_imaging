# `animate-trapezoid`

Create an animation of a SEEMC trapezoidal scan from a saved trajectory file.

The animator replays the electron trajectories while the beam moves across the
trapezoid, making it useful for visualizing where SE and BSE signals originate
during a scan.

## Input

`animate-trapezoid` expects a trajectory file produced by a trajectory-recording
trapezoidal scan:

```text
<prefix>.trajectories.npz
```

For example:

```text
trapezoid_movie.trajectories.npz
```

The normal raster-results `.npz` file by itself is not sufficient; the
trajectory file is required.

## Quick start

If the package entry point is installed:

```bash
animate-trapezoid   trapezoid_movie.trajectories.npz   --output trapezoid_movie.gif   --fps 30   --frames-per-pixel 16   --pause-frames 4   --color-by energy   --vacuum-flight-nm 35
```

When running directly from the source tree, use the example script instead:

```bash
python3 examples/animate_trapezoidal_scan.py   trapezoid_movie.trajectories.npz   --output trapezoid_movie.gif   --fps 30   --frames-per-pixel 16   --pause-frames 4   --color-by energy   --vacuum-flight-nm 35
```

This is a good default configuration for presentation-quality animations.

## MP4 output

To write MP4 instead of GIF:

```bash
animate-trapezoid   trapezoid_movie.trajectories.npz   --output trapezoid_movie.mp4   --fps 30   --frames-per-pixel 16   --pause-frames 4   --color-by energy   --vacuum-flight-nm 35
```

or, from the source tree:

```bash
python3 examples/animate_trapezoidal_scan.py   trapezoid_movie.trajectories.npz   --output trapezoid_movie.mp4   --fps 30   --frames-per-pixel 16   --pause-frames 4   --color-by energy   --vacuum-flight-nm 35
```

MP4 output requires `ffmpeg`.

On macOS, for example:

```bash
brew install ffmpeg
```

If MP4 encoding fails, GIF is the simplest fallback:

```bash
--output trapezoid_movie.gif
```

One ffmpeg error encountered with older versions of the animation script was:

```text
width not divisible by 2
```

This comes from the H.264 encoder requiring even frame dimensions. If an older
script produces this error, use GIF output or update to the current animation
script that handles MP4-compatible frame sizing.

## Main options

| Option | Meaning |
|---|---|
| `INPUT` | Input `*.trajectories.npz` file |
| `--output FILE` | Output animation, normally `.gif` or `.mp4` |
| `--fps N` | Playback frame rate |
| `--frames-per-pixel N` | Number of animation frames spent showing each beam position |
| `--pause-frames N` | Extra frames inserted between beam positions |
| `--color-by energy` | Color trajectories according to electron energy |
| `--vacuum-flight-nm X` | Continue emitted trajectories by `X` nm into vacuum for visualization |

To see all options available in the installed version:

```bash
animate-trapezoid --help
```

or:

```bash
python3 examples/animate_trapezoidal_scan.py --help
```

## Recommended settings

For the animations used in the SEEMC trapezoid work:

```text
fps               = 30
frames-per-pixel  = 16
pause-frames      = 4
color-by          = energy
vacuum-flight-nm  = 35
```

These settings make the beam motion easy to follow while leaving enough time
to see the emitted-electron cascade at every scan position.

### Faster preview

For a quicker test render:

```bash
animate-trapezoid   trapezoid_movie.trajectories.npz   --output preview.gif   --fps 20   --frames-per-pixel 6   --pause-frames 1   --color-by energy   --vacuum-flight-nm 20
```

### Slower presentation version

For a slower animation intended for a talk:

```bash
animate-trapezoid   trapezoid_movie.trajectories.npz   --output trapezoid_presentation.gif   --fps 30   --frames-per-pixel 24   --pause-frames 6   --color-by energy   --vacuum-flight-nm 35
```

## What the timing options do

The total number of animation frames grows approximately with the number of
scan pixels:

```text
frames ≈ number_of_scan_pixels × (frames_per_pixel + pause_frames)
```

Therefore:

- increasing `--frames-per-pixel` makes the electron activity at each pixel
  easier to inspect;
- increasing `--pause-frames` makes the beam stepping from one pixel to the next
  more obvious;
- increasing `--fps` makes the final movie play faster if the number of frames
  is unchanged.

For example, with 201 scan pixels and the recommended settings:

```text
201 × (16 + 4) = 4020 frames
```

At 30 fps, that corresponds to about 134 s of playback.

## Vacuum-flight extension

Electron trajectories normally stop when the transport simulation considers
the electron emitted from the sample.

For visualization, `--vacuum-flight-nm` extends the emitted trajectory into
vacuum:

```bash
--vacuum-flight-nm 35
```

This does **not** change the Monte Carlo transport result or the calculated
yield. It only makes emitted electrons easier to see in the animation.

## Suggested workflow

1. Run a trapezoidal scan with trajectory/history recording enabled.
2. Confirm that a file such as

   ```text
   trapezoid_movie.trajectories.npz
   ```

   was created.
3. Make a fast GIF preview.
4. Adjust `--frames-per-pixel`, `--pause-frames`, or
   `--vacuum-flight-nm` if needed.
5. Render the final GIF or MP4.

A typical final command is:

```bash
animate-trapezoid   trapezoid_movie.trajectories.npz   --output trapezoid_movie.gif   --fps 30   --frames-per-pixel 16   --pause-frames 4   --color-by energy   --vacuum-flight-nm 35
```

## Troubleshooting

### `No such file or directory`

Check that the argument points to the `*.trajectories.npz` file rather than the
normal raster-results file.

### `ffmpeg` not found

Use GIF output or install ffmpeg.

### MP4 fails with `width not divisible by 2`

This is an H.264 frame-size requirement seen with older versions of the
animation script. Use GIF output or update the animation script.

### Animation is too slow

Reduce:

```text
--frames-per-pixel
--pause-frames
```

For testing, `6` frames per pixel and `1` pause frame are usually enough.

### Emitted trajectories are hard to see

Increase:

```bash
--vacuum-flight-nm 35
```

or use a somewhat larger value for visualization.

---

The animation is a visualization of the saved Monte Carlo histories. Changing
animation timing or vacuum-flight settings does not rerun electron transport and
does not change the simulated SE/BSE yields.
