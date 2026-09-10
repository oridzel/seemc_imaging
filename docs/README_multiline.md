# Multi-line trapezoid update

This update keeps the raster one-dimensional (`ny=1`) but allows the beam to
scan across multiple adjacent trapezoidal lines.

## New CLI options

- `--n-lines N` — number of trapezoidal lines, default 1.
- `--pitch-nm P` — center-to-center pitch, default 100 nm.
- `--field-width-nm W` — optional total scan width. If omitted, the script
  automatically covers the full line array plus 40 nm of substrate on each side.

Example:

```bash
python3 examples/trapezoidal_line_scan.py MaterialDatabase.pkl \
  --material Si \
  --energy-ev 1000 \
  --top-width-nm 50 \
  --bottom-width-nm 70 \
  --height-nm 50 \
  --n-lines 3 \
  --pitch-nm 100 \
  --pixels 201 \
  --trajectories 1000 \
  --output three_lines.csv
```

For 3 lines at 100 nm pitch, the line centers are `-100, 0, +100 nm`.

The pitch is center-to-center and must be at least the bottom width, preventing
overlapping trapezoids.

Note: the supplied `trapezoidal_line_scan.py` writes only the yield CSV. The
trajectory-recording scan used by `animate-trapezoid` must also construct
`TrapezoidalLineArray` with the same `n_lines` and `pitch`, and the animator
must render the same geometry.
