"""Record one cascade and write its raw provenance as JSON."""

import argparse
import json

import numpy as np

from seemc_imaging import Sample, simulate_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("material")
    parser.add_argument("--energy", type=float, default=1000.0)
    parser.add_argument("--angle-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="trajectory_history.json")
    args = parser.parse_args()

    sample = Sample(args.material, db_path=args.db_path)
    result = simulate_trajectory(
        sample,
        E0=args.energy,
        angle_rad=np.deg2rad(args.angle_deg),
        rng=np.random.default_rng(args.seed),
        history=True,
        trajectory_id=0,
    )
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result.history.to_dict(), stream, indent=2)
    print(f"TEY={result.tey}; electrons={len(result.history.electrons)}; "
          f"events={len(result.history.events)}")


if __name__ == "__main__":
    main()

