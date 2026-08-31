#!/usr/bin/env python3
"""Run one inversion-kernel evaluation (no PSO or random search)."""

from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mugello_inversion import ModelParameters, evaluate_model, load_observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/single_test.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/single_test.csv")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_path = Path(config["data_file"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    result = evaluate_model(ModelParameters(**config["model"]), load_observations(data_path),
                            intensity_law=config.get("intensity_law", 2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["longitude", "latitude", "observed", "predicted", "distance_km", "kf"])
        for site, predicted, distance, kf in zip(result.observations,
                                                  result.predicted_intensities,
                                                  result.distances_km,
                                                  result.kinematic_values):
            writer.writerow([site.longitude, site.latitude, site.intensity, predicted, distance, kf])
    print(f"sites: {len(result.observations)}")
    print(f"sum-of-squares residual: {result.residual:.0f}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
