#!/usr/bin/env python3
"""Run particle swarm inversion using the Python forward model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake_forward_model import load_intensity_observations
from earthquake_forward_model.inversion import Gene, PARAMETERS, SearchSpace
from earthquake_forward_model.pso import ParticleSwarmSearch, SwarmSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/colline_pisane_pso.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/colline_pisane_pso.json")
    parser.add_argument("--swarm-size", type=int, help="override particle count")
    parser.add_argument("--iterations", type=int, help="override iteration count")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    forward_path = Path(config["forward_config"])
    if not forward_path.is_absolute():
        forward_path = ROOT / forward_path
    forward = json.loads(forward_path.read_text())
    data_path = Path(forward["data_file"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    observations = load_intensity_observations(data_path)

    space = SearchSpace(
        tuple(Gene(name, **config["bounds"][name]) for name in PARAMETERS),
        {key: forward["model"][key] for key in ("samples", "sampling_interval")},
    )
    settings = SwarmSettings(
        config["swarm_size"] if args.swarm_size is None else args.swarm_size,
        config["iterations"] if args.iterations is None else args.iterations,
        config.get("inertia_weight", 0.7),
        config.get("cognitive_weight", 1.5),
        config.get("social_weight", 1.5),
        config.get("maximum_velocity_fraction", 0.2),
    )
    search = ParticleSwarmSearch(
        space, observations, forward.get("intensity_law", 2), config["seed"])
    history = []

    def progress(iteration: int, residual: float) -> None:
        history.append({"stage": "PSO", "iteration": iteration,
                        "best_residual": residual})
        print(f"PSO iteration {iteration}: residual {residual:.0f}", flush=True)

    best = search.run(settings, progress=progress)
    report = {
        "configuration": str(args.config),
        "data_file": str(data_path),
        "seed": config["seed"],
        "swarm_size": settings.swarm_size,
        "iterations": settings.iterations,
        "inertia_weight": settings.inertia_weight,
        "cognitive_weight": settings.cognitive_weight,
        "social_weight": settings.social_weight,
        "maximum_velocity_fraction": settings.maximum_velocity_fraction,
        "unique_forward_evaluations": len(search.cache),
        "pso": [{"run": 1, "residual": best.residual,
                 "model": vars(space.decode(best.genome))}],
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
