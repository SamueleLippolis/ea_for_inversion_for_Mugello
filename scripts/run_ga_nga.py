#!/usr/bin/env python3
"""Run GA and niching GA inversions using the Python forward model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake_forward_model import load_intensity_observations
from earthquake_forward_model.inversion import (
    EvolutionSearch, EvolutionSettings, Gene, PARAMETERS, SearchSpace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/colline_pisane_ga_nga.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/colline_pisane_ga_nga.json")
    parser.add_argument("--population", type=int, help="override population per deme")
    parser.add_argument("--generations", type=int, help="override generations per stage")
    parser.add_argument("--demes", type=int, help="override NGA deme count")
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

    bounds = config["bounds"]
    space = SearchSpace(tuple(Gene(name, **bounds[name]) for name in PARAMETERS),
                        {key: forward["model"][key]
                         for key in ("samples", "sampling_interval")})
    population_size = config["population_size"] if args.population is None else args.population
    generations = config["generations"] if args.generations is None else args.generations
    replacement_count = max(2, round(population_size * config["replacement_fraction"]))
    settings = EvolutionSettings(
        population_size, generations, replacement_count,
        config.get("crossover_probability", 0.9),
        config.get("mutation_probability", 0.06),
        config.get("restart_after", 10))
    demes = config["demes"] if args.demes is None else args.demes
    radius = config["niche_radius"]
    search = EvolutionSearch(space, observations, forward.get("intensity_law", 2),
                             config["seed"])
    history = []

    def progress(stage: str):
        def record(generation: int, deme: int, residual: float) -> None:
            history.append({"stage": stage, "generation": generation,
                            "deme": deme, "best_residual": residual})
            print(f"{stage} generation {generation}, deme {deme}: residual {residual:.0f}",
                  flush=True)
        return record

    ga = search.run(settings, progress=progress("GA"))
    nga = search.run(settings, demes=demes, radius=radius, progress=progress("NGA"))

    def summary(populations):
        return [{"deme": i, "residual": population[0].residual,
                 "model": vars(space.decode(population[0].genome))}
                for i, population in enumerate(populations, 1)]

    report = {
        "configuration": str(args.config),
        "data_file": str(data_path),
        "seed": config["seed"],
        "population_size": population_size,
        "generations_per_stage": generations,
        "replacement_count": replacement_count,
        "restart_after": settings.restart_after,
        "demes": demes,
        "niche_radius": radius,
        "unique_forward_evaluations": len(search.cache),
        "ga": summary(ga),
        "nga": summary(nga),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
