#!/usr/bin/env python3
"""Compare saved GA-NGA and PSO inversion reports on the same search problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake_forward_model import (
    SourceModelParameters, evaluate_forward_model, load_intensity_observations,
)
from earthquake_forward_model.inversion import PARAMETERS


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_problem(report: dict) -> tuple[Path, dict, dict]:
    config = json.loads(project_path(report["configuration"]).read_text())
    forward = json.loads(project_path(config["forward_config"]).read_text())
    return project_path(report["data_file"]).resolve(), config, forward


def candidates(report: dict, stages: tuple[str, ...]) -> list[dict]:
    rows = []
    for stage in stages:
        for candidate in report[stage]:
            identity = candidate["run"] if stage == "pso" else candidate["deme"]
            rows.append({
                "algorithm": stage.upper(),
                "run_or_deme": identity,
                "residual": candidate["residual"],
                "model": candidate["model"],
            })
    return rows


def normalized_model_distance(first: dict, second: dict, bounds: dict) -> float:
    return sum(abs(first[name] - second[name]) /
               (bounds[name]["upper"] - bounds[name]["lower"])
               for name in PARAMETERS) / len(PARAMETERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-nga", type=Path,
                        default=ROOT / "results/colline_pisane_ga_nga.json")
    parser.add_argument("--pso", type=Path,
                        default=ROOT / "results/colline_pisane_pso.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/colline_pisane_optimizer_comparison.json")
    args = parser.parse_args()

    ga_nga = json.loads(args.ga_nga.read_text())
    pso = json.loads(args.pso.read_text())
    ga_data, ga_config, ga_forward = load_problem(ga_nga)
    pso_data, pso_config, pso_forward = load_problem(pso)
    if ga_data != pso_data:
        parser.error("reports use different observation files")
    if ga_config["bounds"] != pso_config["bounds"]:
        parser.error("reports use different parameter bounds or steps")
    if ga_forward.get("intensity_law", 2) != pso_forward.get("intensity_law", 2):
        parser.error("reports use different intensity laws")
    for name in ("samples", "sampling_interval"):
        if ga_forward["model"][name] != pso_forward["model"][name]:
            parser.error(f"reports use different {name} values")

    ga_rows = candidates(ga_nga, ("ga", "nga"))
    pso_rows = candidates(pso, ("pso",))
    if not ga_rows or not pso_rows:
        parser.error("both reports must contain at least one source model")
    observations = load_intensity_observations(ga_data)
    for row in ga_rows + pso_rows:
        model = SourceModelParameters(**row["model"])
        recomputed = evaluate_forward_model(
            model, observations,
            intensity_law=ga_forward.get("intensity_law", 2)).residual
        if recomputed != row["residual"]:
            parser.error(f"{row['algorithm']} {row['run_or_deme']} residual "
                         f"changed: report={row['residual']}, recomputed={recomputed}")
    ranked = sorted(ga_rows + pso_rows,
                    key=lambda row: (row["residual"], row["algorithm"],
                                     row["run_or_deme"]))
    best_ga_nga = min(ga_rows, key=lambda row: row["residual"])
    best_pso = min(pso_rows, key=lambda row: row["residual"])
    delta = best_pso["residual"] - best_ga_nga["residual"]
    site_count = len(observations)
    report = {
        "ga_nga_report": str(args.ga_nga),
        "pso_report": str(args.pso),
        "data_file": str(ga_data),
        "site_count": site_count,
        "intensity_law": ga_forward.get("intensity_law", 2),
        "same_problem": True,
        "ga_nga_search": {
            "seed": ga_nga["seed"],
            "population_size": ga_nga["population_size"],
            "generations_per_stage": ga_nga["generations_per_stage"],
            "demes": ga_nga["demes"],
            "unique_forward_evaluations": ga_nga["unique_forward_evaluations"],
        },
        "pso_search": {
            "seed": pso["seed"],
            "swarm_size": pso["swarm_size"],
            "iterations": pso["iterations"],
            "unique_forward_evaluations": pso["unique_forward_evaluations"],
        },
        "ranked_solutions": ranked,
        "best_ga_nga": best_ga_nga,
        "best_pso": best_pso,
        "pso_minus_ga_nga_residual": delta,
        "pso_minus_ga_nga_parameters": {
            name: best_pso["model"][name] - best_ga_nga["model"][name]
            for name in PARAMETERS
        },
        "normalized_best_source_distance": normalized_model_distance(
            best_pso["model"], best_ga_nga["model"], ga_config["bounds"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print(f"Same data and parameter bounds: yes ({site_count} sites)")
    print(f"Unique forward evaluations: GA-NGA={ga_nga['unique_forward_evaluations']}, "
          f"PSO={pso['unique_forward_evaluations']}")
    for row in ranked:
        print(f"{row['algorithm']} {row['run_or_deme']}: residual {row['residual']:.0f}")
    print(f"PSO minus best GA-NGA residual: {delta:+.0f} (lower is better)")
    print(f"Normalized distance between best sources: "
          f"{report['normalized_best_source_distance']:.4f}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
