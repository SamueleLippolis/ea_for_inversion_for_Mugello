#!/usr/bin/env python3
"""Compare each reported Python inversion source with the Fortran forward model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake_forward_model import (
    SourceModelParameters, evaluate_forward_model, load_intensity_observations,
)
from compare_fortran_python import (
    FORTRAN_PARAMETER_ORDER, extract_fortran_kernel, read_fortran_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/colline_pisane_ga_nga.json",
                        help="JSON report from run_ga_nga.py or run_pso.py")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/colline_pisane_ga_nga_fortran_comparison.json")
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--rtol", type=float, default=2e-4,
                        help="relative KF tolerance")
    parser.add_argument("--distance-tol", type=float, default=2e-3,
                        help="absolute station-distance tolerance in km")
    parser.add_argument("--strict", action="store_true",
                        help="exit with status 1 if full KF parity fails")
    args = parser.parse_args()

    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error(f"Fortran compiler '{args.compiler}' was not found")
    inversion = json.loads(args.input.read_text())
    data_path = Path(inversion["data_file"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    observations = load_intensity_observations(data_path)
    stages = [stage for stage in ("ga", "nga", "pso") if stage in inversion]
    if not stages:
        parser.error("input report contains no GA, NGA, or PSO solutions")

    comparisons = []
    with tempfile.TemporaryDirectory(prefix="inversion-fortran-comparison-") as temporary:
        build = Path(temporary)
        kernel = build / "pqu7_kernel.f"
        executable = build / "fortran_reference"
        output = build / "reference.txt"
        extract_fortran_kernel(ROOT / "fortran_code/pqu7v2.f", kernel)
        subprocess.run([
            compiler, "-O0", "-std=legacy", "-ffixed-line-length-none",
            str(ROOT / "scripts/fortran_reference_driver.f"), str(kernel),
            "-o", str(executable),
        ], check=True)

        for stage in stages:
            for candidate in inversion[stage]:
                identity_key = "run" if stage == "pso" else "deme"
                identity = candidate[identity_key]
                model_dict = candidate["model"]
                model = SourceModelParameters(**model_dict)
                python = evaluate_forward_model(model, observations)
                values = ["0" if name is None else str(model_dict[name])
                          for name in FORTRAN_PARAMETER_ORDER]
                subprocess.run([str(executable), str(data_path), str(output), *values],
                               check=True, cwd=ROOT, capture_output=True)
                f_residual, f_predicted, f_distances, f_kf = read_fortran_results(output)
                if len(f_predicted) != len(observations):
                    raise RuntimeError(f"{stage} {identity_key} {identity}: "
                                       "Fortran site count differs from Python")

                sites = []
                for number, (site, py_int, ft_int, py_dist, ft_dist, py_kf, ft_kf) in enumerate(
                    zip(observations, python.predicted_intensities, f_predicted,
                        python.distances_km, f_distances,
                        python.kinematic_values, f_kf), 1
                ):
                    sites.append({
                        "site": number,
                        "longitude": site.longitude,
                        "latitude": site.latitude,
                        "observed_intensity": site.intensity,
                        "python_intensity": py_int,
                        "fortran_intensity": ft_int,
                        "python_distance_km": py_dist,
                        "fortran_distance_km": ft_dist,
                        "python_kf": py_kf,
                        "fortran_kf": ft_kf,
                        "kf_within_tolerance": math.isclose(
                            py_kf, ft_kf, rel_tol=args.rtol, abs_tol=1e-7),
                    })

                intensity_mismatches = [site["site"] for site in sites
                                        if site["python_intensity"] != site["fortran_intensity"]]
                kf_mismatches = [site["site"] for site in sites
                                 if not site["kf_within_tolerance"]]
                maximum_distance_error = max(abs(site["python_distance_km"] -
                                                 site["fortran_distance_km"])
                                             for site in sites)
                score_parity = (candidate["residual"] == python.residual == f_residual
                                and not intensity_mismatches)
                full_parity = (score_parity and not kf_mismatches
                               and maximum_distance_error <= args.distance_tol)
                comparisons.append({
                    "stage": stage,
                    identity_key: identity,
                    "model": model_dict,
                    "reported_python_residual": candidate["residual"],
                    "recomputed_python_residual": python.residual,
                    "fortran_residual": f_residual,
                    "intensity_mismatch_sites": intensity_mismatches,
                    "kf_mismatch_sites": kf_mismatches,
                    "maximum_distance_difference_km": maximum_distance_error,
                    "score_parity": score_parity,
                    "full_parity": full_parity,
                    "sites": sites,
                })
                print(f"{stage.upper()} {identity_key} {identity}: "
                      f"residual Python={python.residual:.0f}, Fortran={f_residual:.0f}; "
                      f"intensity mismatches={len(intensity_mismatches)}, "
                      f"KF mismatches={len(kf_mismatches)}; "
                      f"full parity={'PASS' if full_parity else 'FAIL'}")

    report = {
        "comparison_type": "Fortran forward model evaluated at Python inversion solutions",
        "inversion_report": str(args.input),
        "data_file": str(data_path),
        "fortran_source": "fortran_code/pqu7v2.f",
        "kf_relative_tolerance": args.rtol,
        "distance_absolute_tolerance_km": args.distance_tol,
        "all_scores_match": all(item["score_parity"] for item in comparisons),
        "all_full_parity": all(item["full_parity"] for item in comparisons),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"report: {args.output}")
    return 1 if args.strict and not report["all_full_parity"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
