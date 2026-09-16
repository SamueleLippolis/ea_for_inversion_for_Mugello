#!/usr/bin/env python3
"""Compile the Fortran kernel and compare it with the Python translation."""

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
    SourceModelParameters,
    evaluate_forward_model,
    load_intensity_observations,
)


FORTRAN_PARAMETER_ORDER = (
    "latitude", "longitude", "strike", "rake", "dip", "depth",
    "length_plus", "mach_plus", "samples", "s_velocity",
    "sampling_interval", "seismic_moment", None, "length_minus", None,
    "mach_minus",
)


def extract_fortran_kernel(source: Path, destination: Path) -> None:
    """Remove only the original main program, retaining its subroutines."""
    lines = source.read_text().splitlines(keepends=True)
    marker = next(i for i, line in enumerate(lines)
                  if line.lower().startswith("      real function feps"))
    destination.write_text("".join(lines[marker:]))


def read_fortran_results(path: Path):
    lines = path.read_text().splitlines()
    summary = lines[0].split()
    count, residual = int(summary[2]), float(summary[3])
    rows = [line.split() for line in lines[1:]]
    if len(rows) != count:
        raise RuntimeError(f"Fortran reported {count} sites but wrote {len(rows)}")
    predicted = tuple(int(row[3]) for row in rows)
    distances = tuple(float(row[4]) for row in rows)
    kf_values = tuple(float(row[5]) for row in rows)
    return residual, predicted, distances, kf_values


def load_sources(path: Path, baseline: dict) -> list[tuple[str, dict]]:
    """Combine the configured source with named parameter variations."""
    sources = [("configured", baseline)]
    seen = {"configured"}
    for entry in json.loads(path.read_text()):
        name = entry["name"]
        changes = entry["model"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"source names must be nonempty and unique: {name!r}")
        unknown = changes.keys() - baseline.keys()
        if unknown:
            raise ValueError(f"{name}: unknown model parameters: {sorted(unknown)}")
        sources.append((name, {**baseline, **changes}))
        seen.add(name)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/colline_pisane_forward_model.json")
    parser.add_argument(
        "--sources", type=Path,
        default=ROOT / "configs/forward_comparison_sources.json",
        help="JSON list of named model variations to compare after the configured source")
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--rtol", type=float, default=2e-4,
                        help="relative tolerance for single-precision Fortran values")
    parser.add_argument("--distance-tol", type=float, default=2e-3,
                        help="absolute station-distance tolerance in km (default: 0.002)")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None:
        print(f"error: Fortran compiler '{args.compiler}' was not found", file=sys.stderr)
        print("Install gfortran, then run this command again.", file=sys.stderr)
        return 2

    config = json.loads(args.config.read_text())
    sources = load_sources(args.sources, config["model"])
    data_path = Path(config["data_file"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    observations = load_intensity_observations(data_path)
    if config.get("intensity_law", 2) != 2:
        print("error: reference driver currently compares Fortran intensity law 2", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="forward-model-parity-") as temporary:
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
        all_passed = True
        for name, model_dict in sources:
            model = SourceModelParameters(**model_dict)
            python_result = evaluate_forward_model(model, observations, intensity_law=2)
            values = ["0" if parameter is None else str(model_dict[parameter])
                      for parameter in FORTRAN_PARAMETER_ORDER]
            subprocess.run([str(executable), str(data_path), str(output), *values],
                           check=True, cwd=ROOT, capture_output=True)
            f_residual, f_predicted, f_distances, f_kf = read_fortran_results(output)
            if len(f_predicted) != len(observations):
                raise RuntimeError(f"{name}: Fortran returned {len(f_predicted)} sites; "
                                   f"Python used {len(observations)}")
            intensity_mismatches = [i for i, (py, ft) in enumerate(
                zip(python_result.predicted_intensities, f_predicted), 1) if py != ft]
            distance_errors = [abs(py-ft) for py, ft in
                               zip(python_result.distances_km, f_distances)]
            kf_mismatches = [i for i, (py, ft) in enumerate(
                zip(python_result.kinematic_values, f_kf), 1)
                if not math.isclose(py, ft, rel_tol=args.rtol, abs_tol=1e-7)]
            passed = (python_result.residual == f_residual and not intensity_mismatches
                      and not kf_mismatches and max(distance_errors) <= args.distance_tol)
            all_passed &= passed
            print(f"{name}: {'PASS' if passed else 'FAIL'} "
                  f"({len(observations)} sites, residual Python={python_result.residual:.0f} "
                  f"Fortran={f_residual:.0f}, intensity mismatches={len(intensity_mismatches)}, "
                  f"KF mismatches={len(kf_mismatches)}, "
                  f"max distance difference={max(distance_errors):.6g} km)")
            if intensity_mismatches:
                for site in intensity_mismatches[:10]:
                    print(f"  site {site} intensity: Python="
                          f"{python_result.predicted_intensities[site-1]} "
                          f"Fortran={f_predicted[site-1]}")
            if kf_mismatches:
                for site in kf_mismatches[:10]:
                    py, ft = python_result.kinematic_values[site-1], f_kf[site-1]
                    print(f"  site {site} KF: Python={py:.9g} Fortran={ft:.9g} "
                          f"relative difference={abs(py-ft)/max(abs(py), abs(ft)):.3g}")

    source_label = "source" if len(sources) == 1 else "sources"
    print(f"PARITY CHECK: {'PASS' if all_passed else 'FAIL'} "
          f"({len(sources)} {source_label}, "
          f"{len(sources) * len(observations)} site predictions)")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
