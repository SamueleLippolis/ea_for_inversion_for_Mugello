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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/colline_pisane_forward_model.json")
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
    model = SourceModelParameters(**config["model"])
    data_path = Path(config["data_file"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    observations = load_intensity_observations(data_path)
    python_result = evaluate_forward_model(
        model, observations, intensity_law=config.get("intensity_law", 2))
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
        values = []
        model_dict = config["model"]
        for name in FORTRAN_PARAMETER_ORDER:
            values.append("0" if name is None else str(model_dict[name]))
        subprocess.run([str(executable), str(data_path), str(output), *values], check=True,
                       cwd=ROOT)
        f_residual, f_predicted, f_distances, f_kf = read_fortran_results(output)

    intensity_mismatches = [i for i, (py, ft) in enumerate(
        zip(python_result.predicted_intensities, f_predicted), 1) if py != ft]
    distance_errors = [abs(py-ft) for py, ft in zip(python_result.distances_km, f_distances)]
    kf_mismatches = [i for i, (py, ft) in enumerate(
        zip(python_result.kinematic_values, f_kf), 1)
        if not math.isclose(py, ft, rel_tol=args.rtol, abs_tol=1e-7)]

    print(f"sites compared: {len(observations)}")
    print(f"residual: Python={python_result.residual:.0f}, Fortran={f_residual:.0f}")
    print(f"predicted-intensity mismatches: {len(intensity_mismatches)}")
    print(f"KF mismatches (rtol={args.rtol:g}): {len(kf_mismatches)}")
    print(f"maximum distance difference: {max(distance_errors):.6g} km")
    if intensity_mismatches:
        print(f"first intensity mismatch sites: {intensity_mismatches[:10]}")
    if kf_mismatches:
        print(f"first KF mismatch sites: {kf_mismatches[:10]}")
    passed = (python_result.residual == f_residual and not intensity_mismatches
              and not kf_mismatches and max(distance_errors) <= args.distance_tol)
    print("PARITY CHECK: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
