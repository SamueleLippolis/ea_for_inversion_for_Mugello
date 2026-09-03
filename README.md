# Earthquake forward model for the 1846 Colline Pisane event

This project models the macroseismic intensities of the 14 August 1846
Colline Pisane earthquake, whose strongest effects occurred around Orciano
Pisano, between Pisa and Livorno.

The Python code is a deterministic translation of the calculation kernel in
`fortran_code/pqu7v2.f`. It is a **forward model**: it receives one proposed
earthquake-source model and calculates the intensity expected at every
observation site. It then calculates the sum-of-squared differences between
the predicted and observed intensities.

The Python implementation does not currently perform a complete inversion. A
complete inversion would repeatedly change the source parameters and call the
forward model to find the parameter set with the smallest misfit. The original
Fortran files include genetic-algorithm code for that search.

## Project structure

- `src/earthquake_forward_model/`: Python forward-model package.
- `configs/`: source-model parameters and input-file selection.
- `data/`: observed macroseismic intensities and original parameter ranges.
- `scripts/`: commands for running the model and comparing Python with Fortran.
- `tests/`: automated checks for the Python implementation.
- `results/`: CSV output produced by the forward model.
- `fortran_code/`: original Fortran model and genetic-algorithm code.
- `fortran_results/`: results produced by the original Fortran workflow.
- `papers/`: scientific reference papers.

## Run the forward model

The included configuration uses the midpoint of the parameter ranges in
`data/set17.txt`:

```bash
python3 scripts/run_forward_model.py
```

The command reads `configs/colline_pisane_forward_model.json` and writes
`results/colline_pisane_forward_model.csv`. Pass another JSON file with
`--config`, or another output location with `--output`, when needed.

Each output row contains the location, observed intensity, predicted intensity,
station distance, and calculated peak kinematic value.

## Run the tests

The test suite has no external Python dependencies:

```bash
python3 -m unittest discover -s tests -v
```

## Compare Python with the original Fortran

If `gfortran` is installed, run:

```bash
python3 scripts/compare_fortran_python.py
```

The script compiles the original Fortran calculation kernel in a temporary
directory and evaluates the same source model in both languages. It compares
all predicted intensities, the residual, station distances, and intermediate
kinematic values. It exits successfully only when the implementations agree.
The original `fortran_code/` directory is not modified.

## Important Python names

- `SourceModelParameters`: parameters describing one proposed earthquake source.
- `IntensityObservation`: one observed intensity and its geographic location.
- `evaluate_forward_model()`: predicts intensities for one source model.
- `ForwardModelResult`: predicted values, distances, and total misfit.
- `load_intensity_observations()`: reads the observation data file.
