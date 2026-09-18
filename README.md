# Earthquake forward model for the 1846 Colline Pisane event

This project models the macroseismic intensities of the 14 August 1846
Colline Pisane earthquake, whose strongest effects occurred around Orciano
Pisano, between Pisa and Livorno.

The Python code is a deterministic translation of the calculation kernel in
`fortran_code/pqu7v2.f`. It is a **forward model**: it receives one proposed
earthquake-source model and calculates the intensity expected at every
observation site. It then calculates the sum-of-squared differences between
the predicted and observed intensities.

The Python implementation also includes a genetic algorithm (GA) followed by
a niching genetic algorithm (NGA), and a particle swarm optimizer (PSO). They
repeatedly evaluate the forward model to find source parameters with smaller
intensity residuals.

## What the inversion can tell us

The dataset contains observed intensities at 108 sites, but no known true
earthquake-source parameters. For a proposed source `x`, the forward model
predicts an intensity `F_i(x)` at site `i`. The site residual is
`F_i(x) - y_i`, where `y_i` is the observed intensity. The objective reported
by the optimizers is the sum of squared site residuals:

```text
SSE(x) = sum_i (F_i(x) - y_i)^2
```

Smaller SSE means a better fit to these observations. The root mean squared
error in intensity units is `sqrt(SSE / 108)`; for example, SSE 108 means an
RMSE of one intensity unit. The midpoint source in the forward-model config is
an example, not ground truth. A low SSE does not prove that the recovered
source is unique or historically correct.

To assess the search algorithms, compare repeated runs with different random
seeds and similar numbers of forward-model evaluations. Record the spread of
final SSE values and source parameters, not only the best run. A synthetic
recovery test can assess source-parameter accuracy: choose a known source,
generate observations with the forward model, then invert them. The real
Colline Pisane data cannot directly measure source-parameter error because
their true source is unknown.

## Project structure

- `src/earthquake_forward_model/`: Python forward-model package.
- `configs/`: source-model parameters and input-file selection.
- `data/`: observed macroseismic intensities and original parameter ranges.
- `scripts/`: commands for running the model and comparing Python with Fortran.
- `tests/`: automated checks for the Python implementation.
- `results/`: forward-model CSV and inversion JSON reports.
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

## Run the GA-NGA inversion

```bash
python3 scripts/run_ga_nga.py
```

The command runs a conventional GA, then an independent NGA search with two
demes. It prints the best residual in each generation and writes the best
model for the GA and each NGA deme, plus the full best-residual history, to
`results/colline_pisane_ga_nga.json`. The random seed and search settings are
in `configs/colline_pisane_ga_nga.json`. For a quick check, use:

```bash
python3 scripts/run_ga_nga.py --population 8 --generations 2 --output /tmp/ga_nga_check.json
```

The implementation uses integer genes on the specified parameter steps,
two-candidate tournament selection, two-point crossover, random allele
replacement, elitism, restart after ten stagnant generations, duplicate repair,
and the Fortran normalized distance rule to keep later demes away from earlier
ones. The algorithm is described in `papers/BSSA-1737.pdf` and implemented in
the original `fortran_code/pgakfdeme9.f`. The Python version uses the 12
independently bounded parameters in
`data/set17.txt`, with explicit steps in the JSON configuration. The original
`pgakfdeme9.f` uses 11 genes and derives rupture lengths from seismic moment
for a different earthquake; the Python search does not impose that coupling.
The default population and generation counts are small enough for a local run;
they are a starting search budget, not evidence of convergence. Increase them
in the JSON file for a scientific inversion and compare repeated seeds.

## Run the PSO inversion

```bash
python3 scripts/run_pso.py
```

This uses the same 12 parameter ranges and forward model as GA-NGA. The
settings are in `configs/colline_pisane_pso.json`; the best source and
best-residual history are written to `results/colline_pisane_pso.json`.
The default search budget is a starting point, not a convergence test.

## Compare GA-NGA and PSO

After both inversions have produced reports, run:

```bash
python3 scripts/compare_ga_nga_pso.py
```

The script checks the observation file and search settings, recomputes each
reported source's SSE, ranks the GA, NGA, and PSO solutions, and writes
`results/colline_pisane_optimizer_comparison.json`. It also reports the number
of distinct forward evaluations used by each search. Use `--ga-nga`, `--pso`,
and `--output` to select different report files. A single run of each method
cannot establish which algorithm is generally more reliable.

### Check inversion sources with Fortran

After running the Python inversion, run the original Fortran **forward kernel**
on its reported best sources:

```bash
python3 scripts/compare_inversion_fortran.py
```

This compiles the Fortran kernel with `gfortran` and writes
`results/colline_pisane_ga_nga_fortran_comparison.json`. The report contains
Fortran and Python residuals, predictions, distances, and KF values at every
site for each reported GA/NGA source. Use `--input` to compare a different
inversion report and `--output` to choose another report location. The command
finishes normally when comparisons are recorded, even if KF values differ;
pass `--strict` to return a failure status on any full-parity mismatch.

This is a forward-model check of the Python inversion's source models. It is
not an independent run of the original Fortran GA. That program requires
PGAPack and its supplied configuration is for a different earthquake and an
11-gene source encoding. The saved `fortran_results/AG.txt` contains results
for 1934 Bihar datasets, so its residuals cannot be compared directly with
the Colline Pisane inversion.

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

The script compiles the original Fortran calculation kernel once in a temporary
directory, then compares five source models across all 108 observation sites:
the configured midpoint and four variations drawn from the bounds in
`data/set17.txt`. The variations are listed in
`configs/forward_comparison_sources.json`. Each entry has a name and the model
parameters that override the configured source. To use another set, pass
`--sources path/to/sources.json`; an empty list compares only the configured
source.

For each source, the report shows the residual, predicted-intensity mismatch
count, kinematic-value (KF) mismatch count, and largest station-distance
difference. Mismatched site numbers and values are printed. The script exits
with a nonzero status if any source exceeds the comparison tolerances.
It also saves the comparison to
`results/colline_pisane_fortran_python_comparison.json`, including the model,
residuals, mismatch sites, and Python and Fortran values at every observation
site. Pass `--output path/to/report.json` to choose another location. The JSON
report is written even when the parity check fails.

With the current five sources, all 540 predicted intensities and all five
residuals agree. The Python model now stops combining rupture samples when the
second rupture ends, matching the Fortran end marker. This resolves the KF
differences for mixed bounds A. Ten KF values across upper bounds and mixed
bounds B still exceed the default relative tolerance of `2e-4`; the largest
remaining relative difference is about 0.24%. The overall parity check
therefore still fails. A diagnostic check found that using Fortran's station
distance and azimuth in Python removes seven of these ten differences. For the
other three, Python agrees with a double-precision Fortran build when both use
the same geometry. The remaining differences are consistent with sensitivity
to Fortran single-precision rounding; the default check keeps its original
tolerance and still reports them. The original `fortran_code/` directory is not
modified.

## Important Python names

- `SourceModelParameters`: parameters describing one proposed earthquake source.
- `IntensityObservation`: one observed intensity and its geographic location.
- `evaluate_forward_model()`: predicts intensities for one source model.
- `ForwardModelResult`: predicted values, distances, and total misfit.
- `load_intensity_observations()`: reads the observation data file.
