# ea_for_inversion_for_Mugello
Solving an inversion problem regarding seismic wases on two earthquakes: California and Mugello 

# Files description

franco results 
- AG.txt = file which cointains GA results

fortran code 
- pqu7v2.f = main code v2
- pqu7v1.f = main code v1 (disused)
- pgakfdeme9.f = GA code 
- pgakf9.conf = support code file of the GAs
- readat.f = subroutine (already in pqu7v1.f)
- reaset.f = subroutine (already in pqu7v1.f)

papers
- BSSA-WiNa.pdf = paper 
- BSSA-1737.pdf = paper 

data 
- lib1846-n108.txt = data of 1846
- set17.txt = ?  

## Python single-model inversion

The deterministic inversion kernel from `fortran_code/pqu7v2.f` is available
in `src/mugello_inversion`. The original Fortran files are kept unchanged.
There is no PSO, GA, or Monte Carlo loop: the script evaluates exactly one
source model, computes predicted macroseismic intensities, and reports the
sum-of-squared residuals.

Run the included Mugello test model (the midpoint of the ranges in `set17.txt`):

```bash
python3 scripts/run_single_inversion.py
```

Results are written to `results/single_test.csv`. Edit
`configs/single_test.json` or pass another JSON file with `--config` to test a
different model. Run the dependency-free test suite with:

```bash
python3 -m unittest discover -s tests -v
```

### Compare Python with the original Fortran

If `gfortran` is installed, the parity script compiles the original Fortran
kernel in a temporary directory and evaluates the same fixed model in both
languages:

```bash
python3 scripts/compare_fortran_python.py
```

It compares all 108 predicted intensities, the sum-of-squares residual, station
distances, and intermediate kinematic-function values. It exits with status 0
only when the comparison passes. The original `fortran_code/` directory is
read-only during this process; the temporary executable and extracted kernel
are automatically removed.

