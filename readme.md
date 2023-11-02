
# **KEA**

**KEA** is a backronym for **A**nalysis **E**ngine (Mar)**K**. It holds all the basic, functional style analysis scripts for turbulence analysis, with the purpose of analysis for the intracluster medium (ICM), but written generally enough to apply to many other regimes.

It provides:
- Powerlaw fitting, intracluster medium mean profile fitting, correlation length fitting
- Applying Poisson noise, and noise analysis methods
- Generation of synthetic fractional Brownian motion fields
- Statistical calculation methods like: structure function, correlation lengths, power spectra via different methods
- Data reading
- Accuracy measurement functions
- Helpful plotting functions
- and much more!

## Todo list

This is my todo list for this refactoring session:
- [x] *simulator/fBm.py*: change the create_fbm() function parameters
- [x] *simulator/exposure.py*: create exposure.py and move exposure and mask generation to here
- [x] *statistic/moments.py*: handle projections, change how the dx/dk is done
- [ ] *statistic/spectra/arevalo_spectra.py*: use scale_filter.py
- [ ] *statistic/spectra/strfn_spectra.py*: add the derivative form of the equivalent structure function power spectrum calculation
- [x] *utils/data_reader.py*: modernize
- [x] *utils/data_writer.py*: modernize
- [x] *utils/funcs.py*: modernize
- [x] *utils/plotting.py*: modernize
- [x] *utils/evaluate.py*: add in MSE, MAPE, and bias methods

Short term goals:
- [ ] *statistics/spectra/arevalo_spectra.py*: add Poisson error
- [ ] *utils/funcs.py*: exponential powerlaw (decay and growth region)

Long term goals of this repository:
- [ ] *simulator/exposure.py*: add synthetic exposure map generation
- [ ] *fitting/profile.py*: Add emcee fitting for the cluster profiles 
- [ ] *fitting*: Correlation scale fitting
- [ ] *noise*: Fix the problems with needing to include the exposure map for generating realizations of the noise. Generalized apply_noise() method
- [ ] *noise*: Fix the problems with the noise floor being off
- [ ] *simulator/fBm.py*: radially dependent power spectrum
- [ ] *simulator/resolution.py*: modernize
- [ ] *statistic/spectra/arevalo_spectra.py*: arevalo_spectra.py, multithreading

Goals I am not sure of yet:
- [ ] *statistic/spectra/arevalo_spectra.py*: N-dimensional version

## File structure
```
Kea/
    fitting/
        Provides fitting routines for mean ICM profiles, correlation lengths, and spectral powerlaws.
    noise/
        Provides methods focused on adding (analyzing) noise to (on) the data
    simulator/
        Provides methods to create synthetic fluctuation fields, exposure maps and masks
    statistics/
        spectra/
            Implementation for several different power spectrum estimation methods
        statfunc/
            Implementation of additional turbulence statistical measures
    utils/
        Additional helpful utilities
    
example_notebooks/
    Example Jupyter notebooks for how to use the various capabilities of the package
tests/
    Unit tests
```

## To install

To install, simply add the following to your `PYTHONPATH` environment variable in your `~/.bashrc` file
```bash
export PYTHONPATH=path/to/Kea:$PYTHONPATH
```
Alternatively, you can call the following at the start of each of your python files that will require Kea:
```Python
import sys
sys.path.append('path/to/Kea/')
```

## Requirements
```
numpy
scipy
skimage
h5py
```
For the mean profile fitting
```
lmfit
emcee
```
For faster correlation, and structure function calculation
```
mpi4py
```
For the *flatsky* power spectrum method
```
pymaster
```
