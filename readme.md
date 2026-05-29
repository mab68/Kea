
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

## File structure
```
Kea/
    docs/
    examples/
    kea/
        __init__.py
        simulator/
            __init__.py
            nongaussian.py
            gaussian.py
        statistics/
            __init__.py
            spectra/
                __init__.py
                fourier.py
                blackman_tukey.py
                difference_of_gaussian.py
                equiv_spectrum.py
            statfunc/
                __init__.py
                strfn.py
                corr.py
        utils/
            __init__.py
            geometry.py
            binning.py
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
