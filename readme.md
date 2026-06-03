
# **KEA**

**KEA** is a backronym for **A**nalysis **E**ngine (Mar)**K**. It holds all the basic, functional style analysis scripts for turbulence analysis, with the purpose of analysis for the intracluster medium (ICM), but written generally enough to apply to many other regimes. For example, this package has been used to analyze solar wind timeseries data, interstellar medium telescope observations, and three-dimensional turbulence simulations. It provides:
- Generation of synthetic stochastic fields
- Statistical calculation methods like: structure function, correlation functions, and power spectra via different methods

<!-- A Kea is the only apline parrot species in the world. Endemic to New Zealand's South Island, Kea are highly intelligent and known for a propensity for vandalism: playfull interacting with humans via stealing their food, pulling apart car rubber windshield seals and wipers, and investigating backpacks of unsuspecting hikers. -->

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
                corrfn.py
                statfunc_numba.py
        utils/
            __init__.py
            geometry.py
            fitting.py
            binning.py
            plotting.py
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
numba
sympy
matplotlib
```
