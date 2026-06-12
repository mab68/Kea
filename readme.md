
[![License](https://img.shields.io/github/license/mab68/kea)](https://github.com/mab68/kea/blob/Kea1.0/LICENSE)

# 🌌 **kea** 🦜⛰️

**kea** is a backronym for **A**nalysis **E**ngine (Mar)**K**. It holds all the basic, functional style analysis scripts for turbulence analysis, with the purpose of analysis for the intracluster medium (ICM), but written generally enough to apply to many other regimes. For example, this package has been used to analyze solar wind timeseries data, interstellar medium telescope observations, and three-dimensional turbulence simulations.

## 🔮 Project Goals

- **Simplify tools that are commonly used in astrophysical plasma turbulence analysis**
- **Standardize and compare power spectral density estimation techniques**
- **Provide tools to test observation processing pipelines**: synthetic field generation methods and masking tools

## 🛠️ To install

To install, first clone this GitHub repository:
```bash
$ git clone https://github.com/mab68/kea
$ cd Kea
```
and then,
```bash
$ pip install -e .
```

Alternatively, simply add the following to your `PYTHONPATH` environment variable in your `~/.bashrc` file:
```bash
export PYTHONPATH=path/to/Kea:$PYTHONPATH
```
OR, you can call the following at the start of each of your python files that will require Kea:
```Python
import sys
sys.path.append('path/to/Kea/')
```

## 📦 Requirements

For the basic set of features provided by **kea**, the following dependencies are required:
```
numpy
scipy
sympy
matplotlib
```

If using GPU calculations **kea** will additionally require:
```
cupy
```
and if using distributed computing:
```
mpi4py
```
These are additional *optional* packages to enable faster calculations.

## 📖 Documentation

Documentation -- including Jupyter notebook tutorials and examples -- are available on [Read the Docs](https://astrokea-docs.readthedocs.io/en/latest/).

## 📄 Citing

If you use **kea** in your research, we would be grateful if you credit (reference) us. We intend to provide references shortly.

## 🤝 Acknowledgements

This project was supported by the Marsden Fund Council from New Zealand Government funding, managed by Royal Society Te Apārangi (No. E4200).

## 📚 References

- Mark A. Bishop, Sean Oughton, Tulasi N. Parashar, Yvette C. Perrott; Direct power spectral density estimation from structure functions without Fourier transforms. Physics of Fluids 1 February 2026; 38 (2): 025107. https://doi.org/10.1063/5.0310561
- https://github.com/mab68/Equivalent_Spectrum

## 📝 TODO:

- [x] Add noise helper functions
- [ ] Finish writing documentation
    - [x] ICM example
        - [ ] Fix L issue...
    - [x] Mock ICM example
    - [ ] PSD method descriptions
    - [ ] Synthesis methods
- [ ] Make into a pip-installable package
- [ ] Implement CI/CD

- [ ] Write JOSS paper
    - [x] Convert thesis chapter
    - [ ] Improvements

## 📁 File structure

```
Kea/
    docs/
        source/
            tutorials/
                synthetic_fields.ipynb
                statistical_functions.ipynb
                powerspectra.ipynb
                powerspectra_1d.ipynb
                masking.ipynb
            examples/
                solar_wind_example.ipynb
                icm_example.ipynb
    src/
        kea/
            simulator/
                noise.py
                mask.py
                synthesize.py
            statistics/
                spectra/
                    blackman_tukey.py
                    difference_of_gaussian.py
                    equiv_spectrum.py
                    fourier.py
                    spectra_base.py
                statfunc/
                    corrfn.py
                    statfunc_base.py
                    strfn.py
            utils/
                binning.py
                compute_config.py
                fitting.py
                functions.py
                geometry.py
                plotting.py
```
