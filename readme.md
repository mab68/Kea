
[![License](https://img.shields.io/github/license/mab68/kea)](https://github.com/mab68/kea/blob/Kea1.0/LICENSE) [![codecov](https://codecov.io/gh/mab68/Kea/graph/badge.svg?token=F55IMEHO85)](https://codecov.io/gh/mab68/Kea)

# 🌌 **kea** 🦜⛰️

**kea** is a backronym for **A**nalysis **E**ngine (Mar)**K**. It holds all the basic, functional style analysis scripts for turbulence analysis, with the purpose of analysis for the intracluster medium (ICM), but written generally enough to apply to many other regimes. For example, this package has been used to analyze solar wind timeseries data, interstellar medium telescope observations, and three-dimensional turbulence simulations.

### 🔮 Project Goals

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
If interpolating using Gaussian process regression:
```
scikit-learn
```
These are additional *optional* packages to enable faster calculations.

## 📖 Documentation

Documentation -- including Jupyter notebook tutorials and examples -- are available on [Read the Docs](https://astrokea-docs.readthedocs.io/en/latest/).

## 📄 Citing

If you use **kea** in your research, we would be grateful if you credit (reference) us.

### 📚 References

- There is an *in prep* paper that uses **kea** to test the different spectral estimation methods.
- There is an *in prep* JOSS paper for this Github repository.

If you use the equivalent spectrum method please also reference:
- Mark A. Bishop, Sean Oughton, Tulasi N. Parashar, Yvette C. Perrott; Direct power spectral density estimation from structure functions without Fourier transforms. Physics of Fluids 1 February 2026; 38 (2): 025107. https://doi.org/10.1063/5.0310561

See, also: https://github.com/mab68/Equivalent_Spectrum

### 🤝 Acknowledgements

This project was supported by the Marsden Fund Council from New Zealand Government funding, managed by Royal Society Te Apārangi (No. E4200).

## Community Guidelines

- At the current stage of development, anyone seeking to contribute to the **kea** codebase is asked to contact the main developers to discuss the best way to integrate their contributions. The codebase is managed on GitHub, and contributions will ultimately result in merging commits into the main branch using a Pull Request.
- It is recommended to fork the repository for active development.
- Please report issues to the GitHub issue tracking tools (https://github.com/mab68/Kea/issues). Issues can also be submitted by email to any of the corresponding authors in the above references.
- If you have any queries or are requesting support, please feel free to reach out using the GitHub issue tracker (https://github.com/mab68/Kea/issues) or via email.

