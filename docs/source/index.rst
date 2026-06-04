
Kea documentation
=================

\ **KEA**\ is a backronym for \ **A**\ nalysis \ **E**\ ngine (Mar)\ **K**\ . It holds all the basic, functional style analysis scripts for turbulence analysis, with the purpose of analysis for the intracluster medium (ICM), but written generally enough to apply to many other regimes. For example, this package has been used to analyze solar wind timeseries data, interstellar medium telescope observations, and three-dimensional turbulence simulations. It provides:

* Generation of synthetic stochastic fields, application of noise, and generation of masks
* Statistical calculation methods like: structure function, correlation functions, and power spectra via different methods

To install
----------

To install, simply add the following to your `PYTHONPATH` environment variable in your `~/.bashrc` file

.. code-block:: bash
    :name: install_kea

    export PYTHONPATH=path/to/Kea:$PYTHONPATH

Alternatively, you can call the following at the start of each of your python files that will require Kea:

.. code-block:: python
    :name: install_kea_2

    import sys
    sys.path.append('path/to/Kea/')

Requirements
------------

The following packages are required by default

.. code-block:: bash
    :name: required_packages

    numpy
    scipy
    sympy
    matplotlib

and to use distributed or GPU processing (respectively):

.. code-block:: bash
    :name: optional:packages

    mpi4py
    cupy

File structure
--------------

.. code-block:: bash
    :name: file_structure

    Kea/
        docs/
        tutorials/
        kea/
            simulator/
                gaussian.py
                masking.py
                noise.py
                nongaussian.py
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
                geometry.py
                plotting.py

User Documentation
------------------

.. toctree::
    :maxdepth: 2

    spectra
    statistics_functions
    simulation
    utility
