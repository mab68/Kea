---
title: 'kea: A Python package of tools for consistent power spectral analysis of data on arbitrary dimensions'

tags:
  - Python
  - astronomy
  - fluid dynamics
  - turbulence
  - synthetic fields
  - correlation function
  - structure function
  - power spectral density
  - intracluster medium
  - solar wind

authors:
  - name: Mark A. Bishop
    orcid: 0009-0002-8645-5139
    equal-contrib: true
    affiliation: 1
  - name: Sean Oughton
    orcid: 0000-0002-2814-7288
    equal-contrib: true
    affiliation: 2
  - name: Tulasi N. Parashar
    orcid: 0000-0003-0602-8381
    equal-contrib: true
    affiliation: 1
  - name: Yvette C. Perrott
    orcid: 0000-0002-6255-8240
    equal-contrib: true
    affiliation: 1

affiliations:
 - name: School of Chemical and Physical Sciences, Victoria University of Wellington, Wellington 6012, New Zealand
   index: 1
 - name: Department of Mathematics, University of Waikato, Hamilton 3240, New Zealand
   index: 2

date: 5 June 2026

bibliography: paper.bib
---

# Summary

Power spectral density techniques are widely used across disciplines but often differ in normalization and data handling, with most fields relying on a limited set of established methods. Historically, researchers have been required to implement their own routines or personally request code from others. This increases barriers to entry for performing analysis and requires a deep understanding of specific Fourier conventions to avoid errors. Without a standardized framework, order-unity normalization differences can creep in, which can lead to physical inferences that are mathematically inconsistent or physically erroneous. To address this, we present a computational package that provides consistently normalized estimators that are adaptable to multiple data types -- including intracluster medium surface brightness fluctuations, in-situ solar wind time-series, and simulation cubes.

# Statement of Need

Autocorrelation functions (ACF), power spectral densities (PSD), and structure functions (SF) serve as the primary mathematical framework for quantifying the statistical properties of stochastic phenomena across varying spatial and temporal scales. In the context of fluid dynamics and astrophysics, these tools allow researchers to decompose complex, multiscale signals into their constituent parts. Of particular interest, especially within the study of (magneto)-hydrodynamic turbulence, are the characteristic scales (such as the integral scale $L$ where energy is injected, and the dissipation scale $\eta$), the power-law indices that define the energy cascade (e.g., the Kolmogorov $-5/3$), and their respective amplitudes which dictate the total turbulent energy budget.

These phenomena are rarely captured in the full continuum of values in the available space in which measurements occur. In other words, in a lot of practical cases, observations are restricted to only 1D slices, 2D projections, or 2D slices. For example, in the case of the solar wind, in-situ measurements are performed by sensors moving in relation to the plasma rest frame. These sensors generate a time-series of data: such as magnetic field (vector field), or density (scalar field) measurements. As another pertinent example, surface brightness observations of the intracluster medium (ICM) observe 2D (emission weighted) projections of scalar fields, or projections of line-of-sight components of a vector field. Often these observations are calibrated or compared against more accessible regimes, or high-fidelity simulations.

The development of robust astrophysical and fluid models to explain these phenomena is inherently an iterative process. As observational data reach higher sensitivities and angular resolutions, researchers increasingly rely on high-fidelity, simulations to probe unobservable dynamics and generate artificial observations. Meaningfully comparing these complex simulations against lower-dimensional, real-world observations necessitates a consistent, universal, and dimensionally-adaptable approach.

Beyond the choice of statistical tool, the reliability of the output is heavily dependent on the nuances of spectral estimation. There exists a vast library of signal processing techniques designed to mitigate the distortions, aliasing, and biases inherent in finite sampling [@Stoica.Moses05; @Maciejewski.etal09; @Sefusatti.etal16]. However, the implementation of these techniques introduces its own set of variables: how the data are windowed to prevent spectral leakage, how the estimates are binned in $k$-space, and how the resulting power is normalized (e.g., ensuring Parseval's theorem is satisfied). For example, there are a total of six different $R$ packages that provide spectrum estimations with varying normalization options [@Barbour.Parker22].

If these methodological choices are not standardized, they can induce spurious physical behavior. For example, improper binning can artificially flatten a spectral slope. This could become a significant issue when comparing across different studies or when cross-correlating observational data with numerical models. Without a rigorous, consistent approach to these estimations, the resulting physical interpretations (such as the injection or dissipation scales) may reflect the limitations of the signal processing rather than the underlying physics.

We thereby introduce `Kea`, a Python package that implements several different dimensionally-agnostic scale-dependent statistical estimation techniques for use in turbulence analysis. `Kea` is available on GitHub via https://github.com/mab68/Kea.

Power-law correlations are a key feature of turbulence (particularly with high Reynolds numbers flows). The scale-to-scale energy transfer associated with the cascade of energy from large scales to smaller scales leads to a power-law power spectrum which is indicative of fractal-like structure where small-scale features are statistically similar to large-scale ones. 

Observing the complete set of information provided by self-similar physical phenomena can be practically challenging. In particular, we highlight that the majority of astronomy and astrophysical observations are driven by the detection and manipulation of photons. The properties of these photons are determined by the conditions of the source (temperature, density, velocity, etc.) of the emitting material and its subsequent interaction with any intervening material. Determining the conditions of the source from the photons alone can be a serious challenge. Noise and resolution constraints may also limit the accuracy of the inference and increase complexity. To understand biases and systematics or comparison of theory to observation, where data is lacking, it can be necessary to compute expected observational properties from controlled surrogate models [@Haworth.etal18; @Simionescu.etal19]. `Kea` additionally provides dedicated utilities for generating synthetic spatial fields.

# State of the Field

While the mathematical definitions of autocorrelation functions, power spectral densities and structure functions are well-established, their application to "real-world" datasets show significant variability in subtle ways. For example, normalizations of the difference-of-Gaussian PSD method differ [@Arevalo.etal12; @Churazov.etal12; @Zhou.etal22]. Additionally, the exact normalization factors change depending on the chosen PSD representation (e.g., angle-averaged, angle-integrated, or amplitude spectra). To resolve these inconsistencies, `Kea` provides dedicated routines for conversion between both these spectral representations and their corresponding normalization conventions.

Currently available software for ICM turbulence analysis are: `turbustat` [@Koch.etal19], and `PITSZI` [@Adam.etal25]. Both of these solutions require 2D data, thus are not available for applications to e.g., solar wind turbulence analysis or simulation cubes. Additionally, they both use a nested sequence of object-oriented class based design with built-in handling with assumptions for their respective fields: the interstellar medium for `turbustat`, and the intracluster medium for `PITSZI`. Hence, modifying for individual or specific needs is challenging.

For SF calculations, `fastSF` [@Sadhukhan.etal21] supports 2D and 3D datasets but strictly requires uniform grids, rendering it incompatible with gapped or irregularly sampled data. While other Python packages like `fluidsf` [@Wagner.etal25] and `PyTurbo_SF` [@Ayouche.etal26] offer SF calculations, they lack a unified framework that couples arbitrary dimensionality handling with gapped-data support.

Finally, tools like `powerbox` [@Murray18], `GaussianRandomFields.jl` [@Robbe23], and `DRDMannTurb` [@Izmailov.etal24] generate synthetic turbulence or Gaussian random fields, they are lagely decoupled from the analytical toolkits required to seamlessly validate empirical estimators against these synthetic baselines.

# Software Design

`Kea` is designed with two main goals: provide standardized tools for robust, multidimensional spectral estimation, and to generate controlled synthetic fields for validating these analytical methods. To efficiently support these objectives, `Kea` is written as a Python package that operates directly on `NumPy` arrays, ensuring it is easily integrable across varying fields of research and workflows. `Kea` additionally has optional support for GPU computation using `CuPy` for structure function, autocorrelation function, and difference-of-Gaussian PSD estimation.

![Spectral estimation capabilities of `Kea`. The top panel displays a 1D synthetic timeseries generated by the software, while the bottom panel illustrates the resulting spectral estimates computed using `Kea`'s implementations of the periodogram (FFT), Blackman-Tukey (BT), equivalent spectrum (ESF), and difference-of-Gaussian (DoG) methods.\label{fig:PSD_example}](plots/PSD_example.png)

`Kea` provides discrete implementations of several key spectral estimation techniques. An application of these methods to a synthetic timeseries is demonstrated in \autoref{fig:PSD_example}. Since the autocorrelation function (ACF), structure function (SF), and the FFT and BT spectral estimates yield $D$-dimensional functions, it is often necessary to bin them to reduce their dimensionality. To address this, `Kea` provides a general binning routine applicable to both the spectral estimates (PSD) and the lag-functions (ACF, SF).

![Synthetic multifractal test fields generated by `Kea`, demonstrating the package's support for 1D (left), 2D (middle), and 3D (right, shown as a 2D slices) spatial data structures. These generation tools allow users to easily create controlled datasets for testing and benchmarking.\label{fig:example_synthesis}](plots/example_synthesis.png)

Synthetic fields have been used extensively to validate methods and measure turbulence in the interstellar medium [@Brunt.Heyer02; @Miville-Deschenes.etal03; @Esquivel.etal03; @Ossenkopf.etal06] and similarly for the ICM [@Vogt.Ensslin05; @ZuHone.etal16; @XrismCollaboration.etal25]. `Kea` provides methods to produce (monofractal and multifractal) synthetic fields with correlations parameterized by power spectral density. These fields are synthesized in Fourier-space and Fourier-transformed to provide real-space fluctuation fields [@Barnsley.etal88; @Lakhal.etal25].

For example, shown in \autoref{fig:example_synthesis} are one, two, and three-dimensional multifractal fields. Thus, the below PSD estimators can be tested for validity with known/expected forms, and their performance tested under constraints of noise and missing data.

# Research Impact Statement

`Kea` bridges the gap between empirical spectral estimation and synthetic field generation, providing a unified framework for researchers analyzing turbulence and continuous spatial data. By operating directly on standard `NumPy` arrays without rigid, domain-specific assumptions, `Kea` serves as an accessible, field-agnostic toolkit for calculating scale-dependent statistics (ACFs, PSDs, and SFs) across 1D, 2D, and 3D datasets.

Crucially, `Kea` enables researchers to rigorously model complex observational biases -- such as noise and missing data gaps -- by generating controlled, stochastic monofractal and multifractal fields. This integrated approach allows users in fields ranging from astrophysics to fluid dynamics to robustly benchmark and validate spectral estimators against synthetic ground truths before applying them to messy, real-world observational data.

In summary, `Kea` offers the following:

* __Domain-agnostic & Lightweight Design:__ Built directly on `NumPy` arrays making it easily integrable into existing workflows across astrophysics, fluid dynamics, and spatial analytics. Has optional additional GPU support with `CuPy`.
* __Flexible Dimensionality Support:__ Native handling for 1D time series, 2D image data, and 3D simulation cubes.
* __Gapped Data Compatibility:__ Provides methods for estimation of structure functions, autocorrelation functions, and power spectral densities on datasets with gaps represented by `NaN`.
* __Unified Analytical Toolkit:__ Provides discrete implementations for periodogram, Blackman-Tukey, equivalent spectrum, and difference-of-Gaussian spectral estimates. 
* __Normalization & Representation Standardization:__ Offers built-in conversion routines between different PSD binning methods and normalizations.
* __Synthetic Field Generation:__ Synthesizes controlled $D$-dimensional stochastic fields (both monofractal and multifractal) parameterized by PSDs to create ground-truth baselines for benchmarking algorithms.

Comprehensive documentation is available at https://astrokea-docs.readthedocs.io/en/latest/index.html with tutorials, and practical examples. The documentation additionally details the discretized scale-dependent statistical estimations, binning routines, and synthetic fields.

# AI Usage Disclosure

The paper was initially drafted in full by the authors. Google Gemini 3.1 was used after the drafting of this manuscript to assist with linguistic polishing to improve manuscript clarification. Google Gemini 3.1 was also used to suggest improvements to the code. All AI-assisted text and code were thoroughly reviewed, tested, and verified by the authors to ensure correctness. The authors assume full responsibility for the final content and scientific accuracy of the manuscript and code.

# Future Work

It is intended for future development of `kea` to implement a general interpretation of spectral estimation using arbitrary filters along with automatic implementation of debiasing using the non-parametric local power-law approximation that is currently only available for the ESF method.

Another area of potential improvement will be the implementation of synthetic fields with anisotropy. This would be useful to test the statistics of projected fluctuations in more complex scenarios.

# Acknowledgements

This project was supported by the Marsden Fund Council from New Zealand Government funding, managed by Royal Society Te Apārangi (No. E4200).

# References

