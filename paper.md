---
title: 'kea: A Python package for galactic dynamics'

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

# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Power spectral density techniques are widely used across disciplines but often differ in normalization and data handling, with most fields relying on a limited set of established methods. Historically, researchers have been required to implement their own routines or personally request code from others. This increases barriers to entry for performing analysis and requires a deep understanding of specific Fourier conventions to avoid errors. Without a standardized framework, order-unity normalization difference often creep in, which can lead to physical inferences that are mathematically inconsistent or physically erroneous. To address this, we present a computational package that provides consistently normalized estimators that are adaptable to multiple data types -- including ICM surface brightness fluctuations, in-situ solar wind time-series, and simulation cubes.

# Statement of need & state of the field

Autocorrelation functions (ACF), power spectral densities (PSD), and structure functions (SF) serve as the primary mathematical framework for quantifying the statistical properties of stochastic phenomena across varying spatial and temporal scales. In the context of fluid dynamics and astrophysics, these tools allow researchers to decompose complex, multiscale signals into their consituent parts. Of particular interest, especially within the study of (magneto)-hydrodynamic turbulence, are the characteristic scales (such as the integral scale $L$ where energy is injected, and the dissipation scale $\eta$), the power-law indices that define the energy cascade (\eg{}the Kolmogorov $-5/3$), and their respective amplitudes which dictate the total turbulent energy budget. 

These phenomena are rarely captured in the full continuum of values in the available space in which measurements occur. In other words, in a lot of practical cases, observations are restricted to only 1D slices, 2D projections, or 2D slices. For example, in the case of the solar wind, in-situ measurements are performed by sensors moving in relation to the plasma rest frame. These sensors generate a time-series of data: such as magnetic field (vector field), or density (scalar field) measurements. As another pertinent example, surface brightness observations of the intracluster medium (ICM) observe 2D (emission weighted) projections of scalar fields, or projections of line-of-sight components of a vector field. See, \eg{}\autoref{sec:sliced_and_projected_data} for some additional discussion. Often these observations are calibrated or compared against more accessible regimes, or high-fidelity simulations.

Beyond the choice of statistical tool, the reliability of the output is heavily dependent on the nuances of spectral estimation. There exists a vast library of signal processing techniques designed to mitigate the distortions, aliasing, and biases inherent in finite sampling \citep[\eg{}][]{Stoica.Moses05, Maciejewski.etal09, Sefusatti.etal16}. However, the implementation of these techniques introduces its own set of variables: how the data are windowed to prevent spectral leakage, how the estimates are binned in $k$-space, and how the resulting power is normalized (\eg{}ensuring Parseval's theorem is satisfied). For example, \cite{Barbour.Parker22} find a total of six different $R$ packages that provide spectrum estimations with varying normalization options.

If these methodological choices are not standardized, they can induce spurious physical behavior. For example, improper binning can artifically flatten a spectral slope. This could become a significant issue when comparing across different studies or when cross-correlating observational data with numerical models. We have already discussed several differences in notations and conventions in \autoref{background:notation_nomenclature} and \autoref{sec:clarifying_units}. Without a rigorous, consistent approach to these estimations, the resulting physical interpretations (such as the inferred viscosity of the ICM) may reflect the limitations of the signal processing rather than the underlying plasma physics.

While the mathematical definitions of autocorrelation functions, power spectral densities and structure functions are well-established, their application to ``real-world'' astrophysical datasets is fraught with nuance. For example, normalizations of the \cite{Arevalo.etal12} (\ie{}difference-of-Gaussian) method differ \citep[\eg{}][]{Arevalo.etal12, Churazov.etal12, Zhou.etal22}.

Current available software for ICM turbulence analysis are: \texttt{turbustat} \citep{Koch.etal19}, and \texttt{PITSZI} \citep{Adam.etal24}. Both of these solutions require 2D data, thus are not readily available for comparisons with \eg{}solar wind turbulence analysis, or simulation cubes. Additionally, they both use a nested sequence of object-oriented class based design with build-in handling with assumptions for their respective fields: the interstellar medium for \texttt{turbustat}, and the intracluster medium for \texttt{PITSZI}. Hence, modifying for individual needs is challenging. SF calculation is available using \texttt{fastSF} \citep{Sadhukhan.etal21}, however, only for uniform 2D and 3D datasets. In other words, gapped datasets are not applicable.

I thereby introduce \texttt{Kea}\footnote{Because Mark doesn't \textbf{Kea}. Or, alternatively: \textbf{A}nalysis \textbf{E}ngine Mar\textbf{K}, backwards.}\footnote{A Kea is the only alpine parrot species in the world. Endemic to New Zealand's South Island, Kea are highly intelligent and known for a propensity for vandalism: playfully interacting with humans via stealing their food, pulling apart car rubber windshield seals and wipers, and investigating backpacks of unsuspecting hikers.}, a Python package that implements several different dimensionally-agnostic PSD estimation techniques for use in turbulence analysis. \texttt{Kea} is available on GitHub via \href{https://github.com/mab68/Kea}{mab68/Kea}.

