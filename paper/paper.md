---
title: 'kea: A Python package for consistent power spectral analysis of data on arbitrary dimensions'

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

Power spectral density techniques are widely used across disciplines but often differ in normalization and data handling, with most fields relying on a limited set of established methods. Historically, researchers have been required to implement their own routines or personally request code from others. This increases barriers to entry for performing analysis and requires a deep understanding of specific Fourier conventions to avoid errors. Without a standardized framework, order-unity normalization difference often creep in, which can lead to physical inferences that are mathematically inconsistent or physically erroneous. To address this, we present a computational package that provides consistently normalized estimators that are adaptable to multiple data types -- including ICM surface brightness fluctuations, in-situ solar wind time-series, and simulation cubes.

# Statement of need & state of the field

Autocorrelation functions (ACF), power spectral densities (PSD), and structure functions (SF) serve as the primary mathematical framework for quantifying the statistical properties of stochastic phenomena across varying spatial and temporal scales. In the context of fluid dynamics and astrophysics, these tools allow researchers to decompose complex, multiscale signals into their consituent parts. Of particular interest, especially within the study of (magneto)-hydrodynamic turbulence, are the characteristic scales (such as the integral scale $L$ where energy is injected, and the dissipation scale $\eta$), the power-law indices that define the energy cascade (e.g., the Kolmogorov $-5/3$), and their respective amplitudes which dictate the total turbulent energy budget. 

These phenomena are rarely captured in the full continuum of values in the available space in which measurements occur. In other words, in a lot of practical cases, observations are restricted to only 1D slices, 2D projections, or 2D slices. For example, in the case of the solar wind, in-situ measurements are performed by sensors moving in relation to the plasma rest frame. These sensors generate a time-series of data: such as magnetic field (vector field), or density (scalar field) measurements. As another pertinent example, surface brightness observations of the intracluster medium (ICM) observe 2D (emission weighted) projections of scalar fields, or projections of line-of-sight components of a vector field. Often these observations are calibrated or compared against more accessible regimes, or high-fidelity simulations.

Beyond the choice of statistical tool, the reliability of the output is heavily dependent on the nuances of spectral estimation. There exists a vast library of signal processing techniques designed to mitigate the distortions, aliasing, and biases inherent in finite sampling [@Stoica.Moses05; @Maciejewski.etal09; @Sefusatti.etal16]. However, the implementation of these techniques introduces its own set of variables: how the data are windowed to prevent spectral leakage, how the estimates are binned in $k$-space, and how the resulting power is normalized (e.g., ensuring Parseval's theorem is satisfied). For example, [@Barbour.Parker22] find a total of six different $R$ packages that provide spectrum estimations with varying normalization options.

If these methodological choices are not standardized, they can induce spurious physical behavior. For example, improper binning can artifically flatten a spectral slope. This could become a significant issue when comparing across different studies or when cross-correlating observational data with numerical models. Without a rigorous, consistent approach to these estimations, the resulting physical interpretations (such as the inferred viscosity of the ICM) may reflect the limitations of the signal processing rather than the underlying plasma physics.

While the mathematical definitions of autocorrelation functions, power spectral densities and structure functions are well-established, their application to "real-world" astrophysical datasets is fraught with nuance. For example, normalizations of the [@Arevalo.etal12] (i.e., difference-of-Gaussian) method differ [@Arevalo.etal12; @Churazov.etal12; @Zhou.etal22].

Current available software for ICM turbulence analysis are: `turbustat` [@Koch.etal19], and `PITSZI` [@Adam.etal24]. Both of these solutions require 2D data, thus are not readily available for comparisons with e.g., solar wind turbulence analysis, or simulation cubes. Additionally, they both use a nested sequence of object-oriented class based design with build-in handling with assumptions for their respective fields: the interstellar medium for `turbustat`, and the intracluster medium for `PITSZI`. Hence, modifying for individual needs is challenging. SF calculation is available using `fastSF` [@Sadhukhan.etal21], however, only for uniform 2D and 3D datasets. In other words, gapped datasets are not applicable. SF calculation is also available in Python with `fluidsf` [REFERENCE].

We thereby introduce `Kea` (because Mark doesn't `kea`. Or, alternatively: **A**nalysis **E**ngine Mar**K**, backwards), a Python package that implements several different dimensionally-agnostic PSD estimation techniques for use in turbulence analysis. `Kea` is available on GitHub via https://github.com/mab68/Kea.

# Implementation Details

The data we obtain in real life is not the continuous data, but rather, a discrete function that is obtained by sampling an underlying continuous function, $y_\mathrm{c}$, at an interval $\Delta x$. This gives the discrete function $y[n]$ and its Fourier transform $\widehat{y}[m]$, defined as the following:
$$
    y[n] = y_\mathrm{c}(n \Delta x) = y_\mathrm{c}(x),
$$
$$
    \widehat{y}[m] = \widehat{y}_\mathrm{c}(m \Delta k) = \widehat{y}_\mathrm{c}(k),
$$
where $x = n \Delta x$, and $k = m \Delta k$ with $m,\, n \in \mathscr{Z}$ (integers).

We assume that $y_\mathrm{c}(x)$ is periodic on the domain $x \in [0,L]$ such that $y_\mathrm{c}(x) = y_\mathrm{c}(x+L)$ and we have sampled $N$ evenly separated points within this domain, our sampling interval is $\Delta x = L/N$. Here, $L$ represents the physical domain and $N$ the number of grid points. The discrete function will also be periodic $y[n] = y[n+N]$ for $n \in [0,N-1]$, $m\in[-N/2, N/2]$. As a result, the Fourier-space function is also periodic $\widehat{y}_\mathrm{c}(k) = y_\mathrm{c}(k + N\Delta k)$ where $\Delta k = \frac{2\pi}{N \Delta x} = \frac{2\pi}{L}$ and $k \in [-\pi N/L, \pi N / L]$. Alternatively, a non-periodic signal is assumed to be $y_\mathrm{c}(x) = 0$ for $x > L$. Note that mathematically, the DFT treats the signal as periodic regardless so the function should be padded with zero's to ensure there is minimal spectral leakage due to a discontinuity at the boundary.

The last step in this process is to acknowledge the following relation for the discretization of the complex exponential term: $xk = (n \Delta x) (m \Delta k) = nm \Delta x 2\pi / (N \Delta x) = 2 \pi n m / N$. Applying the relations stated gives us the discrete form of the continuous signal and its Fourier transform, where we have also replaced $\mathrm{d} x,\, \mathrm{d} k$ with $\Delta x,\, \Delta k$ respectively [@Allen.etal12]:
$$
    y[n] = \Delta k \sum_{m = -N/2}^{N/2} \widehat{y}[m] e^{2\pi i n m / N},
$$
$$
    \widehat{y}[m] = \frac{\Delta x}{2\pi} \sum_{n=0}^{N-1} y[n] e^{-2\pi i n m / N}.
$$

We can derive the generalized $D$-dimensional form of the DFT using the same arguments provided above, which we state as the following:
$$
    y[\mathbf{n}] = \left( \Delta k \right)^{D} \sum_{\mathbf{m}} \widehat{y}[\mathbf{m}] e^{2\pi i \mathbf{n} \cdot \mathbf{m} / N },
$$
$$
    \widehat{y}[\mathbf{m}] = \left( \frac{\Delta x}{2\pi} \right)^{D} \sum_{\mathbf{n}} y[\mathbf{n}] e^{- 2\pi i \mathbf{n} \cdot \mathbf{m} / N },
$$
where the summations are over all $\mathbf{n}$ e.g., $\sum_{\mathbf{n}} = \sum_{n_1=0}^{N_1-1} \sum_{n_2=0}^{N_2 - 1} \dots \sum_{n_D=0}^{N_D - 1}$.

The $D$-dimensional DFT is implemented using the `NumPy.fft` routines which is then multiplied by the correct scaling factor. The DFT is called using the `NumPy.fft.fftn` method, and the wavenumbers are obtained using the `NumPy.fft.fftfreq`, providing the number of sampled data points $N$ (this gives $m/N$ for $m \in [-N/2, N/2]$) which is subsequently multiplied by $2\pi / \Delta x$ to get the *angular* wavenumbers.

The DFT is well-defined when $y[\mathbf{n}] \in \mathscr{R}$, however, particularly it is not always the case that $y[\mathbf{n}]$ has such values for all $\mathbf{n} \in \left\{1,\, \dots,\, N-1 \right\}$ with constant $\Delta x = L/N$. This can be due to a myriad of reasons depending on the observation type, but generally, it will result in "gaps" in the data. To retain uniform sampling, a gap in $y[\mathbf{n}]$ is represented by $\mathsf{NaN}$ (not a number; `NumPy.nan`). However, most implementations of the above discrete Fourier transforms do not behave when $y[\mathbf{n}]$ has $\mathsf{NaN}$ values. Hence, alternative spectral estimation methods are frequently used.

## Lag Functions

First, we introduce the class of functions termed the "lag-function" which use the same backend i.e., performing some operation (where the ACF performs multiplication, and the SF performs subtraction) on data that has been shifted by a lag distance.

### Autocorrelation Function

The autocorrelation function is estimated via:
\begin{equation}
    \label{eqn:discrete_acf}
    R[\mathbf{m}] = \frac{1}{\widetilde{N}[\mathbf{m}]} \sum_{\mathbf{n}} y[\mathbf{n}] y[\mathbf{n} + \mathbf{m}],
\end{equation}
where $\widetilde{N}$ is a normalization by the number of pixels, and $\mathbf{m}$ is a lag-distance in real-space pixels (rather than the wavenumbers, above), and the summation is over the *available* data (i.e., we ignore $\mathsf{NaN}$ values). The shift has to cut both $y[n]$ and $y[m]$ such that only the overlapping sequence (of equal length $N-m$) is present [@Sadhukhan.etal21].

<!-- If the function is periodic, then the lag-distance shift is performed by periodic cycling (using `NumPy.roll`) and $y[n]$ has the same number of data-points, $N$, as $y[n + m]$ for all $m$. Otherwise, the shift has to cut both $y[n]$ and $y[m]$ such that only the overlapping sequence (of equal length $N-m$) is present [@Sadhukhan.etal21]. -->

There are typically two standard ways of normalizing \autoref{eqn:discrete_acf}, either: $\widetilde{N}[m] = N-m-1$, or $\widetilde{N}^{\mathrm{biased}}[m] = N$. Where, for $D$-dimensions, $\widetilde{N}[\mathbf{m}] = \left( N_1 - m_1 - 1 \right) \dots \left( N_D - m_D - 1 \right)$ and $\widetilde{N}^{\mathrm{biased}}[m] = N^D$. If $\widetilde{N}[m] = N-m-1$ is used, then \autoref{eqn:discrete_acf} is sometimes called the standard unbiased ACF estimate. In this case, for large lags the factor $\frac{1}{\widetilde{N}[m]}$ becomes large; because the average is calculated over relatively few products, the estimate is not statistically well-constrained and is prone to high variance. However, practically large $m$ samples inhomogeneities beyond the correlation scale, so only up to a maximum lag of $m_\mathrm{max} = \frac{N}{2}$ (or even $m_\mathrm{max} = \frac{N}{4}$) should be taken. In which case, the unbiased estimator (with $\widetilde{N}[m] = N-m - 1$) has $\approx 50\%-75\%$ of $N$ at $m = m_\mathrm{max}$ and can be statistically reliable. If the biased estimator is used, then \autoref{eqn:discrete_acf} more closely resembles the continuous ACF in the case of non-periodic $y[n]$:
$$
    R[\mathbf{m}] = \left( \frac{\Delta x}{L} \right)^{D} \sum_{\mathbf{n}} y[\mathbf{n}] y[\mathbf{n} + \mathbf{m}] = \frac{1}{N^D} \sum_{\mathbf{m}} y[\mathbf{n}] y[\mathbf{n} + \mathbf{m}],
$$
for $\Delta x = L/N$.

### Structure Function

The SF follows the ACF, as,
$$
    S^{(p)}[\mathbf{m}] = \frac{1}{\widetilde{N}[\mathbf{m}]} \sum_{\mathbf{n}} \left| y[\mathbf{n}] - y[\mathbf{n} + \mathbf{m}] \right|^p,
$$
for the choice of $\widetilde{N}[\mathbf{m}] = (N_1 - m_1 - 1) \dots (N_D - m_D - 1)$, and we ignore $\mathsf{NaN}$ values. The unbiased estimator is typically used with (at most) $m_\mathrm{max} = \frac{N}{2}$.

## Spectral Estimates

Now, we introduce the discrete form of the spectral estimation techniques.

### Periodogram

The periodogram spectral estimate is simply, [@Schuster98]
$$
    E^{\mathrm{FFT}}[\mathbf{m}] = \frac{1}{L^D} \left| \widehat{y}[\mathbf{m}] \right|^2.
$$
This is an estimation of the modal spectrum (hence the vector $\mathbf{m}$ argument) which needs to be subsequently angle-averaged or angle-integrated. When $y[\mathbf{n}]$ has $\mathsf{NaN}$ values, often these values are instead interpolated (e.g., with the average, or linearly).

### Correlogram/Blackman-Tukey

The Blackman-Tukey (BT; sometimes called a correlogram) method [@Blackman.etal60] uses the (discrete) Fourier transformation of the ACF given by,
$$
    E^{\mathrm{BT}} = \Re \left\{\widehat{Q}[\mathbf{m}] \right\},
$$
where $\widehat{Q}[\mathbf{n}]$ is the (discrete) Fourier transform of the ACF multiplied by a window function, $W[\mathbf{n}]$:
$$
    Q[\mathbf{n}] = R[\mathbf{n}] W[\mathbf{n}].
$$
Note that with the unbiased normalization ($\widetilde{N}[m] = N-m-1$), $\widehat{Q}[\mathbf{m}]$ is not guaranteed to be positive semi-definite and there may result in negative spectral estimates which is undesirable in most applications [@Stoica.Moses05].

### Equivalent Spectrum

To estimate a power spectrum using the structure function, first, you must:

- Calculate the second-order structure function: $S^{(2)}[\mathbf{m}]$.
- Average (bin) the second-order structure function over shells of magnitude $m = |\mathbf{m}|$: $\overline{S}^{(2)}[m]$.

Then, the following algorithm is performed:

- Estimate an "uncorrected" equivalent spectrum, $\widetilde{\mathcal{E}}^{\mathrm{ESF}}[m]$ using the following relationship:
    $$
    \mathcal{E}^{\mathrm{ESF}}[m] = \frac{1}{2} \frac{1}{b} (m \Delta x)^2 \frac{\Delta \overline{S}^{(2)}[m]}{\Delta \left(m \Delta x \right)},
    $$
    where $\frac{\Delta \overline{S}^{(2)}[m]}{\Delta \left(n \Delta x \right)}$ represents a finite-difference estimate of the derivative. In the above equation, $m$ still represents the lag-shift. To convert to a wavenumber, a factor $b$ is required.
- Estimate the local power-law slope of the "uncorrected" spectrum.
- Use an analytical expression for the bias for a pure power-law to derive a wavenumber dependent correction factor.
- Return a "debiased" spectrum: $\mathcal{E}^{\mathrm{ESF}}[m]$.

`kea` performs the above algorithm and returns both the "uncorrected" and "debiased" spectral estimates along with the associated (equivalent) wavenumbers.

See, [@Bishop.etal26] for further details and validation.

### Difference of Gaussian

The difference-of-Gaussian method is described for the continuous case with no gaps; there are additional convolutions that can be performed to account for gaps in the data [@Ossenkopf.etal08a; @Arevalo.etal12].

First, we define the real-space scale $\sigma$, and the corresponding pixel-space scale $o$ as,
$$
    \sigma = o \Delta x = b k^{-1} = b \left( m \Delta k \right)^{-1}.
$$
For $k = \left\{\frac{2\pi}{L},\, \dots,\, \frac{\pi N}{L} \right\}$, then $\sigma = \left\{\frac{b L}{\pi N},\, \dots,\, \frac{bL}{2\pi}\right\}$ and $o = \left\{\frac{b}{\pi},\, \dots, \,\frac{bN}{2\pi} \right\}$. Similar to the ESF method, $b=\pi$ would be a natural conclusion to arrive at based on a signal in a domain $L$, however [@Bishop.etal26] suggest otherwise. A common interpretation is $b=\sqrt{2}$ [@Arevalo.etal12].

For the pixel-scale $o$, the normalized scale-filtered field is,
$$
    y_{o}[\mathbf{n}] = M[\mathbf{n}] \Xi[\mathbf{n}] \frac{( y \ast G_{o} )[\mathbf{n}]}{( \Xi \ast G_{o}) [\mathbf{n}] },
$$
where `ndimage.gaussian_filter` is used to perform the discrete convolution (by default, `scipy` is used but if a GPU is configured the `cupyx` library can be used instead). The Gaussian function is truncated at $10\times o$ to increase performance (smaller truncations were found to provide insufficient accuracy in certain circumstances). The function $\Xi[\mathbf{n}]$ represents a (general) floating-point exposure map (for e.g., X-ray surface brightness observations where $\Xi[\mathbf{n}] = 1$ represents a pixel that has relative complete observation) that also defines the boolean mask,
$$
    M[\mathbf{n}] = \begin{cases}
        1 & \text{where $\Xi[\mathbf{n}] > 0$},\\
        0 & \text{otherwise},
    \end{cases}
$$
which ensures we are not counting regions that should be masked. In other words, $\Xi[\mathbf{n}]$ characterizes the locations of the gapped data.

For $\sigma$ values that are nearly equal (with $\xi \approx 10^{-3}$)
$$
    \sigma_1 = o_1 \Delta x = \frac{\sigma}{\sqrt{1 + \xi}},\\
    \sigma_2 = o_2 \Delta x = \sigma \sqrt{1 + \xi},
$$
the scale-filtered variance is,
$$
    V[o] = \sum_{\mathbf{n}} \left( y_{o_1}[\mathbf{n}] - y_{o_2}[\mathbf{n}] \right)^2,
$$
which is then normalized to get the angle-averaged spectrum:
$$
    \overline{E}^{\mathrm{DoG}}[o] = \frac{1}{L^D} \frac{N^D}{\sum_{\mathbf{n}} \Xi[\mathbf{n}]} \frac{V[o]}{\sum_{\mathbf{n}} \left( G_{o_1}[\mathbf{n}] - G_{o_2}[\mathbf{n}] \right) ^2} \frac{\Delta x^{2D}}{\Delta k},
$$
where $o$ is related to the Fourier-space $m = \frac{b}{o}\frac{1}{\Delta x \Delta k}$.

## Binning

Since the ACF, SF, and FFT, and BT spectral estimates provide $D$-dimensional functions, it is useful to bin them to reduce the dimensionality. `Kea` provides a general binning routine that applies to the spectral estimates as well as the lag-functions.

With bin-width $\Delta b = x_+ - x_-$ and $b \in \mathscr{R}$,
$$
    F[b] = \widetilde{\sum}_{x_- \le \left| \mathbf{n} \Delta x \right| < x_+ } f[\mathbf{n}],
$$
where the following options are provided as parameters for the routine:

- Provide a binning function $\widetilde{\sum}$ which is typically e.g., an average, or a summation, or a standard error to describe the variation within the bin.
- Whether to ignore the range of values that are outside the circular or spherical shell (since the data is represented on a square/cubic grid).
- Defining specific minimum and maximum $b$: $b_\mathrm{min}, b_\mathrm{max}$.
- The location of the number that represents the bin e.g., the center of the bin ($b = \left( x_+ + x_- \right)/2$), or the left most value ($b = x_-$).
- Whether to apply a normalization of the bin width: $\Delta b^{-1}$.
- Bin with log-spaced widths. In which case, the bins are uniformly spaced in log-space.
- Specify a fixed specific number of bins, $N_\mathrm{bins}$.


Following the continuous definitions the ACF and SF are typically represented via averaging the circular/spherical shells\footnote{In the case of 1D data, there are exactly 2 values for each $\left| \mathbf{k} \right|$: the positive and negative value.}. Whereas, the angle-integrated and angle-averaged spectra are found when $\widetilde{\sum}$ is the summation and average, respectively:
$$
    \mathcal{E}[b] = \frac{1}{\Delta b} \sum_{k_- \le \left| \mathbf{k} \right| < k_+ } E[\mathbf{m}] \left( \Delta k \right)^D,
$$
$$
    \overline{E}[b] = \frac{1}{K_{D}[b]} \sum_{k_- \le \left| \mathbf{k} \right| < k_+ } E[\mathbf{m}] \left( \Delta k \right)^{D}.
$$
where $K_{D}[b]$ is the number of pixels within the bin-range (thus defining the function that averages) and we apply an additional normalization by the bin-width $\Delta b^{-1}$ (which ensures unit consistency and that $\sum \mathcal{E}[b] \Delta b$ is the total energy). Their relation is found via: $\mathcal{E}[b] = K_{D}[b] \overline{E}[b] / \Delta b$ (assuming the remaining binning properties are kept the same). The bin-shell volume coefficients for centered ($b_1 = b-\Delta b / 2$, $b_2 = b+\Delta b/2$), or left-edge bins ($b_1 = b$, $b_2 = b+\Delta b$) are:
$$
    K_D[b] = \frac{V(b_1) - V(b_2)}{\left( \Delta k \right)^{D}}
$$
where
$$
    V(r) = \frac{r^{D} \pi^{D/2}}{\Gamma\left( D/2 + 1 \right)}
$$
is the volume of a $D$-dimensional hypersphere.


## Synthesis of Stochastic Fields

Power-law correlations are a key feature of turbulence (particularly with high Reynolds numbers flows). The scale-by-scale energy transfer associated with the cascade of energy from large scales to smaller scales leads to a power-law power spectrum which is indicative of fractal-like structure where small-scale features are statistically similar to large-scale ones. 

Observing the complete set of information provided by self-similar physical phenomena can be practically challenging. In particular, we highlight that the majority of astronomy and astrophysical observations are driven by the detection and manipulation of photons. The properties of these photons are determined by the conditions of the source (temperature, density, velocity, etc.) of the emitting material and its subsequent interaction with any intervening material. Determining the conditions of the source from the photons alone can be a serious challenge. Noise and resolution constraints may also limit the accuracy of the inference and increase complexity. To understand biases and systematics or comparison of theory to observation, where data is lacking, it can be necessary to compute expected observational properties from controlled surrogate models [@Haworth.etal18; @Simionescu.etal19].

For example, synthetic fields have been used extensively to validate methods for interstellar turbulence e.g., [@Brunt.Heyer02; @Miville-Deschenes.etal03; @Esquivel.etal03; @Ossenkopf.etal06]. Synthetic fields have similarly been used for the ICM when measuring turbulence e.g., [@Vogt.Ensslin05; @ZuHone.etal16; @XrismCollaboration.etal25].

`kea` provides methods to produce (monofractal and multifractal) synthetic fields with correlations parameterized by power spectral density. Following [@Barnsley.etal88; @Lakhal.etal23], these fields are synthesized in Fourier-space and Fourier-transformed to provide real-space fluctuation fields. Thus, the above PSD estimators can be tested for validity with known/expected forms, and their performance tested under constraints of noise and missing data.


# Future Work

It is intended for future development of `kea` to implement a general interpretation of spectral estimation using arbitrary filters along with automatic implementation of debiasing using the non-parametric local power-law approximation that is currently only available for the ESF method.

Another area of potential improvement will be the implementation of synthetic fields with anisotropy. This would be useful to test the statistics of projected fluctuations in more complex scenarios.


# Acknowledgements

This project was supported by the Marsden Fund Council from New Zealand Government funding, managed by Royal Society Te Apārangi (No. E4200).


# References

