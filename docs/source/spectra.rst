Spectral Density Estimation
---------------------------

Currently **KEA** offers the following spectral estimation techniques:

* Fourier/Periodogram (FFT)
* Blackman Tukley/Correlogram (BT)
* Difference of Gaussian (DoG)
* Equivalent Spectrum (ESF)

The data we obtain from real observations is not continuous data, but rather, a discrete function that is obtained by sampling an underlying continuous function, :math:`y_\mathrm{c}`, at an interval :math:`\Delta x`. This gives the discrete function :math:`y[n]` and its Fourier transform :math:`\widehat{y}[m]`, defined as the following:

.. math::
   y[n] = y_c(n\Delta x) = y_c(x)

.. math::
   \tilde{y}[m] = \tilde{y}_c(m\Delta k) = \tilde{y}_c(k)

where :math:`x = n \Delta x`, and :math:`k = m \Delta k` with :math:`m,\, n \in \mathscr{Z}` (integers). 

We assume that :math:`y_\mathrm{c}(x)` is periodic on the domain :math:`x \in [0,L]` such that :math:`y_\mathrm{c}(x) = y_\mathrm{c}(x+L)` and we have sampled :math:`N` evenly separated points within this domain. Our sampling interval is therefore :math:`\Delta x = L/N`. Here, :math:`L` represents the physical domain and :math:`N` the number of grid points. The discrete function will also be periodic: :math:`y[n] = y[n+N]` for :math:`n \in [0,N-1]`, :math:`m\in[-N/2, N/2]`. As a result, the Fourier-space function is also periodic: :math:`\widehat{y}_\mathrm{c}(k) = y_\mathrm{c}(k + N\Delta k)` where :math:`\Delta k = \frac{2\pi}{N \Delta x} = \frac{2\pi}{L}` and :math:`k \in [-\pi N/L, \pi N / L]`. Alternatively, a non-periodic signal is assumed to be :math:`y_\mathrm{c}(x) = 0` for :math:`x > L`. Note that mathematically, the DFT treats the signal as periodic regardless so the function should be padded with zeros to ensure there is minimal spectral leakage due to a discontinuity at the boundary.

The last step in this process is to acknowledge the following relation for the discretization of the complex exponential term: :math:`xk = (n \Delta x) (m \Delta k) = nm \Delta x 2\pi / (N \Delta x) = 2 \pi n m / N`. Applying the relations stated gives us the discrete form of the continuous signal and its Fourier transform, where we have also replaced :math:`\mathrm{d} x,\, \mathrm{d} k` with :math:`\Delta x,\, \Delta k` respectively (see, e.g., :cite:t:`Allen.etal12`).

The default (fast-Fourier transform) normalization available for **kea** is:

.. math::
   y[\mathbf{n}] = (\Delta k)^{D} \sum_\mathbf{m} \tilde{y}[\mathbf{m}] e^{2\pi i \mathbf{n} \cdot \mathbf{m}/N}

.. math::
   \tilde{y}[\mathbf{m}] = \frac{(\Delta x)^{D}}{(2\pi)^D} \sum_\mathbf{n} y[\mathbf{n}] e^{-2\pi i \mathbf{n} \cdot \mathbf{m}/N}

where the summation is over all :math:`\mathbf{m},\mathbf{n}` e.g.,

.. math::
   \sum_{\mathbf{n}}=\sum_{n_{1}=0}^{N_{1}-1} \sum_{n_{2}=0}^{N_{2}-1} \dots \sum_{n_{3}=0}^{N_{3}-1}.

This normalization is sometimes called the :math:`T` normalization in *kea*. Other normalizations are sometimes used instead; these can be converted to and from using :py:func:`kea.statistics.spectra.convert_normalization_convention()`.

The :math:`D`-dimensional DFT is implemented using Numpy and then multiplied by the correct scaling factors. The DFT is called via :py:func:`NumPy.fft.fftn()`, and the wavenumbers are obtained using the :py:func:`NumPy.fft.fftfreq()`, providing the number of sampled data points :math:`N` (this gives :math:`m/N` for :math:`m \in [-N/2, N/2]`) which is subsequently multiplied by :math:`2\pi/\Delta x` to get the angular wavenumbers.

The DFT is well-defined when :math:`y[\mathbf{n}]\in \mathscr{R}`, however, particularly it is not always the case that :math:`y[\mathbf{n}]` has such values for all :math:`\mathbf{n} \in {1,\dots,N-1}` with constant :math:`\Delta x = L/N`. This can be due to a myriad of reasons depending on the observation type, but generally, it will result in "gaps" in the data. To retain uniform sampling, a gap in :math:`y[\mathbf{n}]` is represented by ``NaN`` (not a number; ``NumPy.NaN``). However, most implementations of the above discrete Fourier transforms do not behave when :math:`y[\mathbf{n}]` has ``NaN`` values. Hence, alternative spectral estimation methods are frequently used.


Periodogram
^^^^^^^^^^^

The periodogram (acronymized as FFT and given by :py:func:`kea.statistics.spectra.fourier_modal_spectrum()`) spectral estimate is simply :cite:p:`Schuster98`:

.. math::
   E^{FFT}[\mathbf{m}] = |\tilde{y}[\mathbf{m}]|^2 \Delta k^{D}

This is an estimation of the *modal* spectrum (hence the vector :math:`\mathbf{m}` argument) which needs to be angle-averaged or angle-integrated. When :math:`y[\mathbf{n}]` has ``NaN``, often these values are instead interpolated (e.g., with the average, or linearly).


Correlogram
^^^^^^^^^^^

The Blackman-Tukey (BT; sometimes called a correlogram) method (:cite:`Blackman.etal60`) uses the (discrete) Fourier transformation of the ACF given by,

.. math::
   E^{\mathrm{BT}} = \Re \left\{\widehat{Q}[\mathbf{m}] \right\},

where :math:`\widehat{Q}[\mathbf{n}]` is the (discrete) Fourier transform of the ACF multiplied by a window function, :math:`W[\mathbf{n}]`:

.. math::
   Q[\mathbf{n}] = R[\mathbf{n}] W[\mathbf{n}].

Note that with the unbiased normalization (:math:`\widetilde{N}[m] = N-m-1``), :math:`\widehat{Q}[\mathbf{m}]` is not guaranteed to be positive semi-definite and there may result in negative spectral estimates which is undesirable in most applications :cite:p:`Stoica.Moses05`.

To perform this method, first call :py:func:`kea.statistics.statfunc.complete_symmetric_correlation_function()` to get the ACF, and then call :py:func:`kea.statistics.spectra.bt_modal_spectrum()` to get the modal spectrum.


Difference of Gaussian
^^^^^^^^^^^^^^^^^^^^^^

The difference-of-Gaussian method is described for the continuous case with no gaps; there are additional convolutions that can be performed to account for gaps in the data :cite:p:`Arevalo.etal12,Ossenkopf.etal08a`.

First, we define the real-space scale :math:`\sigma`, and the corresponding pixel-space scale :math:`o` as,

.. math::
    \sigma = o \Delta x = b k^{-1} = b \left( m \Delta k \right)^{-1}.

For :math:`k = \left\{\frac{2\pi}{L},\, \dots,\, \frac{\pi N}{L} \right\}`, then :math:`\sigma = \left\{\frac{b L}{\pi N},\, \dots,\, \frac{bL}{2\pi}\right\}` and :math:`o = \left\{\frac{b}{\pi},\, \dots, \,\frac{bN}{2\pi} \right\}`. Similar to the ESF method, :math:`b=\pi` would be a natural conclusion to arrive at based on a signal in a domain :math:`L`. A common interpretation is :math:`b=\sqrt{2}`.

For the pixel-scale :math:`o`, the normalized scale-filtered field is,

.. math::
    y_{o}[\mathbf{n}] = M[\mathbf{n}] \Xi[\mathbf{n}] \frac{( y \ast G_{o} )[\mathbf{n}]}{( \Xi \ast G_{o}) [\mathbf{n}] },

where `scipy.ndimage.gaussian_filter` is used to perform the discrete convolution when using CPU calculations and `cupyx.ndimage.gaussian_filter` when using GPU calculations (:py:func:`kea.utils.set_calculation_mode()`). The Gaussian function is truncated at :math:`10\times o` to increase performance (smaller truncations were found to provide insufficient accuracy). The function :math:`\Xi[\mathbf{n}]` represents a (general) floating-point exposure map (for e.g., X-ray surface brightness observations where :math:`\Xi[\mathbf{n}] = 1` represents a pixel that has (relative) complete observation) that also defines the boolean mask,

.. math::
    M[\mathbf{n}] = \begin{cases}
        1 & \text{where $\Xi[\mathbf{n}] > 0$},\\
        0 & \text{otherwise},
    \end{cases}

which ensures we are not counting regions that should be masked. In other words, :math:`\Xi[\mathbf{n}]` characterize the locations of the gapped data.

For :math:`\sigma` values that are nearly equal (with :math:`\xi \approx 10^{-3}`)

.. math::
    \sigma_1 = o_1 \Delta x = \frac{\sigma}{\sqrt{1 + \xi}},\\
    \sigma_2 = o_2 \Delta x = \sigma \sqrt{1 + \xi},

the scale-filtered variance is,

.. math::
    V[o] = \sum_{\mathbf{n}} \left( y_{o_1}[\mathbf{n}] - y_{o_2}[\mathbf{n}] \right)^2,

which is then normalized to get the angle-averaged spectrum:

.. math::
    \overline{E}^{\mathrm{DoG}}[o] = \frac{N^D}{\sum_\mathbf{n} \Xi[\mathbf{n}]} \frac{V[o]}{\sum_\mathbf{n} (G_{o_1}[\mathbf{n}] - G_{o_2}[\mathbf{n}])^2 } N^{-2D},

where :math:`o` is related to the Fourier-space :math:`m = \frac{b}{o}\frac{1}{\Delta x \Delta k}`.


Equivalent Spectrum
^^^^^^^^^^^^^^^^^^^

To estimate a power spectrum using the structure function, first, you must:

- Calculate the second-order structure function: :math:`S^{(2)}[\mathbf{m}]` (using :py:func:`kea.statistics.statfunc.structure_function()`).
- Average (bin) the second-order structure function over shells of magnitude :math:`m = |\mathbf{m}|`: :math:`\overline{S}^{(2)}[m]` (using :py:func:`kea.utils.binning.bin_data()`).

Then, in :py:func:`kea.statistics.spectra.esf_integrated_spectrum()`, the following algorithm is performed :cite:p:`Bishop.etal26`:

- Estimate an "uncorrected" equivalent spectrum, :math:`\widetilde{\mathcal{E}}^{\mathrm{ESF}}[m]` using the following relationship: :math:`\mathcal{E}^{\mathrm{ESF}}[m] = \frac{1}{2} \frac{1}{b} (m \Delta x)^2 \frac{\Delta \overline{S}^{(2)}[m]}{\Delta \left(m \Delta x \right)}`, where :math:`\frac{\Delta \overline{S}^{(2)}[m]}{\Delta \left(n \Delta x \right)}` represents a finite-difference estimate of the derivative. In the above equation, :math:`m` still represents the lag-shift. To convert to a wavenumber, a factor :math:`b` is required.
- Estimate the local power-law slope of the "uncorrected" spectrum.
- Use an analytical expression for the bias for a pure power-law to derive a wavenumber dependent correction factor.
- Return a "debiased" spectrum: :math:`\mathcal{E}^{\mathrm{ESF}}[m]`.

`kea` performs the above algorithm and returns both the "uncorrected" and "debiased" spectral estimates along with the associated (equivalent) wavenumbers.

References
^^^^^^^^^^

.. bibliography::
   :style: plain
   :cited:

Module Contents
^^^^^^^^^^^^^^^

.. automodule:: kea.statistics.spectra
   :members:
   :show-inheritance:
   :undoc-members:
