Spectral Density Estimation
---------------------------

Currently **KEA** offers the following spectral estimation techniques:

* Fourier/Periodogram (FFT)
* Blackman Tukley/Correlogram (BT)
* Difference of Gaussian (DoG)
* Equivalent Spectrum (ESF)

The default (fast-Fourier transform) normalization available for **kea** is:

.. math::
   y[\mathbf{n}] = (\Delta k)^{D} \sum_\mathbf{m} \tilde{y}[\mathbf{m}] e^{2\pi i \mathbf{n} \cdot \mathbf{m}/N}

.. math::
   \tilde{y}[\mathbf{m}] = \frac{(\Delta x)^{D}}{(2\pi)^D} \sum_\mathbf{n} y[\mathbf{n}] e^{-2\pi i \mathbf{n} \cdot \mathbf{m}/N}

where the summation is over all :math:`\mathbf{n}`. The :math:`D`-dimensional DFT is implemented using Numpy and then multiplied by the correct scaling factors. This normalization is sometimes called the :math:`T` normalization in *kea*. Other normalizations are sometimes used instead; these can be converted to and from using :py:func:`kea.statistics.spectra.convert_normalization_convention()`.


Periodogram
^^^^^^^^^^^

The periodogram (acronymized as FFT and given by :py:func:`kea.statistics.spectra.fourier_modal_spectrum()`) spectral estimate is simply:

.. math::
   E^{FFT}[\mathbf{m}] = \frac{1}{L^D} |\tilde{y}[\mathbf{m}]|^2

This is an estimation of the *modal* spectrum (hence the vector :math:`\mathbf{m}` argument) which needs to be angle-averaged or angle-integrated.


Correlogram
^^^^^^^^^^^

The Blackman-Tukey (BT; sometimes called a correlogram) method uses the (discrete) Fourier transformation of the ACF given by,

.. math::
   E^{\mathrm{BT}} = \Re \left\{\widehat{Q}[\mathbf{m}] \right\},

where :math:`\widehat{Q}[\mathbf{n}]` is the (discrete) Fourier transform of the ACF multiplied by a window function, :math:`W[\mathbf{n}]`:

.. math::
   Q[\mathbf{n}] = R[\mathbf{n}] W[\mathbf{n}].

Note that with the unbiased normalization (:math:`\widetilde{N}[m] = N-m-1``), :math:`\widehat{Q}[\mathbf{m}]` is not guaranteed to be positive semi-definite and there may result in negative spectral estimates which is undesirable in most applications.

To perform this method, first call :py:func:`kea.statistics.statfunc.complete_symmetric_correlation_function()` to get the ACF, and then call :py:func:`kea.statistics.spectra.bt_modal_spectrum()` to get the modal spectrum.


Difference of Gaussian
^^^^^^^^^^^^^^^^^^^^^^

The difference-of-Gaussian method is described for the continuous case with no gaps; there are additional convolutions that can be performed to account for gaps in the data.

First, we define the real-space scale $\sigma$, and the corresponding pixel-space scale $o$ as,

.. math::
    \sigma = o \Delta x = b k^{-1} = b \left( m \Delta k \right)^{-1}.

For :math:`k = \left\{\frac{2\pi}{L},\, \dots,\, \frac{\pi N}{L} \right\}`, then :math:`\sigma = \left\{\frac{b L}{\pi N},\, \dots,\, \frac{bL}{2\pi}\right\}` and :math:`o = \left\{\frac{b}{\pi},\, \dots, \,\frac{bN}{2\pi} \right\}`. Similar to the ESF method, :math:`b=\pi` would be a natural conclusion to arrive at based on a signal in a domain :math:`L`. A common interpretation is :math:`b=\sqrt{2}`.

For the pixel-scale $o$, the normalized scale-filtered field is,

.. math::
    y_{o}[\mathbf{n}] = M[\mathbf{n}] \Xi[\mathbf{n}] \frac{( y \ast G_{o} )[\mathbf{n}]}{( \Xi \ast G_{o}) [\mathbf{n}] },

where `scipy.ndimage.gaussian_filter` is used to perform the discrete convolution when using CPU calculations and `cupyx.ndimage.gaussian_filter` when using GPU calculations (:py:func:`kea.utils.set_calculation_mode()`). The Gaussian function is truncated at $10\times o$ to increase performance (smaller truncations were found to provide insufficient accuracy). The function :math:`\Xi[\mathbf{n}]` represents a (general) floating-point exposure map (for e.g., X-ray surface brightness observations where :math:`\Xi[\mathbf{n}] = 1` represents a pixel that has (relative) complete observation) that also defines the boolean mask,

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
    \overline{E}^{\mathrm{DoG}}[o] = \frac{1}{L^D} \frac{N^D}{\sum_{\mathbf{n}} \Xi[\mathbf{n}]} \frac{V[o]}{\sum_{\mathbf{n}} \left( G_{o_1}[\mathbf{n}] - G_{o_2}[\mathbf{n}] \right) ^2} \frac{\Delta x^{2D}}{\Delta k},

where :math:`o` is related to the Fourier-space :math:`m = \frac{b}{o}\frac{1}{\Delta x \Delta k}`.


Equivalent Spectrum
^^^^^^^^^^^^^^^^^^^

To estimate a power spectrum using the structure function, first, you must:

- Calculate the second-order structure function: :math:`S^{(2)}[\mathbf{m}]` (using :py:func:`kea.statistics.statfunc.structure_function()`).
- Average (bin) the second-order structure function over shells of magnitude :math:`m = |\mathbf{m}|`: :math:`\overline{S}^{(2)}[m]` (using :py:func:`kea.utils.binning.bin_data()`).

Then, in :py:func:`kea.statistics.spectra.esf_integrated_spectrum()`, the following algorithm is performed:

- Estimate an "uncorrected" equivalent spectrum, :math:`\widetilde{\mathcal{E}}^{\mathrm{ESF}}[m]` using the following relationship: :math:`\mathcal{E}^{\mathrm{ESF}}[m] = \frac{1}{2} \frac{1}{b} (m \Delta x)^2 \frac{\Delta \overline{S}^{(2)}[m]}{\Delta \left(m \Delta x \right)}`, where :math:`\frac{\Delta \overline{S}^{(2)}[m]}{\Delta \left(n \Delta x \right)}` represents a finite-difference estimate of the derivative. In the above equation, :math:`m` still represents the lag-shift. To convert to a wavenumber, a factor :math:`b` is required.
- Estimate the local power-law slope of the "uncorrected" spectrum.
- Use an analytical expression for the bias for a pure power-law to derive a wavenumber dependent correction factor.
- Return a "debiased" spectrum: :math:`\mathcal{E}^{\mathrm{ESF}}[m]`.

`kea` performs the above algorithm and returns both the "uncorrected" and "debiased" spectral estimates along with the associated (equivalent) wavenumbers.


Module Contents
^^^^^^^^^^^^^^^

.. automodule:: kea.statistics.spectra
   :members:
   :show-inheritance:
   :undoc-members:
