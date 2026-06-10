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

The Blackman-Tukey (BT) uses the FFT of the autocorrelation function:

.. math::
   E^{BT}[\mathbf{m}] = \tilde{R}[\mathbf{m}]

and is given by calling :py:func:`kea.statistics.spectra.bt_modal_spectrum()`.


Difference of Gaussian
^^^^^^^^^^^^^^^^^^^^^^



Equivalent Spectrum
^^^^^^^^^^^^^^^^^^^

Module Contents
^^^^^^^^^^^^^^^

.. automodule:: kea.statistics.spectra
   :members:
   :show-inheritance:
   :undoc-members:
