
"""
per_spectra.py
Implements power spectra calculations using direct FFT conversion and then squaring the absolute value.
This is essentially accomplishing the periodogram method via the FFT.

Functions
---------
modal_spectrum\n
"""

import numpy as np
from scipy import fft

from .. import statistics_base


def modal_spectrum(ar1, ar2=None, wfunc=None, p=0.1, phys_dims=None, norm=True):
    """spectrum(ar, len)

    Computes the spectrum of the given array via Fourier transform.
    The powers of n are really products of each different dimension,
    as they can have different lengths and number of points.

    `$P_{n}(k) = (dk/2pi)**n |dx**n FFTn{ar}|^2$`\n
    `$k_m = 2 pi m / L$`\n
    `$m \in [-N/2, N/2]$`

    This is the classical/Schuster periodogram

    Args:
        ar1 (np.ndarray): Array to compute the spectrum of
        ar2 (np.ndarray): Array if computing the cross spectrum
        wfunc (func): Window function to apply
        p (float): Percentage of the total number to create the window function for
        lenx,leny,lenz (float): System size in x,y,z directions
            to take into account the anisotropy of system if any
    Returns:
        kvec (tuple): Wavenumber arrays
        fek (np.ndarray): Spectrum of the array
    """
    if ar2 is not None:
        assert ar1.shape == ar2.shape, 'Provided arrays are not equal sizes'
    if phys_dims is None:
        phys_dims = [2. * np.pi for _ in range(len(ar1.shape))]
    if wfunc is not None:
        # Compute the modified periodogram
        #print('WARNING: you will need to normalize by an additional factor')
        ar1 = statistics_base.apply_ndim_window(ar1, wfunc, p, norm=norm)
        if ar2 is not None:
            ar2 = statistics_base.apply_ndim_window(ar2, wfunc, p, norm=norm)

    kvec = []
    dx = []
    dk = []
    for i, N in enumerate(ar1.shape):
        dx.append(phys_dims[i]/N)
        dk.append(2.*np.pi/(N*dx[i]))
        kvec.append(fft.fftshift(fft.fftfreq(N))*2.*np.pi/dx[i])

    far1 = np.prod(dx)*fft.fftshift(fft.fftn(ar1))
    if ar2 is not None:
        far2 = np.prod(dx)*fft.fftshift(fft.fftn(ar2))

    if ar2 is not None:
        fek = far1*np.conjugate(far2)
        fek = fek.real
    else:
        fek = np.abs(far1)**2

    # Normalize the power spectrum
    ## NOTE: With BOTH the normalizations, we obtain the expected DFT formula.
    fek = fek / np.sum(np.ones_like(ar1) * np.prod(dx))
    ## NOTE: This normalization is not going to be applied.
    ##      Apply this during the integration.
    #fek = fek * np.prod(dk) / (2.*np.pi)**ar1.ndim

    return tuple(kvec), fek
