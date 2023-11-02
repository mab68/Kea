
"""
corr_spectra.py

Implements power spectra calculations from the autocovariance (the correlogram)

Functions
---------
modal_spectrum\n
"""

import numpy as np
from scipy import fft


from .. import statistics_base
from ..statfunc import statfunc_base, corr


def modal_spectrum(ar, wfunc=None, p=0.1, lenn=None):
    """modal_spectrum(ar, wfunc, p, len)

    Computes the spectrum of the given array via Fourier transform of the autocorrelation.
    The Fourier transform will be computed on the N-dimensional autocorrelation function giving
    an N-dimensional spectrum from the autocorrelation function.

    This function natively calculates the correlogram.
    If the window function `wfunc` is provided then it calculates the Blackman-Tukey spectrum.

    To calculate the cross-spectrum, provide the cross correlation function.

    NOTE: This method assumes that the autocorrelation function is centered in the image.
        It will perform the necessary transformations
    NOTE: This method assumes the provided system-size is correct.
        If you are providing a correlation function from [-L, L]/[-N, N] then
        make sure to DOUBLE `lenn`

    Args:
        ar (np.ndarray): ND Autocorrelation array, ranging from -lag to lag
        wfunc (None,func): The window function to apply to the autocorrelation of `ar`
            If wfunc is provided, we calculate the spectrum via the Blackman-Tukey method
            Multiplying `ar` by the window function
        p (float): Percentage of the total length of the correlation function we are applying
            the window function to. Essentially sets the correlation function to zero past the
            point.
        lenx,leny,lenz (float): System size in x,y,z directions
            to take into account the anisotropy of system if any, default: `2pi`
    Returns:
        kk (tuple): Wavenumber array that is the magnitude of the kvec
        fek (np.ndarray): Spectrum of the array
    """
    ar_cpy = ar.copy()
    shape = np.shape(ar)
    if lenn is None:
        lenn = [2. * np.pi for _ in np.shape(ar)]
    if wfunc is not None:
        # TODO: apply an N-dimensional window function
        raise NotImplementedError
    if wfunc is not None:
        # Compute the modified correlogram i.e. the Blackman-Tukey
        ## TODO: NORMALIZE FOR WINDOWED CORRELOGRAM
        print('WARNING: you will need to normalize')
        ar_cpy = statistics_base.apply_ndim_window(ar_cpy, wfunc, p)

    # Now take the fourier transform of the N-dimensional autocorrelation function
    kvec = []
    # dx = L/N
    dx = []
    # dk = 2pi/Ndx
    dk = []
    for i, N in enumerate(shape):
        dx.append(1.*lenn[i]/N)
        # note: note including 2pi factor because of the normalization
        dk.append(1./(N*dx[i]))
        kvec.append(fft.fftshift(fft.fftfreq(N)) * 2. * np.pi / dx[i])
        #ar_cpy = np.roll(ar_cpy, N//2, axis=i)
    # 2pi normalization is done in dk already
    fek = np.prod(dk) * np.prod(dx) * fft.fftshift(fft.fftn(fft.ifftshift(ar_cpy)))
    return kvec, fek
