
"""
strfn_spectra.py

Converts second order structure function to an equivalent spectra

Functions
---------
omni_spectrum\n
"""

import numpy as np

from scipy.interpolate import interp1d

from .. import statistics_base
from ..statfunc import statfunc_base


def interpolate(k_sf, fek_sf, k_interp):
    """interpolate(k_sf, fek_sf, k_interp)

    Interpolates the structure function spectrum to `k_interp` to match
        the Fourier wavenumbers
    
    Args:
        k_sf (np.ndarray):
        fek_sf (np.ndarray):
        k_interp (np.ndarray):
    Returns:
    """
    f_int = interp1d(np.log10(k_sf), np.log10(fek_sf), kind='linear', fill_value='extrapolate')
    fek_new = f_int(np.log10(k_interp))
    return 10**fek_new


def omni_spectrum(ell, sf2, a=1., b=1.):
    """omni_spectrum(ell, sf2, a, b)
    
    Calculates the equivalent spectrum, the integrated/omni spectrum
        from the derivative of the structure function

    `a`, and `b` are scalings for the ambiguity present in the equivalent
        wavenumber

    Args:
        ell (np.ndarray): 
        sf2 (np.ndarray): Averaged 2nd order structure function
    """

    dS = np.gradient(sf2)
    dell = np.gradient(ell)

    fek = (ell)**2 * (dS/dell) * (1. / 2.) / a
    kk = b / ell

    return kk[::-1], fek[::-1]
