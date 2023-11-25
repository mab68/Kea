
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

def omni_spectrum_2(ar, nar=None, lenn=None):
    if lenn is None:
        lenn = 2.*np.pi
    if nar is None:
        nar = statfunc_base.get_lagvec_magnitude_array(ar.shape)
    nar, sf1d, _ = statistics_base.bin_data(
        nar, ar, mean_func=np.nanmean, min_bin=1., cut_excess=True, ignore_nan=True, norm_bin_size=False)
    N = len(nar)
    dx = lenn/N
    dk = 2.*np.pi/(N*dx)
    kk = 2.*np.pi / (dx * nar)

    dS = np.gradient(sf1d)
    fek = (1./4.) * dS/dx
    return kk, fek

def omni_spectrum(ar, nar=None, lenn=None):
    """omni_spectrum(ar)

    Computes the omni-directional spectrum of the given array via multiplying the
        structure function by the lag space.
    
    NOTE: This will not follow Parseval's theorem, but will follow other power spectra

    Args:
        ar (np.ndarray): N-dimensional second order structure function
        nar (nd.ndarray, None): Array of lag vector magnitudes of the structure function
        lenn (float): Box size
    Returns:
        kk (np.ndarray): Equivalent wavenumber
        fek (np.ndarray): The equivalent SF spectra
    """
    if lenn is None:
        lenn = 2. * np.pi
    if nar is None:
        # We need to make sure we have a lag array
        nar = statfunc_base.get_lagvec_magnitude_array(np.shape(ar))
    else:
        assert np.shape(ar) == np.shape(nar), 'Provided lag array is not the same shape as the array'

    #if len(np.shape(ar)) > 1:
    # We need to make sure we have averaged the structure function down to 1D

    # NOTE: nar is the grid-lags i.e. `n` so need to multiply by dx to get the 'true' distance lags
    #nar, sf1d, _ = statistics_base.bin_data(
    #    nar, ar, mean_func=np.nanmean, bin_center=True, min_bin=1., norm_bin_size=False, cut_excess=True)
    nar, sf1d, _ = statistics_base.bin_data(
        nar, ar, mean_func=np.nanmean, min_bin=1., cut_excess=True, ignore_nan=True, norm_bin_size=False)

    N = len(nar)
    dx = lenn/N
    dk = 2.*np.pi/(N*dx)

    # Either normalize by 1/ndx or 2pi/ndx, where ndx = x
    kk = 2. * np.pi / (dx * nar)

    fek = (dk / (2. * np.pi)) * sf1d / kk
    return kk[::-1], fek[::-1]
