
"""
corr.py

Implements several correlation calculation methods

Functions
---------
process_lags\n
autocovariance_fft\n
autocovariance_loglagged\n
variance_normalize\n
symmetric_function\n
"""


import numpy as np
from scipy import fft

from . import statfunc_base, statfunc_mpi_naive


def process_lags(ar1, ar2, lagvecs, lenn=None, shape=None, periodic=False):
    """process_lags(ar, lagvecs, shape)

    Calculates the mean lagged product of `ar` for each lag in `lagvec`

    Args:
        ar (np.ndarray): 2/3D array to calculate the mean lagged product for
        lagvecs (np.ndarray): Each lag distance
        lenn (tuple): System size
        shape (tuple): Shape of the output correlation function
        periodic (bool): If true, treat the data as periodic
    Returns:
        acf (np.ndarray): 2/3D autocovariance function of `ar` for each lag in `lagvecs`
    """
    assert ar1.shape == ar2.shape, 'Provided arrays are not the same shape'
    if lenn is None:
        lenn = [2. * np.pi for _ in range(ar1.ndim)]
    if shape is None:
        assert ar1.shape == ar2.shape, 'Provided arrays are not the same shape'
        shape = tuple([2*s+1 for s in ar1.shape])
    shifts = None
    if not periodic:
        shifts = statfunc_base.compute_shifted_indices(lagvecs, ar1.shape)
    acf = statfunc_mpi_naive.mpi_pool_lagvecs(ar1, ar2, lagvecs, shifts, shape, statfunc_mpi_naive.correlation, (lenn,))
    return acf

def autocovariance_fft(ar1, ar2=None, shape=None, lenn=None):
    """autocovariance_fft(ar, shape, lenn)

    Calculate the correlation in the Fourier domain

    NOTE: The shape provides the output shape; we still need to calculate the whole `ar.shape`
        autocorrelation function first

    Args:
        ar1 (np.ndarray): Input array to find the autocorrelation function for
        ar2 (np.ndarray): Input array for the cross correlation
        shape (tuple): The return shape of the autocovariance
        bias (bool): If true, calculate the biased sum (1/N) rather than the unbiased (1/(N - lag))
            if None, Then don't apply any normalization
        norm_variance (bool): If true, normalize the autocovariance by the variance (the zeroth-lag element)
    Return:
        corr (np.ndarray): Autocorrelation of `ar` as a function of lag
    """
    if ar2 is None:
        ar2 = ar1
    assert ar1.shape == ar2.shape, 'Provided arrays are not the same shape'
    if lenn is None:
        lenn = [2. * np.pi for _ in range(ar1.ndim)]
    # NOTE: it is intended that dk normalizes by the padded shape rather than the true shape
    dx = [lenn[i]/N for i, N in enumerate(ar1.shape)]
    dk = [1./(N*dx[i]) for i, N in enumerate([2*s+1 for s in ar1.shape])]
    ffta1 = np.prod(dx) * fft.fftn(ar1, s=[2*s+1 for s in ar1.shape], norm='backward')
    ffta2 = np.prod(dx) * fft.fftn(ar2, s=[2*s+1 for s in ar2.shape], norm='backward')
    acf = np.prod(dk) * fft.ifftn(ffta1 * np.conj(ffta2), s=[2*s+1 for s in ar1.shape], norm='forward')
    acf = acf.real
    # Center the image
    acf = np.fft.fftshift(acf)
    if shape is not None:
        acf = statfunc_base.cut_at_lag(acf, shape[0])
    return acf

def autocovariance_fft_nonbias(ar1, ar2=None, shape=None, lenn=None):
    """autocovariance_fft_nonbias(ar, shape, lenn)

    Calculate the correlation in the Fourier domain.
    The returned autocovariance is nonbias, which means the normalization
    is different as we go out to larger lags.

    NOTE: The shape provides the output shape; we still need to calculate the whole `ar.shape`
        autocorrelation function first

    Args:
        ar1 (np.ndarray): Input array to find the autocorrelation function for
        ar2 (np.ndarray): Input array for the cross correlation
        shape (tuple): The return shape of the autocovariance
        bias (bool): If true, calculate the biased sum (1/N) rather than the unbiased (1/(N - lag))
            if None, Then don't apply any normalization
        norm_variance (bool): If true, normalize the autocovariance by the variance (the zeroth-lag element)
    Return:
        corr (np.ndarray): Autocorrelation of `ar` as a function of lag
    """
    if ar2 is None:
        ar2 = ar1
    assert ar1.shape == ar2.shape, 'Provided arrays are not the same shape'
    if lenn is None:
        lenn = [2. * np.pi for _ in range(ar1.ndim)]

    n_array = np.ones([len(lenn)] + [2*s+1 for s in ar1.shape])
    k = statfunc_base.get_lagvecs(tuple([2*s+1 for s in ar1.shape]))
    nk_array = np.ones_like(n_array)
    for i in range(len(lenn)):
        n_array[i,...] *= ar1.shape[i]
        nk_array[i,...] *= (n_array[i,...] - np.abs(k[i]))
    nk_array[nk_array == 0.] = 1.

    dx = np.ones_like(n_array)
    for i in range(len(lenn)):
        dx[i,...] *= lenn[i]/n_array[i,...]

    dk = np.ones_like(n_array)
    for i in range(len(lenn)):
        dk[i] /= (2*ar1.shape[i]+1) * dx[i]

    ffta1 = np.fft.ifftshift(np.prod(dx, axis=0)) * fft.fftn(ar1, s=[2*s+1 for s in ar1.shape], norm='backward')
    ffta2 = np.fft.ifftshift(np.prod(dx, axis=0)) * fft.fftn(ar2, s=[2*s+1 for s in ar2.shape], norm='backward')
    acf = np.prod(dk, axis=0) * fft.ifftn(ffta1 * np.conj(ffta2), s=[2*s+1 for s in ar1.shape], norm='forward')
    acf = acf.real

    acf = np.fft.fftshift(acf) * np.prod(n_array, axis=0) / np.prod(nk_array, axis=0)

    return acf

def variance_normalize(ar, pos=None):
    """variance_normalize(ar):

    Calculates the autocorrelation of the given autocovariance function

    `$C_{corr}(t) = C(t) / C(0)$`

    This is normalizing the autocovariance function by the variance.

    Args:
        ar (np.ndarray): Input autocovariance function
        pos (tuple): Default None, assumes the position is at the center of the array.
            The position of the zero-lag value
    Returns:
        acf (np.ndarray): Variance-normalized autocorrelation function
    """
    if pos is None:
        # Assume the zero-lag is in the center of the image
        pos = tuple([n//2 for n in np.shape(ar)])
    return ar / ar[pos]

def symmetric_function(ar):
    """symmtric_function(ar)

    Converts function by using the symmetric property of the autocorrelation function
        to span negative to positive lags

    Args:
        ar (np.ndarray): Autocorrelation/autocovariance function to symmeterize
    Returns:
        new_ar (np.ndarray): Autocorrelation function going from -lag to lag
    """
    if np.shape(ar) == 0:
        raise NotImplementedError('Invalid values passed %s' % ar)
    newar = np.zeros((2*len(ar),))
    newar[0:len(ar)] = ar[::-1]
    newar[len(ar):2*len(ar)] = ar
    return newar

