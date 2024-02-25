
"""
strfn.py
Implements structure function of arbitrary order calculation

Functions
---------
process_lags\n
strfn_loglagged\n
autocorrelation_to_strfn\n
"""

import numpy as np

from . import statfunc_base, statfunc_mpi_naive


def process_lags(ar1, ar2, lagvecs, lenn=None, shape=None, periodic=False, orders=[2]):
    """process_lags(ar, lagvecs, shape=None, orders)

    Calculates the structure function of order `orders` of `ar` for each lag in `lagvec`

    Args:
        ar1, ar2 (np.ndarray): array to calculate the mean lagged product for
        lagvecs (np.ndarray): Each lag distance
        lenn (tuple): System size
        shape (tuple): Shape of the output correlation function
        orders (list): List of requested orders of the structure functions
        periodic (bool): If true, treat the data as periodic
    Returns:
        sf (np.ndarray): structure function of `ar` for each lag in `lagvecs`
    """
    assert ar1.shape == ar2.shape, 'Provided arrays are not the same shape'
    if lenn is None:
        lenn = [2. * np.pi for _ in range(ar1.ndim)]
    if shape is None:
        shape = tuple([2*s+1 for s in ar1.shape])
    if isinstance(shape, list):
        shape = tuple(shape)
    shape = (len(orders),) + shape
    orders = np.asarray(orders)
    shifts = None
    if not periodic:
        shifts = statfunc_base.compute_shifted_indices(lagvecs, ar1.shape)
    sf = statfunc_mpi_naive.mpi_pool_lagvecs(ar1, ar2, lagvecs, shifts, shape, statfunc_mpi_naive.strfn, (lenn, orders,))
    return sf

def autocorrelation_to_strfn(ar):
    """autocorrelation_to_strfn(ar)

    Calculates the second order structure function using the autocorrelation function

    `$S^2(\ell) = 2 R(0) - 2 R(\ell)$`

    Args:
        ar (np.ndarray): Autocorrelation function
    Returns:
        sf2 (np.ndarray): The second order structure function
    """
    # Compute the correlation function for +ve lags
    pos = tuple([s//2 for s in ar.shape])
    sf2 = 2. * ar[pos] - 2. * ar
    return sf2
