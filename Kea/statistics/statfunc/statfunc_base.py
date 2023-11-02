
"""
lag.py

Provides utility to calculate the lag vectors and functions that use lags

Functions
---------
get_loglags\n
get_lagvecs\n
get_all_lagvecs\n
get_lagvec_magnitude_array\n
compute_shifted_indices\n
cut_at_lag\n
set_min\n
"""

import numpy as np


def get_loglags(max_lag, n_lags, max_half_bin_width=None):
    """get_loglags(max_lag, n_lags, max_half_bin_width)

    Calculates the log lag distances

    Args:
        max_lag (int): The maximum lag to find
        n_lags (int): The total number of binned lags
        max_half_bin_width (None,int): Width of the lag binning
            If None, assume its half the difference to the next lag
    Returns:
        lags (array): The log lags
        lag_bin_width (array): Bin widths for each log-lag
    """
    raise NotImplementedError('Call `statistics_base.get_bins()` for this')
    ## FIXME: THIS ACTUALLY GENERATES n+1 LAGS
    ## FIXME: THIS MIGHT NOT BE WELL DEFINED FOR SOME PARAMETER SPACES
    lag_range = np.exp(np.linspace(0, np.log(max_lag+1), n_lags+2))
    valid_lags = np.unique(lag_range.astype(int))
    lower_lags, upper_lags = valid_lags[:-1], valid_lags[1:]
    if max_half_bin_width is not None:
        lag_bin_width = np.minimum((upper_lags-lower_lags)/2, max_half_bin_width)
    else:
        lag_bin_width = (upper_lags-lower_lags)/2
    lags = upper_lags - lag_bin_width
    return lags, lag_bin_width

def get_lagvecs(lags, lag_bin_width, shape):
    """get_lagvecs(lags, lag_bin_width, shape)

    Generates all the points on annuli of widths `lag_bin_width` for each lag

    Args:
        lags (array): Shape of the lag-space
        lag_bin_width (array): Bin widths for each lag
        shape (tuple): Shape of the 
    Returns:
        lagvecs (np.ndarray): List of all points as vectors in the binned annuli
    """
    vecs = []
    n = len(shape)
    position = np.array(shape).reshape((-1,) + (1,) * n) // 2
    rvecs = np.indices(shape) - position
    rnorms = np.linalg.norm(rvecs, axis=0)
    for l in range(len(lags)):
        vecs.append(
            np.transpose(
                rvecs[:,((lags[l] - lag_bin_width[l]) <= rnorms) == (rnorms <= lags[l])]))
    return np.concatenate(vecs)

def get_lagvecs(shape):
    """get_lagvecs(shape)

    Returns an nd array of the lagvecs with 0 at the center

    Args:
        shape (tuple): Shape of the lag-space
    Returns:
        np.ndarray: Lagvec array in the shape of `shape`
    """
    n = len(shape)
    position = np.array(shape).reshape((-1,) + (1,) * n) // 2
    rvecs = np.indices(shape) - position
    return rvecs

def get_all_lagvecs(shape):
    """get_all_lagvecs(shape)

    Generates all the possible lagvectors

    Args:
        shape (tuple): Shape of the lag-space
    Returns:
        lagvecs (np.ndarray): List of all lags as vectors
    """
    return np.transpose(get_lagvecs(shape).reshape((-1, np.prod(shape))))

def get_lagvec_magnitude_array(shape):
    """get_lagvec_magnitude_array(shape)

    Generates all the magnitudes of the lagvecs (centered around the center)

    Args:
        shape (tuple): Shape of the lag-space
    Returns:
        lagvecs (np.ndarray): N dimensional array (matching shape) of all the magnitudes
            of the lagvectors
    """
    return np.linalg.norm(get_lagvecs(shape), axis=0)

def compute_shifted_indices(lagvecs, shape):
    """compute_shifted_indices(vecs, shape)

    Calculates the cuts for each lag vector

    Args:
        vecs (np.ndarray): List of all the lag distances to take
        shape (tuple): 
    Returns:
        min1, max1 (np.ndarray): Cuts for the first term
        min2, max2 (np.ndarray): Cuts for the second term
    """
    origins = shape*(lagvecs<0)
    counter_origins = shape - origins
    origins_target = counter_origins - lagvecs
    counter_origins_target = origins + lagvecs
    min1 = np.minimum(origins, origins_target).astype(int)
    max1 = np.maximum(origins, origins_target).astype(int)
    min2 = np.minimum(counter_origins, counter_origins_target).astype(int)
    max2 = np.maximum(counter_origins, counter_origins_target).astype(int)
    return min1, max1, min2, max2

def cut_at_lag(ar, lag, all_directions=False):
    """cut_at_lag(ar, lag)

    Cut the lag function to the requested lag from the centre location

    Args:
        ar (np.ndarray): Lag based function (structure function or autocorrelation function)
        lag (int,tuple): The largest lag to include. If tuple, then cut for each direction
        all_directions (bool): If true, the lag request has been specified for the + and - directions
            i.e. lag = ((startx, stopx), (starty, stopy))
    Returns:
        cut_ar (np.ndarray): Lag based function with largest lag == lag
    """
    shape = np.shape(ar)
    ndim = len(shape)
    if not isinstance(lag, tuple):
        lag = tuple([lag for _ in range(ndim)])
    indices = []
    for i in range(ndim):
        if all_directions:
            start = shape[i]//2 - lag[i][0]
            stop = shape[i]//2 + lag[i][1]
        else:
            start = shape[i]//2 - lag[i]
            stop = shape[i]//2 + lag[i]
        indices.append(slice(start, stop))
    cut_ar = ar[tuple(indices)]
    return cut_ar

def one_sided(lags, ar):
    pos = np.where(lags == 0)
    basis_index = tuple([slice(pos[i][0], None) for i in range(len(pos))])
    return ar[basis_index]

def set_min(ar, value=0.):
    """set_min(ar, value=0)

    Sets all values in `ar` smaller than `value` to `value`

    Args:
        ar (np.ndarray): Lag function, e.g. autocorrelation or structure function
        value (float): The value to be the minimum
    Returns:
        min_ar (np.ndarray): The lag function array with minimum as `value`
    """
    ar[ar < value] = value
    return ar
