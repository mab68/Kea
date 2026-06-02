"""
fitting.py

Provides various utility routines to fit slopes etc.
"""

from .binning import bin_data

from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

def log_log_interpolate(x, y, x_interp):
    """log_log_interpolate(x, y, x_interp)\n

    Interpolates in log-log-space
    """
    f_int = interp1d(np.log10(x), np.log10(y), kind='linear', fill_value='extrapolate')
    y_new = f_int(np.log10(x_interp))
    return 10**y_new

def logx_interpolate(x, y, x_interp):
    """logx_interpolate(x, y, x_interp)\n

    Interpolates in log(x)-space
    """
    f_int = interp1d(np.log10(x), y, kind='linear', fill_value='extrapolate')
    y_new = f_int(np.log10(x_interp))
    return y_new

def get_powerlaw(k, fek, num_bins=16, log_space=False):
    """get_powerlaw(k, fek, num_bins, log_space)\n
    
    Calculates the local power law slope

    Args:
        k (np.ndarray): Wavenumbers
        fek (np.ndarray): Spectrum
        num_bins (int/None): Number of bins for binning and interpolation
        log_space (bool): If true, bin equally in log space
    Returns:
        est_alpha (np.narray): Estimate of the local power law slope at each k
    """
    est_alpha = np.gradient(np.log(fek), np.log(k))
    if num_bins is None:
        return est_alpha
    b_k, b_est_alpha, _ = bin_data(
        k, est_alpha, bin_func=np.nanmean, log_space=log_space,
        bin_loc='center', num_bins=num_bins, min_bin=np.nanmin(k), max_bin=np.nanmax(k),
        ignore_nan=True)
    est_alpha = logx_interpolate(b_k, b_est_alpha, k)
    return est_alpha
