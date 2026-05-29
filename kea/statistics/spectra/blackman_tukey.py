"""
blackman_tukey.py
Implements power spectra calculations using the Blackman-Tukey method i.e., the Fourier transform of the
autocorrelation function.

Functions
---------
bt_spectrum\n
"""

from typing import Optional, Union, Callable
import numpy as np

def make_symmetric_acf(
        partial_acf: np.ndarray,
        grid_size: int,
        window: Optional[function] = None):
    """make_symmetric_acf(partial_acf, grid_size, window)
    
    Assumes symmetry and composes the autocorrelation function for some lags
    into the complete lag-space autocorrelation function.

    Args:
        partial_acf (np.ndarray): The partially computed ACF
        grid_size (int): The number of grid points (assumed equal for each dimension)
        window (func): If given, apply the window function
    Returns:
        acf_full (np.ndarray): The complete ACF
    """
    N = grid_size
    Q = partial_acf.reshape((N//2+1, N//2+1))




def bt_spectrum(acf_full, L, N):
    
