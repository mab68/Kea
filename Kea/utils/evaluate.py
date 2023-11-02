
"""
evaluate.py

Provides the MSE, MAPE, and Bias calculations

Functions
---------
mse()\n
mape()\n
bias()\n
"""

import numpy as np


def mse(ar, ar_true, dex=False, x=None, xmin=None, xmax=None):
    """mse(ar, ar_true, dex, x, xmin, xmax)
    
    Calculates the mean squared error of `ar` to `ar_true`

    Args:
        ar (np.ndarray): 
        ar_true (np.ndarray):
        dex (bool): Applies a log to `ar` and `ar_true`
        x (np.ndarray): The wavenumbers the calculation applies to
        xmin (float): Calculates the metric for x >= xmin
        xmax (float): Calculates the metric for x <= xmax
    Returns:
        float: Mean absolute percentage error of `ar` to `ar_true`
    """
    true = ar_true
    estimate = ar
    if x is not None:
        if xmin is not None:
            true = true[x >= xmin]
            estimate = estimate[x >= xmin]
        if xmax is not None:
            true = true[x <= xmax]
            estimate = estimate[x <= xmax]
    if dex:
        true = np.log10(true)
        estimate = np.log10(estimate)
    MAPE = np.nanmean((true - estimate)**2)
    return MAPE

def mape(ar, ar_true, dex=False, x=None, xmin=None, xmax=None):
    """mape(ar, ar_true, dex, x, xmin, xmax)
    
    Calculates the mean absolute percentage error of `ar` to `ar_true`

    Args:
        ar (np.ndarray): 
        ar_true (np.ndarray):
        dex (bool): Applies a log to `ar` and `ar_true`
        x (np.ndarray): The wavenumbers the calculation applies to
        xmin (float): Calculates the metric for x >= xmin
        xmax (float): Calculates the metric for x <= xmax
    Returns:
        float: Mean absolute percentage error of `ar` to `ar_true`
    """
    true = ar_true
    estimate = ar
    if x is not None:
        if xmin is not None:
            true = true[x >= xmin]
            estimate = estimate[x >= xmin]
        if xmax is not None:
            true = true[x <= xmax]
            estimate = estimate[x <= xmax]
    if dex:
        true = np.log10(true)
        estimate = np.log10(estimate)
    MAPE = np.nanmean(np.abs((true - estimate)/true))
    return MAPE

def bias(ar, ar_true, dex=False, x=None, xmin=None, xmax=None):
    """bias(ar, ar_true, dex, x, xmin, xmax)
    
    Calculates the bias error of `ar` to `ar_true`

    Args:
        ar (np.ndarray): 
        ar_true (np.ndarray):
        dex (bool): Applies a log to `ar` and `ar_true`
        x (np.ndarray): The wavenumbers the calculation applies to
        xmin (float): Calculates the metric for x >= xmin
        xmax (float): Calculates the metric for x <= xmax
    Returns:
        float: Mean absolute percentage error of `ar` to `ar_true`
    """
    true = ar_true
    estimate = ar
    if x is not None:
        if xmin is not None:
            true = true[x >= xmin]
            estimate = estimate[x >= xmin]
        if xmax is not None:
            true = true[x <= xmax]
            estimate = estimate[x <= xmax]
    if dex:
        true = np.log10(true)
        estimate = np.log10(estimate)
    bias = np.nanmean(estimate/true)
    return bias
