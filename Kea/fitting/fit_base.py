
"""
fit_base.py

Provides base implementations for fitting power laws (and others)

Functions
---------
fit_powerlaw()\n
fit_correlation_length()\n
"""

import numpy as np

from scipy.optimize import curve_fit

def fit_powerlaw(ax, ay, xi, xf):
    """fit_powerlaw(ax, ay, xi, xf)

    Fits a powerlaw

    Args:
        ax, ay (np.ndarray): The x,y-values of the line have a powerlaw fit to
        xi, xf (float): The start and finish values we want the fit to happen between
    Returns:
        z (tuple): Polynomial fit coefficients
        xx (np.ndarray): Fit x-values
        pwrl (float): The fitted powerlaw value
    """
    idxi = np.argmin(abs(ax[np.isfinite(ax)]-xi))
    idxf = np.argmin(abs(ax[np.isfinite(ax)]-xf))
    xx = np.linspace(xi, xf, 100)
    z = np.polyfit(np.log(ax[idxi:idxf]), np.log(ay[idxi:idxf]), 1)
    p = np.poly1d(z)
    pwrl = np.exp(p(np.log(xx)))
    return z, xx, pwrl

def exp_powerlaw(xx, a1, a2, a3, b1, b2):
    pl = xx**(a1) * np.exp(-(b1/xx)**a2) * np.exp(-(xx/b2)**a3)
    # Set where it is not defined to 0
    pl[~np.isfinite(pl)] = 0.
    # Outside the axis to be 0
    minn = np.where(xx == np.nanmin(xx))
    idx = [m for m in minn]
    idx[0] = slice(0,np.max(xx.shape))
    idx = tuple(idx)
    maxx = np.max(xx[idx])
    pl[xx > maxx] = 0.
    return pl

def fit_exp_powerlaw(datax, datay):
    curve_fit(exp_powerlaw, datax, datay)


def fit_correlation_length():
    # Integral method
    # 1/e method
    raise NotImplementedError
