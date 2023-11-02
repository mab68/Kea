
"""
profile.py

Fitting for ICM mean radial profiles

Functions
---------
"""

import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import lmfit

from ..statistics.statistics_base import bin_data
from ..utils.funcs import beta_func, double_beta_profile


def radial_profile(ar, pos):
    """radial_profile(ar, pos)"""
    grid = np.indices(ar.shape)
    r = np.linalg.norm([grid[i,...] - pos[i] for i in range(ar.ndim)], axis=0)
    l, m, s = bin_data(r, ar, mean_func=np.nanmean, nan_small=True, cut_excess=True)
    return l[1:], m[1:], s[1:]

def compare_double(data, params):
    pos = (int(params['posx']), int(params['posy']))
    x, y, yerr = radial_profile(data, pos)
    model = double_beta_profile(
        (int(params['shapex']), int(params['shapey'])),
        params['I0_1'],
        params['rc_1'],
        params['beta_1'],
        params['I0_2'],
        params['rc_2'],
        params['beta_2'],
        pos
    )
    mx, my, merr = radial_profile(model, pos)
    return (x, y, yerr), (mx, my, merr)

def obj_func_2d(params, data):
    shape = (int(params['shapex']), int(params['shapey']))
    pos = (int(params['posx']), int(params['posy']))
    model = double_beta_profile(
        shape,
        params['I0_1'],
        params['rc_1'],
        params['beta_1'],
        params['I0_2'],
        params['rc_2'],
        params['beta_2'],
        pos
    )
    obj = model - data
    return obj

def fit_model(data, params, method='dual_annealing'):
    """fit_model(data, params)
    
    Args:
        data (np.ndarray):
        params (lmfit.Parameters):
            shapex, shapey
            I0_1
            rc_1
            beta_1
            I0_2
            rc_2
            beta_2
        method (str):
    Returns:
        lmfit.MinimizerResult: 
    """
    result = lmfit.minimize(obj_func_2d, params, kws={'data': data}, method=method, nan_policy='omit')
    return result
