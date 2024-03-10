
"""
funcs.py

Stores common functions like the beta model etc.

Functions
---------
bkn_pow()\n
smooth_pow()\n
beta_pow()\n
beta_profile()\n
double_beta_profile()\n
"""

import numpy as np

from scipy.special import gamma, kv
import scipy.fft as fft
from scipy.ndimage import gaussian_filter


def pure_pow(xx, alphas, **kwargs):
    """pure_pow(xx, alphas)
    
    Generates a pure power law function

        f(x) = x^alpha
    where
        f(0) = 0
    and
        f(x>xmax) = 0
    unless `outside` is True

    Args:
        xx (np.ndarray): Grid
        alphas (tuple): Power law values
    Returns:
        np.ndarray: Pure powerlaw function defined on `xx`
    """
    ## Function
    pl = xx**(alphas[0])
    # Set where it is not defined to 0
    pl[~np.isfinite(pl)] = 0.
    outside = kwargs.get('outside', False)
    if not outside:
        # Outside the axis to be 0
        minn = np.where(xx == np.nanmin(xx))
        idx = [m for m in minn]
        idx[0] = slice(0,np.max(xx.shape))
        idx = tuple(idx)
        maxx = np.max(xx[idx])
        pl[xx > maxx] = 0.
    return pl

def exp_pow(xx, alphas, breaks, **kwargs):
    """exp_pow(xx, alphas)
    
    Generates a power law function with an exponential growth region

    Args:
        xx (np.ndarray): Grid
        alphas (tuple): Power law values
    Returns:
        np.ndarray: Pure powerlaw function defined on `xx`
    """
    ## Function
    bb = breaks[0]
    a1 = alphas[0]
    pl = np.exp(-bb/xx) * xx**(a1)
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

def pow_exp_decay(xx, alphas, breaks, **kwargs):
    """pow_exp_decay(xx, alphas)
    
    Generates a power law function with an exponential decay region

    Args:
        xx (np.ndarray): Grid
        alphas (tuple): Power law values
    Returns:
        np.ndarray: Pure powerlaw function defined on `xx`
    """
    ## Function
    bb = breaks[0]
    a1 = alphas[0]
    a2 = 1.
    if len(breaks) > 1:
        a2 = breaks[1]
    pl = np.exp(-a2*xx/bb) * (xx)**(a1)
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

def smooth_pow(xx, alphas, breaks, delta=0.1, **kwargs):
    """smooth_pow(xx, alphas, breaks, A, **kwargs)
    
    Generates a smooth power law function

    Args:
        xx (np.ndarray):
        alphas (tuple): Power law values
        breaks (tuple): Locatiosn in `x` to transition pwoer laws
        A (float): Amplitude
    Returns:
        np.ndarray: Smoothed power law function
    """
    a = [-alp for alp in alphas]
    a1, a2 = a[0], a[1]
    bb = breaks[0]
    pl = (xx/bb)**(-a1) * (0.5 * (1. + (xx/bb)**(1./delta)))**((a1 - a2)*delta)
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

def beta_pow(xx, beta, rc, A=1., **kwargs):
    """beta_pow(xx, xi, rc, A, **kwargs)

    Generates a beta profile (in Fourier space)

    Args:
        xx (np.ndarray):
        beta (float): power law xi = 3beta/2
        rc (float): Core radius
        A (float): Amplitude
        kwargs (dict):
            method (str):
                fft: Use the fft of the beta profile, then take the fft
                analytical: Use the analytical approximation
    Returns:
        np.ndarray: Beta profile
    """
    method = kwargs.get('method', 'fft')
    #if two_thirds:
    #    return A * (np.pi / 2) * rc**2 * np.exp(- 2. * xx * rc)
    if method == 'fft':
        yy = beta_func(xx.shape, A, rc, beta)
        ff = fft.fftn(yy)
        ff /= np.max(ff)
        ff *= A
        return ff
    if method == 'analytical':
        xi = 3. * beta / 2.
        g = gamma(xi)
        K = kv(0.5 - xi, xx.astype('complex'))
        num = 2.**(1. - xi) * rc**(xi + 0.5) * xx**(xi - 0.5)
        return A * num * K / g
    raise ValueError('Method %s not implemented' % method)


def double_beta_profile(shape, I0_1, rc_1, beta_1, I0_2, rc_2, beta_2, pos):
    """double_beta_profile(shape, I0_1, rc_1, beta_1, I0_2, rc_2, beta_2, pos)
    
    Calculates a double beta distribution function
    `$n(r) = n0 * [1 + (r/r_1)^2]^(-3beta_1/2) + n0 (1 - f)[1 + (r/r_2)^2]^(-3beta_2/2)$`

    Args:
        shape (tuple): Shape of the distribution
        I0_1 (float): Initial value for the first model
        rc_1 (float): Radius of the first beta profile
        beta_1 (float): Decay of the first beta profile
        I0_1 (float): Initial value for the second model
        rc_2 (float): Radius of the second beta profile
        beta_2 (float): Decay for the second beta profile
        pos (tuple): Position for the center
    Returns:
        np.ndarray: Double beta distribution
    """
    model1 = beta_func(shape, I0_1, rc_1, beta_1, pos)
    model2 = beta_func(shape, I0_2, rc_2, beta_2, pos)
    return model1 + model2

def beta_func(shape, m0, rc, beta, pos=None):
    """beta_func(shape, m0, rc, beta, pos)

    Calculates a beta distribution function
    `$n(r) = [1 + (r/rc)^2]^(-3 beta / 2)$`
    
    Args:
        shape (tuple): Shape of the distribution
        m0 (float): Initial value
        rc (int): Radius of the core
        beta (int): Changes the decay of the core distribution
        pos (tuple): Position for the center
    Returns:
        np.ndarray: Beta distribution
    """
    grid = np.indices(shape)
    if pos is None:
        pos = [N//2 for N in shape]
    r = np.linalg.norm([grid[i,...] - pos[i] for i in range(len(shape))], axis=0)
    mean = m0 * (1. + (r/rc)**2)**(-3. * beta / 2.)
    return mean
