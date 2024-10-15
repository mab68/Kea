
"""
strfn_spectra.py

Converts second order structure function to an equivalent spectra

Functions
---------
omni_spectrum\n
"""

import numpy as np

import sympy as sp

from ..statfunc import statfunc_mpi
from .. import statistics_base


def get_powerlaw(k, fek, num_bins=16, log_space=False):
    est_alpha = np.gradient(np.log(fek), np.log(k))
    b_k, b_est_alpha, _ = statistics_base.bin_data(
        k, est_alpha, bin_func=np.nanmean, log_space=log_space,
        bin_loc='left', num_bins=num_bins, min_bin=np.nanmin(k), max_bin=np.nanmax(k),
        ignore_nan=True)
    est_alpha = interpolate.logx_interpolate(b_k, b_est_alpha, k)
    return est_alpha

def convert_to_omni_spectrum(ko, ell, sf2, phys_dims, grid_dims, a, b):
    _dx = phys_dims[0]/grid_dims[0]
    _dk = 2.*np.pi/phys_dims[0]
    dS = np.gradient(sf2, _dx)
    dell = np.gradient(ell, _dx)
    fek = (ell)**2 * (dS/dell) * (1. / 2.) / a
    kk = b / ell
    kk = kk[::-1]
    fek = fek[::-1]
    kmin = np.nanmin(ko)
    kmax = np.nanmax(kk)
    fek = fek[kk > kmin]
    kk = kk[kk > kmin]
    fek = interpolate.log_log_interpolate(kk, fek, ko[ko < kmax])
    return ko[ko < kmax], fek

def debias(est_alpha, b, D):
    sp_beta = sp.symbols('beta', positive=True, real=True)
    sp_D = sp.symbols('D', positive=True, real=True)
    sp_b = sp.symbols('b', positive=True, real=True)
    spec_bias = sp.S(2)*sp.S(2)**(-sp_beta) * sp_b**(sp_beta - sp.S(1)) * (sp.gamma(sp_D/sp.S(2)) * sp.gamma((sp.S(3) - sp_beta) / sp.S(2))/sp.gamma(sp_D/sp.S(2) + sp_beta/sp.S(2) - sp.Rational(1,2)))
    spec_bias = spec_bias.subs({sp_D: sp.S(D), sp_b: sp.S(b)})
    bias_factor = []
    est_alpha[est_alpha > -1.01] = -1.01
    est_alpha[est_alpha < -2.99] = -2.99
    for _a in est_alpha:
        bias_factor.append(np.float64(spec_bias.subs({sp_beta: -_a}).evalf()))
    bias_factor = np.array(bias_factor)
    return bias_factor

def omni_spectrum(ar1, grid_dims, phys_dims, D):
    """
    
    """
    # Calculate the SF
    l, sf = statfunc_mpi.process_lags(ar1)
    # Calculate the magnitude wavenumbers
    ko = 1.
    # Lag-to-wavenumber factors
    a = b = np.sqrt(2.*D + 2.)
    if D == 1:
        a = b = 1.
    # Convert SF to angle-integrated spectrum
    k, fek = convert_to_omni_spectrum(ko, l, sf, phys_dims, grid_dims, a, b)
    a_fek = get_powerlaw(k, fek)
    b_fek = debias(a_fek, b, D)
    return k, fek, b_fek
