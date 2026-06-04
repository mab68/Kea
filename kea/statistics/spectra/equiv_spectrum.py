"""
Implements the equivalent spectrum (ESF) calculated directly using the structure function following the pocedure from:
- Mark A. Bishop, Sean Oughton, Tulasi N. Parashar, Yvette C. Perrott; Direct power spectral density estimation from structure functions without Fourier transforms. Physics of Fluids 1 February 2026; 38 (2): 025107. https://doi.org/10.1063/5.0310561
"""

from kea.utils.fitting import log_log_interpolate, get_powerlaw

from typing import Optional

import numpy as np
import sympy as sp

def filter_bad(
        kk: np.ndarray,
        fek: np.ndarray,
        ko: Optional[np.ndarray]=None):
    """filter_bad(kk, fek, ko)\n

    Helper function to remove bad/unwanted values from the spectrum

    Args:
        kk (np.ndarray): wavenumbers to clean
        fek (np.ndarray): Spectrum to clean
        ko (np.ndarray): 'true' wavenumbers

    Returns:
        (np.ndarray, np.ndarray): Cleaned wavenumbers and spectrum
    """
    # If we provide a 'ko', make sure we don't extend outside of its range
    if ko is not None:
        kmin = np.nanmin(ko)
        kmax = np.nanmax(ko)
        fek = fek[kk >= kmin]
        kk = kk[kk >= kmin]
        fek = fek[kk <= kmax]
        kk = kk[kk <= kmax]
    kk = kk[np.isfinite(fek)]
    fek = fek[np.isfinite(fek)]
    # Remove "un-physical" negative values
    if np.sum(fek <= 0) > 0:
        last_0 = np.where(fek<0)[0][-1]
        kk = kk[last_0:]
        fek = fek[last_0:]
        kk = kk[fek > 0]
        fek = fek[fek > 0]
    # Interpolate onto 'ko' if provided
    if ko is not None:
        fek = log_log_interpolate(kk, fek, ko[ko <= kmax])
        kk = ko[ko <= kmax]
    return kk, fek

def sf_to_spectrum(
        ell: np.ndarray,
        sf2: np.ndarray,
        b: float,
        ko: Optional[np.ndarray]=None):
    """sf_to_spectrum(ell, sf2, phys_dims, grid_dims, b, ko)\n

    Estimates the Fourier spectrum using the (derivative of the) structure function

    Args:
        ell (np.ndarray): Lags
        sf2 (np.ndarray): Angle-averaged second order structure function
        b (float): Wavenumber bias factor
        ko (np.ndarray/None): Discrete wavenumbers for FFT

    Returns:
        (np.ndarray, np.ndarray): Equivalent wavenumbers and uncorrected equivalent spectrum
    """
    dSdell = np.gradient(sf2, ell)
    BfekS = (1./2.) * ell**2 * dSdell / b
    ke = b / ell
    ke, BfekS = ke[::-1], BfekS[::-1]
    ke, BfekS = filter_bad(ke, BfekS, ko)
    return ke, BfekS

def debias(
        est_alpha: np.ndarray,
        b: float,
        D: float):
    """debias(est_alpha, b, D)\n

    The power law bias factor B^{pow}

    Args:
        est_alpha (np.ndarray): Local power law estimate (betas)
        b (float): Wavenumber bias factor
        D (float): Number of dimensions

    Returns:
        np.ndarray: Local power law based bias/correction factor
    """
    sp_beta = sp.symbols('beta', positive=True, real=True)
    sp_D = sp.symbols('D', positive=True, real=True)
    sp_b = sp.symbols('b', positive=True, real=True)
    spec_bias = sp.S(2)*sp.S(2)**(-sp_beta) * sp_b**(sp_beta - sp.S(1)) * (sp.gamma(sp_D/sp.S(2)) * sp.gamma((sp.S(3) - sp_beta) / sp.S(2))/sp.gamma(sp_D/sp.S(2) + sp_beta/sp.S(2) - sp.Rational(1,2)))
    spec_bias_f = sp.lambdify((sp_D, sp_b, sp_beta), spec_bias)
    # clamp the local beta
    est_alpha[est_alpha > -1.01] = -1.01
    est_alpha[est_alpha < -2.99] = -2.99
    bias_factor = spec_bias_f(float(D), b, -est_alpha)
    return bias_factor

def esf_integrated_spectrum(
        physical_lags: np.ndarray,
        structure_function: np.ndarray,
        dimension: int,
        b_factor: Optional[float]=None,
        fourier_wavenumbers: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """esf_spectrum(physical_lags, structure_function, dimension, b_factor, fourier_wavenumbers)\n

    Calculates the equivalent spectrum i.e. an estimate of the (integrated) Fourier power spectrum

    Args:
        physical_lags (np.ndarray): Lags
        structure_function (np.ndarray): Angle-averaged second-order structure function
        dimension (int): Number of dimensions
        b_factor (float/None): The wavenumber bias factor (little-b). If None, uses $b^{est}$.
        fourier_wavenumbers (np.ndarray): FFT based wavenumbers to interpolate the equivalent spectrum onto

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): Equivalent wavenumbers, uncorrected equivalent spectrum, and debiased equivalent spectrum
    """
    if b_factor is None:
        if dimension == 1:
            b_factor = 1.
        else:
            b_factor = np.sqrt(2.*float(dimension) - 2.)
    ke, BfekS = sf_to_spectrum(physical_lags, structure_function, b_factor, fourier_wavenumbers)
    a_fekS = get_powerlaw(ke, BfekS)
    B = debias(a_fekS, b_factor, float(dimension))
    return ke, BfekS, BfekS/B
