"""
Implements the equivalent spectrum (ESF) calculated directly using the structure function following the pocedure from:
- Mark A. Bishop, Sean Oughton, Tulasi N. Parashar, Yvette C. Perrott; Direct power spectral density estimation from structure functions without Fourier transforms. Physics of Fluids 1 February 2026; 38 (2): 025107. https://doi.org/10.1063/5.0310561
"""

from kea.utils import fitting, set_fitting_mode

from typing import Optional

import numpy as np
import sympy as sp

def _filter_bad(
        kk: np.ndarray,
        fek: np.ndarray,
        ko: Optional[np.ndarray]=None):
    """_filter_bad(kk, fek, ko)\n

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
        kk, fek = fitting.interpolate_1d_function(kk, fek, ko[ko <= kmax], x_log=True, y_log=True)
        kk = kk.flatten()
        fek = fek.flatten()
    return kk, fek

def _sf_to_spectrum(
        ell: np.ndarray,
        sf2: np.ndarray,
        b: float):
    """_sf_to_spectrum(ell, sf2, phys_dims, grid_dims, b, ko)\n

    Estimates the Fourier spectrum using the (derivative of the) structure function

    Args:
        ell (np.ndarray): Lags
        sf2 (np.ndarray): Angle-averaged second order structure function
        b (float): Wavenumber bias factor

    Returns:
        (np.ndarray, np.ndarray): Equivalent wavenumbers and uncorrected equivalent spectrum
    """
    dell, dsf2 = fitting.get_derivative(ell, sf2, ell, x_log=True, y_log=True)
    esf = (1./2.) * dell**2 * dsf2 / b
    ke = b/dell
    return ke[::-1], esf[::-1]

def _debias(
        est_alpha: np.ndarray,
        b: float,
        D: float):
    """_debias(est_alpha, b, D)\n

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
        sf_subsample_num: Optional[int]=None,
        powerlaw_subsample_num: Optional[int]=None,
        b_factor: Optional[float]=None,
        fourier_wavenumbers: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """esf_spectrum(physical_lags, structure_function, dimension, b_factor, fourier_wavenumbers)\n

    Calculates the equivalent spectrum i.e. an estimate of the (integrated) Fourier power spectrum

    Args:
        physical_lags (np.ndarray): Lags
        structure_function (np.ndarray): Angle-averaged second-order structure function
        dimension (int): Number of dimensions
        sf_subsample_num (int): Number of points to subsample the SF to
        powerlaw_subsample_num (int): Number of points to subsample the powerlaw estimate to
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

    ## Subsample (in log-space) the SF and lags for better spectrum estimation
    if sf_subsample_num is not None:
        ell_i = np.exp(np.linspace(np.log(np.nanmin(physical_lags)), np.log(np.nanmax(physical_lags)), sf_subsample_num))
        ell_b, sf2_b = fitting.interpolate_1d_function(physical_lags, structure_function, ell_i, x_log=True, y_log=True)
    else:
        ell_b, sf2_b = physical_lags, structure_function

    ## Estimate biased spectrum
    ke, esf = _sf_to_spectrum(ell_b, sf2_b, b_factor)
    print(ke, esf)
    print(*_filter_bad(ke, esf))
    ke, esf = fitting.interpolate_1d_function(*_filter_bad(ke, esf), ke, x_log=True, y_log=True)

    ## Estimate local powerlaw
    dke, desf = fitting.get_local_powerlaw(ke, esf, ke, x_log=True)

    ## Subsample the local powerlaw estimate
    if powerlaw_subsample_num:
        set_fitting_mode(gaussian_process=True)
        ke_i = np.exp(np.linspace(np.log(np.nanmin(dke)), np.log(np.nanmax(dke)), powerlaw_subsample_num))
        dke2, desf2 = fitting.interpolate_1d_function(dke[~np.isnan(desf)], desf[~np.isnan(desf)], ke_i, x_log=True)
        # Re-interpolate onto original ks
        set_fitting_mode(finite_differences=True)
        dke2, desf2 = fitting.interpolate_1d_function(dke2, desf2, ke, x_log=True)
    else:
        dke2, desf2 = dke, desf

    ## Calculate the non-parametric amplitude bias factor
    bias_factor = _debias(desf2, b_factor, float(dimension))
    corrected_esf = esf / bias_factor

    ## Interpolate onto the Fourier wavenumbers if requested:
    if fourier_wavenumbers is not None:
        _, esf = fitting.interpolate_1d_function(ke, esf, fourier_wavenumbers, x_log=True, y_log=True)
        ke, corrected_esf = fitting.interpolate_1d_function(ke, corrected_esf, fourier_wavenumbers, x_log=True, y_log=True)

    ## TODO: Filter out the equivalent nyquist condition
    # nyq = np.min([np.pi*N/L, np.sqrt(2.)*N/L - dk[0]])

    return ke, esf, corrected_esf

