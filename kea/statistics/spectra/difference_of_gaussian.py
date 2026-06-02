"""
difference_of_gaussian.py

Implements the difference-of-Gaussian spectral estimator (DoG) following:
- Arévalo, P., Churazov, E., Zhuravleva, I., Hernández-Monteagudo, C., & Revnivtsev, M. (2012). A Mexican hat with holes: calculating low-resolution power spectra from data with gaps. Monthly Notices of the Royal Astronomical Society, 426(3), 1793-1807.

"""

from ...utils.geometry import validate_shapes, default_physdims, get_dxdk, check_square_dims, get_kvec

from typing import Optional

import numpy as np
from numba import njit, prange


DEFAULT_SCALE_FACTOR = np.sqrt(2.)

@njit(cache=True)
def gaussian_kernel1d(discrete_scale: np.floating) -> np.ndarray:
    radius = int(10.*discrete_scale + 0.5)
    x = np.arange(-radius, radius+1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x/discrete_scale)**2)
    return kernel / np.sum(kernel)

@njit(cache=True)
def gaussian_filter(field: np.ndarray, scale: np.floating):
    
    for axis in range(ndim):
        


@njit(cache=True)
def _calc_stat(
        field: np.ndarray,
        discrete_scale: np.floating,
        exposure_field: Optional[np.ndarray] = None) -> np.floating:

    dimension = field.ndim

    xi = 1e-3
    s1 = discrete_scale / np.sqrt(1. + xi)
    s2 = discrete_scale * np.sqrt(1. + xi)

    if exposure_field is None:
        ## TODO: This can be optimized
        exposure_field = np.ones_like(field)

    # Convolve the image with the Gaussians of scale `s1`
    img_conv_gauss = gaussian_filter(field, s1)
    exp_conv_gauss = gaussian_filter(exposure_field, s1)
    t1 = img_conv_gauss/exp_conv_gauss
    # Convolve the image with Gaussian of scale `s2`
    img_conv_gauss = gaussian_filter(field, s2)
    exp_conv_gauss = gaussian_filter(exposure_field, s2)
    t2 = img_conv_gauss/exp_conv_gauss

    # Calculate the variance of the difference at that scale
    mask = exposure_field > 0.
    masked_diff = exposure_field * mask * (t1 - t2)
    variance = np.nansum(masked_diff*masked_diff)

    # Calculate the variance of the Gaussian filter(s)
    gauss1, gauss2 = gaussian_kernel1d(s1), gaussian_kernel1d(s2)
    gaussian_variance = np.nansum(gauss1**2)**dimension - 2.*np.nansum(gauss1*gauss2)**dimension + np.nansum(gauss2**2)**dimension

    # Generate mask compensation factor (fraction of masked field)
    m_comp = np.prod(np.shape(mask)) / np.nansum(mask)

    # Normalize by fraction of sky and gaussian variance
    return m_comp * variance / gaussian_variance

@njit(parallel=True)
def process_scales(
        field: np.ndarray,
        discrete_scales: np.ndarray,
        exposure_field: Optional[np.ndarray] = None) -> np.ndarray:
    """process_scales(field, exposure_field, scales)\n


    Args:
        field (np.ndarray): The array to compute the scale-statfunc of
        exposure_field (np.ndarray): Additional mask/exposure map
        scales (np.ndarray): List of scales to evaluate at    
    Return:
        out (np.ndarray): Output array
    """
    out = np.zeros(len(discrete_scales), dtype=np.float64)
    for i in prange(len(discrete_scales)):
        s = discrete_scales[i]
        out[i] = _calc_stat(np.ascontiguousarray(field), s, np.ascontiguousarray(exposure_field))
    return out

@validate_shapes('field', 'exposure_field')
@default_physdims('field')
def dog_averaged_spectrum(
        field: np.ndarray,
        exposure_field: Optional[np.ndarray] = None,
        b_factor: Optional[float] = None,
        discrete_scales: Optional[np.ndarray] = None,
        phys_dims: Optional[tuple[float,...]] = None) -> tuple[np.ndarray, np.ndarray]:
    """dog_averaged_spectrum()\n
    
    Args:
        field (np.ndarray): Array to calculate the difference-of-Gaussian spectrum
        exposure_field (np.ndarray): An additional mask/exposure map
        b_factor (float): If set, use a different scale-wavenumber conversion factor
        scales (np.ndarray): (discrete) Gaussian scales (standard deviation)
        phys_dims (tuple): The physical system size in x,y,z,... direction
        mode (str): Mode for the convolution calculation
        xi (float): Determines the difference in the gaussian variance(s)
    Returns:
        equiv_k (np.ndarray): Equivalent wavenumbers
        dogs (np.ndarray): Difference-of-Gaussian spectrum
    """

    grid_dims = field.shape
    if b_factor is None:
        b_factor = DEFAULT_SCALE_FACTOR

    dx, dk = get_dxdk(grid_dims, phys_dims)
    dx, dk = dx[0], dk[0]

    # Default to using the Fourier wavenumber-spaced scales
    if discrete_scales is None:
        wavenumbers = get_kvec(grid_dims, phys_dims)[0]
        discrete_scales = wavenumber_to_discrete_scale(wavenumbers, grid_dims, phys_dims, b_factor)

    dogs = process_scales(field, discrete_scales, exposure_field)
    equiv_k = b_factor / (discrete_scales*dx)
    return equiv_k, dogs

@default_physdims('grid_dims')
@check_square_dims()
def wavenumber_to_discrete_scale(
        wavenumbers: np.ndarray,
        grid_dims: tuple[int,...],
        phys_dims: Optional[tuple[float,...]] = None,
        b_factor: Optional[float] = None) -> np.ndarray:
    """wavenumber_to_discrete_scale(wavenumbers, grid_dims, phys_dims, dimension, b_factor)\n
    
    Converts (Fourier) wavenumbers to a DoG-relevant scale so the DoG spectrum spans the same (equivalent) wavenumbers
    Note: that these values can still be non-integers.

    Args:
        wavenumbers (np.ndarray): Fourier wavenumbers
        grid_dims (tuple): The number of grid-points in each (physical) dimension
        phys_dims (None|tuple): The physical system size in x,y,z,... direction
        b_factor (float): A specific conversion factor to use.
    Returns:
        scales (np.ndarray): "discrete" physical scales
    """
    dx, _ = get_dxdk(grid_dims, phys_dims)
    if b_factor is None:
        b_factor = DEFAULT_SCALE_FACTOR

    scales = b_factor / wavenumbers
    return scales / dx[0]
