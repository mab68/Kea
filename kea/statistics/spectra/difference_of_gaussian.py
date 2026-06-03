"""
difference_of_gaussian.py

Implements the difference-of-Gaussian spectral estimator (DoG) following:
- Arévalo, P., Churazov, E., Zhuravleva, I., Hernández-Monteagudo, C., & Revnivtsev, M. (2012). A Mexican hat with holes: calculating low-resolution power spectra from data with gaps. Monthly Notices of the Royal Astronomical Society, 426(3), 1793-1807.

"""

from ...utils.geometry import validate_shapes, default_physdims, get_dxdk, check_square_dims, get_kvec

from typing import Optional

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage

from astropy.convolution import convolve, Gaussian1DKernel


DEFAULT_SCALE_FACTOR = np.sqrt(2.)

def _calc_stat(
        field: cp.ndarray,
        s1: float,
        s2: float,
        exposure: cp.ndarray) -> np.floating:
    # Convolve the image with the Gaussians of scale `s1`
    filtered_field = ndimage.gaussian_filter(field, s1, mode='constant', cval=0., truncate=10.)
    filtered_exp = ndimage.gaussian_filter(exposure, s1, mode='constant', cval=0., truncate=10.)
    t1 = filtered_field/filtered_exp
    # Convolve the image with the Gaussians of scale `s2`
    filtered_field = ndimage.gaussian_filter(field, s2, mode='constant', cval=0., truncate=10.)
    filtered_exp = ndimage.gaussian_filter(exposure, s2, mode='constant', cval=0., truncate=10.)
    t2 = filtered_field/filtered_exp

    # Calculate the variance of the difference at that scale
    # mask = exposure > 0.
    masked_diff = exposure * (t1 - t2)
    variance = cp.nansum(masked_diff*masked_diff)

    # Generate mask compensation factor (fraction of masked field)
    m_comp = exposure.size / cp.nansum(exposure)

    # Normalize by fraction of sky and gaussian variance
    return m_comp * variance

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
    dimension = field.ndim
    field_gpu = cp.asarray(field, dtype=cp.float64)

    if exposure_field is not None:
        exposure_gpu = cp.asarray(exposure_field, dtype=cp.float64)
    else:
        exposure_gpu = cp.ones_like(field_gpu)

    field_gpu = cp.nan_to_num(field_gpu, nan=0.)
    out = np.zeros(len(discrete_scales), dtype=np.float64)

    for i in range(len(discrete_scales)):
        s = float(discrete_scales[i])

        xi = 1e-3
        s1 = float(s / np.sqrt(1. + xi))
        s2 = float(s * np.sqrt(1. + xi))

        out[i] = _calc_stat(field_gpu, s1, s2, exposure_gpu)

        # Calculate the variance of the Gaussian filter(s)
        # Use separability to make this step faster
        size = np.nanmax([2*int(10.*s1 + 0.5) + 1, 2*int(10.*s2 + 0.5) + 1])
        gauss1, gauss2 = Gaussian1DKernel(s1, x_size=size).array, Gaussian1DKernel(s2, x_size=size).array
        gaussian_variance = (
            np.nansum(gauss1**2)**dimension
            - 2.*np.nansum(gauss1*gauss2)**dimension
            + np.nansum(gauss2**2)**dimension)
        # Normalize by the filter variance
        out[i] = out[i] / gaussian_variance

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
            this array CAN have NaN values.
        exposure_field (np.ndarray): An additional mask/exposure map
            0 represents invalid data
            otherwise, valid/partially valid data
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

    # Default to using the Fourier wavenumber-spaced scales
    if discrete_scales is None:
        wavenumbers = get_kvec(grid_dims, phys_dims)[0]
        discrete_scales = wavenumber_to_discrete_scale(wavenumbers[wavenumbers>0.], grid_dims, phys_dims, b_factor)

    dogs = process_scales(field, discrete_scales, exposure_field) * np.prod(dx)**2 / np.prod(phys_dims)
    equiv_k = b_factor / (discrete_scales*dx[0])
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
