"""
Implements the difference-of-Gaussian spectral estimator (DoG) following:
- Arévalo, P., Churazov, E., Zhuravleva, I., Hernández-Monteagudo, C., & Revnivtsev, M. (2012). A Mexican hat with holes: calculating low-resolution power spectra from data with gaps. Monthly Notices of the Royal Astronomical Society, 426(3), 1793-1807.
"""

from kea.utils.geometry import validate_shapes, default_physdims, get_dxdk, check_square_dims, get_kvec
from kea.utils import get_calculation_mode

from typing import Optional

import numpy as np


DEFAULT_SCALE_FACTOR = np.sqrt(2.)

def _gaussian_1d_kernel(scale: float, radius: int) -> np.ndarray:
    """_gaussian_1d_kernel(scale, radius)\n

    Generates a 1D Gaussian kernel

    Args:
        scale (float): Sigma of the Gaussian
        radius (int): Grid size to generate the Gaussian
    
    Returns:
        np.ndarray: 1D Gaussian kernel
    """
    x = np.arange(-radius, radius+1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x/scale)**2)
    return kernel / np.nansum(kernel)

def _calc_stat(
        field: np.ndarray,
        discrete_scales: np.ndarray,
        exposure_field: Optional[np.ndarray] = None,
        use_gpu: Optional[bool] = False) -> np.ndarray:
    """_calc_stat(field, discrete_scales, exposure_field, use_gpu)\n

    Process scales using the CPU or GPU

    Args:
        field (np.ndarray): Field to compute the statistic for
        discrete_scales (np.ndarray): List of scales to compute the statistic at
        exposure_field (np.ndarray): Additional exposure/mask map
        use_gpu (bool): If true, use the GPU via `cupy`

    Returns:
        np.ndarray: Calculated statistic for given `discrete_scales`
    """
    
    dimension = field.ndim

    if use_gpu:
        # Load GPU libraries
        try:
            import cupy as compute_lib
        except:
            raise ValueError('Cannot use the GPU -- `cupy` not installed')
        try:
            from cupyx.scipy import ndimage as ndimage_lib
        except:
            raise ValueError('Cannot use GPU -- `cupyx` not installed')
        # Move arrays onto GPU
        field_gpu = compute_lib.asarray(field, dtype=compute_lib.float64)
        if exposure_field is not None:
            exposure_gpu = compute_lib.asarray(exposure_field, dtype=compute_lib.float64)
    else:
        # Use standard libraries
        compute_lib = np
        from scipy import ndimage as ndimage_lib
        field_gpu = field
        if exposure_field is not None:
            exposure_gpu = exposure_field

    if exposure_field is None:
        # No exposure field given, so assume a default
        exposure_gpu = compute_lib.ones_like(field_gpu)
        # Make sure the mask is 0 where field has NaNs
        exposure_gpu[compute_lib.isnan(field_gpu)] = 0.

    # Remove NaNs, these are accounted for by convolving the exposure mask too
    field_gpu = compute_lib.nan_to_num(field_gpu, nan=0.)
    # Setup output array
    out_gpu = compute_lib.zeros(len(discrete_scales), dtype=np.float64)
    exp_size = exposure_gpu.size

    for i in range(len(discrete_scales)):
        s = float(discrete_scales[i])

        xi = 1e-3
        s1 = float(s / np.sqrt(1. + xi))
        s2 = float(s * np.sqrt(1. + xi))

        # Convolve the image with the Gaussians of scale `s1`
        filtered_field = ndimage_lib.gaussian_filter(field_gpu, s1, mode='constant', cval=0., truncate=10.)
        filtered_exp = ndimage_lib.gaussian_filter(exposure_gpu, s1, mode='constant', cval=0., truncate=10.)
        t1 = filtered_field/filtered_exp
        # Convolve the image with the Gaussians of scale `s2`
        filtered_field = ndimage_lib.gaussian_filter(field_gpu, s2, mode='constant', cval=0., truncate=10.)
        filtered_exp = ndimage_lib.gaussian_filter(exposure_gpu, s2, mode='constant', cval=0., truncate=10.)
        t2 = filtered_field/filtered_exp

        # Calculate the variance of the difference at that scale
        masked_diff = exposure_gpu * (t1 - t2)
        variance = compute_lib.nansum(masked_diff*masked_diff)

        # Generate mask compensation factor (fraction of masked field)
        m_comp = exp_size / compute_lib.nansum(exposure_gpu)
        # Normalize by fraction of sky and gaussian variance
        out_gpu[i] = m_comp * variance

        # Calculate the variance of the Gaussian filter(s)
        # Use separability to make this step faster 
        # NOTE: Currently we always do this step on the CPU
        size = np.nanmax([2*int(10.*s1 + 0.5) + 1, 2*int(10.*s2 + 0.5) + 1])
        gauss1, gauss2 = _gaussian_1d_kernel(s1, size), _gaussian_1d_kernel(s2, size)
        gaussian_variance = (
            np.nansum(gauss1**2)**dimension
            - 2.*np.nansum(gauss1*gauss2)**dimension
            + np.nansum(gauss2**2)**dimension)

        # Normalize by the filter variance
        out_gpu[i] = out_gpu[i] / gaussian_variance

    if use_gpu:
        # Convert back to numpy array
        out_gpu = compute_lib.asnumpy(out_gpu)
        # Clear GPU memory references
        field_gpu, exposure_gpu, filtered_field, filtered_exp = None, None, None, None
        compute_lib.get_default_memory_pool().free_all_blocks()
        compute_lib.get_default_pinned_memory_pool().free_all_blocks()

    return out_gpu

def process_scales(
        field: np.ndarray,
        discrete_scales: np.ndarray,
        exposure_field: Optional[np.ndarray] = None) -> np.ndarray:
    """process_scales(field, exposure_field, scales)\n

    Calculate the difference of gaussian by processing the scales

    Args:
        field (np.ndarray): The array to compute the scale-statfunc of
        exposure_field (np.ndarray): Additional mask/exposure map
        scales (np.ndarray): List of scales to evaluate at

    Return:
        np.ndarray: Output array
    """
    use_cpu, use_gpu, do_distribute = get_calculation_mode()

    if do_distribute:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except:
            raise ValueError('Cannot distribute lags -- `mpi4py` not configured/installed.')
        rank = comm.Get_rank()
        size = comm.Get_size()
        discrete_scales = discrete_scales[rank::size]
    
    statfunc = _calc_stat(field, discrete_scales, exposure_field, use_gpu)
    return discrete_scales, statfunc

@validate_shapes('field', 'exposure_field')
@default_physdims('field')
def dog_averaged_spectrum(
        field: np.ndarray,
        exposure_field: Optional[np.ndarray] = None,
        b_factor: Optional[float] = None,
        discrete_scales: Optional[np.ndarray] = None,
        phys_dims: Optional[tuple[float,...]] = None) -> tuple[np.ndarray, np.ndarray]:
    """dog_averaged_spectrum(field, exposure_field, b_factor, discrete_scales, phys_dims)\n
    
    Calculate the angle-averaged difference-of-Gaussian spectrum.

    Args:
        field (np.ndarray): Array to calculate the difference-of-Gaussian spectrum
            this array CAN have NaN values.
        exposure_field (np.ndarray): An additional mask/exposure map
            0 represents invalid data
            otherwise, valid/partially valid data
        b_factor (float): If set, use a different scale-wavenumber conversion factor
        scales (np.ndarray): (discrete) Gaussian scales (standard deviation)
        phys_dims (tuple): The physical system size in x,y,z,... direction

    Returns:
        (np.ndarray, np.ndarray): Equivalent wavenumbers and difference-of-Gaussian spectrum
    """

    grid_dims = field.shape
    if b_factor is None:
        b_factor = DEFAULT_SCALE_FACTOR

    dx, dk = get_dxdk(grid_dims, phys_dims)

    # Default to using the Fourier wavenumber-spaced scales
    if discrete_scales is None:
        wavenumbers = get_kvec(grid_dims, phys_dims)[0]
        discrete_scales = wavenumber_to_discrete_scale(wavenumbers[wavenumbers>0.], grid_dims, phys_dims, b_factor)

    discrete_scales, dogs = process_scales(field, discrete_scales, exposure_field)
    dogs = dogs / np.prod(grid_dims)**2
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
        np.ndarray: discrete physical scales
    """
    dx, _ = get_dxdk(grid_dims, phys_dims)
    if b_factor is None:
        b_factor = DEFAULT_SCALE_FACTOR

    scales = b_factor / wavenumbers
    return scales / dx[0]
