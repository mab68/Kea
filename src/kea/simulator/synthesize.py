"""
Provies methodologies to synthesize various types of noise/fields.
"""

from kea.utils.geometry import default_physdims, get_kvec, get_mesh
from kea.utils import functions

from typing import Optional, Callable

import numpy as np

def _normalize(
        field: np.ndarray,
        variance: float) -> np.ndarray:
    """_normalize(field)\n

    Normalize the field a range determined by `variance`

    Args:
        field (np.ndarray): The field to normalize
        variance (float): Variance of the field

    Returns:
        np.ndarray: Normalized field
    """
    field = field - np.nanmean(field)
    curr_var = np.nanmean(np.abs(field)**2)
    return np.sqrt(variance) * field / np.sqrt(curr_var)

def _white_noise(
        grid_dims: tuple[int,...],
        seed: Optional[int]=None) -> np.ndarray:
    """make_white_noise(grid_dims, seed)\n
    
    In Fourier-space, generates white noise i.e., normally distributed, uncorrelated
    complex-valued noise for each pixel in `grid_dims`.

    Args:
        grid_dims (tuple): Fourier-space grid size
        seed (int): Random seed to use

    Returns:
        np.ndarray: Complex-valued white noise
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(*grid_dims) + 1j * np.random.randn(*grid_dims)

def _make_lognormal(
        field: np.ndarray,
        scaling: float) -> np.ndarray:
    """_make_normal(field, scaling)\n

    Exponentiates the field -- generates an exponential field (xFBM).
    
    Args:
        field (np.ndarray): Field to apply
        scaling (float): Additional scaling parameter

    Returns:
        np.ndarray: Exponential field
    """
    return np.exp(field * np.sqrt(scaling))

def _make_correlated_noise(
        field: np.ndarray,
        seed: Optional[int] = None) -> np.ndarray:
    """_make_correlated_noise(field, seed)
    
    Makes (in real-space) a noisy field

    Args:
        field (np.ndarray): Field to apply
        seed (None|int): Random number generator seed
    
    Returns:
        np.ndarray: Noisy field
    """
    if seed is not None:
        np.random.seed(seed)
    normal_noise = np.random.normal(0., np.std(field), np.shape(field))
    df = field * normal_noise
    return df

@default_physdims('grid_dims')
def make_field(
        grid_dims: tuple[int,...],
        spectrum_kwargs: dict[str,float],
        spectrum_function: str|Callable,
        noise: Optional[np.ndarray] = None,
        phys_dims: Optional[tuple[float,...]] = None,
        seed: Optional[int] = None,
        normalize_output: Optional[bool] = False) -> np.ndarray:
    """make_field(grid_dims, spectrum_kwargs, spectrum_function, noise, phys_dims, seed)\n
    
    Generates a Gaussian, monofractal field.

    Args:
        grid_dims (tuple): Number of grid points for each dimension
        spectrum_kwargs (dict): Dictionary of key,value arguments to pass into `spectrum_function`
        spectrum_function (str|Callable): Function that defines the spectrum of the field
        noise (np.ndarray): If provided, the specific noise field to use.
        phys_dims (tuple): Physical scales
        seed (tuple): Random number seed for the random generator
        normalize_output (bool): If true, normalize the output field

    Returns:
        np.ndarray: Synthetic field
    """
    # Generate wavenumber grid
    kvec = get_kvec(grid_dims, phys_dims)
    k_mesh = get_mesh(kvec)

    # If just the name is provided, assume it is in the `utility.functions` file
    if isinstance(spectrum_function, str):
        spectrum_function = getattr(functions, spectrum_function)
    amplitude = spectrum_kwargs.get('amplitude', 1.)
    # Generate the spectrum and then convert to the Fourier-space field (via sqrt)
    field_k = np.sqrt(amplitude * spectrum_function(k_mesh, **spectrum_kwargs).astype('complex'))

    if noise is None:
        # If no noise field provided, assume uncorrelated white noise
        noise = _white_noise(grid_dims, seed)

    # Convert the uncorrelated noise into correlated noise defined by the spectrum
    field_k = noise * field_k
    field_k[np.isnan(field_k)|np.isinf(field_k)] = 0. + 1j * 0.

    # Convert the Fourier-space field into real-space
    field_x = np.fft.ifftn(np.fft.ifftshift(field_k), norm='forward').real

    if normalize_output:
        field_x = _normalize(field_x, 1.)
    return field_x

@default_physdims('grid_dims')
def make_multifractal_field(
        grid_dims: tuple[int,...],
        gauss_spectrum_kwargs: dict[str,float],
        gauss_spectrum_function: str|Callable,
        scaling: float,
        spectrum_kwargs: dict[str,float],
        spectrum_function: str|Callable,
        phys_dims: Optional[tuple[float,...]] = None,
        seeds: Optional[tuple[int,int,int]] = None,
        normalize_output: Optional[bool] = False) -> np.ndarray:
    """make_multifractal_field(grid_dims, gauss_spectrum_kwargs, gauss_spectrum_function, scaling, spectrum_kwargs, spectrum_function, phys_dims, seeds)\n
    
    Generates a multifractal field with parameters described by `spectrum_kwargs` and `scaling`.

    Args:
        grid_dims (tuple): Number of grid points for each dimension
        gauss_spectrum_kwargs (dict): Dictionary of key,value arguments to pass into `gauss_spectrum_function`
        gauss_spectrum_function (str|Callable): Function that defines the spectrum of the gaussian noise
        scaling (float): Scaling parameter
        spectrum_kwargs (dict): Dictionary of key,value arguments to pass into `spectrum_function`
        spectrum_function (str|Callable): Function that defines the spectrum of the multiplicative field
        phys_dims (tuple): Physical scales
        seeds (tuple): Random number seeds to the 3 different random generators
        normalize_output (bool): If true, normalize the output field

    Returns:
        np.ndarray: Multifractal field
    """
    if seeds is None:
        seeds = (None, None, None)

    # Make monofractal field
    gaussian_field = make_field(grid_dims, gauss_spectrum_kwargs, gauss_spectrum_function, None, phys_dims, seeds[0])
    # Make into positive log-normal field
    log_field = _make_lognormal(gaussian_field, scaling)
    # Make into a correlated noise field
    correlated_noise = np.fft.fftshift(np.fft.fftn(_make_correlated_noise(log_field, seed=seeds[1])))
    # Use multiplicative noise to generate a multifractal field
    multifactal_field = make_field(grid_dims, spectrum_kwargs, spectrum_function, correlated_noise, phys_dims, seeds[2], normalize_output)

    return multifactal_field
