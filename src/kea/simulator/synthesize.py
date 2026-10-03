"""
Provies methodologies to synthesize various types of noise/fields.
"""

from kea.utils.geometry import default_physdims, get_kvec, get_mesh, get_dxdk
from kea.utils import functions

from itertools import combinations
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
def _synthesize_field(
        grid_dims: tuple[int,...],
        spectrum_kwargs: dict[str,float],
        spectrum_function: str|Callable,
        noise: Optional[np.ndarray] = None,
        phys_dims: Optional[tuple[float,...]] = None,
        seed: Optional[int] = None,
        normalize_output: Optional[bool] = False) -> np.ndarray:
    """_correlate_field(grid_dims, spectrum_kwargs, spectrum_function, noise, phys_dims, seed)\n
    
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
    _, dk = get_dxdk(grid_dims)
    dK = np.prod(dk)

    # If just the name is provided, assume it is in the `utility.functions` file
    if isinstance(spectrum_function, str):
        spectrum_function = getattr(functions, spectrum_function)
    amplitude = spectrum_kwargs.get('amplitude', 1.)
    # Generate the spectrum and then convert to the Fourier-space field (via sqrt)
    field_k = np.sqrt(amplitude * spectrum_function(k_mesh, **spectrum_kwargs).astype('complex') * dK)

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
def _multifractal_field(
        grid_dims: tuple[int,...],
        gauss_spectrum_kwargs: dict[str,float],
        gauss_spectrum_function: str|Callable,
        scaling: float,
        spectrum_kwargs: dict[str,float],
        spectrum_function: str|Callable,
        phys_dims: Optional[tuple[float,...]] = None,
        seeds: Optional[tuple[int,int,int]] = None,
        normalize_output: Optional[bool] = False) -> np.ndarray:
    """_multifractal_field(grid_dims, gauss_spectrum_kwargs, gauss_spectrum_function, scaling, spectrum_kwargs, spectrum_function, phys_dims, seeds)\n
    
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
    gaussian_field = _synthesize_field(grid_dims, gauss_spectrum_kwargs, gauss_spectrum_function, None, phys_dims, seeds[0])
    # Make into positive log-normal field
    log_field = _make_lognormal(gaussian_field, scaling)
    # Make into a correlated noise field
    noise_real = _make_correlated_noise(log_field, seed=seeds[1])
    noise_real = (noise_real - noise_real.mean()) / np.std(noise_real)
    correlated_noise = np.fft.fftshift(np.fft.fftn(noise_real, norm='ortho'))
    # Use multiplicative noise to generate a multifractal field
    multifactal_field = _synthesize_field(grid_dims, spectrum_kwargs, spectrum_function, correlated_noise, phys_dims, seeds[2], normalize_output)
    return multifactal_field

@default_physdims('grid_dims')
def make_scalar_field(
        grid_dims: tuple[int,...],
        gauss_spectrum_kwargs: dict[str,float],
        gauss_spectrum_function: str|Callable,
        multifractal_scaling: Optional[float]=None,
        spectrum_kwargs: Optional[dict[str,float]]=None,
        spectrum_function: Optional[str|Callable]=None,
        phys_dims: Optional[tuple[float,...]]=None,
        seeds: Optional[tuple[int,...]]=None):
    """make_scalar_field(grid_dims, gauss_spectrum_kwargs, gauss_spectrum_function, scaling, spectrum_kwargs, spectrum_function, phys_dims, seeds)\n
    
    Generates a field with parameters described by `spectrum_kwargs` and `scaling`.

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
        np.ndarray: Scalar monofractal or multifractal field
    """
    make_multifractal = False
    if spectrum_kwargs is not None and spectrum_function is not None and multifractal_scaling is not None:
        make_multifractal = True
    if not make_multifractal:
        scalar_field = _synthesize_field(grid_dims, gauss_spectrum_kwargs, gauss_spectrum_function, phys_dims=phys_dims, seed=seeds)
    else:
        scalar_field = _multifractal_field(grid_dims, gauss_spectrum_kwargs, gauss_spectrum_function, multifractal_scaling, spectrum_kwargs, spectrum_function, phys_dims, seeds)
    return scalar_field

@default_physdims('grid_dims')
def make_solenoidal_field(
        grid_dims: tuple[int,...],
        gauss_spectrum_kwargs: dict[str,float],
        gauss_spectrum_function: str|Callable,
        multifractal_scaling: Optional[float]=None,
        spectrum_kwargs: Optional[dict[str,float]]=None,
        spectrum_function: Optional[str|Callable]=None,
        phys_dims: Optional[tuple[float,...]]=None,
        seeds: Optional[tuple[int,...]]=None) -> tuple[np.ndarray,...]:
    """make_solenoidal_field(grid_dims, spectrum_kwargs, spectrum_function, phys_dims, seeds)\n
    
    Make a divergenceless vector field using vector potentials

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
        (np.ndarray, ..., np.ndarray): Tuple of vector field components
    """
    D = len(grid_dims)
    pairs = list(combinations(range(D), 2))
    if seeds is None:
        seeds = (None,)*len(pairs)

    make_multifractal = False
    if spectrum_kwargs is not None and spectrum_function is not None and multifractal_scaling is not None:
        make_multifractal = True
        seeds = ((None,)*3,)*len(pairs)

    if make_multifractal:
        mod_spectrum_function = spectrum_function
    else:
        mod_spectrum_function = gauss_spectrum_function
    if isinstance(mod_spectrum_function, str):
        mod_spectrum_function = getattr(functions, mod_spectrum_function)

    def potential_spectrum(k_mesh, **kw):
        with np.errstate(divide='ignore', invalid='ignore'):
            P = np.asarray(mod_spectrum_function(k_mesh, **kw)) / ((D - 1) * k_mesh**2)
        P[k_mesh == 0] = 0.
        return P
    
    kgrid = np.meshgrid(*get_kvec(grid_dims, phys_dims), indexing='ij')
    B_k = [np.zeros(grid_dims, dtype=complex) for _ in range(D)]

    for (i,j), s in zip(pairs, seeds):
        if make_multifractal:
            A_ij = make_scalar_field(
                grid_dims, gauss_spectrum_kwargs, gauss_spectrum_function,
                multifractal_scaling, spectrum_kwargs, potential_spectrum,
                phys_dims, seeds=s)
        else:
            A_ij = make_scalar_field(
                grid_dims, gauss_spectrum_kwargs,
                potential_spectrum, phys_dims=phys_dims, seeds=s)
        A_k = np.fft.fftshift(np.fft.fftn(A_ij))
        B_k[i] += 1j * kgrid[j] * A_k
        B_k[j] -= 1j * kgrid[i] * A_k

    return tuple([np.fft.ifftn(np.fft.ifftshift(b)).real for b in B_k])

def dephase_field(field: np.ndarray,
                seed: Optional[int]=None):
    """dephase_field(field)\n

    Dephases a field by re-sampling the phases in Fourier space

    Args:
        field (np.ndarray): The field to dephase

    Returns:    
        np.ndarray: Dephased (Gassianized) field
    """
    if seed is not None:
        np.random.seed(seed)
    field_k = np.fft.fftn(field)
    field_k = np.abs(field_k) * _white_noise(np.shape(field))
    return np.fft.ifftn(field_k).real
