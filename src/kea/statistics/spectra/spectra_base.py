"""
Implements normalization considerations for different Fourier transform conventions
"""

from kea.utils import TWOPI
from kea.utils.geometry import default_physdims, get_dxdk, get_mesh
from kea.utils import binning

from typing import Optional
from enum import IntEnum
import numpy as np

class FourierNorm(IntEnum):
    r"""
    Describes the following normalization conventions:\n

    - T: angular wavenumbers and $(2\pi)^{-D}$ on the forwards transform
    - I: linear wavenumbers and no $(2\pi)^{-D}$ factors (cancelled by the angular-linear wavenumber conversion)
    - C: angular wavenumbers and $(2\pi)^{-D}$ on the backwards transform
    """
    T = 0
    I = 1
    C = 2

class SpectrumType(IntEnum):
    """
    Describes the spectrum type/how the spectrum is binned:\n

    - FIELD: The FT of the field itself. This is NOT a spectrum.
    - MODAL: Represents the PSD in waveVECTOR-space
    - INTEGRATED: Represents the contribution to the PSD that lie between magnitude wavenumbers k and k+dk (assuming left-aligned binning)
    - AVERAGED: Represents the average PSD for magnitude wavenumbers
    - AMPLITUDE: Dimensionless form the the PSD -> standard deviation per log-wavenumber
    """
    FIELD = 0
    MODAL = 1
    INTEGRATED = 2
    AVERAGED = 3
    AMPLITUDE = 4

@default_physdims('modal_spectrum')
def bin_spectrum(
        kvec: tuple[np.ndarray,...],
        modal_spectrum: np.ndarray,
        spec_type: str|SpectrumType,
        phys_dims: Optional[tuple[float,...]]=None,
        **kwargs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """spectrum_integrate(kvec, modal_spectrum, spec_type)\n

    Calculates a 1D binned spectrum from a provided ND modal spectrum `modal_spectrum`.

    Args:
        kvec (tuple): Tuple of wavenumbers representing the axes of the modal spectrum
        modal_spectrum (np.ndarray): The full N-dimensional modal spectrum
        spec_type (str): String indicating what specific spectra type to return.
            integrated: Summed over bin-shells
            averaged: Averaged over bin-shells
            amplitude: Log-spaced variance
        kwargs (dict): Additional parameters to pass onto the binning function
            - min_bin (float): Default 1*dk. Sets the minimum bin value
            - max_bin (float): Default maximum available wavenumber. Sets the maximum bin value
            - log_space (bool): If true, bin in log-spaced wavenumbers
            - ignore_nan (bool): If true, ignore NaN values of the binned result
            - cut_excess (bool): Default True. If true, cut off wavenumbers larger than the basis direction
            - nan_small (bool): Default False. If true, set to nan all bins that have a small number of elements
            - bin_center (bool): Default True. If true, return wavenumbers from the center of the bin region
            - norm_bin_size (bool): Default False. If true, divide by the size of the bin

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): Wavenumber array, binned spectrum, and bin widths
    """

    if isinstance(spec_type, str):
        spec_type = SpectrumType[spec_type.upper()]

    kmesh = get_mesh(kvec)
    if np.shape(kmesh) != np.shape(modal_spectrum):
        raise ValueError('`kvec` does not span `modal_spectrum`.')

    cut_excess = kwargs.get('cut_excess', True)
    nan_small = kwargs.get('nan_small', False)
    # Minimum k is the first non-zero k value = 1*dk])
    _, dk = get_dxdk(modal_spectrum.shape, phys_dims)
    min_k = np.min(dk)
    min_bin = kwargs.get('min_bin', min_k)
    max_bin = kwargs.get('max_bin', None)
    bin_center = kwargs.get('bin_center', True)
    if bin_center:
        bin_center = 'true_center'
    norm_bin_size = kwargs.get('norm_bin_size', None)
    log_space = kwargs.get('log_space', False)
    num_bins = kwargs.get('num_bins', None)
    ignore_nan = kwargs.get('ignore_nan', True)
    max_half_bin_width = kwargs.get('max_half_bin_width', None)
    if spec_type == SpectrumType.INTEGRATED:
        ## Integrate over the bin-shells
        if norm_bin_size is None:
            norm_bin_size = True
        bins, binned_spectrum, bin_widths = binning.bin_data(kmesh, modal_spectrum*np.prod(dk), bin_func=np.nansum,
            cut_excess=cut_excess, nan_small=nan_small, min_bin=min_bin, bin_loc=bin_center,
            norm_bin_size=norm_bin_size, log_space=log_space, num_bins=num_bins, ignore_nan=ignore_nan,
            max_bin=max_bin, max_half_bin_width=max_half_bin_width)
        return bins, binned_spectrum, bin_widths
    elif spec_type == SpectrumType.AVERAGED:
        ## Average over the bin-shells
        if norm_bin_size is None:
            norm_bin_size = False
        bins, binned_spectrum, bin_widths = binning.bin_data(kmesh, modal_spectrum*np.prod(dk), bin_func=np.nanmean,
            cut_excess=cut_excess, nan_small=nan_small, min_bin=min_bin, bin_loc=bin_center,
            norm_bin_size=norm_bin_size, log_space=log_space, num_bins=num_bins, ignore_nan=ignore_nan,
            max_bin=max_bin, max_half_bin_width=max_half_bin_width)
        return bins, binned_spectrum, bin_widths
    else:
        raise ValueError('Spectrum type is not valid: %s' % spec_type)

@default_physdims('grid_dims')
def transform_spectrum(
        k: tuple|np.ndarray,
        fek: np.ndarray,
        dimension: int,
        current_type: str|SpectrumType,
        new_type: str|SpectrumType,
        grid_dims: tuple[int,...],
        phys_dims: Optional[tuple[float,...]] = None,
        bin_widths: Optional[np.ndarray] = None,
        is_centered: Optional[bool] = True) -> tuple[np.ndarray, np.ndarray]:
    """transform_spectrum(k, fek, orig_dim, current_type, new_type)

    Converts between different spectrum types:
    - integrated
    - averaged
    - amplitude

    Args:
        k (tuple|np.ndarray): Wavenumber array
        fek (np.ndarray): (binned) 1D Spectrum of `current_type`
        dimension (int): Dimension of the original data $D$
        current_type (str|SpectrumType): Current spectrum type
        new_type (str|SpectrumType): Requested spectrum type

    Returns:
        (np.ndarray, np.ndarray): `k` and `fek` in the `new_type` convention
    """
    # Convert string inputs to SpectrumType
    if isinstance(current_type, str):
        current_type = SpectrumType[current_type.upper()]
    if isinstance(new_type, str):
        new_type = SpectrumType[new_type.upper()]

    if current_type == new_type:
        # Silently return the original/unchanged spectrum
        return k, fek

    if new_type == SpectrumType.MODAL:
        raise ValueError('Cannot convert 1D spectrum to `MODAL`. Try `AVERAGED`.')
    
    # If modal spectrum, then we need to integrate/average
    if current_type == SpectrumType.MODAL:
        do_amp = False
        if new_type == SpectrumType.AMPLITUDE:
            new_type = SpectrumType.INTEGRATED
        k, fek, bin_widths = bin_spectrum(k, fek, new_type, phys_dims)
        if do_amp:
            k, fek = transform_spectrum(k, fek, dimension, current_type, SpectrumType.AMPLITUDE, grid_dims, phys_dims, bin_widths, is_centered)
        return k, fek

    if new_type == SpectrumType.AMPLITUDE:
        # Convert to the integrated spectrum and then apply the same formula (that is independent of `dimension`)
        if current_type != SpectrumType.INTEGRATED:
            # Convert to the integrated spectrum first
            k, fek = transform_spectrum(k, fek, dimension, current_type, SpectrumType.INTEGRATED, grid_dims, phys_dims, bin_widths, is_centered)
        return k, np.sqrt(k * fek)

    if new_type == SpectrumType.INTEGRATED or new_type == SpectrumType.AVERAGED:
        # Otherwise we need to load the jacobian/shell volume factors
        dx, dk = get_dxdk(grid_dims, phys_dims)
        if bin_widths is None:
            bin_widths = np.ones_like(k)*dk[0]
        binshell_kernel = binning.hypersphere_binshell(k, dimension, bin_widths, grid_widths=dk[0], centered=is_centered)

        if current_type == SpectrumType.INTEGRATED and new_type == SpectrumType.AVERAGED:
            # Converting integrated -> averaged
            fek = fek / binshell_kernel * bin_widths
        else:
            # Otherwise, converting averaged -> integrated
            fek = fek * binshell_kernel / bin_widths

        return k, fek

    raise ValueError('Invalid inputs.')

def convert_normalization_convention(
        kvec: tuple|np.ndarray,
        fek: np.ndarray,
        dimension: int,
        current_norm: str|FourierNorm,
        new_norm: str|FourierNorm,
        spectrum_type: str|SpectrumType) -> tuple[tuple, np.ndarray]:
    r"""convert_normalization_convention(kvec, fek, dimension, current_norm, new_norm, spectrum_type)\n

    Converts between different spectrum normalization conventions:\n
    - T: angular wavenumbers and $(2\pi)^{-D}$ on the forwards transform
    - I: linear wavenumbers and no $(2\pi)^{D}$ factors (cancelled by the angular-linear wavenumber conversion)
    - C: angular wavenumbers and $(2\pi)^{-D}$ on the backwards transform

    Args:
        kvec (tuple): Wavenumber array for each dimension: k_x, k_y, ... 
        fek (np.ndarray): Spectrum of `spectrum_type` and in `current_norm` convention
        dimension (int): Dimension of the original data $D$
        current_norm (FourierNorm): Current normalization convention
        new_norm (FourierNorm): Requested normalization convention
        spectrum_type (SpectrumType): The type of spectrum e.g., modal, amplitude, etc.

    Returns:
        tuple, np.ndarray: `kvec` and `fek` in the `new_norm` convention
    """
    # Convert string inputs to enum fields
    if isinstance(current_norm, str):
        current_norm = FourierNorm[current_norm.upper()]
    if isinstance(new_norm, str):
        new_norm = FourierNorm[new_norm.upper()]
    if isinstance(spectrum_type, str):
        spectrum_type = SpectrumType[spectrum_type.upper()]

    # Not converting anything
    if current_norm == new_norm:
        return kvec, fek

    ## a,b factors
    FACTOR_DICT = {
        FourierNorm.C: (1., -1.),
        FourierNorm.I: (0., -TWOPI),
        FourierNorm.T: (-1., -1.)
    }

    ## Get the factors in terms of the generalized convention
    a, b = FACTOR_DICT[current_norm]
    a_prime, b_prime = FACTOR_DICT[new_norm]

    ## Convert the wavenumber convention
    if isinstance(kvec, tuple):
        ## Tuple of np.ndarray
        kvec = tuple([b*k/b_prime for k in kvec])
    else:
        ## np.ndarray only
        kvec = b*kvec/b_prime

    ## Convert the amplitudes
    if spectrum_type in (SpectrumType.FIELD, SpectrumType.MODAL, SpectrumType.AVERAGED):
        amp = (np.abs(b_prime)/np.abs(b)*TWOPI**(a_prime - a))**(-dimension/2)
    elif spectrum_type == SpectrumType.INTEGRATED:
        amp = (b_prime/b)**(dimension-1) * (np.abs(b_prime)/np.abs(b)*TWOPI**(a_prime - a))**(-dimension/2)
    elif spectrum_type == SpectrumType.AMPLITUDE:
        amp = (b_prime/b)**(dimension/2) * (np.abs(b_prime)/np.abs(b)*TWOPI**(a_prime - a))**(-dimension/4)
    else:
        raise ValueError('Invalid spectrum type: %s' % spectrum_type)

    return kvec, fek/amp

