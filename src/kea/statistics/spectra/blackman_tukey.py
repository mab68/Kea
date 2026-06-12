"""
Implements power spectra calculations using the Blackman-Tukey method i.e., the Fourier transform of the
autocorrelation function.
"""

from kea.utils.geometry import default_physdims, validate_shapes, get_dxdk, get_kvec

from typing import Optional
import numpy as np

@validate_shapes('acf_full')
@default_physdims('acf_full')
def bt_modal_spectrum(
        acf_full: np.ndarray,
        phys_dims: Optional[tuple[float,...]] = None) -> tuple[tuple[np.ndarray,...], np.ndarray]:
    """bt_spectrum(acf_full, phys_dims)\n

    Computes the correlogram/Blackman-Tukey spectrum i.e. an estimate of a (modal) Fourier power spectrum

    Args:
        acf_full (np.ndarray): The complete ACF (has the same grid_dims as the field)
            Note: we assume that the ACF is centered at N//2
        phys_dims (tuple): The physical system size in x,y,z,... direction

    Returns:
        (tuple, np.ndarray): Wavenumber arrays and modal spectrum
    """
    grid_dims = acf_full.shape
    dx, _ = get_dxdk(grid_dims, phys_dims)
    kvec = get_kvec(grid_dims, phys_dims)
    dX = np.prod(dx)
    fek = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(acf_full))).real*dX
    fek = np.abs(fek)
    return kvec, fek
