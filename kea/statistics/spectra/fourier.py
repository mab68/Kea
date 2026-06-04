"""
Implements power spectra calculations using direct FFT conversion and then squaring the absolute value.
This is essentially accomplishing the periodogram method via the FFT.
"""

from kea.utils.geometry import default_physdims, validate_shapes, get_dxdk, get_kvec

from typing import Optional
import numpy as np

@validate_shapes('field_a', 'field_b')
@default_physdims('field_a')
def fourier_modal_spectrum(
        field_a: np.ndarray,
        field_b: Optional[np.ndarray] = None,
        phys_dims: Optional[tuple[float,...]] = None) -> tuple[tuple[np.ndarray,...], np.ndarray]:
    """fourier_spectrum(field_a, field_b, phys_dims)\n

    Computes the N-dimensional (modal) Fourier power spectrum via FFT.
    This is the classical/Schuster periodogram.

    Args:
        field_a (np.ndarray): Array to compute the spectrum of
        field_b (np.ndarray): Array if computing the cross-spectrum
        phys_dims (tuple): The physical system size in x,y,z,... direction

    Returns:
        (tuple, np.ndarray): wavenumber arrays and modal spectrum
    """
    print(phys_dims)
    grid_dims = field_a.shape
    dx, _ = get_dxdk(grid_dims, phys_dims)
    dX = np.prod(dx)
    L = np.prod(phys_dims)

    kvec = get_kvec(grid_dims, phys_dims)
    far1 = dX*np.fft.fftshift(np.fft.fftn(field_a))
    if field_b is not None:
        # Compute the cross-spectrum
        far2 = dX*np.fft.fftshift(np.fft.fftn(field_b))
        fek = far1*np.conjugate(far2)
        fek = fek.real
    else:
        # Compute the auto-spectrum
        fek = np.abs(far1)**2
    return tuple(kvec), fek/L
