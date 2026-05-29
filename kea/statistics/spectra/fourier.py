"""
fourier.py
Implements power spectra calculations using direct FFT conversion and then squaring the absolute value.
This is essentially accomplishing the periodogram method via the FFT.

Functions
---------
fourier_spectrum\n
"""

from ...utils.geometry import default_physdims, validate_shapes

from typing import Optional
import numpy as np

@validate_shapes('field_a', 'field_b')
@default_physdims('field_a')
def fourier_spectrum(
        field_a: np.ndarray,
        field_b: Optional[np.ndarray] = None,
        phys_dims: Optional[tuple] = None) -> tuple[tuple, np.ndarray]:
    """fourier_spectrum(field_a, field_b, phys_dims)\n

    Computes the N-dimensional (modal) Fourier power spectrum via FFT.
    This is the classical/Schuster periodogram.

    Args:
        field_a (np.ndarray): Array to compute the spectrum of
        field_b (np.ndarray): Array if computing the cross-spectrum
        phys_dims (tuple): The physical system size in x,y,z,... direction
    Returns:
        kvec (tuple): Wavenumber arrays, $\\mathbf{k}$
        fek (np.ndarray): Modal spectrum, $E_{D}(\\mathbf{k})$
    """
    kvec = []
    dx = []
    dk = []
    for i, N in enumerate(field_a.shape):
        dx.append(phys_dims[i]/N)
        # Note: this is really dk = 1/Ndx
        dk.append(1./(N*dx[i]))
        kvec.append(np.fft.fftshift(np.fft.fftfreq(N))*2.*np.pi/dx[i])
    far1 = np.prod(dx)*np.fft.fftshift(np.fft.fftn(field_a))
    if field_b is not None:
        # Compute the cross-spectrum
        far2 = np.prod(dx)*np.fft.fftshift(np.fft.fftn(field_b))
        fek = np.prod(dk)*far1*np.conjugate(far2)
        fek = fek.real
    else:
        # Compute the auto-spectrum
        fek = np.prod(dk)*np.abs(far1)**2
    return tuple(kvec), fek
