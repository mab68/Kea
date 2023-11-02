
"""
flatsky_spectra.py

Provides calculations of the power spectrum via assuming the data lies on a small portion of the sphere on the sky
    and then applying spherical harmonics.

Functions
---------
modal_spectrum\n

References:
-----------
[1] Alonso, D., Sanchez, J., Slosar, A., & LSST Dark Energy Science Collaboration. (2019).
    A unified pseudo-C ℓ framework. Monthly Notices of the Royal Astronomical Society, 484(3), 4127-4151.
[2] https://namaster.readthedocs.io/en/latest/
"""

try:
    import pymaster as nmt
except ImportError:
    nmt = None

import numpy as np

from ..statistics_base import get_bins


def modal_spectrum(ar1, ar2=None, mask=None, lenn=None, min_bin=None, max_bin=None, nbins=None,
                   log_space=False, max_half_bin_width=None):
    """modal_spectrum(ar, mask, lenn)

    Use the `pymaster` package to calculate the modal spectrum using the flat-sky approximation
        with spherical harmonics

    Args:
        ar (np.ndarray): 2D array to compute the flat-sky spectrum of
        mask (np.ndarray): Mask to apply to the image
        lenn (tuple):  Size of the domain
    Returns:
        ells (np.ndarray): Effective wavenumbers for the flatsky spectra
        cl00 (np.ndarray): 1D, integrated modal spectrum of the flatsky spectra
        cl_err (np.ndarray): 1D, standard error of bandpower
    """
    if nmt is None:
        raise ImportError('Flatsky spectra requires pymaster')

    ndim = len(np.shape(ar1))
    assert ndim == 2, 'Data must be 2D to cast onto the sky'

    if lenn is None:
        lenn = [2. * np.pi for _ in range(ndim)]
    assert len(lenn) == 2, 'Computational domain length `L` must be 2D not %sD' % len(lenn)
    if mask is None:
        mask = np.ones_like(ar1)

    # Field
    f0 = nmt.NmtFieldFlat(lenn[0], lenn[1], mask, [ar1])
    if ar2 is None:
        f1 = f0
    else:
        f1 = nmt.NmtFieldFlat(lenn[0], lenn[1], mask, [ar2])

    # Bins
    L = np.min(lenn)
    N = np.min(ar1.shape)
    dell = 2. * np.pi / L
    if max_bin is None:
        max_bin = N//2*dell
    if min_bin is None:
        min_bin = dell
    if nbins is None:
        nbins = N//2 - 1
    bins, bin_widths, bin_edges = get_bins(min_bin, max_bin, nbins, max_half_bin_width, log_space)
    l0_bins = bins - bin_widths
    lf_bins = bins + bin_widths

    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_coupled = b.get_effective_ells()

    # Workspace
    w00 = nmt.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f0, f1, b)

    # Compute power spectra
    cl00_coupled = nmt.compute_coupled_cell_flat(f0, f1, b)
    cl00_uncoupled = w00.decouple_cell(cl00_coupled)

    # Compute covariance->variance->stdev~error
    cw = nmt.NmtCovarianceWorkspaceFlat()
    cw.compute_coupling_coefficients(f0, f1, b)
    covar = nmt.gaussian_covariance_flat(
        cw, 0, 0, 0, 0,
        ells_coupled,
        cl00_uncoupled, cl00_uncoupled, cl00_uncoupled, cl00_uncoupled,
        w00)
    cl_err = np.sqrt(np.diagonal(covar))

    return ells_coupled, cl00_uncoupled[0], cl_err
