
"""
scale_filter.py

Implements the Gaussian scale filtering and Fourier wavenumber scale filtering

Functions
---------
fourier_scaleselect\n
gaussian_scale_greater\n
gaussian_scale_less\n
gaussian_scale_equal\n
"""

import numpy as np

import scipy.fft as fft
import scipy.ndimage as ndimage


def fourier_scale(ar, krange):
    """fourier_scale(ar, krange)
    
    Selects frequencies/wavenumbers from `ar` within `krange`

    Args:
        ar (np.ndarray): Data to apply a Fourier filter to
        (kmin, kmax) (tuple): The minimum and maximum wavenumbers
    Returns:
        np.ndarray: Real-space image containing frequencies within krange
    """
    kmin, kmax = krange
    uk2 = fft.fftshift(fft.fftn(ar))
    dx = [2.*np.pi/N for N in ar.shape]
    kvec = [fft.fftshift(fft.fftfreq(N))*2.*np.pi/dx[i] for i, N in enumerate(ar.shape)]
    kmesh = np.meshgrid(*kvec, indexing='xy')
    k = np.linalg.norm(kmesh, axis=0)
    kmask = np.ones_like(k, dtype='bool')
    kmask[np.logical_and(np.abs(k) <= kmax, np.abs(k) >= kmin)] = False
    uk2_filtered = uk2.copy()
    uk2_filtered[kmask] = 0. + 1j * 0.
    ar_filtered = fft.ifftn(fft.ifftshift(uk2_filtered)).real
    return ar_filtered

def gaussian_scale_greater(ar, sigma):
    """gaussian_scale_greater(ar, sigma)

    Convolves the image `ar` with a Gaussian of width (stdev) `sigma`

    Selects scales >= `sigma`
    
    Args:
        ar (np.ndarray): The array to scale filter for
        sigma (float): The width of the Gaussian. NOTE: this is the true sigma `sigma`
    Returns:
        np.ndarray: Scale filtered array
    """
    return ndimage.gaussian_filter(ar, sigma, mode='constant', cval=0.)

def gaussian_scale_less(ar, sigma):
    """gaussian_scale_greater(ar, sigma)

    Convolves the image `ar` with a Gaussian of width `sigma`,
        and subtracts the original by this Gaussian convoled image
    
    Selects scales < `sigma`
    
    Args:
        ar (np.ndarray): The array to scale filter for
        sigma (float): The width of the Gaussian
    Returns:
        np.ndarray: Scale filtered array
    """
    return ar - gaussian_scale_greater(ar, sigma)

def gaussian_scale_equal(ar, sigma, mask=None, xi=1e-3):
    """gaussian_scale_equal(ar, sigma, xi=1e-3)

    Convolves the image `ar` with two Gaussian of almost equal widths
        and takes their difference

    Selects scales = `sigma`
    
    Args:
        ar (np.ndarray): The array to scale filter for
        sigma (float): The width of the Gaussian
        xi (float): Determines the distance between the two widths
    Returns:
        np.ndarray: Scale filtered array
    """
    sig1 = sigma / np.sqrt(1. + xi)
    sig2 = sigma * np.sqrt(1. + xi)
    if mask is None:
        return gaussian_scale_greater(ar, sig1) - gaussian_scale_greater(ar, sig2)
    else:
        img1_conv_gauss1 = gaussian_scale_greater(ar, sig1)
        exp1_conv_gauss1 = gaussian_scale_greater(mask, sig1)
        img1_conv_gauss2 = gaussian_scale_greater(ar, sig2)
        exp1_conv_gauss2 = gaussian_scale_greater(mask, sig2)
        m = np.ones_like(ar)
        m[exp1_conv_gauss1 == 0] = 0
        m[exp1_conv_gauss2 == 0] = 0
        m[mask == 0] = 0
        exp1_conv_gauss1[exp1_conv_gauss1 == 0] = 1.
        exp1_conv_gauss2[exp1_conv_gauss1 == 0] = 1.
        filtered_image1 = m * (img1_conv_gauss1/exp1_conv_gauss1 - img1_conv_gauss2/exp1_conv_gauss2)
        return filtered_image1
