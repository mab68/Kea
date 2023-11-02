
"""
arevalo_spectra.py

Implements the Gaussian difference convolution method from [1], to calculate
    a scale-space (equivalent) power spectrum.

Functions
---------
gaussian_kernel1d\n
f_gaussian_kernel1d\n
img_conv\n
img_conv_gaussian\n
sigma_variance\n
modal_spectrum\n

References:
-----------
[1] Arévalo, P., Churazov, E., Zhuravleva, I., Hernández-Monteagudo, C., & Revnivtsev, M. (2012).
    A Mexican hat with holes: calculating low-resolution power spectra from data with gaps.
    Monthly Notices of the Royal Astronomical Society, 426(3), 1793-1807.
"""

import numpy as np
from scipy import fft

import scipy.ndimage as ndimage
from scipy.interpolate import interp1d

from ..statistics_base import ndim_func
from .. import moments


def gaussian_kernel1d(sig, dx, dsig, M=None):
    """gaussian_kernel1d(sig, dx, ndim, M)

    Generates a sampled Gaussian, the range of points we will sample from
        the true Gaussian `M` is determined by the standard deviation
        `sigma` such that `\( M = C \sig + 1 \)` where `C=5`.

    Args:
        sig (float): The standard deviation of the Gaussian
            NOTE: This is in terms of the sampled standard deviation `o`
        dx (float): Change in `x`
        dsig (float): Change in `sig`
        M (int): Size of the filter (number of points to generate)
    Returns:
        np.ndarray: 1-Dimensional Gaussian in sample-space   
    """
    ## TODO: STILL NEED TO TAKE sig=1
    sigma = float(sig) * dsig
    if M is None:
        M = int(5. * sig + 0.5)
    n = np.arange(-M, M+1, 1)
    x = n * dx
    return np.exp(-x**2 / (2. * sigma**2))/np.sqrt(2. * np.pi * sigma**2)

def f_gaussian_kernel1d(sig, dk, dsig, M=None):
    """f_gaussian_kernel1d(sigma, dk, dsig, ndim, C)

    Generates the sampled Fourier transform of a sampled Gaussian.

    The Fourier transform of a normalized Gaussian is an 
        unnormalized Gaussian with standard deviation `1/sigma`.

    Args:
        sig (float): The standard deviation of the sampled-Gaussian
            NOTE: this is in terms of the sampled standard deviation
            NOTE: this is the standard deviation of the configuration-space Gaussian
        dk (float): Change in `k`
        dsig (float): Change in `sigma`
        ndim (int): Number of dimensions
        M (int): Size of the filter (number of points to generate)
    Returns:
        np.ndarray: 1-Dimensional sampled Gaussian in sample-space
    """
    sigma = sig * dsig
    if M is None:
        M = int(5. * sig + 0.5)
    m = np.arange(-M, M+1, 1)
    k = m * dk
    return np.exp(-k**2 * sigma**2 / 2.)

def img_conv(ar, weights, dx, mode='constant'):
    """img_conv(ar, weights, dx)

    Args:
        ar (np.ndarray):
        weigths (np.ndarray):
        dx (np.ndarray): 
    Returns:
        conv_ar (np.ndarray):
    """
    conv_ar = dx**ar.ndim * ndimage.convolve(ar, weights, mode=mode, cval=0.)
    return conv_ar

def img_conv_gaussian(ar, sigma, mode='constant'):
    """img_conv_gaussian(img, sigma, filter_size)

    Convolves the 2/3D array `ar` with a gaussian of size `filter_size` at stdev `sigma`

    Args:
        ar (np.ndarray): The array to apply the gaussian convolution to
        sigma (float): The stdev of the gaussian. NOTE: this is true sigma `sigma`
    Returns:
        conv_ar (np.ndarray): array convolved with the gaussian
    """
    conv_ar = ndimage.gaussian_filter(ar, sigma, mode=mode, cval=0., truncate=10.)
    return conv_ar

def sigma_variance(o, ar1, exp1, ar2=None, exp2=None, xi=1e-3, mode='constant', lenn=None):
    """sigma_variance(ar, o, xi, exp, lenn)

    Calculates the sigma variance of the array `ar` of scales `sigma`

    The calculation of the filter normalization factor is done in configuration space

    NOTE: The computational domain is assumed to be equal in configuration space `L`
        and sample space `N`

    NOTE: Provided `o` is the modified discretized scale belonging from 1 to N//2:
            `$\sigma = o (L/N) (sqrt(2)/pi)$`

    Args:
        ar1, ar2 (np.ndarray): Array to compute the variance for at the sigma scale
        o (float): Discretized scale
        xi (float): Determines the difference in the gaussian variance
        exp1, exp2 (np.ndarray): Exposure map, combination of the exposure and the mask maps
            where `1.` means take the true pixel, `0.` means mask it out, anything
            in between is to describe the exposure correction.
            The mask is constructed from the exposure map where `exp > 0.`
        lenn (float): Length of the computational domain `L`
    Returns:
        float: Value of the variance at scale `sigma`
    """
    dx = lenn / ar1.shape[0]
    ds = np.sqrt(2) / np.pi
    dsig = dx * ds
    dk = 2. * np.pi / lenn

    o1 = o / np.sqrt(1. + xi)
    o2 = o * np.sqrt(1. + xi)

    #mask1 = exp1 > 0.
    #if exp2 is not None:
    #    mask2 = exp2 > 0.

    ## NOTE: there is actually some intricacy here for the cross spectrum,
    ##       the zeros in `mask1` & `mask2` may not align, leading
    ##       to extra terms that should be cancelling (via multiplication by zero)
    ##       that aren't.
    ## NOTE: One second thought, no, actually the mask is purely the mask
    ##       based off the exposure map,
    ##       so differences in the zeros in `ar1` and `ar2` do not matter
    ##       (are correctly handled).
    # Convolve the image with the two gaussians
    img1_conv_gauss1 = img_conv_gaussian(ar1, o1*ds, mode)
    exp1_conv_gauss1 = img_conv_gaussian(exp1, o1*ds, mode)
    img1_conv_gauss2 = img_conv_gaussian(ar1, o2*ds, mode)
    exp1_conv_gauss2 = img_conv_gaussian(exp1, o2*ds, mode)
    mask1 = exp1 > 0.
    mask1[exp1_conv_gauss1 == 0] = 0
    mask1[exp1_conv_gauss2 == 0] = 0
    exp1_conv_gauss1[exp1_conv_gauss1 == 0] = 1.
    exp1_conv_gauss2[exp1_conv_gauss2 == 0] = 1.
    filtered_image1 = exp1 * mask1 * (img1_conv_gauss1/exp1_conv_gauss1 - img1_conv_gauss2/exp1_conv_gauss2)

    if ar2 is not None:
        img2_conv_gauss1 = img_conv_gaussian(ar2, o1*ds, mode)
        exp2_conv_gauss1 = img_conv_gaussian(exp2, o1*ds, mode)
        img2_conv_gauss2 = img_conv_gaussian(ar2, o2*ds, mode)
        exp2_conv_gauss2 = img_conv_gaussian(exp2, o2*ds, mode)
        mask2 = exp2 > 0.
        mask2[exp2_conv_gauss1 == 0] = 0
        mask2[exp2_conv_gauss2 == 0] = 0
        exp2_conv_gauss1[exp2_conv_gauss1 == 0] = 1.
        exp2_conv_gauss2[exp2_conv_gauss2 == 0] = 1.
        filtered_image2 = exp2 * mask2 * (img2_conv_gauss1/exp2_conv_gauss1 - img2_conv_gauss2/exp2_conv_gauss2)
    else:
        filtered_image2 = filtered_image1
    # The variance requires the dx term by my definitions
    var = np.nansum(filtered_image1 * filtered_image2)

    ## NOTE: we compute the variance of the power spectrum of the filter 
    ##       in configuration space using Parseval's theorem
    #M = np.max([int(10. * o1*dsig + 0.5), int(10. * o2*dsig + 0.5)])
    M = ar1.shape[0]
    cgauss1 = ndim_func(gaussian_kernel1d, ar1.shape, func_args=(o1, dx, dsig, M))
    cgauss2 = ndim_func(gaussian_kernel1d, ar1.shape, func_args=(o2, dx, dsig, M))
    cfilter = cgauss1 - cgauss2
    gauss_var = np.nansum(np.abs(cfilter)**2)

    # Mask compensation
    m_comp = np.prod(np.shape(mask1)) / np.nansum(mask1)

    # Normalize the convolution variance with the Filter variance
    result = m_comp * (dk / (2.*np.pi))**ar1.ndim * var / gauss_var
    return result

# def poisson_error(o, lags, fek, fek0, N, D, xi=1e-3, lenn=None):
#     """poisson_error(o, fek, xi=1e-3, lenn=None)

#     Args:
#         o (float): Discretized scale
#         xi (float): Determines the difference in the gaussian variance
#         lenn (float): Length of the computational domain `L`
#     """
#     dx = lenn / N
#     ds = np.sqrt(2) / np.pi
#     dsig = dx * ds
#     dk = 2. * np.pi / lenn

#     o1 = o / np.sqrt(1. + xi)
#     o2 = o * np.sqrt(1. + xi)

#     # need to interpolate fek to the correct locations
#     cgauss1 = ndim_func(gaussian_kernel1d, N, func_args=(o1, dx, dsig, N))
#     cgauss2 = ndim_func(gaussian_kernel1d, N, func_args=(o2, dx, dsig, N))
#     cfilter = cgauss1 - cgauss2
#     fhat2 = np.abs(fft.fftn(cfilter))**2*np.prod(dk**D)

#     # interpolate fek onto fhat2
#     sym_fek = np.ones((2*len(fek) + 1))
#     sym_lags = np.ones((2*len(fek) + 1))
#     # TODO: do this correctly
#     sym_fek[:len(fek)] = fek[::-1]
#     sym_lags[:len(lags)] = -lags[::-1]
#     sym_fek[len(fek)] = fek0
#     sym_lags[:len(lags)] = 0
#     sym_fek[len(fek):] = fek
#     sym_lags[:len(lags)] = lags
#     interp_f = interp1d(sym_lags, sym_fek)
#     np.indices(fhat2)
#     #interp_f = 

#     num = np.sqrt(np.nansum((fek_interp * fhat2)**2))
#     denom = np.nansum(fhat2)

#     return num/denom

def modal_spectrum(ar1, exp1=None, ar2=None, exp2=None, lags=None, xi=1e-3, mode='constant', lenn=None):
    """modal_spectrum(ar, mask, lags, xi)

    Calculates the spectrum via the summing the sigma-variance method for various sigma values
        provided by `lags`.

    We require the lags provided as a discrete `o \in [1, N//2]`
    where `sigma = o dsig = o (sqrt(2)/pi) (L/N)`, so really, the sigma is `o sqrt(2)/pi`.

    Args:
        ar1, ar2 (np.ndarray): Array to estimate the power spectrum of
        lags (list): List of scales (sigma values)
            NOTE: must be in terms of the standard deviation in sample-space
        xi (float): Determines the difference in the gaussian variance
        exp1, exp2 (np.ndarray): Exposure map, combination of the exposure and the mask maps
        lenn (float): Length of the computational domain `L`
    Returns:
        kvec (tuple): Wavenumber array, the equivalent maximum wavenumber from the filter function
        fek (np.ndarray): Spectrum of the array
    """
    if ar2 is not None:
        assert ar1.shape == ar2.shape, 'Provided arrays must be the same shape'
    assert len(set(ar1.shape)) <= 1, 'Number of points in each dimension is not equal'

    if exp1 is None:
        exp1 = np.ones_like(ar1)
    else:
        assert len(set(exp1.shape)) <= 1, 'Number of points in each dimension is not equal'
        assert set(exp1.shape) == set(ar1.shape), 'Exposure map not equal to given array'
    if ar2 is not None:
        if exp2 is None:
            exp2 = np.ones_like(ar2)
        else:
            assert len(set(exp2.shape)) <= 1, 'Number of points in each dimension is not equal'
            assert set(exp2.shape) == set(ar2.shape), 'Exposure map not equal to given array'

    if lenn is None:
        lenn = 2.*np.pi
    if isinstance(lenn, (list, tuple)):
        assert len(set(lenn)) <= 1, 'System size must be equal in all directions'
        lenn = lenn[0]

    ## k \in [2pi/L, pi N/L]
    ## sig \in [sqrt(2) L/2pi, sqrt(2)L/Npi]
    ## o \in [1, N/2]
    ## for dsig = dx sqrt(2)/pi = L/N sqrt(2)/pi
    if lags is None:
        lags = np.arange(1., ar1.shape[0]//2, 1)

    dx = lenn / ar1.shape[0]
    dsig = dx * np.sqrt(2)/np.pi

    Nsig = len(lags)
    fek = np.zeros(Nsig)
    kk = np.zeros(Nsig)

    for s, o in enumerate(lags):
        kk[s] = np.sqrt(2) / (o * dsig)
        fek[s] = sigma_variance(o, ar1, exp1, ar2, exp2, xi, mode, lenn)
    return kk, fek

def k_to_discrete_lags(k, N, L=2*np.pi):
    """k_to_discrete_lags(k, N, L=2.*np.pi)
    
    The wavenumber:
    :math:`k = (2 \pi / L) n = 2 \pi / \lambda`\n
    The equivalent wavenumber from sigma is:
    :math:`k_{equiv} = sqrt(2) / sigma`
    Which naively leads to:
    :math:`sigma \in [sqrt(2) L/2pi, sqrt(2) L/Npi]`\n
    for assuming `k_{equiv}` is the same as `k`
    :math:`k_{equiv} = k \in [2 pi/L, pi N/L]`

    We instead define the sigma as:
    :math:`sigma = o dsig = o (sqrt(2)/pi) (L/N)`\n
    where `o` is interpreted as a 'discrete' value (although it doesn't have to be):
    :math: `o \in [1, N//2]`
    this also constrains the effective wavenumber `k_{equiv}` to the same domain
    as the true wavenumber `k`. Hence we apply the following transformation
    to convert a given wavenumber `k` to the 'discrete' `o`:

    :math:`dsig = sqrt(2)/pi`\n
    :math:`dx = L/N`\n
    :math:`sigma = o * dsig * dx = sqrt(2) / k`\n
    :math:`=> o = sigma / (dsig * dx)`\n
    :math:`     = (sqrt(2) / k) / (dsig * dx)`

    Args:
        k (np.array): Wavenumber array
        N (int): Number of datapoints
        L (float): Computational domain
    Returns:
        np.array: 'Discrete' sigma values to pass into `modal_spectrum`
    """
    dsig = (np.sqrt(2.) / np.pi) * (L / N)
    lags = np.sqrt(2.) / (k * dsig)
    return lags
