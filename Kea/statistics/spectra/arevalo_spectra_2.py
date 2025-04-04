
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
    sigma = float(sig) * dsig
    if M is None:
        M = int(1000. * sig + 0.5)
    n = np.arange(-M, M+1, 1)
    x = n * dx
    return np.exp(-x**2 / (2. * sigma**2))/np.sqrt(2. * np.pi * sigma**2)
    #G = np.exp(-x**2 / (2. * sigma**2))
    #return G/np.sum(G)

# def img_conv_gaussian(ar, sigma, dx, dsig, M, mode='constant'):
#     input = np.asarray(ar)
#     cval = 0.
#     weights = gaussian_kernel1d(sigma, dx, dsig, M)[::-1]
#     for axis in range(ar.ndim):
#         input = ndimage.correlate1d(input, weights, axis, mode=mode, cval=cval, origin=0)
#     return input

# def img_conv_gaussian(ar, sigma, mode='constant'):
#     """img_conv_gaussian(img, sigma, filter_size)

#     Convolves the 2/3D array `ar` with a gaussian of size `filter_size` at stdev `sigma`

#     Args:
#         ar (np.ndarray): The array to apply the gaussian convolution to
#         sigma (float): The stdev of the gaussian. NOTE: this is true sigma `sigma`
#     Returns:
#         conv_ar (np.ndarray): array convolved with the gaussian
#     """
#     ## TODO: Think about other filters...
#     ## i.e. boxcar
#     conv_ar = ndimage.gaussian_filter(ar, sigma, mode=mode, truncate=10.)
#     return conv_ar

from scipy.signal import windows

# def _kernel1d(sigma, order, radius):
#     x = np.arange(-radius, radius+1)
#     phi_x = np.ones_like(x)
#     phi_x[x > sigma/2.] = 0.
#     phi_x[x < -sigma/2.] = 0.
#     phi_x = phi_x / phi_x.sum()
#     return phi_x

def custom_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=1.0, *, radius=None,
                      window='boxcar'):
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 1.5)
    if radius is not None:
        lw = radius
    weights = windows.get_window(window, lw)[::-1]
    return ndimage.correlate1d(input, weights, axis, output, mode, cval, 0)

def custom_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=1.0, *, radius=None,
                    axes=None, window='boxcar'):
    input = np.asarray(input)
    output = ndimage._ni_support._get_output(output, input)
    axes = ndimage._ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    orders = ndimage._ni_support._normalize_sequence(order, num_axes)
    sigmas = ndimage._ni_support._normalize_sequence(sigma, num_axes)
    modes = ndimage._ni_support._normalize_sequence(mode, num_axes)
    radiuses = ndimage._ni_support._normalize_sequence(radius, num_axes)
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
            for ii in range(num_axes) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode, radius in axes:
            custom_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate, radius=radius, window=window)
            input = output
    else:
        output[...] = input[...]
    return output

def img_conv_gaussian(ar, sigma, mode='constant'):
    conv_ar = custom_filter(ar, sigma, mode=mode, truncate=1.)
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
    ds = np.sqrt(2.) / np.pi
    dsig = dx * ds
    dk = 2. * np.pi / lenn

    o1 = o / np.sqrt(1. + xi)
    o2 = o * np.sqrt(1. + xi)

    # Convolve the image with the two gaussians
    img1_conv_gauss1 = img_conv_gaussian(ar1, o1, mode=mode)
    exp1_conv_gauss1 = img_conv_gaussian(exp1, o1, mode=mode)
    img1_conv_gauss2 = img_conv_gaussian(ar1, o2, mode=mode)
    exp1_conv_gauss2 = img_conv_gaussian(exp1, o2, mode=mode)
    mask1 = exp1 > 0.
    mask1[exp1_conv_gauss1 == 0] = 0
    mask1[exp1_conv_gauss2 == 0] = 0
    exp1_conv_gauss1[exp1_conv_gauss1 == 0] = 1.
    exp1_conv_gauss2[exp1_conv_gauss2 == 0] = 1.
    filtered_image1 = exp1 * mask1 * (img1_conv_gauss1/exp1_conv_gauss1 - img1_conv_gauss2/exp1_conv_gauss2)

    if ar2 is not None:
        img2_conv_gauss1 = img_conv_gaussian(ar2, o1, mode=mode)
        exp2_conv_gauss1 = img_conv_gaussian(exp2, o1, mode=mode)
        img2_conv_gauss2 = img_conv_gaussian(ar2, o2, mode=mode)
        exp2_conv_gauss2 = img_conv_gaussian(exp2, o2, mode=mode)
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
    M = np.max([int(100. * float(o1) + 0.5), int(100. * float(o2) + 0.5)])
    #M = ar1.shape[0]
    cgauss1 = gaussian_kernel1d(o1, dx, dx, M)
    cgauss2 = gaussian_kernel1d(o2, dx, dx, M)
    ## NOTE: Gaussian impulses are separable, so we can just integrate a 1D impulse D times
    gauss_var = (np.nansum(cgauss1**2))**(ar1.ndim) - 2.*(np.nansum(cgauss1*cgauss2))**(ar1.ndim) + (np.nansum(cgauss2**2))**(ar1.ndim)

    # Mask compensation
    m_comp = np.prod(np.shape(mask1)) / np.nansum(mask1)

    # Normalize the convolution variance with the Filter variance
    result = m_comp * var / gauss_var
    return result

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

    dx = lenn / ar1.shape[0]
    dsig = dx * np.sqrt(2.)/np.pi

    Nsig = len(lags)
    fek = np.zeros(Nsig)
    kk = np.zeros(Nsig)

    for s, o in enumerate(lags):
        #kk[s] = np.sqrt(2.) / (o * dsig)
        if ~np.isfinite(o):
            continue
        kk[s] = np.sqrt(2.*ar1.ndim + 6.) / (2. * o * dx)
        fek[s] = sigma_variance(o, ar1, exp1, ar2, exp2, xi, mode, lenn)
    return kk, fek / np.sum(np.ones_like(ar1) * dx**ar1.ndim)

def k_to_discrete_lags(k, N, L=2.*np.pi, D=2):
    """k_to_discrete_lags(k, N, L=2.*np.pi, D=2)

    Args:
        k (np.array): Wavenumber array
        N (int): Number of datapoints
        L (float): Computational domain
        D (int): Number of dimensions
    Returns:
        np.array: 'Discrete' sigma values to pass into `modal_spectrum`
    """
    #dsig = (np.sqrt(2.) / np.pi) * (L / N)
    #lags = np.sqrt(2.) / (k * dsig)
    #return lags
    dx = L / N
    #ds = np.sqrt(2.) / np.pi
    #lags = 2. / np.sqrt(2.*k**2 - D + 1.)
    lags = np.sqrt(2. * D + 6.) / (2. * k)
    return lags / dx# / ds

# k = np.sqrt(2./sigma**2 + (D-1.)/2.)
# s = 2. / np.sqrt(2.*k**2 - D + 1.)
