
"""
fbm.py

Provides methods for generating random fields of specific power law (inspired by [1,2])

Functions
---------
generate_field()\n
dephase_data()\n

References:
-----------
[1] Koch, Eric W., Erik W. Rosolowsky, Ryan D. Boyden, Blakesley Burkhart, Adam Ginsburg, Jason L. Loeppky, and Stella SR Offner.
    "TurbuStat: Turbulence statistics in python." The Astronomical Journal 158, no. 1 (2019): 1.
[2] https://turbustat.readthedocs.io/en/latest/
[3] Barnsley, M. F., Devaney, R. L., Mandelbrot, B. B., Peitgen, H. O., Saupe, D., Voss, R. F., ... & McGuire, M. (1988).
    The science of fractal images (Vol. 1, p. 312). New York: Springer.
[4] Bates, M. L., Whitworth, A. P., & Lomax, O. D. (2020).
    Characterizing lognormal fractional-Brownian-motion density fields with a convolutional neural network.
    Monthly Notices of the Royal Astronomical Society, 493(1), 161-170.
"""

import itertools

import numpy as np

import scipy.fft as fft

from ..utils import funcs

def create_fbm(grid_dims, phys_dims, alphas, breaks, gfunc='pure_pow', func_kwargs={}):
    """create_fbm(grid_dims, phys_dims, alphas, breaks, gfunc, func_kwargs)

    Generates an arbitrary fBm field

    Args:
        grid_dims (tuple): Number of grid points in each dimension
        phys_dims (tuple): Physical domain in each direction
        alphas (tuple): Power laws
        breaks (tuple): Locations for the power law to change
        gfunc (str/func): Generation function, or string of the function name from `utils/funcs.py`
        func_kwargs (dict): Additional function arguments to pass to `gfunc`
    Returns:
        nd.ndarray: fBm field
    """
    assert len(set(grid_dims)) <= 1, 'Must have the same number of points in each dimension'
    if phys_dims is None:
        phys_dims = [2. * np.pi for _ in phys_dims]
    assert len(set(phys_dims)) <= 1, 'Must be the same length in each dimension'

    dx = [phys_dims[i]/grid_dims[i] for i in range(len(grid_dims))]
    dxD = np.prod(dx)
    dk = [1./phys_dims[i] for i in range(len(phys_dims))]
    dkD = np.prod(dk)

    # Generate wavenumbers
    kvec = [np.fft.fftfreq(grid_dims[i])*2.*np.pi/dx[i] for i in range(len(grid_dims))]
    kgrid = np.meshgrid(*kvec, indexing='ij')
    kk = np.linalg.norm(kgrid, axis=0)

    if isinstance(gfunc, str):
        gfunc = getattr(funcs, gfunc)

    # Generate fBM in Fourier space
    output = gfunc(kk, alphas, **func_kwargs).astype('complex')
    #C = 2. / dkD
    C = np.nanmean(output) / np.prod(grid_dims)
    output = np.sqrt(output / C)

    # Random phases
    phases = np.random.uniform(0, 2.*np.pi, size=grid_dims)
    #phases = phases - fft.fftshift(phases)
    phases = np.cos(phases) + 1j * np.sin(phases)
    #phases /= np.sqrt(np.sum(phases**2) / float(phases.size))
    output = output * phases
    #output = output

    ## Ensure the function is Hermitian
    # => The FT is real-valued
    # DC value
    zero_idx = tuple([0 for _ in grid_dims])
    output[zero_idx] = 0. + 1j * 0.

    # The nyquist components without a partner must be their own real
    args = [[n//2, 0] for n in grid_dims]
    idx = list(itertools.product(*args))
    for i in range(len(idx)):
        output[tuple(idx[i])] = output[tuple(idx[i])].real + 1j * 0.

    # l_args = [[slice(1, n//2 + 1), 0] for n in grid_dims]
    # l_idx = list(itertools.product(*l_args))
    # r_args = [[slice(n, n//2 - 1, -1), 0] for n in grid_dims]
    # r_idx = list(itertools.product(*r_args))
    # for i in range(len(l_idx)):
    #     output[tuple(l_idx[i])] = np.conjugate(output[tuple(r_idx[i])])

    # NOTE: There is redundant computation here
    l_args = [[slice(1, n//2 + 1), slice(n, n//2 - 1, -1), 0, n//2] for n in grid_dims]
    l_idx = list(itertools.product(*l_args))
    r_args = [[slice(n, n//2 - 1, -1), slice(1, n//2 + 1), 0, n//2] for n in grid_dims]
    r_idx = list(itertools.product(*r_args))
    for i in range(len(l_idx)):
        output[tuple(l_idx[i])] = np.conjugate(output[tuple(r_idx[i])])

    # if len(grid_dims) == 2:
    #     for i in range(0, N//2+1):
    #         for j in range(0, N//2+1):
    #             i0 = 0 if i == 0 else N-i
    #             j0 = 0 if j == 0 else N-j
    #             output[i,j] = np.conjugate(output[i0,j0])
    #             output[i0,j] = np.conjugate(output[i,j0])

    # if len(grid_dims) == 3:
    #     for i in range(0, N//2+1):
    #         for j in range(0, N//2+1):
    #             for k in range(0, N//2+1):
    #                 i0 = 0 if i == 0 else N-i
    #                 j0 = 0 if j == 0 else N-j
    #                 k0 = 0 if k == 0 else N-k
    #                 output[i,j,k] = np.conjugate(output[i0,j0,k0])
    #                 output[i0,j,k] = np.conjugate(output[i,j0,k0])
    #                 output[i,j0,k] = np.conjugate(output[i0,j,k0])
    #                 output[i,j,k0] = np.conjugate(output[i0,j0,k])

    return fft.ifftn(output, norm='backward').real#output

def dephase_data(ar):
    """dephase_data(ar)

    Retains the same power law as the data, but randomizes the phases.
    This creates an fBM field from some raw data.

    Args:
        ar (np.ndarray): Data to dephase
    Returns:
        np.ndarray: Dephased data    
    """
    far = fft.fftn(ar)
    phases = np.random.uniform(0, 2.*np.pi, size=ar.shape)
    far_dephased = np.abs(far) * (np.cos(phases) + 1j * np.sin(phases))
    fbm_field = fft.ifftn(far_dephased).real
    return fbm_field

