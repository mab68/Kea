
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
"""

import numpy as np

import scipy.fft as fft

from ..utils import funcs


def create_fbm(grid_dims, phys_dims, alphas, breaks, gfunc='smooth_pow', func_kwargs={}):
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
    output = gfunc(kk, alphas, breaks, **func_kwargs).astype('complex')
    C = 2./dkD
    output = np.sqrt(output * C)
    # Random phases
    phases = np.random.uniform(0, 2.*np.pi, size=grid_dims)
    phases = np.cos(phases) + 1j * np.sin(phases)
    output = output * phases
    #output = output
    output[np.isnan(output)] = 0. + 1j * 0.0

    # DC components must have no imaginary components
    #index = [0 for _ in dims]
    #output[tuple(index)] = output[tuple(index)].real + 1j * 0.0
    #index[-1] = -1
    #output[tuple(index)] = output[tuple(index)].real + 1j * 0.0

    # Convert to configuration-space
    fbm_field = dkD * fft.ifftn(output, norm='forward').real
    return fbm_field

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

