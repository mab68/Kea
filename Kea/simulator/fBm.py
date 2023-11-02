
"""
data_generator.py

Provides methods for generating random fields of specific power law (inspired by [1,2])

Functions
---------
generate_field()\n
dephase_data()\n
generate_mask()\n
generate_exposure()\n

References:
-----------
[1] Koch, Eric W., Erik W. Rosolowsky, Ryan D. Boyden, Blakesley Burkhart, Adam Ginsburg, Jason L. Loeppky, and Stella SR Offner.
    "TurbuStat: Turbulence statistics in python." The Astronomical Journal 158, no. 1 (2019): 1.
[2] https://turbustat.readthedocs.io/en/latest/
"""

import numpy as np

import scipy.fft as fft

from ..utils import funcs


def create_fbm(dims, alphas, breaks, lenn=None, seed=1234, phase='random', power=True, gfunc='smooth_pow', **kwargs):
    """create_fbm(dims, alphas, breaks, A, lenn, seed, kwargs)

    Generates an arbitrary fBm field

    Args:
        dims (tuple): Number of points in each dimension
        alphas (tuple): Power laws
        breaks (tuple): Locations for the power law to change slope
        A (float): The amplitude of the spectrum to return
        xn (float): Noise scale
        lenn (tuple): Length of the domain for each dimension
        seed (float): Random number generator seed
        phase (str): How to deal with the complex phase information
            random (str): Completely randomize the phase information
        gfunc (func/str): Generation function, or string of the function name from `simulator.funcs.py`
        kwargs (dict): information to pass onto the generation functions
    Returns:
        np.ndarray: fBM field of specific powerlaw
    """
    np.random.seed(seed)
    assert len(set(dims)) <= 1, 'Must have the same number of points in each dimension'
    if lenn is None:
        lenn = [2. * np.pi for _ in dims]
    assert len(set(lenn)) <= 1, 'Must be the same length in each dimension'

    dx = [lenn[i]/dims[i] for i in range(len(dims))]
    dxD = np.prod(dx)
    dk = [1./lenn[i] for i in range(len(dims))]
    dkD = np.prod(dk)

    # Generate wavenumbers
    kvec = [np.fft.fftfreq(dims[i])*2.*np.pi/dx[i] for i in range(len(dims))]
    kgrid = np.meshgrid(*kvec, indexing='ij')
    kk = np.linalg.norm(kgrid, axis=0)

    if isinstance(gfunc, str):
        gfunc = getattr(funcs, gfunc)

    # Generate fBM in Fourier space
    output = gfunc(kk, alphas, breaks, **kwargs).astype('complex')
    C = 2./dkD
    if power:
        output = np.sqrt(output * C)
    if phase == 'random':
        # Random phases
        phases = np.random.uniform(0, 2.*np.pi, size=dims)
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

def dephase_data(ar, seed=1234):
    """dephase_data(ar)

    Retains the same power law as the data, but randomizes the phases.
    This creates an fBM field from some raw data.

    Args:
        ar (np.ndarray): Data to dephase
    Returns:
        np.ndarray: Dephased data    
    """
    np.random.seed(seed)
    far = fft.fftn(ar)
    phases = np.random.uniform(0, 2.*np.pi, size=ar.shape)
    far_dephased = np.abs(far) * (np.cos(phases) + 1j * np.sin(phases))
    fbm_field = fft.ifftn(far_dephased).real
    return fbm_field

def generate_mask(shape, mask_type='random', p=0.5, seed=1234):
    """generate_mask(shape)

    Creates a mask that would remove 'bad' data. 

    Args:
        shape (tuple): Shape of the mask
        mask_type (str): random, outside, fbm
        p (float): Fraction of data to remove
    Returns:
        np.ndarray: Mask to indicate bad data
    """
    assert len(set(shape)) <= 1, 'Must have the same number of points in each dimension'
    np.random.seed(seed)
    mask = np.ones(shape)

    if mask_type == 'random':
        n = len(shape)
        grid = np.indices(shape)
        def create_circle(pos, r):
            rvecs = [grid[i,...] - pos[i] for i in range(n)]
            c = np.linalg.norm(rvecs, axis=0)
            return c <= r
        total = np.prod(shape)
        while np.sum(mask)/total > p:
            pos = [np.random.randint(0, shape[i]) for i in range(n)]
            r = np.random.randint(1, shape[0]//16)
            m = create_circle(pos, r)
            mask[m] = 0.
    elif mask_type == 'outside':
        r_cut = np.sqrt(p * shape[0]**2 / np.pi)
        grid = np.indices(shape)
        grid = [grid[i,...] - n//2 for i, n in enumerate(shape)]
        idx = np.linalg.norm(grid, axis=0) > r_cut
        mask[idx] = 0
    elif mask_type == 'fbm':
        m2_p = 1.
        val = 0.
        while m2_p > p:
            mask = generate_field(shape, [2., -11./3.], [2.], seed=423)
            mask[np.abs(mask) < val] = 0
            mask[np.abs(mask) >= val] = 1
            m2_p = np.sum(mask)/np.prod(shape)
            val = val + 0.2
    else:
        raise NotImplementedError('Unknown mask type (%s)' % mask_type)

    return mask

def generate_exposure(shape, seed=1234):
    """generate_exposure(shape, seed)

    Args:
        shape (tuple): Shape of the exposure map
    Returns:
        np.ndarray: Exposure map
    """
    assert len(set(shape)) <= 1, 'Must have the same number of points in each dimension'
    np.random.seed(seed)
    exp = np.ones(shape)

    n = len(shape)
    grid = np.indices(shape)

    def create_circle(pos, r):
        rvecs = [grid[i,...] - pos[i] for i in range(n)]
        c = np.linalg.norm(rvecs, axis=0)
        return c <= r

    total = np.prod(shape)
    while np.sum(exp) < 10.*total:
        pos = [np.random.randint(0, shape[i]) for i in range(n)]
        r = np.random.randint(1, shape[0]//16)
        m = create_circle(pos, r)
        exp[m] = exp[m] + 1.

    exp = (exp - np.min(exp))/np.ptp(exp)
    return exp
