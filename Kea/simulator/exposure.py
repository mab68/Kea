
"""
exposure.py

Provides methods for generating masks, and exposure maps

Functions
---------
generate_mask()\n
generate_exposure()\n
"""

import numpy as np

from .fbm import create_fbm


def create_mask(shape, mask_type='random', p=0.5):
    """create_mask(shape, mask_type, p)

    Creates a mask that would remove 'bad' data. 

    Args:
        shape (tuple): Shape of the mask
        mask_type (str): random, outside, fbm
        p (float): Fraction of data to remove
    Returns:
        np.ndarray: Mask to indicate bad data
    """
    assert len(set(shape)) <= 1, 'Must have the same number of points in each dimension'
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
            mask = create_fbm(shape, [2., -11./3.], [2.], seed=423)
            mask[np.abs(mask) < val] = 0
            mask[np.abs(mask) >= val] = 1
            m2_p = np.sum(mask)/np.prod(shape)
            val = val + 0.2
    else:
        raise NotImplementedError('Unknown mask type (%s)' % mask_type)

    return mask

def create_exposure(shape):
    raise NotImplementedError()
