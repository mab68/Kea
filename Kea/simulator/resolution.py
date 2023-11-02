
"""
modify_resolution.py

Provides methods to modify the resolution, mimicking rtelescope resolution constraints

Functions
---------
pad_image\n
apply_pool_resolution\n
apply_blur_resolution\n
"""

import numpy as np

import skimage
from skimage import filters

## NOTE: skimage.filters.difference_of_gaussians(image, low_sigma, high_sigma=None, *, mode='nearest', cval=0, channel_axis=None, truncate=4.0, multichannel=False)
# https://stackoverflow.com/questions/36263738/why-is-gaussian-filter-different-between-cv2-and-skimage
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html#gsc.tab=0


def pad_image(ar, pad_size):
    """pad_ar(ar, pad_size)

    Pads the array `ar` with zeros by `pad_size` on all directions

    Args:
        ar (np.ndarray): The array to pad
        pad_size (int): The size of the pad to add to all axes
    Returns:
        padded_ar (np.ndarray): Array with padded values
    """
    # TODO: PAD IMAGE VARIABLY BASED ON X,Y,Z PAD SIZE
    pad_size = int(pad_size)
    padded_ar = np.pad(ar, pad_size, constant_values=0)
    return padded_ar

def apply_pool_resolution(ar, block_size, pool_func=np.mean, resize=False, resize_kwargs={}):
    """apply_pool_resolution(ar, block_size)

    Use `pool_func` to pool nearby pixels of size `block_size` to reduce the image resolution

    Args:
        ar (np.ndarray): Data array to apply resolution to
        block_size (tuple): Size of the resolution blocks
        pool_func (func): Function for grouping the data
        resize (bool): Resize back into the original size of `ar`
    Returns:
        new_ar (np.ndarray): Reduced resolution image (of the same size)
    """
    assert len(np.shape(ar)) == len(block_size), 'Block size dimensions (%s) must match data dimensions (%s)' % (len(np.shape(block_size)), len(np.shape(ar)))
    if block_size == (0, 0) or block_size == (0, 0, 0):
        return ar
    orig_shape = np.shape(ar)
    reduced_ar = skimage.measure.block_reduce(ar, block_size, pool_func)
    if resize:
        new_ar = skimage.transform.resize(reduced_ar, orig_shape, **resize_kwargs)
        return new_ar
    return reduced_ar

def apply_blur_resolution(ar, sigma):
    """apply_blur_resolution(ar, sigma)

    Apply a gaussian blur on `ar` by standard deviation `sigma`

    Args:
        ar (np.ndarray): Data array to apply resolution to
        sigma (float): Gaussian filter standard deviation
    Returns:
        new_ar (np.ndarray): Reduced resolution image (of the same size)
    """
    return filters.gaussian(ar.astype(float), sigma)

