
"""
moments.py

Calculates the moments of an array.

Functions
---------
mean\n
var\n
make_moment\n
"""

import numpy as np


def mean(ar, shape=None, lenn=None, axis=None, take_abs=True):
    """mean(ar, axis)

    Calculates the mean of `ar` along axis `axis`

    Args:
        ar (np.ndarray): The array to calculate the moment of (assuming mean 0)
        shape (tuple): Sampled number of points
        lenn (tuple): Box size
        axis (tuple): Either tuple indicating several axis to calculate the moment over.
            Or a single axis, or None indicating summing over every axis.
    Returns:
        mean (np.ndarray): 1st order moment, i.e. the mean
    """
    return make_moment(ar, 1, shape=shape, lenn=lenn, axis=axis, take_abs=take_abs)

def var(ar, shape=None, lenn=None, axis=None, take_abs=True):
    """var(ar, axis)

    Calculates the variance `ar` along axis `axis`

    `$V(x) = <u(x)^2>$`

    Args:
        ar (np.ndarray): The array to calculate the moment of (assuming mean 0)
        shape (tuple): Sampled number of points
        lenn (tuple): Box size
        axis (tuple): Either tuple indicating several axis to calculate the moment over.
            Or a single axis, or None indicating summing over every axis.
    Returns:
        mean (np.ndarray): 2st order moment, i.e. the variance
    """
    return make_moment(ar, 2, shape=shape, lenn=lenn, axis=axis, take_abs=take_abs)

def make_moment(ar, moment, shape=None, lenn=None, axis=None, take_abs=True):
    """make_moment(ar, moment, axis)

    Calculates the variance `ar` along axis `axis` with moment order/power `p`

    `$V(x) = <|u(x)|^p> = \int |u(x)|^p dx = dx \sum |u(x)|^p = L/N \sum |u(x)|^p$`

    Args:
        ar (np.ndarray): The array to calculate the moment of
        moment (int): The moment. What power to take of `ar`
        shape (tuple): Sampled number of points
        lenn (tuple): Box size
        axis (tuple): Either tuple indicating several axis to calculate the moment over.
            Or a single axis, or None indicating summing over every axis.
    Returns:
        mean (np.ndarray): nth order moment
    """
    if shape is None:
        shape = np.shape(ar)
    if lenn is None:
        lenn = [2. * np.pi for _ in range(len(shape))]

    if hasattr(shape, '__iter__'):
        dx = [lenn[i]/N for i, N in enumerate(shape)]
        if axis is not None:
            dx = [dx[i] for i in axis]
        dx = np.prod(dx)
    else:
        dx = np.prod(lenn) / shape

    if take_abs:
        ar = np.abs(ar)

    moment = np.nansum(ar**moment * dx, axis=axis)
    moment = moment / np.nansum(np.ones_like(ar) * dx, axis=axis)
    return moment
