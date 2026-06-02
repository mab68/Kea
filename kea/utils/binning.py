"""
binning.py

Provides common methods for binning

Functions
---------
- ndim_func
- bin_data
- get_bins
- hypersphere_binshell
"""

from .geometry import volume_hypersphere

from typing import Optional, Callable

import numpy as np
from scipy.stats import binned_statistic

def ndim_func(
        wfunc: Callable,
        shape: tuple,
        func_args: Optional[tuple]=()):
    """ndim_func(wfunc, shape, func_args)\n

    Generates a N-Dimensional function by taking the outer product
        of the 1-dimensional `wfunc` for as many dimensions in `shape`
    
    Args:
        wfunc (func): Generation function in 1D
        shape (tuple): The shape of the n-dimensional space
        func_args (tuple): Arguments to pass into `wfunc`
    Returns:
        np.ndarray: N-dimensional function    
    """
    f = [wfunc(*func_args) for _ in shape]
    return np.prod(np.array(np.ix_(*f), dtype=object))

def bin_data(
        pos_array: np.ndarray,
        ar: np.ndarray,
        bin_func: Optional[Callable] = np.nanmean,
        cut_excess: Optional[bool] = False,
        nan_small: Optional[bool] = False,
        min_bin: Optional[float] = None,
        max_bin: Optional[float] = None,
        bin_loc: Optional[str] = 'center',
        norm_bin_size: Optional[bool] = False,
        log_space: Optional[bool] = False,
        num_bins: Optional[int] = None,
        ignore_nan: Optional[bool] = False,
        max_half_bin_width: Optional[float] = None):
    """bin_data(...)\n

    Bin `ar` with the domain `pos_array` using the `bin_func`.
    This reduces an ND array down to 1 dimension through the `bin_func`.

    Args:
        pos_array (np.ndarray): The position array to bin magnitudes of
        ar (np.ndarray): The ND array to bin down to 1D
        bin_func (func): The function used to bin `ar`
        cut_excess (bool): If true, remove binned lags greater than one of the basis directions
        nan_small (bool): If true, set the binned function to nan for when the number of points is small
        min_bin (float): Smallest bin value, if None, then automatically pick
        max_bin (float): Largest bin value.
            if None, then get the largest of `pos_array`
            if `basis`, then choose the largest in the basis
        bin_loc (str):
            center: Places the bin in the center of the bin range.
            true_center: Places the bins in the center of the available data,
                this fixes problems close to 0.
            left: Places the bins at the left bin-edge.
        norm_bin_size (bool): If true, divide the binned functions by the size of their respective bins
        log_space (bool): If true, use log-spacings
        num_bins (float): Number of bins
        ignore_nan (bool): Remove nan values
        max_half_bin_width (None/float): Default None. If set, this is the maximum (half) bin width allowed
    Returns:
        bins (np.array): The bins of the computed statistics
        ar1D (np.array): Mean statistic of the binning from ND to 1D on the bins
        width (np.array): Bin widths
    """
    pos_array = (pos_array.copy()).round(decimals=10)

    # Find the basis of the position array
    pos = np.where(pos_array == 0)
    if len(pos[0]) == 0:
        basis_index = 0
        narbasis = pos_array
        nn = pos_array
    else:
        basis_index = [pos[i][0] for i in range(len(pos))]
        basis_index[0] = Ellipsis
        narbasis = pos_array[tuple(basis_index)]
        nn = narbasis[pos[0][0]:]

    # Calculate the bin space
    # Assume that the array is evenly spaced (this is an assumption made with everything)
    if min_bin is None:
        # If no minimum bin specified, then automatically choose one
        min_bin = np.nanmin(nn)

    if max_bin is None:
        max_bin = np.nanmax(pos_array)
    if isinstance(max_bin, str) and max_bin == 'basis':
        max_bin = np.nanmax(narbasis)

    if num_bins is None:
        binsize = np.diff(nn)[0]
        nbins = int(np.round(max_bin/binsize) + 1)
    else:
        nbins = num_bins

    bins, bin_widths, be = get_bins(min_bin, max_bin, nbins, max_half_bin_width, log_space)

    # Set bad, the regions outside the binning domain because scipy includes them into the binnings
    car = ar.copy()
    car[pos_array < min_bin] = np.nan
    car[pos_array > max_bin] = np.nan

    # Compute the binnings
    ar1d, bin_edges, _ = binned_statistic(pos_array.ravel(), car.ravel(), bins=be, statistic=bin_func)

    if bin_loc == 'center':
        # Set the bins to the mid point of the bin edges
        bins1d = (bin_edges[1:] + bin_edges[:-1])/2.
    elif bin_loc == 'true_center':
        # Set the bins to the mid point of the actual `pos_array` data
        new_bins = []
        for i in range(len(bin_edges)-1):
            bmin, bmax = bin_edges[i].round(decimals=10), bin_edges[i+1].round(decimals=10)
            mask = np.where(np.logical_and(pos_array >= bmin, pos_array < bmax))
            if pos_array[mask].size == 0:
                new_bins.append((bmin + bmax)/2.)
            else:
                new_bins.append((np.nanmin(pos_array[mask]) + np.nanmax(pos_array[mask]))/2.)
        bins1d = np.array(new_bins)
    else:
        # Set the bins to the start of the bin edges
        bins1d = bin_edges[:-1]

    if max_half_bin_width is not None:
        # We should remove the bad bin edges (for the bins we don't want)
        mask = np.isin(bins1d, bins)
        ar1d = ar1d[mask]
        bins1d = bins1d[mask]

    # Divide the functions by their bin sizes
    # NOTE: The bin widths are actually half the bin widths
    if max_half_bin_width:
        # We have already calculate this
        width = 2.*bin_widths
    else:
        # Otherwise, calculate from binned_statistic bin_edges
        lower_bins, upper_bins = bin_edges[:-1], bin_edges[1:]
        width = (upper_bins - lower_bins)
        if max_half_bin_width is not None:
            width = np.minimum(width, 2.*max_half_bin_width)

    if norm_bin_size:
        ar1d = ar1d / width

    if nan_small:
        # Compute the counts, so we can ignore bad statistics
        cts, _, _ = binned_statistic(pos_array.ravel(), car.ravel(), bins=be, statistic='count')
        mask = cts <= 1
        ar1d[mask] = np.nan

    if cut_excess:
        # Cut off lags above the basis directions
        ar1d = ar1d[bins1d <= nn[-1]]
        width = width[bins1d <= nn[-1]]
        bins1d = bins1d[bins1d <= nn[-1]]

    if ignore_nan:
        # Remove nan values
        mask = np.isfinite(ar1d)
        ar1d = ar1d[mask]
        width = width[mask]
        bins1d = bins1d[mask]

    return bins1d, ar1d, width

def get_bins(
        min_bin: float,
        max_bin: float,
        nbins: int,
        max_half_bin_width: Optional[float]=None,
        log_space: Optional[bool]=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """get_bins(min_bin, max_bin, nbins, max_half_bin_width)\n

    Generates the bin centers, widths and edges for use with the binning functions

    Args:
        min_bin (float): Minimum bin number
        max_bin (float): Maximum bin number
        nbins (int): The number of bins
        max_half_bin_width (float): The maximum (half) width of the bins
        log_space (bool): If true, use log separations for the bins
    Returns:
        bins (np.ndarray): List of bin centers
        half_bin_width (np.ndarray): List of (half) bin widths for each bin center
        bin_edges (np.ndarray): Unique bin edges
    """
    if log_space:
        bin_range = np.exp(np.linspace(np.log(min_bin), np.log(max_bin), nbins+1))
    else:
        bin_range = np.linspace(min_bin, max_bin, nbins+1)

    ## NOTE: Inherited code converts to integers; there are probably problems
    ##        with doing this... and also not doing this...
    ## valid_bins = np.unique(bin_range.astype(int))
    valid_bins = np.unique(bin_range.round(decimals=4))

    lower_bins, upper_bins = valid_bins[:-1], valid_bins[1:]

    bin_width = (upper_bins - lower_bins) / 2.
    if max_half_bin_width is not None:
        bin_width = np.minimum(bin_width, max_half_bin_width)

    bins = upper_bins - bin_width
    # consequences of not using the above note
    bin_edges = np.unique(np.concatenate((bins-bin_width, bins+bin_width)).round(decimals=4))
    return bins, bin_width, bin_edges

def hypersphere_binshell(
        bins: np.ndarray,
        dimension: int | float,
        bin_widths: np.ndarray,
        grid_widths: tuple,
        centered: Optional[bool] = False) -> np.ndarray:
    """hypersphere_binshell(bins, dimension, bin_widths, grid_widths, centered)\n

    Calculates the number of grid elements within a hyperspherical bin shell for arbitrary dimension $D$.

    Args:
        bins (np.ndarray): Bin array (radius)
        dimension (int|float): Euclidean dimension
        bin_widths (np.ndarray): Bin width size
        grid_widths (tuple): Element widths for each dimension (grid size)
        centered (bool): If true, bins represent the center of the shell, otherwise the inner radius
    Returns:
        bin_shell (np.ndarray): Hypersphere bin-shell for each bin (radius) with given widths
    """
    if centered:
        r_inner = bins - bin_widths / 2.
        r_outer = bins + bin_widths / 2.
    else:
        r_inner = bins
        r_outer = bins + bin_widths
    
    shell_volume = volume_hypersphere(r_outer, dimension) - volume_hypersphere(r_inner, dimension)
    grid_volume_scale = np.prod(grid_widths)**dimension
    return shell_volume / grid_volume_scale
