
"""
statistics_base.py

Provides common methods for the spectra and statfuncs

Functions
---------
ndim_func\n
apply_ndim_window\n
bin_data\n
"""

import numpy as np

from scipy.stats import binned_statistic


def ndim_func(wfunc, shape, func_args=()):
    """ndim_func(wfunc, shape, func_args)

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

def apply_ndim_window(ar, wfunc, p=1., norm=False):
    """apply_ndim_window(ar, wfunc, p=1.)
    
    Applies an n-dimensional window function `wfunc` onto data `ar` of length `p*len(ar)`

    Args:
        ar (np.ndarray):
        wfunc (func): The window generation function in 1D
        p (float): The size of the window function as percentage of the length of the data
    Returns:
        data (np.ndarray): Windowed data
    """
    assert np.prod(ar.shape) == ar.shape[0]**(ar.ndim)
    if callable(wfunc):
        wf = ndim_func(wfunc, ar.shape, func_args=(int(ar.shape[0] * p), True,))
    else:
        wf = wfunc
    if p != 1.0:
        #pad_wf = np.pad(wf, int((1. - p) * ar.shape[0] / 2.))
        pad_wf = np.zeros_like(ar)
        cpos = [int(n/2) for n in ar.shape]
        indices = tuple([slice(int(cpos[i] - wf.shape[i]/2), int(cpos[i] + wf.shape[i]/2)) for i in range(ar.ndim)])
        pad_wf[indices] = wf
    else:
        pad_wf = wf
    data = ar.copy() * pad_wf
    if norm:
        data /= np.var(pad_wf)
    return data

def bin_data(nar, ar, bin_func=np.nanmean,
             cut_excess=False, nan_small=False, min_bin=None,
             max_bin=None, bin_loc='center', norm_bin_size=False, log_space=False,
             num_bins=None, ignore_nan=False, max_half_bin_width=None):
    """bin_data(...)

    Bin `ar` with the domain `nar` using the `bin_func`.
    This reduces an ND array down to 1 dimension through the `bin_func`.

    Args:
        nar (np.ndarray): The position array to bin magnitudes of
        ar (np.ndarray): The ND array to bin down to 1D
        bin_func (func): The function used to bin `ar`
        cut_excess (bool): If true, remove binned lags greater than one of the basis directions
        nan_small (bool): If true, set the binned function to nan for when the number of points is small
        min_bin (float): Smallest bin value, if None, then automatically pick
        max_bin (float): Largest bin value.
            if None, then get the largest of `nar`
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
    nar = (nar.copy()).round(decimals=10)

    # Find the basis of the position array
    pos = np.where(nar == 0)
    if len(pos[0]) == 0:
        basis_index = 0
        narbasis = nar
        nn = nar
    else:
        basis_index = [pos[i][0] for i in range(len(pos))]
        basis_index[0] = Ellipsis
        narbasis = nar[tuple(basis_index)]
        nn = narbasis[pos[0][0]:]

    # Calculate the bin space
    # Assume that the array is evenly spaced (this is an assumption made with everything)
    if min_bin is None:
        # If no minimum bin specified, then automatically choose one
        min_bin = np.nanmin(nn)

    if max_bin is None:
        max_bin = np.nanmax(nar)
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
    car[nar < min_bin] = np.nan
    car[nar > max_bin] = np.nan

    # Compute the binnings
    ar1d, bin_edges, _ = binned_statistic(nar.ravel(), car.ravel(), bins=be, statistic=bin_func)

    if bin_loc == 'center':
        # Set the bins to the mid point of the bin edges
        bins1d = (bin_edges[1:] + bin_edges[:-1])/2.
    elif bin_loc == 'true_center':
        # Set the bins to the mid point of the actual `nar` data
        new_bins = []
        for i in range(len(bin_edges)-1):
            bmin, bmax = bin_edges[i].round(decimals=10), bin_edges[i+1].round(decimals=10)
            mask = np.where(np.logical_and(nar >= bmin, nar < bmax))
            if nar[mask].size == 0:
                new_bins.append((bmin + bmax)/2.)
            else:
                new_bins.append((np.nanmin(nar[mask]) + np.nanmax(nar[mask]))/2.)
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
        cts, _, _ = binned_statistic(nar.ravel(), car.ravel(), bins=be, statistic='count')
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

def get_bins(min_bin, max_bin, nbins, max_half_bin_width=None, log_space=False):
    """get_bins(min_bin, max_bin, nbins, max_half_bin_width)

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
    valid_bins = np.unique(bin_range.round(decimals=10))

    lower_bins, upper_bins = valid_bins[:-1], valid_bins[1:]

    bin_width = (upper_bins - lower_bins) / 2.
    if max_half_bin_width is not None:
        bin_width = np.minimum(bin_width, max_half_bin_width)

    bins = upper_bins - bin_width
    # consequences of not using the above note
    bin_edges = np.unique(np.concatenate((bins-bin_width, bins+bin_width)).round(decimals=10))
    return bins, bin_width, bin_edges

def kern(bins, D, db, dx):
    """kern(bins, D, db, dx)

    Calculates the number of elements within a bin of width db

    Args:
        bins (np.ndarray): Bin array
        D (int): Euclidean dimension
        db (np.ndarray): Bin width size
        dx (tuple,list): Element widths for each dimension, grid size
    Returns:
        np.ndarray: The number of elements within a bin of width db
    """
    if D == 1:
        return 2.
    elif D == 2:
        return (2. * np.pi * bins * db + np.pi * db**2) / np.prod(dx)**2
    elif D == 3:
        return (4./3.) * np.pi * (3. * bins * db**2 + 3. * bins**2 * db + db**3) / np.prod(dx)**3
    else:
        raise ValueError('Not implemented for dimension %s' % D)

def kern_center(bins, D, db, dx):
    """kern_center(bins, D, db, dx)

    Calculates the number of elements within a bin of width db
        when `bins` indicates the center of the bin, and we
        go from bin-db/2 to bin+db/2.

    Args:
        bins (np.ndarray): Bin array
        D (int): Euclidean dimension
        db (np.ndarray): Bin width size
        dx (tuple,list): Element widths for each dimension, grid size
    Returns:
        np.ndarray: The number of elements within a bin of width db
    """
    if D == 1:
        return 2.
    elif D == 2:
        return (2. * np.pi * bins * db) / np.prod(dx)**2
    elif D == 3:
        return (4./3.) * np.pi * (3. * bins**2 * db + db**3 / 4.) / np.prod(dx)**3
    else:
        raise ValueError('Not implemented for dimension %s' % D)

def nanstderr(x):
    ## NOTE: half is because the power spectrum is symmetric
    #raise NotImplementedError('The power spectrum is symmetric, so there would be half as many independent wavenumbers.')
    return np.nanstd(x)#/np.sqrt(0.5*x.size)
