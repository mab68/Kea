"""
Provides various utility routines to fit slopes etc.
"""

from kea.utils import get_fitting_mode, binning

from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


def interpolate_1d_function(x: np.ndarray,
                            y: np.ndarray,
                            x_interp: Optional[np.ndarray]=None,
                            x_log: Optional[bool]=False,
                            y_log: Optional[bool]=False) -> tuple[np.ndarray, np.ndarray]:
    """fit_1d_function(x, y, x_interp)\n

    Fits/interpolates a 1D function onto `x_interp`.

    Args:
        x (np.ndarray): 1D coordinate grid
        y (np.ndarray): 1D function
        x_interp (np.ndarray): 1D coordinate grid to interpolate to

    Returns:
        np.ndarray: Estimate of 1D function
    """
    do_fd, do_gpr = get_fitting_mode()
    if x_interp is None:
        mags = int(np.ceil(np.log10(np.nanmax(x)/np.nanmin(x[x>0]))))
        if x_log:
            x_interp = 10**(np.linspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)), 10*mags))
        else:
            x_interp = np.linspace(np.nanmin(x), np.nanmax(x), 10*mags)

    if x_log:
        x = np.log10(x)
        x_interp = np.log10(x_interp)
    if y_log:
        y = np.log10(y)

    if do_gpr:
        # Do Gaussian process regression for interpolating
        # Import required libraries
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        # Convert to correct shapes for `sklearn`
        x_interp = x_interp.reshape(-1, 1)
        x_train = x.reshape(-1, 1)
        y_train = y.reshape(-1, 1)
        # ConstantKernel: Scales the amplitude of the function
        # RBF: Handles the smooth, underlying non-linear curve
        # WhiteKernel: absorbs the high-frequency noise
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) \
                + WhiteKernel(noise_level=1.)
        # Run optimizer
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gp.fit(x_train, y_train)
        # Evaluate mean on the fine (interpolation) grid
        y_result = gp.predict(x_interp, return_std=False)
        y_result = y_result.flatten()
        x_interp = x_interp.flatten()
    else:
        # Do simple linear interpolation
        f_int = interp1d(x, y, kind='linear', fill_value='extrapolate')
        y_result = f_int(x_interp)

    if x_log:
        x_result = 10**x_interp
    else:
        x_result = x_interp
    if y_log:
        y_result = 10**y_result
    return x_result, y_result

def get_derivative(x: np.ndarray,
                   y: np.ndarray,
                   x_interp: Optional[np.ndarray]=None,
                   x_log: Optional[bool]=False,
                   y_log: Optional[bool]=False) -> tuple[np.ndarray, np.ndarray]:
    """get_derivative(x, y)\n
    
    Calculates the derivative using Gaussian process regression.
    
    Args:
        x (np.ndarray): 1D coordinate grid
        y (np.ndarray): 1D function
        x_interp (np.ndarray): 1D coordinate grid to interpolate to

    Returns:
        (np.ndarray, np.narray): Estimate of the derivative
    """
    _, do_gpr = get_fitting_mode()
    if do_gpr:
        # If set to `do_gpr`, then we will always interpolate using Gaussian process regression
        # Fit Gaussian process
        x_interp, y_interp = interpolate_1d_function(x, y, x_interp, x_log, y_log)
        # Calculate derivative
        return x_interp, np.gradient(y_interp, x_interp)
    else:
        # Otherwise, just do basic finite differencing
        if x_interp is None:
            # No interpolation given, so just finite difference on the data available
            x_interp = x
            y_interp = y
        else:
            # Otherwise, linearly interpolate
            x_interp, y_interp = interpolate_1d_function(x, y, x_interp, x_log, y_log)
        # Calculate derivative
        return x_interp, np.gradient(y_interp, x_interp)

def get_local_powerlaw(x: np.ndarray,
                       y: np.ndarray,
                       x_interp: Optional[np.ndarray]=None,
                       x_log: Optional[bool]=False) -> tuple[np.ndarray, np.ndarray]:
    """get_powerlaw(x, y)\n
    
    Calculates the local power law slope.

    Args:
        x (np.ndarray): 1D Coordinate grid
        y (np.ndarray): 1D function
        x_interp (np.ndarray): 1D coordinate grid to interpolate to
        x_log (bool):

    Returns:
        (np.ndarray, np.narray): Estimate of the local power law slope at each x
    """
    if x_interp is not None:
        x_interp = np.log(x_interp)
    x_log, dy = get_derivative(np.log(x), np.log(y), x_interp=x_interp)
    return np.exp(x_log), dy
