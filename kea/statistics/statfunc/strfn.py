"""
Calculates the autocorrelation function for arbitrary dimension $D$.
"""

from kea.statistics.statfunc.statfunc_base import StatMetric, process_lags
from kea.utils.geometry import validate_shapes, get_all_lagvecs

from typing import Optional
import numpy as np

@validate_shapes('field')
def structure_function(
        field: np.ndarray,
        discrete_lags: Optional[np.ndarray] = None,
        max_lag: Optional[int] = None,
        longitudinal: Optional[bool] = False,
        orders: Optional[tuple] = (2,)) -> tuple[np.ndarray, np.ndarray]:
    """structure_function(field, max_lag, longitudinal, orders)\n

    Args:
        field (np.ndarray): Array to calculate the (auto)-structure function
        lags (np.ndarray): The lags (indices) to evaluate the ACF at
        max_lag (int): Maximum lag (as grid index) to go to
        longitudinal (bool): Whether to calculate along 1D
        orders (tuple): List of SF orders to calculate

    Returns:
        (np.ndarray, np.ndarray): The array of (discrete) lags and computed SFs
    """
    # If looking for the longitudinal/transverse SF, then just iterate over "1D" lags
    D = field.ndim
    if longitudinal:
        D = 1

    # Generate the appropriate lags
    if discrete_lags is None:
        if max_lag is None:
            max_lag = np.min(np.shape(field))//2
        lagvec_shape = (max_lag,)*D
        discrete_lags = get_all_lagvecs(lagvec_shape)

    # Calculate SF
    discrete_lags, sf = process_lags(field, discrete_lags, StatMetric.STRFN, np.array(orders))
    return discrete_lags, sf
