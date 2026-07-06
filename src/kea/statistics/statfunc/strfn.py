"""
Calculates the autocorrelation function for arbitrary dimension $D$.
"""

from kea.statistics.statfunc.statfunc_base import StatMetric, process_lags
from kea.utils.geometry import validate_shapes, get_lagvecs

from typing import Optional
import numpy as np

@validate_shapes('field')
def structure_function(
        field: np.ndarray,
        discrete_lags: Optional[np.ndarray] = None,
        max_lag: Optional[int] = None,
        longitudinal: Optional[bool] = False,
        orders: Optional[tuple] = (2,),
        n_pt: Optional[int] = 2) -> tuple[np.ndarray, np.ndarray]:
    """structure_function(field, discrete_lags, max_lag, longitudinal, orders)\n

    Args:
        field (np.ndarray): Array to calculate the (auto)-structure function
        discrete_lags (np.ndarray): The lags (indices) to evaluate the ACF at
        max_lag (int): Maximum lag (as grid index) to go to
        longitudinal (bool): Whether to calculate along 1D
        orders (tuple): List of SF orders to calculate
        n_pt (int): Number of points in the SF stencil, ONLY 2,3,4,5 implemented.

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
        discrete_lags = get_lagvecs(lagvec_shape)

    STRFN_CONFIG = {
        2: StatMetric.STRFN,
        3: StatMetric.STRFN_3PT,
        4: StatMetric.STRFN_4PT,
        5: StatMetric.STRFN_5PT,
    }
    if n_pt is None:
        n_pt = 2
    if n_pt not in STRFN_CONFIG:
        raise ValueError('%s-pt SF stencil not implemented' % str(n_pt))
    stat_metric = STRFN_CONFIG[n_pt]

    # Calculate SF
    discrete_lags, sf = process_lags(field, discrete_lags, stat_metric, np.array(orders))
    return discrete_lags, sf
