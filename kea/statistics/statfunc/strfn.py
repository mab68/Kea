"""
strfn.py

Calculates the autocorrelation function for arbitrary dimension $D$.
"""

from .statfunc_numba import process_lags, StatMetric
from ...utils.geometry import validate_shapes, get_all_lagvecs

from typing import Optional
import numpy as np

def statfunc_strfn(
        field_a: np.ndarray,
        field_b: np.ndarray,
        p: int = 2) -> np.floating:
    assert np.shape(field_a) == np.shape(field_b), 'Provided fields have different shapes'
    return np.nanmean(np.abs(field_a - field_b)**p)

@validate_shapes('field')
def structure_function(
        field: np.ndarray,
        max_lag: Optional[int] = None,
        longitudinal: Optional[bool] = False,
        orders: Optional[tuple] = (2,)) -> tuple[np.ndarray, np.ndarray]:
    """structure_function(field, max_lag, longitudinal, orders)

    Args:
        field (np.ndarray): Array to calculate the (auto)-structure function
        max_lag (int): Maximum lag (as grid index) to go to
        longitudinal (bool): Whether to calculate along 1D
        orders (tuple): List of SF orders to calculate
    Returns:
        lags (np.ndarray): The array of lags
        sf (np.ndarray): Computed SFs. [Note, this might need to be reshaped]
    """
    # If looking for the longitudinal/transverse SF, then just iterate over "1D" lags
    D = field.ndim
    if longitudinal:
        D = 1

    # Generate the appropriate lags
    if max_lag is None:
        max_lag = np.min(np.shape(field))//2
    lagvec_shape = tuple([max_lag for _ in range(D)])
    lags = get_all_lagvecs(lagvec_shape)

    # Calculate SF
    sf = process_lags(field, lags, StatMetric.STRFN, np.array(orders))
    return lags, sf
