"""
statfunc_cupy.py

This implementation does not provide a periodic calculation.

Functions
---------
- process_lags
"""

from .statfunc_base import StatMetric

from typing import Optional
import numpy as np
import cupy as cp

def process_lags(
    field: np.ndarray,
    lags: np.ndarray,
    stat_metric: StatMetric,
    powers: Optional[np.ndarray]=None) -> np.ndarray:
    """process_lags(field, lags, stat_metric, powers)
    
    Args:
        field (np.ndarray): The array to compute the lag-statfunc of
        lags (np.ndarray): List of lags
        stat_metric (StatMetric): CORR, BIAS_CORR, or STRFN
        powers (tuple): What powers (of the StatMetric.STRFN) to calculate
    Returns:
        out (np.ndarray): Output array
    """

    if powers is None:
        powers = np.array([2,])
    num_powers = len(powers)
    powers_gpu = cp.asarray(powers, dtype=cp.int64)
    out_gpu = cp.zeros((len(lags), num_powers), dtype=cp.float64)
    field_gpu = cp.asarray(field, dtype=cp.float64)
    
    shape = field_gpu.shape
    num_dims = len(shape)
    total_elements = field_gpu.size

    for i, lag in enumerate(lags):
        _dims = len(lag)
        s1 = [slice(0,shape[d]) for d in range(num_dims)]
        s2 = [slice(0,shape[d]) for d in range(num_dims)]
        for d in range(_dims):
            s1[d] = slice(lag[d],shape[d])
            s2[d] = slice(0,shape[d]-lag[d])
        
        view1 = field_gpu[tuple(s1)]
        view2 = field_gpu[tuple(s2)]

        denom = view1.size if stat_metric in (StatMetric.CORR, StatMetric.STRFN) else total_elements

        if stat_metric in (StatMetric.CORR, StatMetric.BIAS_CORR):
            out_gpu[i, 0] = np.nansum(view1 * view2) / denom
        elif stat_metric == StatMetric.STRFN:
            diff = cp.abs(view1 - view2)
            diff_powered = diff[..., None]**powers_gpu
            total_diffs = cp.nansum(diff_powered, axis=tuple(range(diff_powered.ndim - 1)))
            out_gpu[i,:] = total_diffs / denom

    return cp.asnumpy(out_gpu)
