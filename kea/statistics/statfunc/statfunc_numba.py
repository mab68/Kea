"""
statfunc_numba.py

The implementation of the statfunc calculations using numba for increased processing speed.
This implementation does not provide a periodic calculation.
"""

from typing import Optional
from enum import IntEnum
import numpy as np
from numba import njit, prange

class StatMetric(IntEnum):
    CORR = 1
    BIAS_CORR = 2
    STRFN = 3

@njit
def _calc_stat(
    field: np.ndarray,
    shape: np.ndarray,
    s1_start: np.ndarray,
    s2_start: np.ndarray,
    overlap_shape: np.ndarray,
    stat_metric: StatMetric,
    powers: tuple) -> np.ndarray:
    """
    Computes the statistic for a given lag across arbitrary dimensions using flat index mapping.
    
    shape: The original shape of the field
    s1_start: Array of start indices for slice 1
    s2_start: Array of start indices for slice 2
    overlap_shape: The shape of the overlapping region
    """
    num_powers = len(powers)
    totals = np.zeros(num_powers, dtype=np.float64)

    # 1. Compute total number of elements in the overlapping bounding box
    total_elements = 1
    for dim_size in overlap_shape:
        total_elements *= dim_size
    if total_elements == 0:
        return totals

    ndim = len(shape)
    
    # 2. Pre-calculate strides for the original field to do fast index math
    # field.strides gives bytes; we want item counts for flat indexing
    field_strides = np.zeros(ndim, dtype=np.int64)
    current_stride = 1
    for d in range(ndim - 1, -1, -1):
        field_strides[d] = current_stride
        current_stride *= shape[d]

    # 3. A single flat loop handles ALL dimensions
    for flat_idx in range(total_elements):
        
        # Unravel flat_idx into multi-dimensional coordinates relative to the overlap box
        # and simultaneously calculate the absolute flat pointer for both slices
        idx_s1 = 0
        idx_s2 = 0
        remainder = flat_idx
        
        for d in range(ndim):
            stride_overlap = 1
            for k in range(d + 1, ndim):
                stride_overlap *= overlap_shape[k]
                
            coord_d = remainder // stride_overlap
            remainder %= stride_overlap
            
            # Map back to the absolute coordinate of each slice
            idx_s1 += (s1_start[d] + coord_d) * field_strides[d]
            idx_s2 += (s2_start[d] + coord_d) * field_strides[d]

        # Use flat indexing (.ravel() equivalent at pointer level)
        val1 = field.reshape(-1)[idx_s1]
        val2 = field.reshape(-1)[idx_s2]
        
        if stat_metric == StatMetric.CORR or stat_metric == StatMetric.BIAS_CORR:
            # Multiply the two values
            totals[0] += val1 * val2
        elif stat_metric == StatMetric.STRFN:
            # Calculate the absolute difference to the given powers
            diff = np.abs(val1 - val2)
            for idx in range(num_powers):
                p = powers[idx]
                totals[idx] += diff**p

    # Turn the summation into a mean (divide by the number of data-points)
    if stat_metric == StatMetric.CORR or stat_metric == StatMetric.STRFN:
        return totals / total_elements
    # Otherwise, we will divide by the total number as a BIASED estimator
    if stat_metric == StatMetric.BIAS_CORR:
        return totals / np.prod(shape)
    # Otherwise, don't do any normalization
    return totals

@njit(parallel=True)
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
    out = np.zeros((len(lags), num_powers), dtype=np.float64)
    shape = np.array(field.shape)
    N_total = np.prod(shape)
    ndim = len(shape)
    
    # Parallel loop over all provided lags
    for i in prange(len(lags)):
        l = lags[i]

        # Allocate small, fixed-size arrays on the stack (Numba optimizes this away)
        s1_start = np.zeros(ndim, dtype=np.int64)
        s2_start = np.zeros(ndim, dtype=np.int64)
        overlap_shape = np.zeros(ndim, dtype=np.int64)

        # Calculate bounding boxes for N-dimensions
        for d in range(ndim):
            lag_d = l[d]
            s1_start[d] = lag_d
            s2_start[d] = 0
            overlap_shape[d] = shape[d] - lag_d

        # Call our generalized flat-loop calculator
        out[i] = _calc_stat(field, shape, s1_start, s2_start, overlap_shape, stat_metric, powers)

    return out
