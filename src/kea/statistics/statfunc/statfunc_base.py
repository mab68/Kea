"""
Generic function to calculate structure function or correlation function (all in real-space).
"""

from kea.utils import get_calculation_mode

from typing import Optional
from enum import IntEnum

import numpy as np

class StatMetric(IntEnum):
    """
    The statistic function to calculate:
    - CORR: Typical correlation function
    - BIAS_CORR: Biased correlation function
    - STRFN: Structure function of arbitrary order
    """
    CORR = 1
    BIAS_CORR = 2
    STRFN = 3

def _calc_stat(
        field: np.ndarray,
        lags: np.ndarray,
        stat_metric: StatMetric,
        powers: Optional[np.ndarray] = None,
        use_gpu: Optional[bool] = False) -> np.ndarray:
    """process_lags(field, lags, stat_metric, powers)\n

    Process the lags using the CPU or GPU
    
    Args:
        field (np.ndarray): Field to compute statistic for
        lags (np.ndarray): List of lag vectors
        stat_metric (StatMetric): The statistic to calculate
        powers (None|np.ndarray): If calculating the structure function, what orders to calculate
        use_gpu (bool): If true, use the GPU via `cupy`

    Returns:
        (np.ndarray, np.ndarray): Lags and calculated statistic function at those lags
    """

    if powers is None:
        powers = np.array([2,])

    if use_gpu:
        # Using GPU so use cupy
        try:
            import cupy as compute_lib
        except:
            raise ValueError('Cannot use the GPU -- `cupy` not installed')

        powers_gpu = compute_lib.asarray(powers, dtype=compute_lib.int64)
        field_gpu = compute_lib.asarray(field, dtype=compute_lib.float64)
    else:
        # If using the CPU, use numpy instead of cupy
        compute_lib = np
        powers_gpu = powers
        field_gpu = field
    out_gpu = compute_lib.zeros((len(lags), len(powers)), dtype=compute_lib.float64)
    
    shape = field_gpu.shape
    num_dims = len(shape)
    total_elements = np.prod(shape)

    for i, lag in enumerate(lags):
        _dims = len(lag)
        s1 = [slice(0,shape[d]) for d in range(num_dims)]
        s2 = [slice(0,shape[d]) for d in range(num_dims)]
        for d in range(_dims):
            s1[d] = slice(lag[d],shape[d])
            s2[d] = slice(0,shape[d]-lag[d])
        
        view1 = field_gpu[tuple(s1)]
        view2 = field_gpu[tuple(s2)]

        # denom = view1.size if stat_metric in (StatMetric.CORR, StatMetric.STRFN) else total_elements

        if stat_metric == StatMetric.CORR:
            out_gpu[i,0] = compute_lib.nanmean(view1 * view2)
        elif stat_metric == StatMetric.BIAS_CORR:
            out_gpu[i,0] = compute_lib.nansum(view1 * view2) / total_elements
        elif stat_metric == StatMetric.STRFN:
            diff = compute_lib.abs(view1 - view2)
            diff_powered = diff[..., None]**powers_gpu
            out_gpu[i,:] = compute_lib.nanmean(diff_powered, axis=tuple(range(diff_powered.ndim - 1)))

        # if stat_metric in (StatMetric.CORR, StatMetric.BIAS_CORR):
        #     out_gpu[i, 0] = compute_lib.nansum(view1 * view2)
        # elif stat_metric == StatMetric.STRFN:
        #     diff = compute_lib.abs(view1 - view2)
        #     diff_powered = diff[..., None]**powers_gpu
        #     total_diffs = compute_lib.nansum(diff_powered, axis=tuple(range(diff_powered.ndim - 1)))
        #     out_gpu[i,:] = total_diffs / denom

    if use_gpu:
        out_gpu = compute_lib.asnumpy(out_gpu)
        # Clear GPU memory references
        powers_gpu, field_gpu = None, None
        view1, view2 = None, None
        if stat_metric == StatMetric.STRFN:
            diff, diff_powered = None, None
        compute_lib.get_default_memory_pool().free_all_blocks()
        compute_lib.get_default_pinned_memory_pool().free_all_blocks()
    return out_gpu

def process_lags(
        field: np.ndarray,
        lags: np.ndarray,
        stat_metric: StatMetric,
        powers: Optional[np.ndarray]=None) -> np.ndarray:
    """process_lags(field, lags, stat_metric, powers)\n

    Process the lags using the set calculation modes
    
    Args:
        field (np.ndarray): Field to compute statistic for
        lags (np.ndarray): List of lag vectors
        stat_metric (StatMetric): The statistic to calculate
        powers (None|np.ndarray): If calculating the structure function, what orders to calculate

    Returns:
        (np.ndarray, np.ndarray): Lags and calculated statistic function at those lags
    """
    use_cpu, use_gpu, do_distribute = get_calculation_mode()

    if do_distribute:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except:
            raise ValueError('Cannot distribute lags -- `mpi4py` not configured/installed.')
        rank = comm.Get_rank()
        size = comm.Get_size()
        lags = lags[rank::size]

    statfunc = _calc_stat(field, lags, stat_metric, powers, use_gpu)
    return lags, statfunc
