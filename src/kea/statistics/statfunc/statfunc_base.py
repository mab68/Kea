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
    CORR = 0
    BIAS_CORR = 1
    STRFN = 2
    STRFN_3PT = 3
    STRFN_4PT = 4
    STRFN_5PT = 5

def _calc_stat(
        field_a: np.ndarray,
        lags: np.ndarray,
        stat_metric: StatMetric,
        field_b: Optional[np.ndarray] = None,
        powers: Optional[np.ndarray] = None,
        use_gpu: Optional[bool] = False) -> np.ndarray:
    """process_lags(field, lags, stat_metric, powers)\n

    Process the lags using the CPU or GPU
    
    Args:
        field_a (np.ndarray): Field to compute statistic for
        lags (np.ndarray): List of lag vectors
        stat_metric (StatMetric): The statistic to calculate
        field_b (np.ndarray): Field to compute cross-statistic for
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
        field_gpu = compute_lib.asarray(field_a, dtype=compute_lib.float64)
        if field_b is not None:
            field_bgpu = compute_lib.asarray(field_b, dtype=compute_lib.float64)
    else:
        # If using the CPU, use numpy instead of cupy
        compute_lib = np
        powers_gpu = powers
        field_gpu = field_a
        field_bgpu = field_b

    out_gpu = compute_lib.zeros((len(lags), len(powers)), dtype=compute_lib.float64)
    
    shape = field_gpu.shape
    num_dims = len(shape)
    total_elements = np.prod(shape)

    # Maps stat_metric -> (num_points, coefficients, scale)
    STRFN_CONFIGS = {
        StatMetric.STRFN: (2, [1.0, -1.0], 1.0),
        StatMetric.STRFN_3PT: (3, [1.0, -2.0, 1.0], 1.0 / 3.0),
        StatMetric.STRFN_4PT: (4, [1.0, -3.0, 3.0, -1.0], 1.0 / 10.0),
        StatMetric.STRFN_5PT: (5, [1.0, -4.0, 6.0, -4.0, 1.0], 1.0 / 35.0),
    }
    # Correlation function follows the 2pt strfn
    num_pts = STRFN_CONFIGS[stat_metric][0] if stat_metric in STRFN_CONFIGS else 2

    for i, lag in enumerate(lags):
        _dims = len(lag)

        # Check if the npt SF creates views outside the typical bounds
        # if so, we ignore these lags
        skip_lag = False
        for d in range(_dims):
            if (num_pts - 1) * lag[d] >= shape[d]:
                skip_lag = True
                break
        if skip_lag:
            out_gpu[i,:] = compute_lib.nan
            continue

        # Dynamically create slices into the array based on lag vector
        views_a = []
        if field_b is not None:
            views_b = []
        for p in range(num_pts):
            slices = []
            for d in range(num_dims):
                if d < _dims:
                    start = (num_pts - 1 - p) * lag[d]
                    end = shape[d] - p * lag[d]
                    slices.append(slice(start,end))
                else:
                    slices.append(slice(0,shape[d]))
            views_a.append(field_gpu[tuple(slices)])
            if field_b is not None:
                views_b.append(field_bgpu[tuple(slices)])

        if stat_metric == StatMetric.CORR:
            if field_b is not None:
                out_gpu[i,0] = compute_lib.nanmean((views_a[0]*views_b[1] + views_a[1]*views_b[0])/2.)
            else:
                out_gpu[i,0] = compute_lib.nanmean(views_a[0]*views_a[1])
        elif stat_metric == StatMetric.BIAS_CORR:
            if field_b is not None:
                out_gpu[i,0] = compute_lib.nansum((views_a[0]*views_b[1] + views_a[1]*views_b[0])/2.) / total_elements
            else:
                out_gpu[i,0] = compute_lib.nansum(views_a[0]*views_a[1]) / total_elements
        elif stat_metric in STRFN_CONFIGS:
            _, coeffs, scale = STRFN_CONFIGS[stat_metric]
            diff = sum(c*v for c,v in zip(coeffs, views_a))
            diff_abs = compute_lib.abs(diff)
            diff_powered = scale * (diff_abs[...,None] ** powers_gpu)
            out_gpu[i,:] = compute_lib.nanmean(diff_powered, axis=tuple(range(diff_powered.ndim - 1)))

    if use_gpu:
        out_gpu = compute_lib.asnumpy(out_gpu)
        # Clear GPU memory references
        powers_gpu, field_gpu, field_bgpu = None, None, None
        view1, view2 = None, None
        if stat_metric == StatMetric.STRFN:
            diff, diff_powered = None, None
        compute_lib.get_default_memory_pool().free_all_blocks()
        compute_lib.get_default_pinned_memory_pool().free_all_blocks()
    return out_gpu

def process_lags(
        field_a: np.ndarray,
        lags: np.ndarray,
        stat_metric: StatMetric,
        field_b: Optional[np.ndarray]=None,
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

    statfunc = _calc_stat(field_a, lags, stat_metric, field_b=field_b, powers=powers, use_gpu=use_gpu)
    return lags, statfunc
