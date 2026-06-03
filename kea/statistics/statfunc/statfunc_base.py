"""
statfunc_base.py
"""

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

class CalculationMode(IntEnum):
    """
    Set the calculate mode for statfuncs:
    - USE_CPU (do computations only on the CPU)
    - USE_GPU (use cupy to do computations on the GPU)
    - DISTRIBUTE_LAGS (distribute lags across computers/cores via MPI)

    Set multiple calculation modes using a pair-tuple e.g., (DISTRIBUTE_LAGS, USE_CPU)
    """
    USE_CPU = 0
    USE_GPU = 1
    DISTRIBUTE_LAGS = 2

CALCULATION_MODE_FLAGS = (CalculationMode.USE_CPU,)
def set_calculation_mode(
        use_cpu: Optional[bool]=True,
        use_gpu: Optional[bool]=False,
        distribute_lags: Optional[bool]=False) -> tuple[CalculationMode] | tuple[CalculationMode,CalculationMode]:
    """set_calculation_mode(use_cpu, use_gpu, distribute_lags)\n

    Set the calculate mode for statfuncs:
    - USE_CPU (do computations only on the CPU)
    - USE_GPU (use cupy to do computations on the GPU)
    - DISTRIBUTE_LAGS (distribute lags across computers/cores via MPI)

    Set multiple calculation modes using a pair-tuple e.g., (DISTRIBUTE_LAGS, USE_CPU)

    Args:
        use_cpu (bool): If true, use the CPU for calculations
        use_gpu (bool): If true, use the GPU for calculations
        distribute_lags (bool): If true, distribute the lag-array calculations across available processes
    Returns:
        tuple: Current flags
    """
    if use_cpu and use_gpu:
        raise ValueError('Cannot do both GPU and CPU computations')
    if (not use_cpu and distribute_lags) or (not use_gpu and distribute_lags):
        raise ValueError('Need to use CPU or GPU')
    if (not use_cpu) and (not use_gpu) and (not distribute_lags):
        raise ValueError('No flags set')
    flags = []
    if use_cpu:
        flags.append(CalculationMode.USE_CPU)
    if use_gpu:
        flags.append(CalculationMode.USE_GPU)
    if distribute_lags:
        flags.append(CalculationMode.DISTRIBUTE_LAGS)
    CALCULATION_MODE_FLAGS = tuple(flags)
    print('Set statistic function calculation flags: {CALCULATION_MODE_FLAGS}')
    return CALCULATION_MODE_FLAGS

def _process_gpu(
        field: np.ndarray,
        lags: np.ndarray,
        stat_metric: StatMetric,
        powers: Optional[np.ndarray]=None):
    """process_lags(field, lags, stat_metric, powers)\n

    Process the lags using the GPU (cupy)
    
    Args:
        field (np.ndarray):
        lags (np.ndarray): List of lag vectors
        stat_metric (StatMetric): The statistic to calculate
        powers (None|np.ndarray): If calculating the structure function, what orders to calculate
    Returns:
        (np.ndarray, np.ndarray): Lags and calculated statistic function at those lags
    """
    import cupy as cp
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

def _process_cpu(
        field: np.ndarray,
        lags: np.ndarray,
        stat_metric: StatMetric,
        powers: Optional[np.ndarray]=None):
    """process_lags(field, lags, stat_metric, powers)\n

    Process the lags using the CPU (native numpy)
    
    Args:
        field (np.ndarray):
        lags (np.ndarray): List of lag vectors
        stat_metric (StatMetric): The statistic to calculate
        powers (None|np.ndarray): If calculating the structure function, what orders to calculate
    Returns:
        (np.ndarray, np.ndarray): Lags and calculated statistic function at those lags
    """
    if powers is None:
        powers = np.array([2,])
    num_powers = len(powers)
    out = np.zeros((len(lags), num_powers))
    
    shape = field.shape
    num_dims = len(shape)
    total_elements = field.size

    for i, lag in enumerate(lags):
        _dims = len(lag)
        s1 = [slice(0,shape[d]) for d in range(num_dims)]
        s2 = [slice(0,shape[d]) for d in range(num_dims)]
        for d in range(_dims):
            s1[d] = slice(lag[d],shape[d])
            s2[d] = slice(0,shape[d]-lag[d])
        
        view1 = field[tuple(s1)]
        view2 = field[tuple(s2)]

        denom = view1.size if stat_metric in (StatMetric.CORR, StatMetric.STRFN) else total_elements

        if stat_metric in (StatMetric.CORR, StatMetric.BIAS_CORR):
            out[i, 0] = np.nansum(view1 * view2) / denom
        elif stat_metric == StatMetric.STRFN:
            diff = np.abs(view1 - view2)
            diff_powered = diff[..., None]**powers
            total_diffs = np.nansum(diff_powered, axis=tuple(range(diff_powered.ndim - 1)))
            out[i,:] = total_diffs / denom

    return out

def process_lags(
        field: np.ndarray,
        lags: np.ndarray,
        stat_metric: StatMetric,
        powers: Optional[np.ndarray]=None) -> np.ndarray:
    """process_lags(field, lags, stat_metric, powers)\n

    Process the lags using the set calculation modes
    
    Args:
        field (np.ndarray):
        lags (np.ndarray): List of lag vectors
        stat_metric (StatMetric): The statistic to calculate
        powers (None|np.ndarray): If calculating the structure function, what orders to calculate
    Returns:
        (np.ndarray, np.ndarray): Lags and calculated statistic function at those lags
    """
    do_distribute = CalculationMode.DISTRIBUTE_LAGS in CALCULATION_MODE_FLAGS
    do_cpu = CalculationMode.USE_CPU in CALCULATION_MODE_FLAGS
    do_gpu = CalculationMode.USE_GPU in CALCULATION_MODE_FLAGS

    if do_distribute:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except:
            raise ValueError('Cannot distribute lags -- `mpi4py` not configured/installed.')
        rank = comm.Get_rank()
        size = comm.Get_size()
        lags = lags[rank::size]

    if do_gpu:
        process_function = _process_gpu
        try:
            import cupy as cp
        except:
            raise ValueError('Cannot do GPU calculations -- `cupy` not configured/installed.')
    else:
        process_function = _process_cpu

    statfunc = process_function(field, lags, stat_metric, powers)
    return lags, statfunc

