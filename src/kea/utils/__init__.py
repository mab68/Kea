
from kea.utils import compute_config
from .compute_config import CalculationMode, FittingMode

from typing import Optional

import numpy as np

TWOPI = 2.*np.pi

def set_calculation_mode(
        use_cpu: Optional[bool]=False,
        use_gpu: Optional[bool]=False,
        do_distribute: Optional[bool]=False) -> tuple[CalculationMode] | tuple[CalculationMode,CalculationMode]:
    """set_calculation_mode(use_cpu, use_gpu, do_distribute)\n

    Set the calculate mode for statfuncs:
    - USE_CPU (do computations only on the CPU)
    - USE_GPU (use cupy to do computations on the GPU)
    - DO_DISTRIBUTE (distribute lags across computers/cores via MPI)

    Set multiple calculation modes using a pair-tuple e.g., (DO_DISTRIBUTE, USE_CPU)

    Args:
        use_cpu (bool): If true, use the CPU for calculations
        use_gpu (bool): If true, use the GPU for calculations
        do_distribute (bool): If true, distribute the lag-array calculations across available processes

    Returns:
        tuple: Current flags
    """
    if use_cpu and use_gpu:
        raise ValueError('Cannot do both GPU and CPU computations')
    if (not use_cpu) and (not use_gpu):
        raise ValueError('Need to use CPU or GPU')
    if (not use_cpu) and (not use_gpu) and (not do_distribute):
        raise ValueError('No flags set')
    flags = []
    if use_cpu:
        flags.append(CalculationMode.USE_CPU)
    if use_gpu:
        flags.append(CalculationMode.USE_GPU)
    if do_distribute:
        flags.append(CalculationMode.DO_DISTRIBUTE)
    compute_config.CALCULATION_MODE_FLAGS = tuple(flags)
    print('Set statistic function calculation flags:', tuple(flags))
    return compute_config.CALCULATION_MODE_FLAGS

def get_calculation_mode():
    """get_calculation_mode()\n
    
    Returns the calculation mode
    - USE_CPU (do computations only on the CPU)
    - USE_GPU (use cupy to do computations on the GPU)
    - DO_DISTRIBUTE (distribute lags across computers/cores via MPI)

    Returns:
        tuple: use_cpu, use_gpu, do_distribute
    """
    do_distribute = CalculationMode.DO_DISTRIBUTE in compute_config.CALCULATION_MODE_FLAGS
    use_cpu = CalculationMode.USE_CPU in compute_config.CALCULATION_MODE_FLAGS
    use_gpu = CalculationMode.USE_GPU in compute_config.CALCULATION_MODE_FLAGS
    return use_cpu, use_gpu, do_distribute

def set_fitting_mode(
        finite_differences: Optional[bool]=False,
        gaussian_process: Optional[bool]=False):
    """set_fitting_mode()\n
    
    Set the technique to use for fitting functions

    - DO_FINITE_DIFFERENCE (do finite difference calculations)
    - DO_GAUSSIAN_REGRESSION (do Gaussian process regression)

    Args:
        finite_differences (bool):
        gaussian_process (bool):
    
    Returns:
        bool: Current flag
    """
    if not finite_differences and not gaussian_process:
        raise ValueError('Need either `finite differences` or `gaussian process`')
    if finite_differences and gaussian_process:
        raise ValueError('Cannot do both `finite differences` and `gaussian process`')

    if finite_differences:
        compute_config.FITTING_MODE_FLAGS = FittingMode.DO_FINITE_DIFFERENCE
    if gaussian_process:
        compute_config.FITTING_MODE_FLAGS = FittingMode.DO_GAUSSIAN_REGRESSION

    return compute_config.FITTING_MODE_FLAGS

def get_fitting_mode():
    """get_fitting_mode()\n

    Returns:
        tuple: do_finite_differences, do_gaussian_process
    """
    return (FittingMode.DO_FINITE_DIFFERENCE == compute_config.FITTING_MODE_FLAGS, FittingMode.DO_GAUSSIAN_REGRESSION == compute_config.FITTING_MODE_FLAGS)

