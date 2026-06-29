"""
Stores the calculation mode flags.

Handles CPU or GPU implementations as well as distributed the list of lags across processes using MPI.

To set which calculation mode to use, call `set_calculation_mode()`.
"""

from typing import Optional
from enum import IntEnum

class CalculationMode(IntEnum):
    """
    Set the calculate mode for statfuncs:

    - USE_CPU (do computations only on the CPU)
    - USE_GPU (use cupy to do computations on the GPU)
    - DO_DISTRIBUTE (distribute lags/scales across computers/cores via MPI)

    Set multiple calculation modes using a pair-tuple e.g., (DO_DISTRIBUTE, USE_CPU)
    """
    USE_CPU = 0
    USE_GPU = 1
    DO_DISTRIBUTE = 2

class FittingMode(IntEnum):
    """
    Set the technique to use for fitting functions

    - DO_FINITE_DIFFERENCE (do finite difference calculations)
    - DO_GAUSSIAN_REGRESSION (do Gaussian process regression)
    """
    DO_FINITE_DIFFERENCE = 0
    DO_GAUSSIAN_REGRESSION = 1

## Holds the current statfunc calculation mode
CALCULATION_MODE_FLAGS = (CalculationMode.USE_CPU,)
FITTING_MODE_FLAGS = FittingMode.DO_FINITE_DIFFERENCE
