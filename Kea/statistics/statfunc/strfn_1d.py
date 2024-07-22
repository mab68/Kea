

import numpy as np

from . import statfunc_base, statfunc_mpi_1d_large


def process_lags(ar, max_lag=None, periodic=False):
    if max_lag is None:
        max_lag = len(ar)//2
    return statfunc_mpi_1d_large.mpi_pool_lagvecs(ar, max_lag, periodic)

