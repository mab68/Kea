

import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    mpi4py = None
    comm = None

from .. import moments


def process_lags(ar, lags):
    out = np.zeros(len(lags))
    N = len(ar)
    for i, l in enumerate(lags):
        out[i] = np.nanmean(np.abs(ar[l:] - ar[:N-l])**2)
    return out

def process_lags_per(ar, lags):
    out = np.zeros(len(lags))
    N = len(ar)
    for i, l in enumerate(lags):
        out[i] = np.nanmean(np.abs(ar - np.roll(ar, l, axis=0))**2)
    return out

def mpi_pool_lagvecs(ar, max_lag, periodic):
    p_func = process_lags
    if periodic:
        p_func = process_lags_per
    # Not running in mpi, so just compute the statfunc
    if (comm is None) or comm.Get_size() == 1:
        lags = np.arange(0, max_lag, 1)
        return lags, p_func(ar, lags)
    # Compute the structure function without any communication whatsoever
    rank = comm.Get_rank()
    size = comm.Get_size()
    lags = np.arange(rank, max_lag, size)
    print('starting %s:%s' % (comm.Get_rank(), len(lags)))
    # Compute lag function
    statfunc = p_func(ar, lags)
    return lags, statfunc
