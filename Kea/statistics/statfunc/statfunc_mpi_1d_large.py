

import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    mpi4py = None
    comm = None

from .. import moments


def process_lags(ar, lags, periodic):
    out = np.zeros(len(lags))
    N = len(ar)
    for i, l in enumerate(lags):
        if periodic:
            X = ar[:]
            Xp = np.roll(ar[:], l, axis=0)
        else:
            X = ar[l:]
            Xp = ar[:N-l]
        out[i] = np.nanmean(np.abs(X - Xp)**2)
    return out

def mpi_pool_lagvecs(ar, max_lag, periodic):
    lags = np.arange(0, max_lag, 1)
    # Not running in mpi, so just compute the statfunc
    if (comm is None) or comm.Get_size() == 1:
        return lags, process_lags(ar, lags, periodic)
    if comm.Get_rank() == 0:
        split_lags = np.array_split(lags, comm.Get_size())
    else:
        split_lags = None
    # Tell each process what lags they should calculate
    split_shifts = comm.scatter(split_lags, root=0)
    print('starting %s:%s' % (comm.Get_rank(), len(split_shifts)))
    # Compute lag function
    statfunc = process_lags(ar, split_shifts, periodic)
    # Wait until everyone has finished processing
    comm.Barrier()
    # Retrieve and combine functions
    rcv = np.zeros(len(lags))
    comm.Gather(statfunc, rcv, root=0)
    if comm.Get_rank() == 0:
        return lags, rcv
    else:
        return None, None
