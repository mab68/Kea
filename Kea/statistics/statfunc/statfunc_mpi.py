
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    mpi4py = None
    comm = None


def correlation(X1, X1p):
    return np.nanmean(X1 * X1p)

def strfn(X1, X1p, p):
    return np.nanmean(np.abs(X1 - X1p)**p)

def process_lags(ar, lagvecs, periodic=False):
    """
    
    """
    raise NotImplementedError()

def mpi_pool_lagvecs(ar, lagvecs, periodic=False):
    """
    
    """
    raise NotImplementedError()

