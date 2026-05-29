"""
corr.py

Calculates the autocorrelation function for arbitrary dimension $D$.

NOTE: $\vec{\ell}$ will be only the *positive* lags.
"""

from typing import Optional, Callable
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    print('NO MPI')
    mpi4py = None
    comm = None

def process_lags(
        field: np.ndarray,
        lags: np.ndarray,
        statfunc: Callable,
        wrap: Optional[bool]=False) -> np.ndarray:
    """process_lags(field, lags, statfunc, wrap)

    Computes the statfunc for the given lags

    Args:
        field (np.ndarray): The field to compute the (auto)-statfunc for
        lags (np.ndarray): List of lag-shifts
        statfunc (func): The function to call that provides the composition of the lag-nonlag field
        wrap (bool): If true, roll the field
    Return:
        out (np.ndarray): The statfunc for the given lags
    """
    out = np.zeros(len(lags))
    N = np.shape(field)
    D = len(N)
    # Iterate through each provided lag
    for i, l in enumerate(lags):
        _D = len(l)
        if wrap:
            # If wrap, then the field is periodic and we can roll
            field_l = field.copy()
            for d in range(0, _D):
                field_l = np.roll(field_l, l[d], axis=d)
            out[i] = statfunc(field, field_l)
        else:
            # Otherwise, cut out (use only) the overlapping segments
            s1, s2 = [slice(0,N[d]) for d in range(D)], [slice(0,N[d]) for d in range(D)]
            for d in range(1, _D+1):
                s1[d-1] = slice(l[_D-d],N[D-d])
                s2[d-1] = slice(0,N[D-d]-l[_D-d])
            out[i] = statfunc(field[tuple(s1)], field[tuple(s2)])
    return out

def get_all_lagvecs(shape):
    """get_all_lagvecs(shape)
    
    Get all the lagvectors as a list required to compute the positive lag quadrant of the SF
    """
    return np.transpose(np.indices(shape).reshape((-1, np.prod(shape))))

def mpi_statfunc(ar, max_lag=None, longitudinal=False):
    """mpi_sf(ar, max_lag)

    Calculates the second order SF by naively distributing the lags amongst the available
        processors. Returns the lags, and SF computed by the current processor
    You will need to save the outputs, and recombine into one large SF later on.

    Args:
        ar (np.ndarray): Numpy ND array structure to compute the SF for
        max_lag (int): Maximum lag to go to
        longitudinal (bool): If true, compute the longitudinal SF only
    Returns:
        l (np.ndarray): Lags corresponding to the SF for the current processor
        sf (np.ndarray): SF computed by the current processor
    """
    if max_lag is None:
        max_lag = np.min(np.shape(ar))//2
    D = ar.ndim
    if longitudinal:
        D = 1
    max_lag = tuple([max_lag for _ in range(D)])
    p_func = process_lags
    # Not running in mpi, so just compute the statfunc
    if (comm is None) or comm.Get_size() == 1:
        lags = get_all_lagvecs(max_lag)
        return lags, p_func(ar, lags)
    # Compute the structure function without any communication whatsoever
    rank = comm.Get_rank()
    size = comm.Get_size()
    lags = get_all_lagvecs(max_lag)[rank::size]
    print('starting %s:%s' % (comm.Get_rank(), len(lags)))
    # Compute lag function
    statfunc = p_func(ar, lags)
    return lags, statfunc
