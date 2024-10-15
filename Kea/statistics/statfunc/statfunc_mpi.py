
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    mpi4py = None
    comm = None

def wrap_sf(orders=[2]):
    def process_sf(ar, lags):
        """process_sf(ar, lags)

        Calculate the structure function for the given lags and orders
        """
        out = np.zeros((len(lags), len(orders)))
        N = np.shape(ar)
        D = len(N)
        for i, l in enumerate(lags):
            s1, s2 = [], []
            for d in range(1,D+1):
                s1.append(slice(l[D-d],N[D-d]))
                s2.append(slice(0,N[D-d]-l[D-d]))
            for o, p in enumerate(orders):
                out[i,o] = np.nanmean(np.abs(ar[tuple(s1)] - ar[tuple(s2)])**p)
        return out
    return process_sf

def wrap_acf():
    def process_acf(ar, lags):
        """process_acf(ar, lags)

        Calculate the correlation function for the given lags and orders
        """
        out = np.zeros(len(lags))
        N = np.shape(ar)
        D = len(N)
        for i, l in enumerate(lags):
            s1, s2 = [], []
            for d in range(1,D+1):
                s1.append(slice(l[D-d],N[D-d]))
                s2.append(slice(0,N[D-d]-l[D-d]))
            out[i] = np.nanmean(ar[tuple(s1)] * ar[tuple(s2)])
        return out
    return process_acf

def get_all_lagvecs(shape):
    """get_all_lagvecs(shape)
    
    Get all the lagvectors as a list required to compute the positive lag quadrant of the SF
    """
    return np.transpose(np.indices(shape).reshape((-1, np.prod(shape))))

def process_lags(ar, max_lag=None, lag_func=wrap_sf()):
    """process_lags(ar, max_lag, lag_func)

    Calculates the second order SF by naively distributing the lags amongst the available
        processors. Returns the lags, and SF computed by the current processor
    You will need to save the outputs, and recombine into one large SF later on.

    Args:
        ar (np.ndarray): Numpy ND array structure to compute the SF for
        max_lag (int/None): Maximum lag to go to. If None, use N//2
        lag_func (func): Function to compute. Default is SF of order 2.
    Returns:
        l (np.ndarray): Lags corresponding to the SF for the current processor
        sf (np.ndarray): SF computed by the current processor
    """
    if max_lag is None:
        max_lag = np.min(np.shape(ar))//2
    max_lag = tuple([max_lag for _ in range(ar.ndim)])
    # Not running in mpi, so just compute the statfunc
    if (comm is None) or comm.Get_size() == 1:
        lags = get_all_lagvecs(max_lag)
        return lags, lag_func(ar, lags)
    # Compute the structure function without any communication whatsoever
    rank = comm.Get_rank()
    size = comm.Get_size()
    lags = get_all_lagvecs(max_lag)[rank::size]
    print('starting %s:%s' % (comm.Get_rank(), len(lags)))
    # Compute lag function
    statfunc = lag_func(ar, lags)
    return lags, statfunc
