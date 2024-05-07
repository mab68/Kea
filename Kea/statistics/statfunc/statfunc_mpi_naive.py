
"""
statfunc_mpi_naive.py
Implements multiprocessing for correlation and structure function calculation

This 'naive' implementation means that you run your python as:
>>> mpiexec -n P_NUM python3 PYTHON_FILE

Everything before the correlation calculation will be done simulateneously,
for each process. Meaning that loading files will be done seperately for each process
as well as performing computations, except for the statfunc processes in this file. 
This file allows for the computation to be spread across those different processes once
they reach this file but everything before hand will be done repeatedly. 
This is a waste of memory and could be impossible for large simulation files that need to be processed.

A better method would be to use the `statfunc_mpi_naive` implementation and just setup the problem before hand
using the mpi utilities in `analysis_engine.utils.mpi` to read the required files from master and then
set them up as shared files.
>>> from analysis_engine.utils.mpi import mpi
>>> import analysis_engine.utils.data_reader as dr
>>> arz = mpi.get_shared_window(dr, {'keys': KEYS_LIST, 'folder_name': 'FOLDER_NAME', 'file_name': 'FILE_NAME'})

To run as a single process, call `process_lags()`, providing the entire `shifts` array to the function
>>> process_lags(...)

Functions
---------
correlation\n
strfn\n
process_lags\n
mpi_pool_lagvecs\n
"""


## TODO: ALSO PROVIDE A METHOD TO CALCULATE THE INTEGRATED SPACE
## i.e. BINNING ALL THE LAGS IN THIS STEP RATHER THAN SAVING THE 3D/2D SPACE AND DOING IT AFTER

import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    mpi4py = None
    comm = None

from .. import moments


def correlation(X1, X1p, X2, X2p, shape, lenn):
    """correlation(X1, X2, shape, *lfuncargs)

    Calculates the correlation between `X1` and `X2` via multiplication

    Calculates the cross-correlation `1/2 <f g'> + 1/2 <g f'>`

    Args:
        X1, X2 (np.ndarray): The two arrays to calculate the correlation
        shape (tuple): Sampled number of total points
    Returns:
        corr (float): Correlation of `X1` and `X2`
    """
    ## NOTE: For the correlation function, we would prefer the bias form
    ## TODO: double check this...
    if 0 in np.shape(X1) or 0 in np.shape(X2):
        return 0.
    shape = np.nansum(np.isfinite(X1) & np.isfinite(X2p) & np.isfinite(X2) & np.isfinite(X1p))
    a1 = moments.mean(X1 * X2p, shape=shape, take_abs=False, lenn=lenn)
    a2 = moments.mean(X2 * X1p, shape=shape, take_abs=False, lenn=lenn)
    return 0.5 * (a1 + a2)

def strfn(X1, X1p, X2, X2p, shape, lenn, orders):
    """strfn(X1, X2, shape, orders)

    Calculates the structure function between `X1` and `X2` of each requested order in `orders`

    Calculates the cross-structure function `1/2 <(f - g')^p> + 1/2 <(g - f')^p>` for general powers `p`

    Args:
        X1, X2 (np.ndarray): The two arrays to calculate the correlation
        shape (tuple): Sampled number of total points
        orders (np.array): The orders of the structure function to calculate
    Returns:
        strfn (np.ndarray): List of structure functions of `orders`
    """
    ## NOTE: For structure function, we would prefer the nonbias form
    if 0 in np.shape(X1) or 0 in np.shape(X2):
       ## NOTE: Maybe we should check why we need this...
       ## somewhere in the shifts, we are calculating with things are shape (0, N) etc...
       return [0. for _ in orders]
    sfs = []
    shape = np.nansum(np.isfinite(X1) & np.isfinite(X2p) & np.isfinite(X2) & np.isfinite(X1p))
    for o in orders:
        a1 = moments.make_moment(X1 - X2p, moment=o, shape=shape, take_abs=True, lenn=lenn)
        a2 = moments.make_moment(X2 - X1p, moment=o, shape=shape, take_abs=True, lenn=lenn)
        sfs.append(0.5 * a1 + 0.5 * a2)
        #sfs.append(np.nanmean(np.abs(X1 - X2p)**o))
    return sfs

def process_lags(ar1, ar2, lagvecs, shifts, shape, lagvec_func, lagvec_args):
    """process_lags(ar, lagvecs, shifts, shape, lagvec_func, lagvec_args)

    Processes lags in shifts. Where shifts has been given for each process.
    If no multiprocessing, then just give all of shifts to a single process.

    Args:
        ar1 (np.ndarray): The data array to process
        ar2 (np.ndarray): The data array to process
        lagvecs (np.ndarray): Each lag as a vector
        shifts (tuple): Cuts of the array for each given lagvec for this process to compute
        shape (tuple): Array of the true shape
        lagfunc (func): Lag function
        funcargs (tuple): Args for the lag function
        out (np.ndarray): Function output numpy array
        i (int): Current lag we are processing  
    Returns:
        out (np.ndarray): Processes `lagvec_func` applied for each shift on `ar`
    """
    out = np.zeros(shape=shape)
    orig_shape = np.nansum(np.isfinite(ar1) & np.isfinite(ar2))
    #min1, max1, min2, max2 = shifts
    for i in range(lagvecs.shape[0]):
        if shifts is not None:
            x1_indices = tuple([slice(shifts[0][i,j],shifts[1][i,j]) for j in range(ar1.ndim)])
            x2_indices = tuple([slice(shifts[2][i,j],shifts[3][i,j]) for j in range(ar1.ndim)])
            x1i = tuple(np.array(x1_indices)[::-1])
            x2i = tuple(np.array(x2_indices)[::-1])
            X1 = ar1[x1i]
            X1p = ar1[x2i]
            X2 = ar2[x1i]
            X2p = ar2[x2i]
        else:
            X1 = ar1
            X1p = X1.copy()
            X2 = ar2
            X2p = X2.copy()
            for j in range(ar1.ndim):
                X1p = np.roll(X1p, lagvecs[i][j], axis=j)
                X2p = np.roll(X2p, lagvecs[i][j], axis=j)
        if len(lagvec_args) == 2:
            # If lagfunc has an additional term, i.e. strfn order
            func_indices = tuple([Ellipsis] + [lagvecs[i][j] + (shape[j+1]//2) for j in range(ar1.ndim)])
        else:
            func_indices = tuple([lagvecs[i][j] + (shape[j]//2) for j in range(ar1.ndim)])
            func_indices = tuple(np.array(func_indices)[::-1])
        out[func_indices] = lagvec_func(X1, X1p, X2, X2p, orig_shape, *lagvec_args)
    return out

def mpi_pool_lagvecs(ar1, ar2, lagvecs, shifts, shape, lagvec_func, lagvec_args):
    """mpi_pool_lagvecs(ar, lagvecs, shifts, shape, lagvec_func, lagvec_args)

    Args:
        ar (np.ndarray): The data array to process
        lagvecs (np.ndarray): Each lag as a vector
        shifts (tuple): Cuts of the array for each lagvec
        shape (tuple): Array of the true shape
        lagvec_func (func): Lag function
        lagvec_args (tuple): Args for the lag function
    Returns:
        lagf (np.ndarray): The resulting computed lag function array
    """
    # Not running in mpi, so just compute the statfunc
    if (comm is None) or comm.Get_size() == 1:
        return process_lags(ar1, ar2, lagvecs, shifts, shape, lagvec_func, lagvec_args)
    # Tell each process what lags they should calculate
    if comm.Get_rank() == 0:
        slices = shifts
    else:
        slices = None
    slices = comm.scatter(slices, root=0)
    # Compute lag function
    statfunc = process_lags(ar1, ar2, lagvecs, slices, shape, lagvec_func, lagvec_args)
    # Wait until everyone has finished processing
    comm.Barrier()
    # Retrieve and combine functions
    statfunc = comm.Gather(statfunc, statfunc, root=0)
    return statfunc
