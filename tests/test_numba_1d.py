
import sys
sys.path.append('/home/mark.bishop/Documents/Kea')

import kea
from kea.simulator import mff
from kea.statistics.statfunc import mpi_acf, statfunc_numba

import numpy as np
import time

D = 1
#1_000, 10_000, 100_000, 
N_range = [500_000, 1_000_000]
NUM_ITERS = 10

numba_times = np.zeros((NUM_ITERS,len(N_range)))
naive_times = np.zeros((NUM_ITERS,len(N_range)))

for j, N in enumerate(N_range):
    kk = mff.make_grid(N, D)
    lags = np.zeros((N//2, D))

    for i in range(NUM_ITERS):
        field = mff.make_mff(kk[1], 2.*np.pi, 2./3., N)

        mpi_lags = mpi_acf.get_all_lagvecs(tuple([N//2 for _ in range(D)]))
        start_naive = time.perf_counter()
        res_naive = mpi_acf.process_lags(field, mpi_lags)
        end_naive = time.perf_counter()
        naive_times[i,j] = end_naive - start_naive

        print('FINISHED NAIVE', naive_times[i,j])

        start_numba = time.perf_counter()
        res_numba = statfunc_numba.process_lags(field, mpi_lags, statfunc_numba.StatMetric.CORR)[:,0]
        end_numba = time.perf_counter()
        numba_times[i,j] = end_numba - start_numba

        print('FINISHED NUMBA', numba_times[i,j])

        print(np.allclose(res_naive, res_numba))
        print(numba_times[i,j], naive_times[i,j])

    print('-------------------------------------------')
    print(np.mean(numba_times[:,j]), np.mean(naive_times[:,j]))
    print('===========================================')

np.save('numba_times_1d.npy', numba_times)
np.save('naive_times_1d.npy', naive_times)
