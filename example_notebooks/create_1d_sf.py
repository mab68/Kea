
import Kea
from Kea.utils import plotting
from Kea.simulator import fbm
from Kea.statistics import statistics_base, moments, scale_filter
from Kea.statistics.spectra import strfn_spectra, spectra_base, arevalo_spectra, per_spectra, corr_spectra
from Kea.statistics.statfunc import strfn_1d, strfn, statfunc_base
from Kea.fitting import fit_base

c = plotting.COLOR_CYCLE_6

import matplotlib as mpl
plotting.modify_rc()
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from scipy.signal import windows

import itertools

import h5py

from mpi4py import MPI
comm = MPI.COMM_WORLD

rank = comm.Get_rank()

D = 1

N = 10000
L = 5.*np.pi
grid_dims = [N for _ in range(D)]
phys_dims = [L for _ in range(D)]
twopi = 2.*np.pi
dk = twopi/L
dx = L/N
BREAK = 100 * dk

if rank == 0:
    f_f = fbm.create_fbm([N for _ in range(D)], phys_dims, (-1., -5./3.), gfunc='smooth_pow', func_kwargs={'breaks': (BREAK,), 'delta': 0.25})
    f0 = h5py.File('T.Fr.h5', 'a')
    f0.create_dataset('T.Fr', data=f_f)
    f0.close()

#l, sf = strfn_1d.process_lags(f_f, N//2, periodic=False)
#l = l * dx

# lv_p = statfunc_base.get_all_lagvecs([2*n+1 for n in grid_dims])
# sf_p = strfn.process_lags(f_f, f_f, lv_p, lenn=phys_dims, shape=[2*n+1 for n in grid_dims], orders=[2], periodic=False)[0]
# lvm_p = statfunc_base.get_lagvec_magnitude_array([2*n+1 for n in grid_dims]) * dx
# l_p, sf2_p, w_p = statistics_base.bin_data(lvm_p, sf_p, bin_func=np.nanmean, cut_excess=True, nan_small=False, min_bin=dx, max_bin=L/2., num_bins=N//3, bin_loc='true_center')
