
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

f0 = h5py.File('T.Fr.h5', 'r')
field = f0['T.Fr']

l, sf = strfn_1d.process_lags(field, N//2)

if rank == 0:
    np.save('l.npy', l)
    np.save('sf.npy', sf)

