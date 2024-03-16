import Kea
from Kea.utils import plotting
from Kea.simulator import fbm
from Kea.statistics import statistics_base
from Kea.statistics.spectra import strfn_spectra, spectra_base, arevalo_spectra, per_spectra
from Kea.statistics.statfunc import strfn, statfunc_base, statfunc_mpi_naive
from Kea.fitting import fit_base

import matplotlib as mpl
plotting.modify_rc()
import matplotlib.pyplot as plt

import numpy as np

D = 2
N = 256
L = 2.*np.pi
grid_dims = [N for _ in range(D)]
phys_dims = [L for _ in range(D)]

dx = L/N
dk = 2.*np.pi/L

min, max = -(D+3.), -(D-1.)
NUM = 20
alphas = np.linspace(max, min, NUM)

FOLDER = ''
DATA_FOLDER = FOLDER + 'data/'

fields = np.load(DATA_FOLDER + 'fields%s_%s.npy' % (D,N))
models = np.load(DATA_FOLDER + 'models%s_%s.npy' % (D, N))
kfeks = np.load(DATA_FOLDER + 'kfeks%s_%s.npy' % (D, N))
k_arevaloz = np.load(DATA_FOLDER + 'karevalo%s_%s.npy' % (D, N))
