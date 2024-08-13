
import numpy as np

from Kea.statistics.spectra import per_spectra, spectra_base, strfn_spectra
from Kea.statistics.statfunc import statfunc_base, strfn
from Kea.statistics import statistics_base
from Kea.utils import plotting, data_writer

import sympy as sp

import h5py as h5

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plotting.modify_rc()

twopi = 2.*np.pi

from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits


f0 = fits.open('/home/m/Documents/science_codes/Data/LMC/LMC_SED_Mass.fits')
field1 = f0[0].data[625-512:625+512,625-512:625+512]


field0 = f0[0].data[625-512:625+512,625-512:625+512]
field0[np.isnan(field0)] = 0.

N = 1024#int(f0[0].header['NAXIS1'])
L = float(f0[0].header['CDELT2'])*N

D = 2

phys_dims = (L, L)
grid_dims = (N, N)

dk = twopi/L
dx = L/N

lv_p = statfunc_base.get_all_lagvecs(grid_dims)
sf_p = strfn.process_lags(field0, field0, lv_p, lenn=phys_dims, shape=grid_dims, orders=[2], periodic=False)[0]
#lvm_p = statfunc_base.get_lagvec_magnitude_array([n+1 for n in grid_dims]) * dx
#l_p, sf2_p, w_p = statistics_base.bin_data(lvm_p, sf_p, bin_func=np.nanmean, cut_excess=True, nan_small=False,
#                                           min_bin=dx, max_bin=L/2., bin_loc='center', log_space=True, num_bins=N//4)

np.save('sf2d.npy', sf_p)

