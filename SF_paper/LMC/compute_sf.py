
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
import astropy.units as u

from scipy.signal import windows

import time

start_time = time.time()

f0 = fits.open('/home/m/Documents/science_codes/Data/LMC/LMC_SED_Mass.fits')

da = (float(f0[0].header['CDELT2']) * u.deg).to(u.rad)
dist = 50.*u.kpc
dx = (dist * np.tan(da)).value
print(dx*u.kpc)
print(1024*dx*u.kpc)

N = 1024#int(f0[0].header['NAXIS1'])
L = dx*N
D = 2
phys_dims = (L, L)
grid_dims = (N, N)
dk = twopi/L

original = f0[0].data[625-512:625+512,625-512:625+512]
uncert = f0[1].data[625-512:625+512,625-512:625+512]

from Kea.statistics import scale_filter
zerod = original.copy()
zerod[np.isnan(zerod)] = 0.
zerod2 = scale_filter.gaussian_scale_greater(zerod, 4)
zerod2[np.isnan(original)] = np.nan

windowed = zerod2 * statistics_base.ndim_func(windows.tukey, (N, N), (N, 0.5))

lv_p = statfunc_base.get_all_lagvecs(grid_dims)
sf_p = strfn.process_lags(windowed, windowed, lv_p, lenn=phys_dims, shape=grid_dims, orders=[2], periodic=False)[0]

np.save('sf2d_PSF_4.npy', sf_p)

print('Compute time', time.time() - start_time)

#20183.099205732346[s]
