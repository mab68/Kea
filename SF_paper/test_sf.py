
import Kea
from Kea.utils import plotting
from Kea.simulator import fbm
from Kea.statistics import statistics_base, moments, scale_filter
from Kea.statistics.spectra import strfn_spectra, spectra_base, arevalo_spectra, per_spectra, corr_spectra
from Kea.statistics.statfunc import strfn, statfunc_base, statfunc_mpi_naive, corr
from Kea.fitting import fit_base

c = plotting.COLOR_CYCLE_6

import matplotlib as mpl
plotting.modify_rc()
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from scipy.signal import windows

import itertools

from mpi4py import MPI
comm = MPI.COMM_WORLD

D = 1

N = 10000
L = 5.*np.pi

N_f = 4*N
L_f = 4*L

grid_dims = [N for _ in range(D)]
grid_dims_f = [N_f for _ in range(D)]

phys_dims = [L for _ in range(D)]
phys_dims_f = [L_f for _ in range(D)]

twopi = 2.*np.pi

dk_f = twopi/L_f
dk = twopi/L

dx_f = L_f/N_f
dx = L/N

BREAK = 100 * dk_f


## GENERATE BIG FIELD
f_f = fbm.create_fbm([N_f for _ in range(D)], phys_dims_f, (-1.1, -5./3.), gfunc='smooth_pow', func_kwargs={'breaks': (BREAK,), 'delta': 0.25})
## GENERATE SUBSET FIELD
f = f_f[N_f//2-N//2:N_f//2+N//2]
f = f - np.mean(f)
## APPLY FIELD
f_p = windows.tukey(N, 0.1) * f


kD_f, fekD_f = per_spectra.modal_spectrum(f_f, phys_dims=phys_dims_f)
fekD_f = fekD_f * (dk_f/twopi)**D
k_f, fek_f, _ = spectra_base.spectrum_integrate(kD_f, fekD_f, spec_type='omni', lenn=phys_dims_f, num_bins=N//2)

kD_p, fekD_p = per_spectra.modal_spectrum(f_p, phys_dims=phys_dims)
fekD_p = fekD_p * (dk/twopi)**D
k_p, fek_p, _ = spectra_base.spectrum_integrate(kD_p, fekD_p, spec_type='omni', lenn=phys_dims, num_bins=N//2)

kD, fekD = per_spectra.modal_spectrum(f, phys_dims=phys_dims)
fekD = fekD * (dk/twopi)**D
k, fek, _ = spectra_base.spectrum_integrate(kD, fekD, spec_type='omni', lenn=phys_dims, num_bins=N//2)

