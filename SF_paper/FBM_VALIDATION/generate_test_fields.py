
import Kea
from Kea.utils import plotting, data_reader, data_writer, funcs
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
import matplotlib.patches as patches

import numpy as np

from scipy.signal import windows

import itertools

import h5py

import os

D = [1, 2]#, 3]
N1 = [10000]
N2 = [2048]
N3 = [512]

A_factor = 2

L = 2.*np.pi

#DATA_DIR=/nfs/scratch/bishopm1/data
#DATA_DIR = os.environ['DATA_DIR']
DATA_DIR = '/home/m/Documents/science_codes/Kea/SF_paper/'
NAME_PREFIX = DATA_DIR + 'FBM_VALIDATION/test/'

for d in D:
    if d == 1:
        N = N1
    elif d == 2:
        N = N2
    else:
        N = N3

    for n in N:
        NAME = NAME_PREFIX + str(d) + '_' + str(n) + '_A2' + '/in/'
        if not os.path.exists(NAME):
            os.makedirs(NAME)
        NAME = NAME + 'T.Fr'

        grid_dims = [n for _ in range(d)]
        large_grid_dims = [A_factor*n for _ in range(d)]
        phys_dims = [L for _ in range(d)]
        large_phys_dims = [A_factor*L for _ in range(d)]

        H = ((5./3.) - 1.) / 2.
        alph = d + 2.*H

        kmin = dk = 2.*np.pi / L
        break_k = 5. * dk
        kmax = np.pi * n / L
        diss_k = 0.5 * kmax

        # GENERATE:
        #   - A-periodic N^D
        #   - Periodic (4*N)^D
        #       x? This is not needed, we compare to periodic N^D instead
        #   - Periodic N^D
        #   - Injected spectra N^D
        #       x This will be generated when needed

        # Periodic (4*N)^D
        Lper_field = fbm.create_fbm(
            [A_factor*n for _ in range(d)],
            [A_factor*L for _ in range(d)],
            (-alph,), gfunc='pow_exp', func_kwargs={'breaks': (break_k, diss_k)}, A=2.)

        print('generated periodic field %s^%s (%s)' % (4*n, d, alph))
        print('var:', moments.var(Lper_field, large_grid_dims, large_phys_dims))

        Lper_file = data_writer.save_h5(Lper_field, NAME + '.Lper.h5', 'T.Fr')

        # A-periodic N^D
        if d == 1:
            a_field = Lper_field[(A_factor*n)//2-n//2:(A_factor*n)//2+n//2]
        elif d == 2:
            a_field = Lper_field[(A_factor*n)//2-n//2:(A_factor*n)//2+n//2,(A_factor*n)//2-n//2:(A_factor*n)//2+n//2]
        elif d == 3:
            a_field = Lper_field[(A_factor*n)//2-n//2:(A_factor*n)//2+n//2,(A_factor*n)//2-n//2:(A_factor*n)//2+n//2,(A_factor*n)//2-n//2:(A_factor*n)//2+n//2]

        print('var:', moments.var(a_field, grid_dims, phys_dims))
        var = moments.var(a_field, grid_dims, phys_dims)
        Aper_file = data_writer.save_h5(a_field, NAME + '.Aper.h5', 'T.Fr')

        # Periodic N^D
        per_field = fbm.create_fbm(
            [n for _ in range(d)],
            [L for _ in range(d)],
            (-alph,), gfunc='pow_exp', func_kwargs={'breaks': (break_k, diss_k)}, A=var)

        print('generated periodic field %s^%s (%s)' % (n, d, alph))
        print('var:', moments.var(per_field, grid_dims, phys_dims))

        per_file = data_writer.save_h5(per_field, NAME + '.per.h5', 'T.Fr')

        print('saved field %s^%s' % (n, d))
        Lper_file.close()
        Aper_file.close()
        per_file.close()
