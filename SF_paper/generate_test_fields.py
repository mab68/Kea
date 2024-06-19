
import Kea
from Kea.utils import plotting, data_reader, data_writer
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

D = [2]#[2, 3]
N2 = [512]#[64, 128, 256, 512, 1024, 2048]
N3 = []#[64, 128]

L = 2.*np.pi
dk = 2.*np.pi/L

break_k = 2.5 * dk
diss_k = 1000. * dk

#DATA_DIR=/nfs/scratch/bishopm1/data
#DATA_DIR = os.environ['DATA_DIR']
DATA_DIR = '/home/m/Documents/science_codes/Kea/SF_paper'
NAME_PREFIX = DATA_DIR + '/SF_TEST/'

for d in D:
    if d == 2:
        N = N2
    else:
        N = N3

    for n in N:
        NAME = NAME_PREFIX + str(d) + '_' + str(n) + '_A2' + '/in/'
        if not os.path.exists(NAME):
            os.makedirs(NAME)
        NAME = NAME + 'T.Fr.h5'

        grid_dims = [n for _ in range(d)]
        phys_dims = [L for _ in range(d)]

        H = ((5./3.) - 1.) / 2.
        alph = d + 2.*H

        field = fbm.create_fbm(grid_dims, phys_dims, (-alph,), gfunc='pow_exp', func_kwargs={'breaks': (break_k, diss_k)}, A=2.5e-5)
        print('generated field %s^%s (%s)' % (n, d, alph))
        print('var:', moments.var(field, grid_dims, phys_dims))
        file = data_writer.save_h5(field, NAME, 'T.Fr')
        print('saved field %s^%s' % (n, d))
        file.close()
