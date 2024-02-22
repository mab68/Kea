import Kea
from Kea.utils import plotting
from Kea.simulator import fbm
from Kea.statistics import statistics_base
from Kea.statistics.spectra import strfn_spectra, spectra_base, arevalo_spectra
from Kea.statistics.statfunc import strfn, statfunc_base, statfunc_mpi_naive
from Kea.fitting import fit_base

import matplotlib as mpl
plotting.modify_rc()
import matplotlib.pyplot as plt

import numpy as np

D = 2
N = 512
L = 2.*np.pi
grid_dims = [N for _ in range(D)]
phys_dims = [L for _ in range(D)]

min, max = -(D+3.), -1.
NUM = 20
alphas = np.linspace(max, min, NUM)
try:
    fields = np.load('data/fields%s.npy' % D)
except:
    np.random.seed(1234)
    fields = []
    for a in alphas:
        fields.append(fbm.create_fbm(grid_dims, phys_dims, (a,), (1.,)))
    np.save('data/fields%s.npy' % D, fields)

try:
    models = np.load('data/models%s.npy' % D)
except:
    models = []
    for i, a in enumerate(alphas):
        k, _, _ = spectra_base.calculate_integrated_spectrum((fields[i],), spec_type='modal', lenn=phys_dims)
        models.append(fbm.funcs.smooth_pow(k, (a,), (1.,)))
    np.save('data/models%s.npy' % D, models)

try:
    kfeks = np.load('data/kfeks%s.npy' % D)
except:    
    kfeks = []
    for f in fields:
        k, fek, _ = spectra_base.calculate_integrated_spectrum((f,), spec_type='modal', lenn=phys_dims)
        kfeks.append((k, fek))
    np.save('data/kfeks%s.npy' % D, kfeks)

try:
    sfs = np.load('data/sfs%s.npy' % D)
except:
    sfs = []
    for i, f in enumerate(fields):
        print('strfn', i)
        lv2 = statfunc_base.get_all_lagvecs([2*n+1 for n in f.shape])
        sf2 = strfn.process_lags(f, f, lv2, periodic=True, lenn=phys_dims, shape=tuple([2*n+1 for n in f.shape]), orders=[2])[0]
        sfs.append(sf2)
    np.save('data/sfs%s.npy' % D, sfs)

try:
    lsf = np.load('data/lsf%s.npy' % D)
except:
    lsf = []
    for i, fs in enumerate(sfs):
        print('bin strfn', i)
        lvm = statfunc_base.get_lagvec_magnitude_array(fs.shape)
        l, sf, _ = statistics_base.bin_data(lvm, fs, mean_func=np.nanmean, cut_excess=True, min_bin=1., ignore_nan=True)
        lsf.append((l, sf))
    np.save('data/lsf%s.npy' % D, lsf)

try:
    k_arevaloz = np.load('data/karevalo%s.npy' % D)
except:
    k_arevaloz = []
    for i, f in enumerate(fields):
        print('arevalo', i)
        lags = arevalo_spectra.k_to_discrete_lags(kfeks[i][0], N, L)
        k, fek = arevalo_spectra.modal_spectrum(f, lenn=phys_dims, lags=lags)
        k_arevaloz.append((k, fek))
    np.save('data/karevalo%s.npy' % D, k_arevaloz)

