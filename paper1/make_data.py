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

try:
    from paper1 import load_data
except FileNotFoundError:
    pass

D = load_data.D
N = load_data.N
L = load_data.L
grid_dims = [N for _ in range(D)]
phys_dims = [L for _ in range(D)]

twopi = 2.*np.pi

dx = L/N
dk = twopi/L

FOLDER = load_data.FOLDER
DATA_FOLDER = load_data.FOLDER

alphas = load_data.alphas

try:
    fields = np.load(DATA_FOLDER + 'fields%s_%s.npy' % (D,N))
except FileNotFoundError:
    np.random.seed(1234)
    fields = []
    for a in alphas:
        fields.append(fbm.create_fbm(grid_dims, phys_dims, (a,), gfunc='pure_pow'))
    np.save(DATA_FOLDER + 'fields%s_%s.npy' % (D, N), fields)

try:
    models = np.load(DATA_FOLDER + 'models%s_%s.npy' % (D, N))
except FileNotFoundError:
    models = []
    for i, a in enumerate(alphas):
        kD, fekD = per_spectra.modal_spectrum(fields[i], phys_dims=phys_dims)
        kmesh = spectra_base.wavenumber_mesh(kD)
        model = fbm.funcs.pure_pow(kmesh, (a,))
        model = model / np.nansum(model * (dk/twopi)**D)

        k_model, fek_model, w = statistics_base.bin_data(kmesh, model,
            bin_func=np.nanmean, min_bin=dk, max_bin=None,
            ignore_nan=False, norm_bin_size=False, bin_loc='true_center',
            cut_excess=True, nan_small=False)
        
        models.append((k_model, fek_model, w))
    np.save(DATA_FOLDER + 'models%s_%s.npy' % (D, N), models)

try:
    kfeks = np.load(DATA_FOLDER + 'kfeks%s_%s.npy' % (D, N))
except FileNotFoundError:    
    kfeks = []
    for i, a in enumerate(alphas):
        kD, fekD = per_spectra.modal_spectrum(fields[i], phys_dims=phys_dims)
        kmesh = spectra_base.wavenumber_mesh(kD)

        k, fek, w = statistics_base.bin_data(kmesh, fekD,
            bin_func=np.nanmean, min_bin=dk, max_bin=None,
            ignore_nan=False, norm_bin_size=False, bin_loc='true_center',
            cut_excess=True, nan_small=False)

        kfeks.append((k, fek, w))
    np.save(DATA_FOLDER + 'kfeks%s_%s.npy' % (D, N), models)

try:
    k_arevaloz = np.load(DATA_FOLDER + 'karevalo%s_%s.npy' % (D, N))
except FileNotFoundError:
    k_arevaloz = []
    for i, f in enumerate(fields):
        print('Arevalo', i)
        lags = arevalo_spectra.k_to_discrete_lags(kfeks[i][0], N, L, D)
        k, fek = arevalo_spectra.modal_spectrum(f, lenn=phys_dims, lags=lags, mode='wrap')
        k_arevaloz.append((k, fek))
    np.save(DATA_FOLDER + 'karevalo%s_%s.npy' % (D, N), k_arevaloz)

# try:
#     sfs = np.load('data/sfs%s_%s.npy' % (D, N))
# except:
#     sfs = []
#     for i, f in enumerate(fields):
#         print('strfn', i)
#         lv2 = statfunc_base.get_all_lagvecs([2*n+1 for n in f.shape])
#         sf2 = strfn.process_lags(f, f, lv2, periodic=True, lenn=phys_dims, shape=tuple([2*n+1 for n in f.shape]), orders=[2])[0]
#         sfs.append(sf2)
#     np.save('data/sfs%s_%s.npy' % (D, N), sfs)

# try:
#     lsf = np.load('data/lsf%s_%s.npy' % (D, N))
# except:
#     lsf = []
#     for i, fs in enumerate(sfs):
#         print('bin strfn', i)
#         lvm = statfunc_base.get_lagvec_magnitude_array(fs.shape) * dx
#         l, sf, _ = statistics_base.bin_data(lvm, fs, bin_func=np.nanmean, cut_excess=True, min_bin=0., ignore_nan=True)
#         lsf.append((l, sf))
#     np.save('data/lsf%s_%s.npy' % (D, N), lsf)
