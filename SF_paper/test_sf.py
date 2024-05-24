
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

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

D = 2

N = 64#2**10
print(N**D)
L = 5.*np.pi

N_f = 4*N
print(N_f**D)
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

BREAK = 25 * dk_f


def SmoothySpec(a,start=None,end=None,width=3):
   b=a.copy()
   if start is None: start = 0
   if end is None: end = len(a)
   for i in range(start,end):
      #b[i+1::][:-1] = 0.25*b[i::][:-2]+0.5*b[i+1::][:-1]+0.25*b[i::][2:]
      b[i+1:-1] = 0.25*b[i:-2] + 0.5*b[i+1:-1] + 0.25*b[i+2:]
   return b


NAME = 'small_test.h5'
GENERATE_FIELD = False
MAKE_PLOT = False

if rank == 0:
    if GENERATE_FIELD:
        ## GENERATE AND SAVE BIG FIELD
        f_f = fbm.create_fbm([N_f for _ in range(D)], phys_dims_f, (-2.01, -8./3.), gfunc='smooth_pow', func_kwargs={'breaks': (BREAK,), 'delta': 0.25})
        print('full field generated')
        file = data_writer.save_h5(f_f, NAME, 'full')
        f_f = None
        ## GENERATE SUBSET FIELD FROM BIG FIELD
        idx = tuple([slice(N_f//2-N//2, N_f//2+N//2) for _ in range(D)])
        f = file['full'][idx]
        f = f - np.mean(f)
        f_p = f * statistics_base.ndim_func(windows.tukey, (N, N), (N, 0.01))
        print('windowed subset generated')
        ## SAVE SUBSET FIELD
        file.close()
        file = data_writer.save_h5(f_p, NAME, 'subset')
    else:
        file = h5py.File(NAME, 'r')
        print('loaded')

    if MAKE_PLOT:
        ## MAKE 2D PLOT
        fig, ax = plt.subplots(1, 2, figsize=(3.5, 3.5), dpi=512)
        cmin, cmax = plotting.get_minmax(file['full'])
        ax[0].imshow(file['full'], origin='lower', cmap='bwr', vmin=cmin, vmax=cmax)
        ax[1].imshow(file['subset'], origin='lower', cmap='bwr', vmin=cmin, vmax=cmax)
        for axx in ax:
            axx.set_xticks([])
            axx.set_yticks([])
        rect = patches.Rectangle((N_f//2-N//2, N_f//2-N//2), N, N, linewidth=1., edgecolor='black', linestyle=':', facecolor='none')
        ax[0].add_patch(rect)
        fig.savefig('plots/full_subset_fbm.pdf', transparent=True, bbox_inches='tight', pad_inches=0)

        ## MAKE SPECTRUM PLOT
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=512)
        kD_f, fekD_f = per_spectra.modal_spectrum(file['full'][...].astype(float), phys_dims=phys_dims_f)
        fekD_f = fekD_f * (dk_f/twopi)**D
        ko_f, feko_f, w_f = spectra_base.spectrum_integrate(kD_f, fekD_f, spec_type='omni', lenn=phys_dims_f)
        _, feke_f, _ = spectra_base.spectrum_integrate(kD_f, fekD_f, spec_type='error', lenn=phys_dims_f)
        feke_f = np.sqrt((2.*np.pi*ko_f*feke_f)**2)

        ax.fill_between(ko_f, feko_f-feke_f, feko_f+feke_f, color='black', alpha=0.25, edgecolor='none')
        ax.loglog(ko_f, feko_f, color='black', linewidth=1.)

        kD_p, fekD_p = per_spectra.modal_spectrum(file['subset'][...].astype(float), phys_dims=phys_dims)
        fekD_p = fekD_p * (dk/twopi)**D
        ko_p, feko_p, w_p = spectra_base.spectrum_integrate(kD_p, fekD_p, spec_type='omni', lenn=phys_dims)
        _, feke_p, _ = spectra_base.spectrum_integrate(kD_p, fekD_p, spec_type='error', lenn=phys_dims)
        feke_p = np.sqrt((2.*np.pi*ko_p*feke_p)**2)

        ax.fill_between(ko_p, feko_p-feke_p, feko_p+feke_p, color='blue', alpha=0.25, edgecolor='none')
        ax.loglog(ko_p, feko_p, color='blue', linewidth=1., alpha=0.5)

        feko_s = np.exp(SmoothySpec(np.log(feko_f)))
        ax.loglog(ko_f, feko_s, color='gray', linewidth=1.)

        ax.axvline(BREAK, linewidth=1., color='gray', linestyle=(0,(5,5)))

        #ax.set_ylim((1e-10, 1e0))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='y', direction='in', which='minor')
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='x', direction='in', which='minor')
        ax.grid(linestyle=':', alpha=0.3, linewidth=0.5, color='gray')

        ax.set_ylabel(r'Periodogram, $\tilde{\mathcal{E}}^P(k)$')
        ax.set_xlabel(r'Wavenumber, $k$')
        fig.savefig('plots/full_subset_fbm_spectrum.pdf', transparent=True, bbox_inches='tight', pad_inches=0)

        file.close()

print('init %s' % rank)
comm.Barrier()

file = h5py.File(NAME, 'r')
print(np.shape(file['subset']))
lv_p = statfunc_base.get_all_lagvecs([n+1 for n in grid_dims])
print(np.shape(lv_p))
lv, sf_p = strfn.process_lags(file['subset'], file['subset'], lv_p,
                          lenn=phys_dims, shape=[n+1 for n in grid_dims], orders=[2], periodic=False)
## TODO: return lagvecs for each core, and sf for each core
## TODO: then we will need to combine properly

comm.Barrier()
print('Fin. %s' % rank)

if rank == 0:
    #print(np.shape(lv))
    #lvm = statfunc_base.get_lagvec_magnitude_array([n+1 for n in grid_dims])*dx
    #sf_A = np.zeros([n+1 for n in grid_dims])
    #for i in range(len(shifts)):
    #    shifts[i]
    #    
    #    func_indices = tuple([lagvecs[i][j] + (shape[j]//2) for j in range(ar1.ndim)])
    #    func_indices = tuple(np.array(func_indices)[::-1])
    #np.save('sf.npy', sf_p)
    #np.save('shifts.npy', shifts)

file.close()
