
import numpy as np

from Kea.statistics.spectra import per_spectra, spectra_base
from Kea.utils import plotting

import h5py as h5

import matplotlib.pyplot as plt
plotting.modify_rc()


file = h5.File('SF_TEST/2_64/in/T.Fr.h5', 'r')
field = np.array(file['T.Fr'])

kD, fekD = per_spectra.modal_spectrum(field, phys_dims=(2.*np.pi, 2.*np.pi))
k, fek, _ = spectra_base.spectrum_integrate(kD, fekD, spec_type='modal', lenn=(2.*np.pi, 2.*np.pi))

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=256)

ax.loglog(k, fek, color='blue', linewidth=1., alpha=0.5)

ax.loglog(k, 1e2*k**(-11./3.), color='black', linewidth=0.5, linestyle=(0,(5,5)))

#ax.set_ylim((1e-11, 1e1))
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

plt.show()
