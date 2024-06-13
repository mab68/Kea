
import numpy as np

from Kea.statistics.spectra import per_spectra, spectra_base
from Kea.statistics.statfunc import statfunc_base, strfn
from Kea.statistics import statistics_base
from Kea.utils import plotting, data_writer

import h5py as h5

import time

## CALCULATE APERIODIC SF

file = h5.File('SF_TEST/2_512/in/T.Fr.h5', 'r')
field = np.array(file['T.Fr'])

L = 2.*np.pi
N = 512
D = 2
grid_dims = [N for _ in range(D)]
phys_dims = [L for _ in range(D)]
dx = L/N

lv_f = statfunc_base.get_all_lagvecs([n for n in grid_dims])

start = time.time()

sf_f = strfn.process_lags(field, field, lv_f, lenn=phys_dims, shape=[n for n in grid_dims], orders=[2], periodic=False)[0]

end = time.time()
print(end - start)

file3 = data_writer.save_h5(sf_f, 'SF_TEST/2_512/out/SF_Grid_scalar_mine.h5', 'SF_Grid_scalar2')
file3.close()
file.close()

## CALCULATE PERIODIC SF

file = h5.File('SF_TEST/2_512/in/T.Fr.h5', 'r')
field = np.array(file['T.Fr'])

L = 2.*np.pi
N = 512
D = 2
grid_dims = [N for _ in range(D)]
phys_dims = [L for _ in range(D)]
dx = L/N

lv_f = statfunc_base.get_all_lagvecs([n for n in grid_dims])

start = time.time()

sf_f = strfn.process_lags(field, field, lv_f, lenn=phys_dims, shape=[n for n in grid_dims], orders=[2], periodic=True)[0]

end = time.time()
print(end - start)

file3 = data_writer.save_h5(sf_f, 'SF_TEST/2_512/out/SF_Grid_scalar_mine_periodic.h5', 'SF_Grid_scalar2')
file3.close()
file.close()
