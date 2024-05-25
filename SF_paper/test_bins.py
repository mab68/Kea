
import numpy as np

from Kea.statistics import statistics_base
from Kea.statistics.statfunc import statfunc_base

D = 3
N = 1024
L = 2.*np.pi

grid_dims = [N for _ in range(D)]


min_bin, max_bin = 0., L

linear_all = statistics_base.get_bins(0, L, N)
print(len(linear_all[0]))

lv_p = statfunc_base.get_all_lagvecs([n+1 for n in grid_dims])
print(len(lv_p))
