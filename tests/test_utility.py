
from kea.utils import fitting, functions, geometry, get_calculation_mode, set_calculation_mode, compute_config, CalculationMode

import numpy as np

import pytest

def test_sphere_1d():
    D = 1
    r, db = 1000., 1.
    r2 = r + db
    assert np.isclose(2., geometry.volume_hypersphere(r2, D) - geometry.volume_hypersphere(r, D)), 'Incorrect shell volume'

def test_sphere_2d():
    D = 2
    r, db = 1000., 1.
    r2 = r + db
    r_center = (r + r2)/2.
    assert np.isclose(2.*np.pi*r_center, geometry.volume_hypersphere(r2, D) - geometry.volume_hypersphere(r, D)), 'Incorrect shell volume'

def test_sphere_3d():
    D = 3
    r, db = 1000., 1.
    r2 = r + db
    r_center = (r + r2)/2.
    assert np.isclose(4.*np.pi*r_center**2, geometry.volume_hypersphere(r2, D) - geometry.volume_hypersphere(r, D)), 'Incorrect shell volume'

def test_compute_config():
    set_calculation_mode(use_cpu=False, use_gpu=True)
    assert (False, True, False) == get_calculation_mode(), 'Did not properly update compute config'
    assert (CalculationMode.USE_GPU,) == compute_config.CALCULATION_MODE_FLAGS, 'Did not properly update compute config'
    set_calculation_mode(use_cpu=True, use_gpu=False)
    assert (True, False, False) == get_calculation_mode(), 'Did not properly update compute config'
    assert (CalculationMode.USE_CPU,) == compute_config.CALCULATION_MODE_FLAGS, 'Did not properly update compute config'
    set_calculation_mode(use_cpu=True, use_gpu=False, do_distribute=True)
    assert (True, False, True) == get_calculation_mode(), 'Did not properly update compute config'
    assert (CalculationMode.USE_CPU,CalculationMode.DO_DISTRIBUTE) == compute_config.CALCULATION_MODE_FLAGS, 'Did not properly update compute config'

def test_physdims():
    geometry.DEFAULT_PHYS_SCALE = 1.
    assert geometry.DEFAULT_PHYS_SCALE == 1., 'Did not update physical scale'

def test_powerlaw():
    k = np.linspace(1., 10., 1000)
    E = functions.pure_powerlaw(k, powerlaw=2.)
    est_pow = fitting.get_local_powerlaw(k, E, x_log=True)[1]
    assert np.allclose(2.*np.ones(len(k)), est_pow), 'Did not estimate correct power-law'
