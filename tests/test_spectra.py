
from kea.statistics.spectra import *

import numpy as np

import pytest


####################################################################
# Make sure that we can actually generate 1,2,3 dimensional shapes #
####################################################################

def test_fft_shape_1d():
    N, D = 16, 1
    FIELD_SHAPE = (N,)*D
    kvec, fek = fourier_modal_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (D, N), 'Wavenumbers not expected shape'
    assert np.shape(fek) == FIELD_SHAPE, 'Modal spectrum not expected shape'

def test_fft_shape_2d():
    N, D = 16, 2
    FIELD_SHAPE = (N,)*D
    kvec, fek = fourier_modal_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (D, N), 'Wavenumbers not expected shape'
    assert np.shape(fek) == FIELD_SHAPE, 'Modal spectrum not expected shape'

def test_fft_shape_3d():
    N, D = 16, 3
    FIELD_SHAPE = (N,)*D
    kvec, fek = fourier_modal_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (D, N), 'Wavenumbers not expected shape'
    assert np.shape(fek) == FIELD_SHAPE, 'Modal spectrum not expected shape'

def test_bt_shape_1d():
    N, D = 16, 1
    FIELD_SHAPE = (N,)*D
    kvec, fek = bt_modal_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (D, N), 'Wavenumbers not expected shape'
    assert np.shape(fek) == FIELD_SHAPE, 'Modal spectrum not expected shape'

def test_bt_shape_2d():
    N, D = 16, 2
    FIELD_SHAPE = (N,)*D
    kvec, fek = bt_modal_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (D, N), 'Wavenumbers not expected shape'
    assert np.shape(fek) == FIELD_SHAPE, 'Modal spectrum not expected shape'

def test_bt_shape_3d():
    N, D = 16, 3
    FIELD_SHAPE = (N,)*D
    kvec, fek = bt_modal_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (D, N), 'Wavenumbers not expected shape'
    assert np.shape(fek) == FIELD_SHAPE, 'Modal spectrum not expected shape'

def test_dog_shape_1d():
    N, D = 16, 1
    FIELD_SHAPE = (N,)*D
    kvec, fek = dog_averaged_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (N//2-1,), 'Wavenumbers not expected shape'
    assert np.shape(fek) == (N//2-1,), 'Spectrum not expected shape'

def test_dog_shape_2d():
    N, D = 16, 2
    FIELD_SHAPE = (N,)*D
    kvec, fek = dog_averaged_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (N//2-1,), 'Wavenumbers not expected shape'
    assert np.shape(fek) == (N//2-1,), 'Spectrum not expected shape'

def test_dog_shape_3d():
    N, D = 16, 3
    FIELD_SHAPE = (N,)*D
    kvec, fek = dog_averaged_spectrum(np.ones(FIELD_SHAPE))
    assert np.shape(kvec) == (N//2-1,), 'Wavenumbers not expected shape'
    assert np.shape(fek) == (N//2-1,), 'Spectrum not expected shape'


