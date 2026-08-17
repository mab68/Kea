
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

#######################################################
# Test spectral densities provide appropriate results #
#######################################################
def test_fft_noise():
    field = np.random.normal(0., 1., (256,))
    kvec, fek = fourier_modal_spectrum(field)
    assert np.isclose(np.nanmean(fek), np.nanmean(np.abs(field)**2)*(1./256.)), 'Not expected spectra'

def test_fft_sine():
    x = np.linspace(0., 100., 1000)*(2.*np.pi/100.)
    field = np.sin(100.*x)
    kvec, fek = fourier_modal_spectrum(field)
    assert np.argmax(fek[500:]) == 100., 'Not expected spectra'

#######################################################
# Test for bugfixes                                   #
#######################################################
def test_exposure_bugfix():
    N = 256
    D = 1
    field = np.ones((N,)*D)
    exposure = np.ones_like(field)
    kvec, fek = dog_averaged_spectrum(field, exposure)
    assert np.shape(kvec) == (N//2-1,), 'Wavenumbers not expected shape'
    assert np.shape(fek) == (N//2-1,), 'Spectrum not expected shape'

