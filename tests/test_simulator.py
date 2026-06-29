
from kea.simulator import make_field, make_multifractal_field, create_mask, apply_gaussian, apply_poisson

import numpy as np

import pytest

####################################################################
# Make sure that we can actually generate 1,2,3 dimensional shapes #
####################################################################

def test_field_shape_1():
    FIELD_SHAPE = (16,)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    assert np.shape(field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_field_shape_2():
    FIELD_SHAPE = (16,16)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    assert np.shape(field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_field_shape_3():
    FIELD_SHAPE = (16,16,16)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    assert np.shape(field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_mfield_shape_1():
    FIELD_SHAPE = (16,)
    field = make_multifractal_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', 0.5, {'powerlaw': -5./3.}, 'pure_powerlaw', seeds=(0,1,2))
    assert np.shape(field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_mfield_shape_2():
    FIELD_SHAPE = (16,16)
    field = make_multifractal_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', 0.5, {'powerlaw': -5./3.}, 'pure_powerlaw', seeds=(0,1,2))
    assert np.shape(field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_mfield_shape_3():
    FIELD_SHAPE = (16,16,16)
    field = make_multifractal_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', 0.5, {'powerlaw': -5./3.}, 'pure_powerlaw', seeds=(0,1,2))
    assert np.shape(field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_mask_shape_1():
    MASK_SHAPE = (16,)
    mask = create_mask(MASK_SHAPE)
    assert np.shape(mask) == MASK_SHAPE

def test_mask_shape_2():
    MASK_SHAPE = (16,16)
    mask = create_mask(MASK_SHAPE)
    assert np.shape(mask) == MASK_SHAPE

def test_mask_shape_3():
    MASK_SHAPE = (16,16,16)
    mask = create_mask(MASK_SHAPE)
    assert np.shape(mask) == MASK_SHAPE

def test_noise_shape_1():
    FIELD_SHAPE = (16,)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    noisy_field = apply_gaussian(field, 1.)
    assert np.shape(noisy_field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_noise_shape_2():
    FIELD_SHAPE = (16,16)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    noisy_field = apply_gaussian(field, 1.)
    assert np.shape(noisy_field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_noise_shape_3():
    FIELD_SHAPE = (16,16,16)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    noisy_field = apply_gaussian(field, 1.)
    assert np.shape(noisy_field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_poisson_shape_1():
    FIELD_SHAPE = (16,)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    noisy_field = apply_poisson(field, 1.)
    assert np.shape(noisy_field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_poisson_shape_2():
    FIELD_SHAPE = (16,16)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    noisy_field = apply_poisson(field, 1.)
    assert np.shape(noisy_field) == FIELD_SHAPE, 'Did not produce the expected shape'

def test_poisson_shape_3():
    FIELD_SHAPE = (16,16,16)
    field = make_field(FIELD_SHAPE, {'powerlaw': -5./3.}, 'pure_powerlaw', seed=0)
    noisy_field = apply_poisson(field, 1.)
    assert np.shape(noisy_field) == FIELD_SHAPE, 'Did not produce the expected shape'
