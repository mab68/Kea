
from kea.statistics.statfunc import *

import numpy as np

import pytest

####################################################################
# Make sure that we can actually generate 1,2,3 dimensional shapes #
####################################################################
def test_p_corr_shape_1d():
    N, D = 16, 1
    FIELD_SHAPE = (N,)*D
    lvec, acf = partial_correlation_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == ((N//2)**D,D), 'Lag vectors not expected shape'
    assert np.shape(acf) == ((N//2)**D,1), 'ACF not expected shape'

def test_p_corr_shape_2d():
    N, D = 16, 2
    FIELD_SHAPE = (N,)*D
    lvec, acf = partial_correlation_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == ((N//2)**D,D), 'Lag vectors not expected shape'
    assert np.shape(acf) == ((N//2)**D,1), 'ACF not expected shape'

def test_p_corr_shape_3d():
    N, D = 16, 3
    FIELD_SHAPE = (N,)*D
    lvec, acf = partial_correlation_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == ((N//2)**D,D), 'Lag vectors not expected shape'
    assert np.shape(acf) == ((N//2)**D,1), 'ACF not expected shape'

def test_c_corr_shape_1d():
    N, D = 16, 1
    FIELD_SHAPE = (N,)*D
    lvec, acf = complete_symmetric_correlation_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == FIELD_SHAPE + (D,), 'Lag vectors not expected shape'
    assert np.shape(acf) == FIELD_SHAPE, 'ACF not expected shape'

def test_c_corr_shape_2d():
    N, D = 16, 2
    FIELD_SHAPE = (N,)*D
    lvec, acf = complete_symmetric_correlation_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == FIELD_SHAPE + (D,), 'Lag vectors not expected shape'
    assert np.shape(acf) == FIELD_SHAPE, 'ACF not expected shape'

def test_c_corr_shape_3d():
    N, D = 16, 3
    FIELD_SHAPE = (N,)*D
    lvec, acf = complete_symmetric_correlation_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == FIELD_SHAPE + (D,), 'Lag vectors not expected shape'
    assert np.shape(acf) == FIELD_SHAPE, 'ACF not expected shape'

def test_sf_shape_1d():
    N, D = 16, 1
    FIELD_SHAPE = (N,)*D
    lvec, acf = structure_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == ((N//2)**D,D), 'Lag vectors not expected shape'
    assert np.shape(acf) == ((N//2)**D,1), 'SF not expected shape'

def test_sf_shape_2d():
    N, D = 16, 2
    FIELD_SHAPE = (N,)*D
    lvec, acf = structure_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == ((N//2)**D,D), 'Lag vectors not expected shape'
    assert np.shape(acf) == ((N//2)**D,1), 'SF not expected shape'

def test_sf_shape_3d():
    N, D = 16, 3
    FIELD_SHAPE = (N,)*D
    lvec, acf = structure_function(np.ones(FIELD_SHAPE))
    assert np.shape(lvec) == ((N//2)**D,D), 'Lag vectors not expected shape'
    assert np.shape(acf) == ((N//2)**D,1), 'SF not expected shape'

#######################
# Test implementation #
#######################
def test_sf_increment():
    N, D = 16, 1
    field = np.array(np.arange(0, N))
    lvec, sf = structure_function(field)
    assert sf[0] == 0., 'SF did not produce the correct value'
    assert sf[1] == 1., 'SF did not produce the correct value'

def test_corr_1d():
    VAL = 5.
    N, D = 16, 1
    field = VAL*np.ones((N,)*D)
    lvec, acf = partial_correlation_function(field)
    assert np.allclose(acf, (VAL**2)*np.ones(((N//2)**D,1))), 'ACF did not produce the correct value'

def test_corr_2d():
    VAL = 5.
    N, D = 16, 2
    field = VAL*np.ones((N,)*D)
    lvec, acf = partial_correlation_function(field)
    assert np.allclose(acf, (VAL**2)*np.ones(((N//2)**D,1))), 'ACF did not produce the correct value'

def test_corr_3d():
    VAL = 5.
    N, D = 16, 3
    field = VAL*np.ones((N,)*D)
    lvec, acf = partial_correlation_function(field)
    assert np.allclose(acf, (VAL**2)*np.ones(((N//2)**D,1))), 'ACF did not produce the correct value'
