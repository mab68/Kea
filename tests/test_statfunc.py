
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

