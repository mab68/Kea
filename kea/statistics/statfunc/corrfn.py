"""
corrfn.py

Calculates the autocorrelation function for arbitrary dimension $D$.
"""

from .statfunc_numba import process_lags, StatMetric
from ...utils.geometry import default_physdims, validate_shapes, get_all_lagvecs

from typing import Optional
import numpy as np
import itertools

def statfunc_biased_acf(
        field_a: np.ndarray,
        field_b: np.ndarray):
    assert np.shape(field_a) == np.shape(field_b), 'Provided fields have different shapes'
    return np.nansum(field_a * field_b)/np.prod(np.shape(field_a))

def statfunc_acf(
        field_a: np.ndarray,
        field_b: np.ndarray):
    assert np.shape(field_a) == np.shape(field_b), 'Provided fields have different shapes'
    return np.nanmean(field_a * field_b)

@validate_shapes('field')
def complete_symmetric_correlation_function(
        field: np.ndarray,
        max_lag: Optional[int] = None,
        longitudinal: Optional[bool] = False,
        biased: Optional[bool] = False) -> tuple[np.ndarray, np.ndarray]:
    """complete_symmetric_correlation_function(partial_acf, grid_size, window)\n
    
    Assumes symmetry and composes the autocorrelation function for some lags
    into the complete lag-space autocorrelation function.

    For D-dimensions there are $2^{D}$ quadrants (2D)/octants (3D) -- orthants. Because
    the ACF has inversion symmetry ($R(r) = R(-r)$), we only need to compute $2^{D-1}$
    orthants and the other half is filled by reversing the computed orthants. We
    use binary numbers to represent the coordinate flips.

    Args:
        field (np.ndarray): Array to calculate the (auto)-structure function
        max_lag (int): Maximum lag (as grid index) to go to
        longitudinal (bool): Whether to calculate along 1D
        biased (bool): If true, apply biased normalization
    Returns:
        lags (np.ndarray): The array of lags
        acf (np.ndarray): Computed ACF. [Note, this might need to be reshaped depending on your requirements]
    """
    # If looking for the longitudinal/transverse SF, then just iterate over "1D" lags
    D = field.ndim
    if longitudinal:
        D = 1

    stat_metric = StatMetric.CORR
    if biased:
        stat_metric  = StatMetric.BIAS_CORR

    # Generate the appropriate lags
    N = np.min(np.shape(field))
    if max_lag is None:
        max_lag = N//2 + 1
    assert max_lag < N, 'Wanting maximum lag outside available range'
    shape_nd = (N,)*D
    mid = N//2
    lagshape_nd = (max_lag,)*D
    lags = get_all_lagvecs(lagshape_nd)
    print(shape_nd, lagshape_nd)

    ## Initialize the full N_dimensional ACF array
    Q_full = np.zeros(shape_nd)

    ## Generate ALL orthant bitmask(s)
    ## 0 means keep input, 1 means flip input
    all_orthants = list(itertools.product([0,1], repeat=D))
    ## Sort for unique orthants by taking half
    ## We elect for all orthants that have 0 in the first axis
    unique_orthants = [orth for orth in all_orthants if orth[0] == 0]

    for orthant in unique_orthants:

        # 1. Prepare the field by flipping the appropriate axes (where bitmask == 1)
        flipped_field = field.copy()
        flip_axes = tuple([axis for axis, flip in enumerate(orthant) if flip == 1])
        if flip_axes:
            flipped_field = np.flip(flipped_field, axis=flip_axes)

        # 2. Compute the partial ACF
        Q_part = process_lags(np.ascontiguousarray(flipped_field), lags, stat_metric)[:,0]
        Q_part = Q_part.reshape(lagshape_nd)

        # 3. Stitch this orthant into the complete ACF
        # Strip boundary edges due to size mismatches
        strip = tuple(slice(None, -1) if flip == 0 else slice(None) for flip in orthant)
        # Construct target slice (how we insert Q_part into Q_full)
        target_slices = tuple(slice(mid, None) if flip == 0 else slice(None, mid+1) for flip in orthant)
        # Construct how we read Q_part
        read_slices = tuple(slice(None) if flip == 0 else slice(None, None, -1) for flip in orthant)
        # Copy the partial into full
        Q_full[target_slices] = Q_part[strip][read_slices]

        # 4. Enforce inversion symmetry by filling in the spatially opposite quadrant
        # This orthant has bits inverted
        opp_orthant = tuple(1 - flip for flip in orthant)
        strip = tuple(slice(None, -1) if flip == 0 else slice(None) for flip in opp_orthant)
        target_slices = tuple(slice(mid, None) if flip == 0 else slice(None, mid+1) for flip in opp_orthant)
        read_slices = tuple(slice(None) if flip == 0 else slice(None, None, -1) for flip in opp_orthant)
        # Copy partial into opposite quadrant of full
        Q_full[target_slices] = Q_part[strip][read_slices]

    return lags, Q_full

@validate_shapes('field')
def partial_correlation_function(
        field: np.ndarray,
        lags: Optional[np.ndarray] = None,
        max_lag: Optional[int] = None,
        longitudinal: Optional[bool] = False,
        biased: Optional[bool] = False) -> tuple[np.ndarray, np.ndarray]:
    """correlation_function(field, max_lag, longitudinal, biased)\n

    Calculate a (partial) auto-correlation function

    Args:
        field (np.ndarray): Array to calculate the (auto)-structure function
        max_lag (int): Maximum lag (as grid index) to go to
        longitudinal (bool): Whether to calculate along 1D
        biased (bool): If true, apply biased normalization
    Returns:
        lags (np.ndarray): The array of lags
        acf (np.ndarray): Computed ACF. [Note, this might need to be reshaped depending on your requirements]
    """
    # If looking for the longitudinal/transverse SF, then just iterate over "1D" lags
    D = field.ndim
    if longitudinal:
        D = 1

    # Generate the appropriate lags
    if max_lag is None:
        max_lag = np.min(np.shape(field))//2
    lagvec_shape = (max_lag,)*D
    lags = get_all_lagvecs(lagvec_shape)

    # Calculate SF
    stat_metric = StatMetric.CORR
    if biased:
        stat_metric = StatMetric.BIAS_CORR
    acf = process_lags(field, lags, stat_metric)
    return lags, acf
