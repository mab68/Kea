"""
normalization.py

Implements normalization considerations for different Fourier transform conventions
"""

from ...utils.geometry import TWOPI

from typing import Optional
from enum import IntEnum
import numpy as np

class FourierNorm(IntEnum):
    T = 0
    I = 1
    C = 2

class SpectrumType(IntEnum):
    MODAL = 1
    INTEGRATED = 2
    AVERAGED = 3
    AMPLITUDE = 4




# def transform_spectrum(
#         k: np.ndarray,
#         fek: np.ndarray,
#         dimension: int,
#         current_type: SpectrumType,
#         new_type: SpectrumType) -> np.ndarray:
#     """transform_spectrum(k, fek, orig_dim, current_type, new_type)

#     Converts between different spectrum types e.g., integrated, averaged, amplitude, ...

#     Args:
#         k (np.ndarray): Wavenumber array
#         fek (np.ndarray): (binned) 1D Spectrum of `current_type`
#         dimension (int): Dimension of the original data $D$
#         current_type (SpectrumType): Current spectrum type
#         new_type (SpectrumType): Requested spectrum type
#     Returns:
#         fek (np.ndarray): `fek` in the `new_type` convention
#     """

def convert_normalization_convention(
        kvec: tuple,
        fek: np.ndarray,
        dimension: int,
        current_norm: FourierNorm,
        new_norm: FourierNorm,
        spectrum_type: SpectrumType) -> tuple[tuple, np.ndarray]:
    """convert_normalization_convention(kvec, fek, dimension, current_norm, new_norm, spectrum_type)\n

    Converts between different spectrum normalization conventions

    Args:
        kvec (tuple): Wavenumber array for each dimension: k_x, k_y, ... 
        fek (np.ndarray): Spectrum of `spectrum_type` and in `current_norm` convention
        dimension (int): Dimension of the original data $D$
        current_norm (FourierNorm): Current normalization convention
        new_norm (FourierNorm): Requested normalization convention
        spectrum_type (SpectrumType): The type of spectrum e.g., modal, amplitude, etc.
    Returns:
        kvec (tuple), fek (np.ndarray): `kvec` and `fek` in the `new_norm` convention
    """
    # Not converting anything
    if current_norm == new_norm:
        return kvec, fek

    # Convert the wavenumber convention
    # These are the twopi wavenumber conventions
    if new_norm == FourierNorm.T or new_norm == FourierNorm.C:
        if current_norm == FourierNorm.I:
            kvec = tuple([TWOPI*k for k in kvec])
    # These are the unity wavenumber conventions
    if new_norm == FourierNorm.I:
        if current_norm == FourierNorm.T or current_norm == FourierNorm.C:
            kvec = tuple([k/TWOPI for k in kvec])

    # Generate conversion tables,
    # The factor to multiply `fek` with to get the new spectrum
    if spectrum_type == SpectrumType.MODAL or spectrum_type == SpectrumType.AVERAGED:
        CONVERSION_TABLE = {
            (FourierNorm.T, FourierNorm.I): TWOPI**(dimension),
            (FourierNorm.T, FourierNorm.C): TWOPI**(dimension),
            (FourierNorm.I, FourierNorm.C): 1.,
        }
    elif spectrum_type == SpectrumType.INTEGRATED:
        CONVERSION_TABLE = {
            (FourierNorm.T, FourierNorm.I): TWOPI**(1),
            (FourierNorm.T, FourierNorm.C): TWOPI**(dimension),
            (FourierNorm.I, FourierNorm.C): TWOPI**(1-dimension),
        }
    elif spectrum_type == SpectrumType.AMPLITUDE:
        CONVERSION_TABLE = {
            (FourierNorm.T, FourierNorm.I): 1.,
            (FourierNorm.T, FourierNorm.C): TWOPI**(dimension),
            (FourierNorm.I, FourierNorm.C): TWOPI**(dimension/2.),
        }
    else:
        raise ValueError('Invalid spectrum type: {spectrum_type}')

    invert = False
    conv_tuple = (current_norm, new_norm)
    # If not in the table, then the inverted factor should be
    if conv_tuple not in CONVERSION_TABLE:
        conv_tuple = (new_norm, current_norm)
        invert = True

    factor = CONVERSION_TABLE[conv_tuple]
    if invert:
        factor = 1./factor
    fek = fek * factor

    return kvec, fek

