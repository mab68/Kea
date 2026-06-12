"""
Provides functions to generate noise -- in the sense of telescope image sensing noise, rather than the noise fields in synthesize.

In other words, apply noise to an observation.
"""

from typing import Optional

import numpy as np


def apply_poisson(
        field: np.ndarray,
        target_SNR: float) -> np.ndarray:
    """apply_poisson(field)\n

    Applies Poisson noise to each pixel of the field, where each pixel determines
    the expected number of events in a fixed-time interval.

    Args:
        field (np.ndarray): Field to apply noise to; Note that it should
            be that field>=0 for every pixel.
        target_SNR (float): Target signal to noise ratio -- technically applies to the
            largest value in `field`

    Returns:
        np.ndarray: Noisy field
    """
    peak_signal = target_SNR**2
    field_min = np.nanmin(field)
    field_max = np.nanmax(field)
    field_range = field_max - field_min
    ## Numpy requires positive values, we rescale the field
    image_scaled = (field - field_min) / field_range * peak_signal
    noisy_image_scaled = np.random.poisson(image_scaled)
    ## Numpy returns a discrete field that represents the counted photons
    noisy_image = (noisy_image_scaled / peak_signal) * field_range + field_min
    ## Convert back to the scale of the original field
    return noisy_image

def apply_gaussian(
        field: np.ndarray,
        standard_deviation: float|np.ndarray,
        mean: Optional[float|np.ndarray]=0.) -> np.ndarray:
    """apply_gaussian(field, standard_deviation, mean)\n

    Adds Gaussian noise to an image with given `mean` and `standard_deviation`.

    Args:
        field (np.ndarray): Field to apply noise to
        standard_deviation (float|np.ndarray): Standard deviation of the Gaussian distribution
            to sample additive noise. This can vary spatially.
        mean (float|np.ndarray): Mean of the Gaussian distribution (default = 0). This
            can vary spatially.

    Returns:
        np.ndarray: Noisy field
    """
    if isinstance(standard_deviation, float):
        standard_deviation = np.ones_like(field)*standard_deviation
    if isinstance(mean, float):
        mean = np.ones_like(field)*mean
    return field + np.random.normal(mean, standard_deviation)

