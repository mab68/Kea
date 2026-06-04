"""
Provides common mathematical formulae, primarily for power-spectra.
"""

import numpy as np

def get_raise_parameters(
        names: tuple[str,...],
        kwargs: dict[str,float]) -> tuple:
    """get_raise_parameters(names, kwargs)\n

    Checks in `kwargs` for parameters that are not None. If they are
    then raise a ValueError.

    Args:
        names (tuple): List of parameter names
        kwargs (dict): Dictionary of parameters
    
    Returns:
        tuple: Tuple of parameters matching the order in `names`
    """
    parameters = []
    for key in names:
        value = kwargs.get(key, None)
        if value is None:
            raise ValueError('Expected valid parameter: `%s`' % key)
        parameters.append(value)
    return tuple(parameters)

def pure_powerlaw(
        grid: np.ndarray,
        **kwargs: float):
    """pure_powerlaw(grid, **kwargs)\n

    Generates a pure power-law function with index `powerlaw`.
    Note: It is unphysical to have a pure power-law function.
    Additionally, for discrete/finite data, there MUST be
    (an effective) break scale,
    thus the generated function is not truly a pure power-law.

    Args:
        grid (np.ndarray): Value grid
        powerlaw (float): Pure power-law index

    Returns:
        np.ndarray: Evaluated equation on `grid`
    """
    # Make sure we are given a powerlaw and 2 break scales
    powerlaw = get_raise_parameters(('powerlaw',), kwargs)
    result = grid**(powerlaw)
    # Set unphysical values to 0
    result[~np.isfinite(result)] = 0.
    return result

def broken_powerlaw(
        grid: np.ndarray,
        **kwargs: float):
    """broken_powerlaw(grid, **kwargs)\n
    
    Generates a broken power-law that has a smooth transition.
    Provide 2 powerlaws (`powerlaw_1`, `powerlaw_2`), a break scale (`break`),
    and transition/smoothness factor (`delta`).

    Args:
        grid (np.ndarray): Value grid
        powerlaw_1 (float): First power-law index
        powerlaw_2 (float): Second power-law index
        break_scale (float): Scale where the transition occurs
        smoothness (None|float): Factor that smooths the transition (increasing the distance between the pure power-law regions)
    
    Returns:
        np.ndarray: Evaluated equation on `grid`
    """
    powerlaw_1, break_scale, powerlaw_2 = get_raise_parameters(('powerlaw_1', 'break_scale', 'powerlaw_2'), kwargs)
    smoothness = kwargs.get('smoothness', 1.)
    result = (grid/break_scale)**(-powerlaw_1) * (0.5 * (1. + (grid/break_scale)**(1./smoothness)))**((powerlaw_1 - powerlaw_2)*smoothness)
    # Set unphysical values to 0
    result[~np.isfinite(result)] = 0.
    return result

def powerlaw_with_exponentials(
        grid: np.ndarray,
        **kwargs: float):
    """powerlaw_with_exponentials(grid, **kwargs)\n

    Generates a power-law region in-between exponential growth and decay regions.
    Provide a powerlaw (`powerlaw`) and 2 break scales (`break_1`, `break_2`).
    Additional parameters can be provided as `exponential_factor_1`, `exponential_factor_2`
    which gives the powers of the exponential terms (this defaults to 2).

    Args:
        grid (np.ndarray): Value grid
        powerlaw (float): Pure power-law index
        break_1 (float): Scale that transitions from exponential growth to power-law
        break_2 (float): Scale that transitions from power-law to exponential decay
        exp_factor_1 (None|float): Additional factor; makes the exponential grow faster
        exp_factor_2 (None|float): Additional factor; makes the exponential decay faster

    Returns:
        np.ndarray: Evaluated equation on `grid`
    """
    # Make sure we are given a powerlaw and 2 break scales
    powerlaw, break_1, break_2 = get_raise_parameters(('powerlaw', 'break_1', 'break_2'), kwargs)
    # Additional factors that can be provided, otherwise default to 2
    exp_factor_1 = kwargs.get('exponential_factor_1', 2)
    exp_factor_2 = kwargs.get('exponential_factor_2', 2)
    result = grid**(powerlaw) * np.exp(-(break_1/grid)**exp_factor_1) * np.exp(-(grid/break_2)**exp_factor_2)
    # Set unphysical values to 0
    result[~np.isfinite(result)] = 0.
    return result
