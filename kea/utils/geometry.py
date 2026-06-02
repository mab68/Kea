"""
geometry.py

Provides common methods associated with the geometry e.g., grid size calculation etc.

Functions
---------
- validate_shapes
- default_physdims
- volume_hypersphere
- get_all_lagvecs
- get_dxdk
- get_kvec
"""

from typing import Optional
from functools import wraps
import inspect
import numpy as np
from scipy.special import gamma

# The default physical scale that is automatically applied if none are given for each function call
# It is assumed that this applies equally for each cartesian axes
TWOPI = 2.*np.pi
DEFAULT_PHYS_SCALE = 2.*np.pi

def validate_shapes(*param_names: str):
    """
    Decorator to validate that specified NumPy array parameters:
    1. Are not None.
    2. Have at least 1 dimension.
    3. Share the exact same shape with each other.
    """
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Bind args and kwargs to their named parameters
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            extracted_arrays = []
            
            # Look up only the parameters requested by the user
            for name in param_names:
                if name not in bound_args.arguments:
                    raise AttributeError(f"Decorator targeted '{name}', but it's not in the function signature.")
                
                val = bound_args.arguments[name]

                # If a secondary field like field_b is optional and remains None, skip it
                if val is None:
                    continue
                
                if not isinstance(val, np.ndarray):
                    raise TypeError(f"Parameter '{name}' must be a numpy ndarray.")
                
                extracted_arrays.append((name, val))
            
            if len(extracted_arrays) == 0:
                raise ValueError('No valid arrays provided')
            
            # --- Perform Assertions ---
            if extracted_arrays:
                # 1. Check dimensions on the first specified array
                first_name, first_arr = extracted_arrays[0]
                if first_arr.ndim < 1:
                    raise ValueError(f"Field '{first_name}' has no dimensions.")
                
                # 2. Compare shapes of all subsequent specified arrays
                for next_name, next_arr in extracted_arrays[1:]:
                    if first_arr.shape != next_arr.shape:
                        raise ValueError(
                            f"Shape mismatch: '{first_name}' {first_arr.shape} "
                            f"does not match '{next_name}' {next_arr.shape}."
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator

def default_physdims(array_param_name: str):
    """
    Decorator to automatically initialize physics dimensions to `DEFAULT_PHYS_SCALE` if they are not initialized
    multiplied by the number of dimensions of a target array.
    """
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Extract the array and current phys_dims
            array_field = bound_args.arguments.get(array_param_name)
            current_phys_dims = bound_args.arguments.get('phys_dims')
            
            # Only calculate defaults if phys_dims is explicitly None
            # AND we have a valid array to calculate ndim from
            if current_phys_dims is None:
                # Get number of dims from another provided parameter
                if isinstance(array_field, np.ndarray):
                    num_dims = array_field.ndim
                elif isinstance(array_field, tuple):
                    num_dims = len(array_field)
                else:
                    raise NotImplementedError('No other valid array provided')
                # Inject the computed tuple back into the function arguments
                bound_args.arguments['phys_dims'] = tuple([DEFAULT_PHYS_SCALE for _ in range(num_dims)])

            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator

def check_square_dims():
    """
    Decorator to automatically check if `grid_dims` and `phys_dims` are equal in dimension
    and have the same values in each dimension (respectively)
    """
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Extract the array and current phys_dims
            grid_dims = bound_args.arguments.get('grid_dims')
            phys_dims = bound_args.arguments.get('phys_dims')

            if len(grid_dims) != len(phys_dims):
                raise ValueError('Provided dimensions not consistent')
            dimension = len(grid_dims)
            if grid_dims != (grid_dims[0],) * dimension:
                raise ValueError('Grid dimensions are not square')
            if phys_dims != (phys_dims[0],) * dimension:
                raise ValueError('Physical dimensions are not square')

            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator

def volume_hypersphere(
        radius: float,
        dimension: int | float) -> float:
    """volume_hypersphere(radius, dimension)\n
    
    Calculates the volume of a $D$-dimensional hypersphere with radius $r$.

    Args:
        radius (float): Radius of the hypersphere
        dimension (int): (Euclidean) dimension of the hypersphere
    Returns:
        volume (float): Volume of the hypersphere
    """
    D = float(dimension)
    return (np.pi**(D/2.) / gamma(D/2. + 1.)) * radius**D

def get_mesh(
        axes_vec: tuple[np.ndarray,...]) -> np.ndarray:
    """get_mesh(axes_vec)\n

    Computes the n-dimensional (magnitude) mesh from a list of grid axes
    
    Args:
        axes_vec (tuple): axes associated with each dimension
    Returns:
        norm_mesh (np.ndarray): Magnitude mesh
    """
    mesh = np.meshgrid(*axes_vec, indexing='xy')
    norm_mesh = np.linalg.norm(mesh, axis=0)
    return norm_mesh

def get_all_lagvecs(
        grid_dims: tuple[float,...]) -> np.ndarray:
    """get_all_lagvecs(grid_dims)\n
    
    Get all the lagvectors as a list required to compute the positive lag quadrant of the ACF/SF

    Args:
        grid_dims (tuple): Number of grid points in each dimension (N_x, N_y, ...)
    Returns:
        lagvecs (np.ndarray): List of vector indices that represent lag-shifts covering the `grid_dims` domain
    """
    return np.transpose(np.indices(grid_dims).reshape((-1, np.prod(grid_dims))))

@default_physdims('grid_dims')
def get_dxdk(
        grid_dims: tuple[float,...],
        phys_dims: Optional[tuple[float,...]] = None) -> tuple[tuple[float,...], tuple[float,...]]:
    """get_dxdk(grid_dims, phys_dims)\n

    Gets the (constant) discrete increments for the physical and wavenumber space

    Args:
        grid_dims (tuple): Number of grid points in each dimension (N_x, N_y, ...)
        phys_dims (tuple): The physical length scale for each dimension (L_x, L_y, ...)
    Returns:
        dx (tuple): Grid spacing/sampling rate for each dimension e.g., dx = L_x/N_x, ...
        dk (tuple): Wavenumber sampling rate for each dimension e.g., dk_x = 2*pi/L_x, ...
    """
    dx, dk = [], []
    for i, N in enumerate(grid_dims):
        dx.append(phys_dims[i]/N)
        dk.append(TWOPI/(N*dx[i]))
    return tuple(dx), tuple(dk)

@default_physdims('grid_dims')
def get_kvec(
        grid_dims: tuple[float,...],
        phys_dims: Optional[tuple[float,...]] = None) -> tuple[np.ndarray,...]:
    """get_kvec(grid_dims, phys_dims)\n

    Gets wavenumbers for each dimension: k_x, k_y, ...
    
    Args:
        grid_dims (tuple): Number of grid points in each dimension (N_x, N_y, ...)
        phys_dims (tuple): The physical length scale for each dimension (L_x, L_y, ...)
    Returns:
        kvec (tuple[np.ndarray]): Wavenumber array for each dimension: k_x, k_y, ...
    """
    dx, _ = get_dxdk(grid_dims, phys_dims)
    kvec = []
    for i, N in enumerate(grid_dims):
        kvec.append(np.fft.fftshift(np.fft.fftfreq(N)*TWOPI/dx[i]))
    return tuple(kvec)

