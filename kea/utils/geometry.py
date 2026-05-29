
from typing import Optional
from functools import wraps
import inspect
import numpy as np

# The default physical scale that is automatically applied if none are given for each function call
# It is assumed that this applies equally for each cartesian axes
DEFAULT_PHYS_SCALE = 2.*np.pi

def get_all_lagvecs(
        shape: tuple) -> np.ndarray:
    """get_all_lagvecs(shape)\n
    
    Get all the lagvectors as a list required to compute the positive lag quadrant of the ACF/SF
    """
    return np.transpose(np.indices(shape).reshape((-1, np.prod(shape))))

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
    Decorator to automatically initialize physics dimensions to 2*pi if they are not initialized
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
            if current_phys_dims is None and isinstance(array_field, np.ndarray):
                num_dims = array_field.ndim
                computed_dims = tuple([DEFAULT_PHYS_SCALE for _ in range(num_dims)])
                
                # Inject the computed tuple back into the function arguments
                bound_args.arguments['phys_dims'] = computed_dims
                
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator
