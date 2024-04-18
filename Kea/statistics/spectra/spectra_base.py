
"""
spectra_base.py

Provides universal spectrum methods

Functions
---------
calculate_integrated_spectrum\n
translate_spectrum\n
spectrum_integrate\n
wavenumber_mesh\n
"""

import numpy as np


from .. import statistics_base


def calculate_spectrum(ar1, ar2=None, method='periodogram', spec_type='omni', grid_dims=None, phys_dims=None, **int_kwargs):
    """calculate_spectrum(ar1, ar2, method, spec_type, grid_dims, phys_dims, **int_kwargs)
    
    Calculates the integrated spectrum (cross-spectrum if `ar2` is defined) using the requested methods

    Returns:
        ki (np.ndarray): Binned wavenumbers
        feki (np.ndarray): Binned power spectrum
        wi (np.ndarray): Bin widths
    """
    D = ar1.ndim
    if method == 'periodogram':
        from . import per_spectra as periodogram
        st = spec_type
        if spec_type == 'amplitude':
            st = 'omni'
        kD, fekD = periodogram.modal_spectrum(ar1, ar2, phys_dims=phys_dims)
        ki, feki, wi = spectrum_integrate(kD, fekD, spec_type=st, phys_dims=phys_dims, **int_kwargs)
        if spec_type == 'amplitude':
            bin_loc = int_kwargs.get('bin_loc', 'center')
            feki = translate_spectrum(ki, feki, wi, st, spec_type, grid_dims, phys_dims, D, bin_loc == 'center')
        return ki, feki, wi

    raise NotImplementedError('Not implemented method %s' % method)

def translate_spectrum(b, fek, db, orig_type, new_type, grid_dims, phys_dims, orig_dim=None, bin_center=True):
    """translate_spectrum(k, fek, orig_type, new_type)
    
    Transforms a binned spectrum of `orig_type` to `new_type`

    Args:
        k (np.array): Binned wavenumbers
        fek (np.ndarray): Binned 1D spectrum of `orig_type`
        orig_type (str): String indicating the type of `fek` i.e. 'modal'
        new_type (str): String indicating output type i.e. 'omni'
        orig_dim (int): Original dimension of `fek`, for translating binned spectra
    Returns:
        np.ndarray: New binned 1D spectrum of `new_type`
    """
    dx = [phys_dims[i]/grid_dims[i] for i in range(len(grid_dims))]
    dk = [2.*np.pi/L for L in phys_dims]

    # The amplitude calculation is the same for each dimension, if using the omni spectrum
    if new_type == 'amplitude':
        if orig_type != 'omni':
            fek = translate_spectrum(b, fek, orig_type, 'omni')
        return np.sqrt(b * fek)
    if orig_type == 'amplitude':
        return translate_spectrum(b, fek**2/b, 'omni', new_type)

    # 2D and 3D jacobians/shell areas are different
    if bin_center:
        jacob = statistics_base.kern_center(b, orig_dim, db, dk)
    else:
        jacob = statistics_base.kern(b, orig_dim, db, dk)

    # modal -> omni: multiply by the shell areas
    if orig_type == 'modal' and new_type == 'omni':
        return jacob * fek
    # omni -> modal spectrum: divide by the shell areas
    if orig_type == 'omni' and new_type == 'modal':
        return fek / jacob
    raise ValueError('Bad original (%s)->new spectrum (%s) conversion' % (orig_type, new_type))

def spectrum_integrate(kvec, mspec, spec_type='omni', lenn=None, **kwargs):
    """spectrum_integrate(kvec, mspec, spec_type)

    Calculates an integrated spectrum from a provided ND modal spectrum `mspec`.

    The omni-spectrum is summed over spherical annuli/shells. For non 3 dimensional arrays,
        take care of what your expected power spectrum will be\n
        as taking polar coordinates will provide different factors of `k`\n
        i.e. for two dimensions $\\varepsilon(k) = 2 pi k E(k)$.

    The modal-spectrum is averaged over spherical annuli/shells.
        Like the omni-spectrum, take care of what your expected power spectrum will be\n
        as $E(k) = \\varepsilon(k)/2pi k$ for 2D modal spectrum.

    The reduced-spectrum is summed over N-1 dimensions.

    Args:
        kvec (tuple): Tuple of wavenumbers
        mspec (np.ndarray): The full N-dimensional modal spectrum
        spec_type (str): String indicating what specific spectra type to return.
            omni: Summed modal spectrum
            modal: Averaged N-dimensional modal spectrum
            reduced: Summed modal spectrum along N-1 axes as a function of `axis` wavenumbers
                axis (int): The axis wavenumbers to be a function of
            perp: NOT_IMPLEMENTED
        kwargs (dict): Additional parameters to pass onto the binning function
            See `statistics_base.bin_data()`
    Returns:
        bins (np.array): Binned wavenumber array
        spec (np.ndarray): Binned spectrum
        widths (np.ndarray): Bin widths
    """
    kmesh = wavenumber_mesh(kvec, phys_dims=None)
    assert np.shape(kmesh) == np.shape(mspec), 'kvec does not span mspec'
    cut_excess = kwargs.get('cut_excess', True)
    nan_small = kwargs.get('nan_small', False)
    if lenn is None:
        lenn = [2.*np.pi for _ in range(mspec.ndim)]
    # Minimum k is the first non-zero k value = 1*dk
    min_k = np.min([2.*np.pi/lenn[i] for i in range(mspec.ndim)])
    min_bin = kwargs.get('min_bin', min_k)
    max_bin = kwargs.get('max_bin', None)
    bin_loc = kwargs.get('bin_loc', 'true_center')
    norm_bin_size = kwargs.get('norm_bin_size', True)
    log_space = kwargs.get('log_space', False)
    num_bins = kwargs.get('num_bins', None)
    ignore_nan = kwargs.get('ignore_nan', False)
    max_half_bin_width = kwargs.get('max_half_bin_width', None)
    if spec_type == 'omni':
        ## The omni spectrum is the integrated modal spectrum
        norm_bin_size = kwargs.get('norm_bin_size', True)
        bins, ispec, istd = statistics_base.bin_data(kmesh, mspec, bin_func=np.nansum,
            cut_excess=cut_excess, nan_small=nan_small, min_bin=min_bin, bin_loc=bin_loc,
            norm_bin_size=norm_bin_size, log_space=log_space, num_bins=num_bins, ignore_nan=ignore_nan,
            max_bin=max_bin, max_half_bin_width=max_half_bin_width)
        return bins, ispec, istd
    elif spec_type == 'modal':
        ## The 1D modal spectrum is the averaged ND modal spectrum
        norm_bin_size = kwargs.get('norm_bin_size', False)
        bins, ispec, istd = statistics_base.bin_data(kmesh, mspec, bin_func=np.nanmean,
            cut_excess=cut_excess, nan_small=nan_small, min_bin=min_bin, bin_loc=bin_loc,
            norm_bin_size=norm_bin_size, log_space=log_space, num_bins=num_bins, ignore_nan=ignore_nan,
            max_bin=max_bin, max_half_bin_width=max_half_bin_width)
        return bins, ispec, istd
    elif spec_type == 'error':
        bins, ispec, istd = statistics_base.bin_data(kmesh, mspec, bin_func=statistics_base.nanstderr,
            cut_excess=cut_excess, nan_small=nan_small, min_bin=min_bin, bin_loc=bin_loc,
            norm_bin_size=norm_bin_size, log_space=log_space, num_bins=num_bins, ignore_nan=ignore_nan,
            max_bin=max_bin, max_half_bin_width=max_half_bin_width)
        return bins, ispec, istd
    elif spec_type == 'reduced':
        ## The reduced spectrum is the integrated (along N-1 axes) spectrum
        assert 'axis' in kwargs, 'Cannot compute reduced spectra without an axis'
        axis = kwargs['axis']
        assert axis <= len(kvec) - 1, 'Reduced axis not in bounds' 
        kbin = kvec[axis]
        sum_axes = []
        for i in range(len(kvec)):
            if i != axis:
                sum_axes.append(i)
        redspec = np.sum(mspec, axis=tuple(sum_axes))
        redstd = np.var(mspec, axis=tuple(sum_axes))
        return kbin, redspec, redstd
    else:
        raise ValueError('Spectrum type %s is not valid' % spec_type)

def wavenumber_mesh(kvec, phys_dims=None):
    """wavenumber_mesh(kvec)

    Computes the ndimensional wavenumber magnitude mesh

    Args:
        kvec (tuple): Wavenumbers for each basis direction
        phys_dims (tuple,list): If set, we normalize the wavenumber mesh by dk,
            effectively giving the wavenumber indices
    Returns:
        km (np.ndarray): Wavenumber magnitude mesh
    """
    if phys_dims is None:
        dk = np.float64(1.)
    else:
        dk = np.prod([2.*np.pi/L for L in phys_dims])
    kmesh = np.meshgrid(*kvec, indexing='xy') / dk
    km = np.linalg.norm(kmesh, axis=0)
    return km
