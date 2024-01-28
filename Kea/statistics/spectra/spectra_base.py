
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


def calculate_integrated_spectrum(arz, method='periodogram', spec_type='omni', lenn=None, convert=False, lags=None, int_kwargs={}):
    """calculate_spectrum(arz, method, spec_type)

    Calculates the integrated spectrum of all the arrays in `arz` using the requested methods

    Args:
        arz (tuple): Tuple of arrays to compute the integrated spectra of each
        method (str): Default periodogram. What spectrum method to use
            periodogram: Use the periodogram method
            correlogram: Use the correlogram method
            strfn: Use the equivalent structure function method
            arevalo: Use the Arevalo method
            flatsky: Use the `pymaster` flatsky method
        spec_type (str): Default omni. What method of integration to use
            omni: Calculate the omni-spectrum
            modal: Calculate the 1-D modal-spectrum
            amplitude: Calculate the amplitude
        convert (bool): If true, for strfn and correlogram convert the array to the respective forms
        lenn (tuple): Computational domain
        int_kwargs (dict): Kwargs to pass onto the integration method
    Returns:
        kx (np.ndarray): 1 dimensional Wavenumber array
        fek (np.ndarray): Individual integrated spectra for each `ar` in `arz`
        fektrace (np.ndarray): Summation of the individual integrated spectra
    """
    if method == 'periodogram' or method == 'correlogram':
        if method == 'periodogram':
            from . import per_spectra as spectra
        if method == 'correlogram':
            from . import corr_spectra as spectra
            from ..statfunc import corr
        st = spec_type
        if spec_type == 'amplitude':
            st = 'omni'
        specs = []
        for ar in arz:
            if method == 'correlogram' and convert:
                ar = corr.autocovariance_fft(ar.copy(), shape=[N//2 for N in ar.shape])
            kxn, fxn = spectra.modal_spectrum(ar, lenn=lenn)
            kx, fx, _ = spectrum_integrate(kxn, fxn, spec_type=st, lenn=lenn, **int_kwargs)
            if spec_type == 'amplitude':
                fx = translate_spectrum(kx, fx, st, spec_type)
            specs.append(fx)
        return kx, *specs, np.sum(specs, axis=0)
    elif method == 'strfn':
        #assert spec_type == 'omni', 'Can only calculate the omni-spectrum for %s method' % method
        from . import strfn_spectra as spectra
        from ..statfunc import corr, strfn
        specs = []
        for ar in arz:
            if convert:
                acf = corr.autocovariance_fft(ar.copy(), shape=[N//2 for N in ar.shape])
                ar = strfn.autocorrelation_to_strfn(acf)
            kx, fx = spectra.omni_spectrum(ar, lenn=lenn)
            if spec_type != 'omni':
                fx = translate_spectrum(kx, fx, 'omni', spec_type, orig_dim=ar.ndim)
            specs.append(fx)
        return kx, *specs, np.sum(specs, axis=0)
    elif method == 'strfn2':
        #assert spec_type == 'omni', 'Can only calculate the omni-spectrum for %s method' % method
        from . import strfn_spectra as spectra
        from ..statfunc import corr, strfn
        specs = []
        for ar in arz:
            if convert:
                acf = corr.autocovariance_fft(ar.copy(), shape=[N//2 for N in ar.shape])
                ar = strfn.autocorrelation_to_strfn(acf)
            kx, fx = spectra.omni_spectrum_2(ar, lenn=lenn)
            if spec_type != 'omni':
                fx = translate_spectrum(kx, fx, 'omni', spec_type, orig_dim=ar.ndim)
            specs.append(fx)
        return kx, *specs, np.sum(specs, axis=0)
    elif method == 'arevalo':
        from . import arevalo_spectra as spectra
        specs = []
        for ar in arz:
            kx, fx = spectra.modal_spectrum(ar, lenn=lenn, lags=lags)
            if spec_type != 'modal':
                fx = translate_spectrum(kx, fx, 'modal', spec_type)
            specs.append(fx)
        return kx, *specs, np.sum(specs, axis=0)
    elif method == 'flatsky':
        #assert spec_type == 'modal', 'Can only calculate the modal-spectrum for %s method' % method
        if method == 'flatsky':
            from . import flatsky_spectra as spectra
        specs = []
        for ar in arz:
            kx, fx, _ = spectra.modal_spectrum(ar, lenn=lenn)
            if spec_type != 'modal':
                fx = translate_spectrum(kx, fx, 'modal', spec_type)
            specs.append(fx)
        return kx, *specs, np.sum(specs, axis=0)
    else:
        raise NotImplementedError

def translate_spectrum(k, fek, orig_type, new_type, orig_dim=None):
    """translate_spectrum(k, fek, orig_type, new_type)
    
    Transforms a binned spectrum of `orig_type` to `new_type`

    Args:
        k (np.array): Binned wavenumbers
        fek (np.ndarray): Binned 1D spectrum of `orig_type`
        orig_type (str): String indicating the type of `fek` i.e. 'modal'
        new_type (str): String indicating output type i.e. 'omni'
        orig_dim (int): Original dimension of `fek`, for translating binned spectra
    Returns:
        np.ndarray?: New binned 1D spectrum of `new_type`
    """
    # The amplitude calculation is the same for each dimension, if using the omni spectrum
    if new_type == 'amplitude':
        if orig_type != 'omni':
            fek = translate_spectrum(k, fek, orig_type, 'omni')
        return np.sqrt(k * fek)
    if orig_type == 'amplitude':
        return translate_spectrum(k, fek**2/k, 'omni', new_type)

    # 2D and 3D jacobians/shell areas are different
    dim = orig_dim
    if dim is None:
        dim = fek.ndim
    if dim == 1:
        jacob = 1.
    if dim == 2:
        jacob = 2. * np.pi * k
    if dim == 3:
        jacob = 4. * np.pi * k**2

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
            cut_excess (bool): Default True. If true, cut off wavenumbers larger than the basis direction
            nan_small (bool): Default False. If true, set to nan all bins that have a small number of elements
            min_bin (float): Default 1.0. Sets the minimum bin value
            bin_center (bool): Default False. If true, return wavenumbers from the center of the bin region
            norm_bin_size (bool): Default False. If true, divide by the size of the bin
    Returns:
        bins (np.array): Wavenumber array
        spec (np.ndarray): Integrated spectrum
        std (np.ndarray): Standard deviation for the integration of the spectrum
    """
    kmesh = wavenumber_mesh(kvec, phys_dims=lenn)
    assert np.shape(kmesh) == np.shape(mspec), 'kvec does not span mspec'
    cut_excess = kwargs.get('cut_excess', True)
    nan_small = kwargs.get('nan_small', False)
    if lenn is None:
        lenn = [2.*np.pi for _ in range(mspec.ndim)]
    # Minimum k is the first non-zero k value = 1*dk
    min_k = np.min([2.*np.pi/lenn[i] for i in range(mspec.ndim)])
    min_k = 1.
    min_bin = kwargs.get('min_bin', min_k)
    max_bin = kwargs.get('max_bin', None)
    bin_center = kwargs.get('bin_center', True)
    norm_bin_size = kwargs.get('norm_bin_size', False)
    log_space = kwargs.get('log_space', False)
    num_bins = kwargs.get('num_bins', None)
    ignore_nan = kwargs.get('ignore_nan', False)
    max_half_bin_width = kwargs.get('max_half_bin_width', None)
    if spec_type == 'omni':
        ## The omni spectrum is the integrated modal spectrum
        bins, ispec, istd = statistics_base.bin_data(kmesh, mspec, mean_func=np.nansum, std_func=nanstderr,
            cut_excess=cut_excess, nan_small=nan_small, min_bin=min_bin, bin_center=bin_center,
            norm_bin_size=norm_bin_size, log_space=log_space, num_bins=num_bins, ignore_nan=ignore_nan,
            max_bin=max_bin, max_half_bin_width=max_half_bin_width)
        return bins, ispec, istd
    elif spec_type == 'modal':
        ## The 1D modal spectrum is the averaged ND modal spectrum
        bins, ispec, istd = statistics_base.bin_data(kmesh, mspec, mean_func=np.nanmean, std_func=nanstderr,
            cut_excess=cut_excess, nan_small=nan_small, min_bin=min_bin, bin_center=bin_center,
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

def nanstderr(x):
    ## NOTE: half is because the power spectrum is symmetric
    return np.nanstd(x)/np.sqrt(0.5*x.size)

def wavenumber_mesh(kvec, phys_dims=None):
    """wavenumber_mesh(kvec)

    Computes the ndimensional wavenumber magnitude mesh

    Args:
        kvec (tuple): Wavenumbers for each basis direction
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
