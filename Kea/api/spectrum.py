

from enum import Enum

import numpy as np

from scipy import fft

from Kea.statistics.spectra import arevalo_spectra, corr_spectra, flatsky_spectra, per_spectra, spectra_base, strfn_spectra
from Kea.statistics.statfunc import statfunc_base, corr, strfn

from .statistic import Statistic


class SpectrumMethod(Enum):
    PERIODOGRAM = 1
    CORRELOGRAM = 2
    AREVALO = 3
    FLATSKY = 4
    STRFN = 5


class Spectrum(Statistic):

    # List of strings for the spectrum type. I.e. ['x1proj_vx1']
    varname = None

    # Original wavenumbers
    kvec = None
    # Original spectrum
    fekD = None
    # Dimensionality of the data
    ndim = None
    # Physical domain
    phys_domain = None
    # The method used to compute the spectrum
    spectype = None
    # dk = 2pi/L
    dk = None

    def __init__(self, varname, kvec, fekD, ndim, phys_domain, spectype):
        """Spectrum(varname, kvec, fekD, ndim, phys_domain, spectype)
        
        Args:
            varname (str): Name of the variable the spectrum is of
            kvec (tuple): D-dimensional wavenumbers in each coordinate direction
            fekD (np.ndarray): D-dimensional power spectrum
            ndim (int): Number of dimensions in the original data
            phys_domain (tuple): physical domain for each dimension
            spectpye (SpectrumMethod): Method used to compute the spectrum
        """
        super().__init__()
        self.varname = varname
        self.kvec = kvec
        self.fekD = fekD
        self.ndim = ndim
        self.phys_domain = phys_domain
        self.spectype = spectype
        self.dk = [2.*np.pi/L for L in phys_domain]

    def translate(self, current_type, new_type):
        """translate(current_type, new_type)
        
        Converts an integrated spectrum,
        Types:
            - Modal
            - Omni
            - Amplitude

        Args:
            current_type (str): The current integrated spectrum to convert
            new_type (str): What to convert the integrated spectrum to
        """
        k = self.__getattribute__('%s_k' % current_type)
        fek = self.__getattribute__('%s_fek' % current_type)
        knew, feknew = spectra_base.translate_spectrum(k, fek, current_type, new_type, self.ndim)
        return knew, feknew

    def integrate(self, int_type='modal', int_kwargs={}):
        """integrate(int_type, int_kwargs)
        
        Integrates and saves the D-dimensional modal spectrum

        Args:
            int_type (str): String indicating what specific spectra to return
                omni: Summed/integrated modal spectrum
                modal: Averaged D-dimensional modal spectrum
                reduced: NOT IMPLEMENTED
                    Summed modal spectrum along N-1 axis as a function of `axis` wavenumber
                    axis (int): The axis wavenumbers to be a function of
            int_kwargs (dict): Arguments to pass onto the integration function
                cut_excess (bool): Default True. If true, cut off wavenumbers larger than the basis direction
                nan_small (bool): Default False. If true, set to nan all bins that have a small number of elements
                min_bin (float): Default 1.0. Sets the minimum bin value
                bin_center (bool): Default False. If true, return wavenumbers from the center of the bin region
                norm_bin_size (bool): Default False. If true, divide by the size of the bin
        Returns:
            int_k, int_fek
        """
        if self.spectype == SpectrumMethod.AREVALO:
            raise ValueError('Cannot integrate the Arevalo method')
        if self.spectype == SpectrumMethod.STRFN:
            raise ValueError('Cannot integrate the structure function method')

        if int_type == 'reduced':
            raise NotImplementedError

        kbin, specbin, stdbin = spectra_base.spectrum_integrate(self.kvec, self.fekD, int_type, self.phys_domain, **int_kwargs)

        kwidth = None
        self.__setattr__('%s_k' % int_type, kbin)
        self.__setattr__('%s_fek' % int_type, specbin)
        self.__setattr__('%s_err' % int_type, stdbin)
        self.__setattr__('%s_kwidth' % int_type, kwidth)

        return kbin, specbin

    @staticmethod
    def compute_spectrum(ar1, ar2=None, exp1=None, exp2=None, lenn=None,
                         method=SpectrumMethod.PERIODOGRAM, method_kwargs={}, varname=None):
        """compute_spectrum(ar1, ar2, exp1, exp2, lenn, method, method_kwargs, varname)
        
        Args:
            ar1 (np.ndarray):
            ar2 (np.ndarray):
            exp1 (np.ndarray):
            exp2 (np.ndarray):
            lenn (tuple):
            method (SpectrumMethod):
            method_kwargs (dict):
            varname (str):
        Returns:
            Spectrum: Spectrum class holding the spectrum information
        """
        ndim = ar1.ndim
        if ar2 is not None:
            assert ndim == ar2.ndim, 'Provided arrays have different dimensions (%s, %s)' % (ndim, ar2.ndim)
            assert ar1.shape == ar2.shape, 'Provided arrays are not the same size (%s, %s)' % (ar1.shape, ar2.shape)
        if method == SpectrumMethod.PERIODOGRAM:
            # Calculate the periodogram modal spectrum
            kvec, fekD = per_spectra.modal_spectrum(ar1, ar2, lenn=lenn, **method_kwargs)
        elif method == SpectrumMethod.CORRELOGRAM:
            # Calculate the correlation function first
            lv_acf = statfunc_base.get_all_lagvecs([2*n+1 for n in ar1.shape])
            acf = corr.process_lags(ar1, ar2, lv_acf, periodic=True, lenn=lenn, shape=tuple([2*n+1 for n in ar1.shape]))
            acf = statfunc_base.cut_at_lag(acf, np.min(ar1.shape)//2)
            # Then calculate the spectrum
            kvec, fekD = corr_spectra.modal_spectrum(acf, lenn=lenn)
        elif method == SpectrumMethod.FLATSKY:
            modal_k, modal_fek, modal_err = flatsky_spectra.modal_spectrum(ar1, ar2, lenn=lenn)
            kvec, fekD = None, None
            ## TODO: save the error
        elif method == SpectrumMethod.AREVALO:
            N = np.min(ar1.shape)
            L = lenn[np.where(ar1.shape == N)]
            k = fft.fftshift(fft.fftfreq(N))*2.*np.pi/(L/N)
            lags = arevalo_spectra.k_to_discrete_lags(k, N, L)
            modal_k, modal_fek = arevalo_spectra.modal_spectrum(ar2, ar2=ar2, lags=lags, lenn=lenn, **method_kwargs)
            kvec, fekD = None, None
            ## TODO: Would the wavenumber 'error'/binning be the width of the difference of Gaussians?
        elif method == SpectrumMethod.STRFN:
            # Calculate the structure function first
            lv_sf = statfunc_base.get_all_lagvecs([2*n+1 for n in ar1.shape])
            sf2 = strfn.process_lags(ar1, ar2, lv_sf, periodic=False, lenn=lenn, shape=tuple([2*n+1 for n in ar1.shape]), orders=[2])[0]
            # Then calculate the spectrum
            omni_k, omni_fek = strfn_spectra.omni_spectrum(sf2, lenn=lenn)
        else:
            raise ValueError('Bad method type (%s)' % method)
        spectrum = Spectrum(varname, kvec, fekD, ndim, lenn, method)

        if method == SpectrumMethod.AREVALO or method == SpectrumMethod.FLATSKY:
            spectrum.__setattr__('modal_k', modal_k)
            spectrum.__setattr__('modal_fek', modal_fek)
        if method == SpectrumMethod.STRFN:
            spectrum.__setattr__('omni_k', omni_k)
            spectrum.__setattr__('omni_fek', omni_fek)
        return spectrum

