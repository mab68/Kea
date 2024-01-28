

from enum import Enum

import numpy as np

from Kea.statistics import moments

from .spectrum import Spectrum, SpectrumMethod


class DataType(Enum):
    """
    Enum specifying the datatype for use in the data
    """

    SYNTHETIC = 1
    SIMULATION = 2
    OBSERVATION = 3


class Data():
    """
    Base data object, holds the information and additional universal functions
    """

    # Coord names
    COORD_NAMES = ['x1', 'x2', 'x3']

    # DataType
    datatype = None

    # Data location
    basename = None
    basefolder = None

    # Number of dimensions
    ndim = None


    def __init__(self, datatype, basename):
        """
        Args:
            datatype (DataType): Enum specifying the datatype in this Data object
        """
        self.datatype = datatype
        self.basename = basename
        if self.basename is not None:
            self.basefolder = basename + '/'

    def set_dims(self, ndim, grid_dims, phys_dims):
        """set_dims(grid_dims, lenn_dims)

        Sets the physical and grid domains

        Args:
            ndim (int): Sets the number of base dimensions
            grid_dims (tuple): Sets the grid domain
            lenn_dims (tuple): Sets the computational/physical domain
        """
        # Need to setup the physical domain etc.
        assert len(grid_dims) == len(phys_dims), 'Grid (%s) does not have the same dimensions as the computational domain (%s)' % (len(grid_dims), len(phys_dims))
        self.ndim = ndim
        for i, g in enumerate(grid_dims):
            self.__setattr__('n%s' % self.COORD_NAMES[i], g)
        for i, l in enumerate(phys_dims):
            self.__setattr__('L%s' % self.COORD_NAMES[i], l)

    def get_dims(self):
        """get_dims()

        Gets the grid and physical domains

        Returns:
            shape, lenn: Tuples of the grid, and physical domains
        """
        shape = []
        lenn = []
        for i in range(self.ndim):
            shape.append(self.__getattribute__('n%s' % self.COORD_NAMES[i]))
            lenn.append(self.__getattribute__('L%s' % self.COORD_NAMES[i]))
        return shape, lenn

    def project(self, var, axis, weights=None, moment=1, save_var=True):
        """project(var, axis)

        Creates a projection of `var`
        
        Args:
            var (str): Name of the variable to project
            axis (tuple, int): The axis to project the data over
            weights (str): 
            avg_func (func): The function to use to project, default: np.nansum
            save_var (bool): If true, save the variable as `x%sproj_%s % (axis,var)`
        """
        if hasattr(axis, '__iter__'):
            axis = tuple(axis)
        else:
            axis = (axis,)
        ar = self.__getattribute__(var)
        shape, lenn = self.get_dims()
        proj_ar = moments.make_moment(ar, moment, shape=shape, lenn=lenn, axis=axis, take_abs=False)
        if save_var:
            coord_str = ''
            for i in axis:
                coord_str += self.COORD_NAMES[i]
            ar_name = '%sproj_%s' % (coord_str, var)
            print('Created %s' % ar_name)
            self.__setattr__(ar_name, proj_ar)
        return proj_ar

    def spectrum(self, varz, method='periodogram'):
        """spectrum(varz)

        Create a power spectrum using the variable

        Args:
            varz (str, tuple): The variable to take the power spectrum of, 2-tuple, take the cross-spectrum
            method (str): What spectrum estimation method to use: periodogram, correlogram, arevalo, flatsky, structure function
        Returns:
            Spectrum: Spectrum class holding the power spectrum
        """
        try:
            method = SpectrumMethod[method.upper()]
        except:
            raise ValueError('Invalid method: %s' % method)
    
        if isinstance(varz, str):
            v1, v2 = varz, None
        else:
            if len(varz) != 2:
                raise ValueError('Invalid variable: %s' % varz)
            v1, v2 = varz

        ar1 = self.__getattribute__(v1)
        ar2 = None
        if v2 is not None:
            ar2 = self.__getattribute__(v2)

        _, lenn = self.get_dims()

        return Spectrum.compute_spectrum(
            ar1, ar2, lenn=lenn, method=method, varname=varz
        )


    def apply_noise():
        raise NotImplementedError()

    def add_mean():
        raise NotImplementedError()

    def modify_resolution():
        raise NotImplementedError()
    
    def dephase_data():
        raise NotImplementedError()
    
    def generate_mask():
        raise NotImplementedError()
    
    def generate_exposure():
        raise NotImplementedError()


