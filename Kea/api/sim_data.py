
from enum import Enum

import numpy as np

from .data import Data, DataType

from Kea.utils.data_reader import read_h5_data


class SimulationType(Enum):
    """
    Enum specifying the simulation type, what variables may
        be expected, and how to load the data.
    """

    H5_SEAN = 1
    H5_TULASI = 2
    PLUTO = 3


class SimulationData(Data):

    COORD_REMAPPER = {'x': 'x1', 'y': 'x2', 'z': 'x3'}

    simtype = None
    basename = None
    basefolder = None

    def __init__(self, simtype, basename, loadvarz=None, loadtime=None, phys_dims=None):
        """
        Args:
            simtype (SimulationType): Enum indicating the simulation type
            basename (str): Base filename for the data location
            phys_dims (tuple): Tuple containing the physical domain of the simulation
            loadvarz (tuple): Tuple of strings to load from the simulation
            loadtime (float): Snapshot to load from the simulation
        """
        super().__init__(DataType.SIMULATION, basename)
        self.simtype = simtype
        if loadvarz is not None:
            loaded_varz = self.load_vars(loadvarz, loadtime)
            ar = self.__getattribute__(loaded_varz[0])
            if phys_dims is None:
                phys_dims = [2.*np.pi for _ in range(ar.ndim)]
            self.set_dims(ar.ndim, ar.shape, phys_dims)

    def load_var(self, var, time):
        if self.simtype == SimulationType.H5_SEAN:
            if time is not None:
                tt = str(time).zfill(2)
            else:
                tt = '04'
            if 'V' in var:
                vv = 'v'
            elif 'rho' in var:
                vv = 'rho'
            elif 'B' in var:
                vv = 'b'
            ar = read_h5_data([var], self.basefolder, 'z3-mhd3-%s-uno-Xsp-%s.h5' % (tt, vv))[0]
            nv = var.lower()
            nv = nv.replace('x', self.COORD_REMAPPER['x'])
            nv = nv.replace('y', self.COORD_REMAPPER['y'])
            nv = nv.replace('z', self.COORD_REMAPPER['z'])
            self.__setattr__(nv, ar)
            print('Loaded %s' % nv)
        elif self.simtype == SimulationType.H5_TULASI:
            if time is not None:
                tt = str(time).zfill(3)
            else:
                tt = '170'
            ar = read_h5_data([var], self.basefolder, 'MHD3D_256_%s.h5' % tt)[0]
            nv = var.lower()
            nv = nv.replace('x', self.COORD_REMAPPER['x'])
            nv = nv.replace('y', self.COORD_REMAPPER['y'])
            nv = nv.replace('z', self.COORD_REMAPPER['z'])
            self.__setattr__(nv, ar)
            print('Loaded %s' % nv)
        else:
            raise NotImplementedError
        return nv

    def load_vars(self, vars, time):
        loaded_varz = []
        for var in vars:
            loaded_varz.append(self.load_var(var, time))
        return loaded_varz

