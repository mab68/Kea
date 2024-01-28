

from Kea.simulator import fbm
from Kea.utils import data_reader, data_writer

from .data import Data, DataType


class SyntheticData(Data):

    basename = None

    def __init__(self, basename=None):
        """
        Args:
            basename (None,str): Optional parameter if we want to load in a synthetic file
        """
        super().__init__(DataType.SYNTHETIC, basename)
        if self.basename is not None:
            # Load in the data here
            raise NotImplementedError('I have not implemented the data saving/loading feature yet')
            #data_reader.read_np_data(basename)

    @staticmethod
    def create_fbm(var, grid_dims, phys_dims, fbm_kwargs, func_kwargs={}, basename=None):
        """create_fbm(var, grid_dims, phys_dims, fbm_kwargs, func_kwargs, basename)

        Creates an fbm field for `var` and then instantiates a SyntheticData object

        Args:
            var (str): Variable name, to be able to access the fBm field via `data.var`
            grid_dims (tuple): Number of datapoints in each dimension
            phys_dims (tuple): Computational size in each dimension
            fbm_kwargs (dict): Arguments to pass onto the fBm field creator
                alphas (tuple): Power laws
                breaks (tuple): Locations for the power law to change slope
            func_kwargs (dict): Arguments to pass onto the function in the fbm field creator
            basename (str): if set, saves the fbm data to the basename folder
        Returns:
            data (SyntheticData): Data object holding the fbm field
        """

        ar = fbm.create_fbm(grid_dims, phys_dims=phys_dims, func_kwargs=func_kwargs, **fbm_kwargs)

        if basename is not None:
            # Need to save the fBm data
            fn = '%s.dbl' % var
            if basename is not None:
                fn = '/' + fn
            data_writer.pluto_data_file(ar, basename + fn)
            # Then load the data
            data = SyntheticData(basename)
            return data

        data = SyntheticData(basename)
        data.__setattr__(var, ar)
        data.set_dims(len(grid_dims), grid_dims, phys_dims)

        return data
