
from enum import Enum

from .data import Data, DataType


class ObservationType(Enum):
    """
    Enum specifying the observation
    """

    CHANDRA = 1
    PLANCK = 2
    SITELLE = 3


class ObservationData(Data):
    
    obstype = None
    basename = None

    def __init__(self, obstype, basename):
        """
        Args:
            obstype (ObservationType): Enum indication the observation type
            basename (str): Base filename for the data location
        """
        super().__init__(DataType.OBSERVATION)
        self.obstype = obstype
        self.basename = basename
