
import sys
sys.path.append('/home/mark.bishop/Documents/Kea')
import kea
from kea.statistics import spectra

import numpy as np

spectra.fourier_spectrum(np.ones(500))
