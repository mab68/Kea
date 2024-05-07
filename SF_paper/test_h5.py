
import numpy as np

import h5py

## Things to note: 
## 1) This does not load anything into memory
##          We should be able to pass this information around
##          in my code without much trouble (specially for mpi)
file = h5py.File('/home/m/Documents/science_codes/Data/cmhd3d/iso-nu73-mhd3-01-uno-Xsp-rho.h5', 'r')

x = file['cmhd3d-mpi']['rho']

print(x)

## 2) Slicing the array will load the slice into memory
#print(type(file['cmhd3d-mpi']['rho'][:,:,0:1024]))

file.close()
