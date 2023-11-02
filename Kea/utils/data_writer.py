
"""
data_writer.py
Provides helper functions for writing data or converting data formats

Functions
---------
numpy_to_h5\n
simdat_to_h5\n
numpy_to_npy\n
"""

import numpy as np


def pluto_grid_file(grid, filename='grid0.out'):
    """pluto_grid_file(grid, filename)
    
    Generates a PLUTO grid file

    Convention is to use: filename=`grid0.out`

    Args:
        grid (tuple): Grid tuple containing ((Nx, Lx), (Ny, Ly), (Nz, Lz))
        filename (str): Filename to save the file to
    """
    f = open(filename, 'w')
    f.write('# GEOMETRY:    CARTESIAN\n')
    for i in range(len(grid)):
        N = grid[i][0]
        L = grid[i][1]
        f.write('%d\n' % N)
        x = np.linspace(0., L, N+1)
        for j in range(N):
            xmin = x[j]
            xmax = x[j+1]
            f.write('%d %12.6e  %12.6e\n' % (j+1, xmin, xmax))
    f.close()

def pluto_data_file(ar, filename='ar0.dbl'):
    """pluto_data_file(ar, filename)
    
    Generates a PLUTO binary data file

    Convention is to use: filename=`VAR0.dbl`

    Args:
        ar (np.ndarray): The array to save to `filename` as raw binary
        filename (str): Filename to save the file to
    """
    ar.astype('double').tofile(filename)

def numpy_to_npy(ar, filename):
    """numpy_to_npy(ar, filename)
    
    Save numpy array to file

    Args:
        ar (np.ndarray): The array to save `filename`
        filename (str): Filename to save the file to
    """
    np.save(filename, ar)
