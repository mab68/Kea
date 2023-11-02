
"""
read_data.py

Provides helper functions for reading data

Functions
---------
read_h5_data\n
read_npy_data\n
"""

import os

import h5py

import numpy as np


def read_h5_data(keys, folder_name=None, file_name=None):
    """read_h5_data(keys, folder_name=None, file_name=None)

    Returns data from h5 file as numpy array

    Args:
        keys (list [N]): Keys to extract from the h5 file
        folder_name (str): Location to find the file
        file_name (str): Name of the file to import data from
    Returns:
        return_vals (list [N]): List of the file data if the key was found
    """
    if folder_name is None:
        # Default to working directory
        folder_name = os.path.dirname(os.path.realpath(__file__))
    if file_name is None:
        # Default to first file in folder_name
        all_filesnames = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f)) and f.endswith('.h5')]
        file_name = all_filesnames[0]
    print('Reading file:', file_name)
    file = h5py.File(folder_name + file_name, 'r')
    sformat = False
    if 'cmhd3d-mpi' in file:
        # Sean's HDF5 Format
        file = file['cmhd3d-mpi']
        sformat = True
    return_vals = []
    for k in keys:
        if k in file.keys():
            val = file[k][...].astype(float)
            if sformat:
                val = val[:,:,0:256]
            return_vals.append(val)
        else:
            raise ValueError('Key %s not in file %s' % (k, file_name))
    #file.close()
    return return_vals

def read_np_data(file_name, folder_name=None):
    """read_np_data(file_name, folder_name=None)

    Loads the file `folder_name/file_name.npy` as a numpy array

    Args:
        file_name (str): Name of the file
        folder_name(str): Folder the file is located in. If none, defaults to working directory
    Returns:
        ar (np.ndarray): Numpy array
    """
    if folder_name is None:
        # Default to working directory
        folder_name = os.path.dirname(os.path.realpath(__file__))
    ext = '.npy'
    if ext in file_name:
        # Append extension if not already on file_name
        ext = ''
    fn = folder_name + file_name + ext
    print('Reading file: ', fn)
    return np.load(fn)

