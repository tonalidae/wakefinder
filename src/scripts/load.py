import numpy as np
import pytest

def txt(filename):
    """ Read data from a text file.
    Parameters
    ----------
    filename : str
        Name of the file to be read.
        Returns
        -------
        txt_data : array
            Data read from the text file.
    """
    txt_data = np.loadtxt('../../data/'+filename+'.txt', delimiter=' ')
    return txt_data
#More functions to read data in different formats can be added here

def orbit(filename):
    """ Read orbit data from a text file.
    Parameters
    ----------
    filename : str
        Name of the file to be read.
        Returns
        -------
        orbit_data : array
            Data read from the text file.
    """
    orbit_data = np.loadtxt('../../data/'+filename+'.txt', delimiter=' ', usecols=range(0, 12))
    return orbit_data
