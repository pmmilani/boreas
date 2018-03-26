#-------------------------------- main.py file -----------------------------------------#

"""
Main file - entry point to the code. This file coordinates all other files and 
implements all the functionality directly available to the user.
"""

# import statements
import numpy as np
from pkg_resources import get_distribution
from rafofc.models import MLModel
from rafofc.tecplot_data import TPDataset, RANSDataset

def printInfo():
    """
    Makes sure everything is properly installed.
    
    We print a welcome message, the version of the package, and attempt to load
    the pre-trained model to make sure the data file is there.
    """
    
    print('Welcome to RaFoFC - Random Forest for Film Cooling package!')
    
    # Get distribution version
    dist = get_distribution('rafofc')
    print('Version: {}'.format(dist.version))
    
    # Try to load the model and print information about it
    print('Attempting to load the default model...')
    rafo = MLModel()
    print('Defaul model was found and can be loaded properly.')
    rafo.printDescription()


def testTecplot(path, zone=None):
    """
    Testing function
    """  
    
    data = TPDataset(path, zone)
    data.normalize()
    data.calculateDerivatives()
    data.saveDataset("out_" + path)
    rans_data = data.extractQuantityArrays()
    
    return rans_data



