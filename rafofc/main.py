#--------------------------- main.py file ------------------------------------#
## Main file - entry point to the code. This file coordinates all other files
# and implements all the functionality directly available to the user.

# import statements
import numpy as np
from pkg_resources import get_distribution
from . import helpers
from . import models


"""
This simple function can be called by the user to check that everything was
installed properly. We print a welcome message, the version of the package,
and attempt to load the pre-trained model to make sure the data file is there.
"""
def PrintInfo():
    print('Welcome to RaFoFC - Random Forest for Film Cooling package!')
    
    # Get distribution version
    dist = get_distribution('rafofc')
    print('Version: {}'.format(dist.version))
    
    # Try to load the model
    print('Attempting to load the default model...')
    rafo = models.MLModel()
    print('Defaul model was found and can be loaded properly.')



    
def FixedAddition():
    a = np.array([0,1,2])
    b = np.array([1,2,3])
    c = helpers.MyAdditionRELU(a,b)
    print("Array addition: {}".format(c))

def LoadDiffusivity(path=None):
    x = np.random.normal(size=(1000, 19))
    rafo = models.MLModel(path)    
    y = rafo.Predict(x)
    assert y.shape == (1000,), 'Wrong shape!'
    print("Prediction done successfully!")