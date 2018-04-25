#--------------------------------- test_all.py -----------------------------------------#
"""
Test script containing several functions that will be invoked in unit testing. They are
built so the user can just call 'pytest'
"""

import os
import numpy as np
from rafofc.main import printInfo, applyMLModel
from rafofc.models import MLModel


def test_print_info():
    """
    This function calls the printInfo helper from main to make sure that it does not
    crash and returns 1 as expected.
    """
    
    assert printInfo() == 1
    
    
def test_loading_default_ml_model():
    """
    This function checks to see whether we can load the default model.
    """            
        
    # Test loading
    rafo_custom = MLModel() 
    rafo_custom.printDescription()
    
    # Test predicting
    n_test = 100
    N_FEATURES = 19
    x = -100.0 + 100*np.random.rand(n_test, N_FEATURES)
    y = rafo_custom.predict(x)
    assert y.shape == (n_test, )

    
def test_loading_custom_ml_model():
    """
    This function checks to see whether we can load the (embedded) custom model.
    """            
    
    # Make the path relative to the location of the present script
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "ML_2.pckl")
    
    # Test loading
    rafo_custom = MLModel(filepath=filename) 
    rafo_custom.printDescription()
    
    # Test predicting
    n_test = 100
    N_FEATURES = 19
    x = -100.0 + 100*np.random.rand(n_test, N_FEATURES)
    y = rafo_custom.predict(x)
    assert y.shape == (n_test, )
