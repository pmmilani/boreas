# ----------------------------- models.py ------------------------------------#
# This file contains all the classes used to load and interact with the machine
# learning model that will make predictions on a turbulent diffusivity


# ------------ Import statements
# joblib is used to load trained models from disk
from sklearn.externals import joblib 
from sklearn.ensemble import RandomForestRegressor
import os
import pkg_resources


"""
This class is a diffusivity model that maps from local flow variables to a 
turbulent diffusivity. Here is where a pre-trained machine learning model
comes is.
"""
class MLModel:

    """ 
    This initializes the class by loading a previously trained model 
    that was saved using joblib. The user can instantiate it without 
    arguments to load the default model, shipped with the package.
    Alternatively, you can instantiate it with one argument representing
    the path to another saved model.
    """   
    def __init__(self, filepath=None):
        
        # if no path is provided, load the default model
        if filepath is None: 
            path = 'data/defaultML.pckl' # location of the default model
            filepath = pkg_resources.resource_filename(__name__, path)
            
            error_msg = ("When attempting to load the default model from disk," 
                         + " no file was found in path {}.".format(filepath)
                         + " Check your installation.")
        
        # Here a path is provided (relative to the working directory), so just
        # load that file
        else:
            error_msg = ("When attempting to load a custom model from disk, "
                         + "no file was found in path {}.".format(filepath) 
                         + " Make sure that the file exists.")
        
        
        assert os.path.isfile(filepath), error_msg # make sure the file exists  
        
        # saved as private variables or the class 
        self.__description, self.__model = joblib.load(filepath)     
            
    """
    This function is called to predict the diffusivity given the features.
    It assumes that the underlying implementation (random forests from sklearn)
    contain a method called predict. 
    x.shape = (N_POINTS, N_FEATURES)
    """
    def Predict(self, x):
        return self.__model.predict(x)
        
    """
    This function is called to print out the string that is attached to the
    model. This can be called to make sure that we are loading the model that
    we want.
    """
    def PrintDescription(self):
        print(self.__description)
        
        
