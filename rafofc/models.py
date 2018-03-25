# ---------------------------------- models.py -----------------------------------------#
# This file contains all the classes used to load and interact with the machine learning
# model that will make predictions on a turbulent diffusivity


# ------------ Import statements
# joblib is used to load trained models from disk
from sklearn.externals import joblib 
from sklearn.ensemble import RandomForestRegressor
import os
import pkg_resources


"""
This class is a diffusivity model that maps from local flow variables to a turbulent  
diffusivity. Here is where a pre-trained machine learning model comes in.
"""
"""
JL Comment: It's typical in python for the comments to go under the method name or class name, not above it.
This is important for practical reasons as well (e.g. doctests), not just visual reasons.
https://www.python.org/dev/peps/pep-0257/
"""
class MLModel:

    """ 
    This initializes the class by loading a previously trained model that was saved
    using joblib. The user can instantiate it without arguments to load the default
    model, shipped with the package. Alternatively, you can instantiate it with one
    argument representing the path to another saved model.
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
    This function is called to predict the diffusivity given the features. It assumes 
    that the underlying implementation (random forests from sklearn) contain a method 
    called predict. 
    x.shape = (N_POINTS, N_FEATURES)
    """
    def Predict(self, x):
        """
        JL Comment: It's really unusual in python to have methods that have capital names.  It's more
        common to use lower case for methods and reserve capitalization for classes.  This would mean
        that you can have a method called printDescripton or print_description, but rarely PrintDescription
        or Predict.

        JL Comment: This would be a good place for a doctest.  Make sure that predict returns what you'd expect on
        a point for which you know the answer.  A great way to do this is to have it make prediction on an easy point,
        then include this as the correct answer in the doctest.
        For information on doctests, check here: https://docs.python.org/2/library/doctest.html
        """
        return self.__model.predict(x)

        
    """
    This function is called to print out the string that is attached to the model. 
    This can be called to make sure that we are loading the model that we want.
    """
    def PrintDescription(self):
        print(self.__description)
        
        
