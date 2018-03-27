#----------------------------------- models.py -----------------------------------------#
""" 
This file contains all the classes used to load and interact with the machine learning
model that will make predictions on a turbulent diffusivity
"""


# ------------ Import statements
from sklearn.externals import joblib # joblib is used to load trained models from disk
from sklearn.ensemble import RandomForestRegressor
import os
import pkg_resources



class MLModel:
    """
    This class is a diffusivity model that maps from local flow variables to a turbulent  
    diffusivity. Here is where a pre-trained machine learning model comes in.
    """

    def __init__(self, filepath=None):
        """
        Constructor for MLModel class        
                    
        This initializes the class by loading a previously trained model that was saved
        using joblib. The user can instantiate it without arguments to load the default
        model, shipped with the package. Alternatively, you can instantiate it with one
        argument representing the path to another saved model. 
        
        The model loaded from disk has to be created using joblib.dump() with protocol 2 
        (compatible with python 2.7), and it has to be a list [string, model] where the 
        model itself is the second element, and the first element is a string describing
        the model.
        
        Arguments:
        filepath -- optional, the path from where to load the pickled ML model. If not 
                    supplied, will load the default model.
        """ 
        
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
            
    
    def predict(self, x):    
        """
        Predicts the diffusivity given the features. 
        
        It assumes that the underlying implementation (default: random forests from
        sklearn) contains a method called predict.
        
        Arguments:
        x -- numpy array (num_useful, N_FEATURES) of features
        
        Returns:
        y -- numpy array (num_useful,) for the turbulent diffusivity predicted at each 
             cell
        """       
        
        print("ML model loaded: {}".format(self.__description))
        print("Predicting turbulent diffusivity using ML model...", end="", flush=True)
        y = self.__model.predict(x)
        print(" Done")
        return y
    
    def printDescription(self):
        """
        Prints descriptive string attached to the loaded model.
        
        This function is called to print out the string that is attached to the model. 
        This can be called to make sure that we are loading the model that we want.
        """
        
        print(self.__description)
        
        
