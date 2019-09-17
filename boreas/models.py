#----------------------------------- models.py -----------------------------------------#
""" 
This file contains all the classes used to load and interact with the machine learning
model that will make predictions on a turbulent diffusivity
"""

# ------------ Import statements
import joblib # joblib is used to load trained models from disk
from sklearn.ensemble import RandomForestRegressor
import os
import pkg_resources
import numpy as np
import timeit
import tensorflow as tf
from boreas import constants
from tbnns.main import TBNNS
from tbnns.utils import suppressWarnings


class MLModel:
    """
    This is a super-class of models that predict turbulent diffusivity given
    local flow features
    """

    def __init__(self):
        """
        Constructor for MLModel class
        """        
        self._description = "Empty model"
        self._model = None
    
    
    def loadFromDisk(self):
        """
        Loads a previously trained model from disk.
        """
        print("Loading method not implemented yet!")
        pass

    
    def train(self):
        """       
        Trains a model to perform regression on y given x       
        """
        print("Training method not implemented yet!")
        pass    
    
    
    def predict(self):    
        """
        Predicts the diffusivity/Prt given features
        """
        print("Predicting method not implemented yet!")
        pass
    
    def printDescription(self):
        """
        Prints descriptive string attached to the loaded model.
        
        This function is called to print out the string that is attached to the model. 
        This can be called to make sure that we are loading the model that we want.
        """
        
        assert isinstance(self._description, str)        
        print(self._description)
        
        
class RFModel_Isotropic(MLModel):
    """
    This class is a diffusivity model that maps from local flow variables to a isotropic
    turbulent Prandtl number, using a Random Forest model.
    """
    
    
    def loadFromDisk(self, filepath=None):
        """
        Loads a previously trained model from disk.
        
        Loads a previously trained model that was saved using joblib. The user can call
        it without arguments to load the default model, shipped with the package. 
        Alternatively, you can call it with one argument representing the relative path
        to another saved model. The model loaded from disk has to be created using 
        joblib.dump() and it has to be a list [string, model] where the model itself is
        the second element, and the first element is a string describing that model.
        
        Arguments:
        filepath -- optional, the path from where to load the pickled ML model. If not 
                    supplied, will load the default model.
        """
    
        # if no path is provided, load the default model
        if filepath is None: 
            path = 'data/defaultRF.pckl' # location of the default model
            filepath = pkg_resources.resource_filename(__name__, path)
            
            error_msg = ("When attempting to load the default RF model from disk," 
                         + " no file was found in path {}.".format(filepath)
                         + " Check your installation.")
        
        # Here a path is provided (relative to the working directory), so just
        # load that file
        else:
            error_msg = ("When attempting to load a custom RF model from disk, "
                         + "no file was found in path {}.".format(filepath) 
                         + " Make sure that the file exists.")
        
        
        assert os.path.isfile(filepath), error_msg # make sure the file exists  
        
        # saved as private variables or the class 
        self._description, self._model = joblib.load(filepath)
        assert isinstance(self._model, RandomForestRegressor), "Not a scikit-learn RF!"

    
    def train(self, x, y, description, savepath,
              n_trees=None, max_depth=None, min_samples_split=None):
        """       
        Trains a random forest model to regress on log(y) given x
        
        This function trains a random forest (implemented in scikit-learn) to predict
        the turbulent Prandtl number given mean flow features. The features are x, and
        y contains gamma = 1/Prt. We predict log(y) since it is a ratio.
        
        Arguments:
        x -- numpy array containing the features at training points, of shape
             (n_useful, N_FEATURES). 
        y -- numpy array containing labels gamma = 1/Prt, of shape (n_useful,)
        description -- string containing a short, written description of the model,
                       which is important when the model is loaded and used at a 
                       later time.
        savepath -- string containing the path in which the model is saved to disk
        n_trees -- optional. Hyperparameter of the random forest, contains number of
                   trees to use. If None (default), reads value from constants.py
        max_depth -- optional. Hyperparameter of the random forest, contains maximum
                     depth of each tree to use. If None (default), reads value from
                     constants.py 
        min_samples_split -- optional. Hyperparameter of the random forest, contains
                             minimum number of samples at a node required to split. Can
                             either be an int (number itself) or a float (ratio of total
                             examples). If None (default), reads value from constants.py
        """
        
        # Run sanity checks first
        assert x.shape[0] == y.shape[0], "Number of examples don't match!"
        assert x.shape[1] == constants.N_FEATURES, "Wrong number of features!"
        
        # Read default parameters from constants.py if None is provided
        if n_trees is None:
            n_trees = constants.N_TREES
        if max_depth is None:
            max_depth = constants.MAX_DEPTH
        if min_samples_split is None:
            min_samples_split = constants.MIN_SPLIT
            
        # Initialize class with description and fresh estimator
        self._description = description
        self._model = RandomForestRegressor(n_estimators=n_trees,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            n_jobs=-1) # n_jobs=-1 means use all CPUs
        
        # Train and time
        print("Training Random Forest on {} points ({})".format(x.shape[0], description))
        print("n_trees={}, max_depth={}, min_samples_split={}".format(n_trees,max_depth,
                                                                      min_samples_split))
        print("This may take several hours. Training...", end="", flush=True)
        tic=timeit.default_timer() # timing
        self._model.fit(x, np.log(y))
        toc=timeit.default_timer()        
        print(" Done! It took {:.1f} min".format((toc - tic)/60.0))
        
        # Save model to disk in specified location
        joblib.dump([self._description, self._model], savepath, 
                    compress=constants.COMPRESS, protocol=constants.PROTOCOL)    
    
    
    def predict(self, x):    
        """
        Predicts Pr_t given the features using random forest model. 
        
        It assumes that the underlying implementation (default: random forests from
        scikit-learn) contains a method called predict. It also assumes that the model
        was trained to predict log(alpha_t / nu_t), so Pr_t = nu_t/alpha_t = 1/exp(y)
        
        Arguments:
        x -- numpy array (num_useful, N_FEATURES) of features
        
        Returns:
        y -- numpy array (num_useful,) for the turbulent Prandtl number predicted at
             each cell
        """       
        
        assert isinstance(self._model, RandomForestRegressor)
        assert x.shape[1] == constants.N_FEATURES, "Wrong number of features!"
        
        print("ML model loaded: {}".format(self._description))
        print("Predicting Pr-t using ML model...", end="", flush=True)
        y = self._model.predict(x)
        Prt = 1.0/np.exp(y)
        print(" Done!")
        return Prt       
 
 
class TBNNModel_Anisotropic(MLModel):
    """
    This class implements a model that predicts a tensorial (anisotropic) diffusivity
    using a tensor basis neural network (TBNN-s).
    """
    
    def __init__(self):
        """
        Constructor for TBNNModel_Anisotropic class. Adds the extra tf.Session()
        and suppresses the tensorflow warnings
        """ 
        
        super().__init__() # performs regular initialization        
        suppressWarnings()        
        self._tfsession = tf.Session() # initializes tensorflow session
        
    
    def loadFromDisk(self, filepath=None):
        """
        Loads a previously trained model from disk.
        
        Loads a previously trained model that was saved using joblib. The user can call
        it without arguments to load the default model, shipped with the package. 
        Alternatively, you can call it with one argument representing the relative path
        to another saved model. The model loaded from disk has to be created using 
        joblib.dump() and it has to be a list [string, model] where the model itself is
        the second element, and the first element is a string describing that model.
        
        Arguments:
        filepath -- optional, the path from where to load the pickled ML model. If not 
                    supplied, will load the default model. Note that for TBNN-s, we need
                    the file containing the metadata AND a tensorflow checkpoint file
                    containing parameters. filepath holds the location of the former, 
                    which contains the location of the latter.
        """
    
        # if no path is provided, load the default model
        if filepath is None: 
            path = 'data/defaultTBNNs.pckl' # location of the default model
            filepath = pkg_resources.resource_filename(__name__, path)
            
            error_msg = ("When attempting to load the default TBNN-s model from disk," 
                         + " no file was found in path {}.".format(filepath)
                         + " Check your installation.")
        
        # Here a path is provided (relative to the working directory), so just
        # load that file
        else:
            error_msg = ("When attempting to load a custom TBNN-s model from disk, "
                         + "no file was found in path {}.".format(filepath) 
                         + " Make sure that the file exists.")
        
        
        assert os.path.isfile(filepath), error_msg # make sure the file exists  
        
        # saved as private variables or the class 
        self._description, model_list = joblib.load(filepath)        
        FLAGS, saved_path, feat_mean, feat_std = model_list
        
        # Now, initialize TBNN-s and load parameters.        
        if filepath is None: # need to correct the directory is default model is loaded
            saved_path = pkg_resources.resource_filename(__name__, saved_path)
        self._model = TBNNS(FLAGS, saved_path, feat_mean, feat_std)        
        self._model.loadParameters(self._tfsession)
        
    
    def predict(self, x_features, tensor_basis):
        """
        Predicts alpha_ij given the features using TBNN-s model. 
        
        It uses the tbnns library which has the proper implementation. The model
        predicts a dimensionless diffusivity matrix (3x3) on each cell of the domain.
        To obtain the dimensional diffusivity, multiply alphaij by nu_t.
        
        Arguments:
        x_features -- numpy array (num_useful, N_FEATURES) of features
        tensor_basis -- numpy array (num_useful, N_BASIS, 3, 3) of tensor basis
        
        Returns:
        alphaij -- numpy array (num_useful, 3, 3) for the dimensionless tensor 
                   diffusivity in each cell
        """       
        
        assert isinstance(self._model, TBNNS), "Model is not a TBNN-s!"
        assert x_features.shape[1] == constants.N_FEATURES, "Wrong number of features!"
        assert tensor_basis.shape[1] == constants.N_BASIS, "Wrong number of tensor basis!"
        assert tensor_basis.shape[0] == x_features.shape[0], \
                               "number of features and tensor basis do not match!"
        
        print("ML model loaded: {}".format(self._description))
        print("Predicting Pr-t using ML model...", end="", flush=True)
        alphaij = self._model.getTotalDiffusivity(self._tfsession, x_features, 
                                                  tensor_basis)       
        print(" Done!")
        
        return alphaij

    
    def printParams(self):
        """
        This function prints all the trainable parameters in the loaded TBNN-s model. Use
        this for sanity checking and for seeing how big is a loaded model. Can only call
        this after model has been initialized.
        """
        
        assert isinstance(self._model, TBNNS), "Model is not a TBNN-s!"       
        self._model.printTrainableParams()