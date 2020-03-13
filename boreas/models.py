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
from boreas import constants
from tbnns.tbnns import TBNNS
from tbnns.utils import cleanDiffusivity
from joblib import Parallel, parallel_backend


def makePrediction(model_type, model_path, ensemble, x, tb=None, std_flag=False):
    """
    Predicts pr_t/alpha_ij from model specification and features. 
    
    This function constructs either the RF or TBNN-s class, loads
    them from disk, and predicts the turbulent quantity using it. It
    also uses a model ensemble if ensemble=True
    
    Arguments:
    model_type -- string, either "RF" or "TBNNS"
    model_path -- string or list, containing model path or list of model
                  paths to be loaded
    ensemble -- boolean, either True of False, determining whether a single instance
                or a model ensemble should be used.
    x -- numpy array containing features, shape (num_useful, ).
    tb -- optional, numpy array containing tensor basis for prediction of shape
          (num_useful, num_basis, 3, 3). Only used for TBNN-s model.
    std_flag -- optional, boolean array that instructs the function to return the
                standard deviation of the diffusivity INSTEAD of the diffusivity.
                Only works when ensemble=True.
    
    Returns:
    pr_t or alpha_ij -- numpy array (num_useful, ) or (num_useful, 3, 3) for the 
                       machine-learned turbulence quantity.
    """   
    
    if model_type == "RF":        
        if ensemble:
            num_models = len(model_path)
            prt_mean = np.zeros((x.shape[0],))
            prt_second_moment = np.zeros((x.shape[0],))
            print("Ensemble of {} models will be used.".format(num_models))
            
            for model in model_path:            
                rf = RFModelIsotropic()
                rf.loadFromDisk(model)
                prt = rf.predict(x)
                prt_mean += (1.0/num_models) * prt
                if std_flag: prt_second_moment += (1.0/num_models) * (prt**2)
            
            if std_flag: 
                prt_ML = np.sqrt(prt_second_moment - prt_mean**2)
            else:
                prt_ML = prt_mean        
        else:
            # Initialize model from disk and predict turbulent Prandtl number. If 
            # model_path is None, just load the default model from disk.
            rf = RFModelIsotropic()
            rf.loadFromDisk(model_path)
            prt_ML = rf.predict(x)

        if std_flag:
            print("The standard deviation across the ensemble is returned!")
            
        return prt_ML
    
    elif model_type == "TBNNS":
    
        if ensemble:
            num_models = len(model_path) 
            alphaij_mean = np.zeros((x.shape[0], 3, 3))
            alphaij_second_moment = np.zeros((x.shape[0], 3, 3))
            print("Ensemble of {} models will be used.".format(num_models))
            
            for model in model_path:
                nn = TBNNSModelAnisotropic()
                nn.loadFromDisk(model, verbose=True)
                alpha = nn.predict(x, tb, clean=False)                
                alphaij_mean += (1.0/num_models) * alpha
                if std_flag: alphaij_second_moment += (1.0/num_models) * (alpha**2)
            
            if std_flag: 
                alphaij_ML = np.sqrt(alphaij_second_moment - alphaij_mean**2)
            else:
                alphaij_ML = alphaij_mean     
                
        else:            
            # Initialize model from disk and predict tensorial diffusivity. If 
            # model_path is None, just load the default model from disk. 
            nn = TBNNSModelAnisotropic()
            nn.loadFromDisk(model_path, verbose=True)
            alphaij_ML = nn.predict(x, tb, clean=False)
        
        if std_flag:
            print("The standard deviation across the ensemble is returned!")
        if not std_flag:
            # I set clean=False and only clean at the end to speed up the ensemble
            x_star = (x - nn._model.features_mean) / nn._model.features_std # normalize
            alphaij_ML = cleanDiffusivity(alphaij_ML, test_inputs=x_star, 
                                          prt_default=constants.PR_T,
                                          gamma_min=1.0/constants.PRT_CAP)                                      
        return alphaij_ML
    
    else:
        print("Error! model_type must be either 'RF' or 'TBNNS'")


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
        Trains the regression model given data from high fidelity simulation.      
        """
        print("Training method not implemented yet!")
        pass    
    
    
    def save(self):
        """       
        Saves a trained model to disk so it can be used later. Call after train.
        """        
        print("Saving method not implemented yet!")
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
        
        
class RFModelIsotropic(MLModel):
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

    
    def train(self, x, y, n_trees=None, max_depth=None, min_samples_split=None, 
              n_jobs=None):
        """       
        Trains a random forest model to regress on log(y) given x
        
        This function trains a random forest (implemented in scikit-learn) to predict
        the turbulent Prandtl number given mean flow features. The features are x, and
        y contains gamma = 1/Prt. We predict log(y) since it is a ratio.
        
        Arguments:
        x -- numpy array containing the features at training points, of shape
             (n_useful, N_FEATURES). 
        y -- numpy array containing labels gamma = 1/Prt, of shape (n_useful,)        
        n_trees -- optional. Hyperparameter of the random forest, contains number of
                   trees to use. If None (default), reads value from constants.py
        max_depth -- optional. Hyperparameter of the random forest, contains maximum
                     depth of each tree to use. If None (default), reads value from
                     constants.py 
        min_samples_split -- optional. Hyperparameter of the random forest, contains
                             minimum number of samples at a node required to split. Can
                             either be an int (number itself) or a float (ratio of total
                             examples). If None (default), reads value from constants.py
        n_jobs -- optional. Number of processors to use when training the RF (notice that
                  training is embarassingly parallel). If None (default behavior), then
                  the value is read from constants.py. See manual for 
                  RandomForestRegressor class; if this is -1, all processors are used.
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
        if n_jobs is None:
            n_jobs = constants.N_PROCESSORS
        
        with parallel_backend('multiprocessing'):
            # Initialize class with fresh estimator
            self._model = RandomForestRegressor(n_estimators=n_trees,
                                                max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                n_jobs=n_jobs)
            
            # Train and time
            print("Training Random Forest on {} points".format(x.shape[0]))
            print("n_trees={}, max_depth={}, min_samples_split={}".format(n_trees,
                                                                        max_depth,
                                                                      min_samples_split))
            print("This may take several hours. Training...", end="", flush=True)
            tic=timeit.default_timer() # timing
            self._model.fit(x, np.log(y))
            toc=timeit.default_timer()        
            print(" Done! It took {:.3f} min".format((toc - tic)/60.0))
    

    def save(self, description, savepath):
        """       
        Saves a trained random forest model to disk so it can be used later
        
        Arguments:        
        description -- string containing a short, written description of the model,
                       which is important when the model is loaded and used at a 
                       later time.
        savepath -- string containing the path in which the model is saved to disk        
        """
        
        self._description = description
        print("Model description: {}".format(description), flush=True)
        print("Saving RF to {}...".format(savepath), end="", flush=True)
        # Save model to disk in specified location
        joblib.dump([self._description, self._model], savepath, 
                    compress=constants.COMPRESS, protocol=constants.PROTOCOL)
        print(" Done!", flush=True)
    
    
    def predict(self, x):    
        """
        Predicts Pr_t given the features using random forest model. 
        
        It assumes that the underlying implementation (default: random forests from
        scikit-learn) contains a method called predict. It also assumes that the model
        was trained to predict log(alpha_t / nu_t), so Pr_t = nu_t/alpha_t = 1/exp(y)
        
        Arguments:
        x -- numpy array (num_useful, N_FEATURES) of features
        
        Returns:
        prt -- numpy array (num_useful,) for the turbulent Prandtl number predicted at
             each cell
        """       
        
        assert isinstance(self._model, RandomForestRegressor)
        assert x.shape[1] == constants.N_FEATURES, "Wrong number of features!"
        
        print("ML model loaded: {}".format(self._description))
        print("Predicting Pr-t using RF model...", end="", flush=True)
        tic=timeit.default_timer() # timing
        with parallel_backend('multiprocessing'):
            y = self._model.predict(x)
        prt = 1.0/np.exp(y)
        toc=timeit.default_timer()        
        print(" Done! It took {:.3f} min".format((toc - tic)/60.0))
        return prt      
 
 
class TBNNSModelAnisotropic(MLModel):
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
        self._model = TBNNS() # holds the instance of the model
        
    
    def loadFromDisk(self, filepath=None, verbose=False):
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
        verbose -- optional, boolean flag indicating whether to print out model parameters
                   and flags upon loading. False by default.
        """
        
        # if no path is provided, load the default model
        if filepath is None:            
            path = 'data/defaultTBNNs.pckl' # location of the default model
            filepath = pkg_resources.resource_filename(__name__, path)            
            error_msg = ("When attempting to load the default TBNN-s model from disk," 
                         + " no file was found in path {}.".format(filepath)
                         + " Check your installation.")
            
            # this function is defined to correct the path where the tensorflow
            # checkpoint containing parameters of the model are saved
            def fn_modify(saved_path): 
                saved_path = os.path.join('data',saved_path)
                saved_path = pkg_resources.resource_filename(__name__, saved_path)
                return saved_path
        
        # Here a path is provided (relative to the working directory), so just
        # load that file
        else:
            error_msg = ("When attempting to load a custom TBNN-s model from disk, "
                         + "no file was found in path {}.".format(filepath) 
                         + " Make sure that the file exists.")
            
            fn_modify = None 
        
        assert os.path.isfile(filepath), error_msg # make sure the file exists          
                
        # load model from disk and return the description 
        self._description = self._model.loadFromDisk(filepath, verbose=verbose,
                                                     fn_modify=fn_modify)
        
    
    def predict(self, x_features, tensor_basis, clean=True):
        """
        Predicts alpha_ij given the features using TBNN-s model. 
        
        It uses the tbnns library which has the proper implementation. The model
        predicts a dimensionless diffusivity matrix (3x3) on each cell of the domain.
        To obtain the dimensional diffusivity, multiply alphaij by nu_t.
        
        Arguments:
        x_features -- numpy array (num_useful, N_FEATURES) of features
        tensor_basis -- numpy array (num_useful, N_BASIS, 3, 3) of tensor basis
        clean -- boolean, whether to clean the output diffusivity
        
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
        print("Predicting tensor diffusivity using TBNN-s model...", flush=True)
        alphaij, _ = self._model.getTotalDiffusivity(x_features, tensor_basis,
                                                     prt_default=constants.PR_T,
                                                     gamma_min=1.0/constants.PRT_CAP,
                                                     clean=clean)
        print("Done!")
        
        return alphaij

    
    def printParams(self):
        """
        This function prints all the trainable parameters in the loaded TBNN-s model. Use
        this for sanity checking and for seeing how big is a loaded model. Can only call
        this after model has been initialized.
        """
        
        assert isinstance(self._model, TBNNS), "Model is not a TBNN-s!"       
        self._model.printModelInfo()    