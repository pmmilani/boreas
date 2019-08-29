#----------------------------------- processing.py -------------------------------------#
"""
This file contains utility functions to process a case into useful quantities (stored 
in numpy arrays). These include calculating features, Pr_t, should_use, etc.
"""

# ------------ Import statements
import tecplot
import numpy as np
import time # needed by tqdm
from tqdm import tqdm # progress bar
from rafofc import constants


def calcInvariants(gradU, gradT, n_features):
    """
    This function calculates the invariant basis at one point.

    Arguments:
    gradU -- 2D tensor with local velocity gradient (numpy array shape (3,3))
    gradT -- array with local temperature gradient (numpy array shape (3,))
    n_features -- number of features for the ML model
    
    Returns:
    invariants -- array of shape (n_features-2,) that contains the invariant basis
                  from the gradient tensors 
                  that are used by the ML model to make a prediction at the current point
    
    # from Zheng (1994)
    """

    S = (gradU + np.transpose(gradU)) # symmetric component
    R = (gradU - np.transpose(gradU)) # anti-symmetric component
     
    # For speed, pre-calculate these
    S2 = np.linalg.multi_dot([S, S])
    R2 = np.linalg.multi_dot([R, R])
    S_R2 = np.linalg.multi_dot([S, R2])    
    
    ### Fill basis 0-12 
    invariants = np.empty(n_features-2)
    
    # Velocity gradient only (0-5)
    invariants[0] = np.trace(S2)
    invariants[1] = np.trace(np.linalg.multi_dot([S2, S]))
    invariants[2] = np.trace(R2)
    invariants[3] = np.trace(S_R2)
    invariants[4] = np.trace(np.linalg.multi_dot([S2, R2]))
    invariants[5] = np.trace(np.linalg.multi_dot([S2, R2, S, R]))
    
    # Velocity + temperature gradients (6-12)
    invariants[6] = np.linalg.multi_dot([gradT, gradT])
    invariants[7] = np.linalg.multi_dot([gradT, S, gradT])
    invariants[8] = np.linalg.multi_dot([gradT, S2, gradT])
    invariants[9] = np.linalg.multi_dot([gradT, R2, gradT])
    invariants[10] = np.linalg.multi_dot([gradT, S, R, gradT])
    invariants[11] = np.linalg.multi_dot([gradT, S2, R, gradT])
    invariants[12] = np.linalg.multi_dot([gradT, R, S_R2, gradT])
    
    return invariants
    
    
def calculateShouldUse(mfq, threshold):
    """
    This function is called to determine in which points we should make a prediction.

    We basically take a threshold (typically 1e-3) and fill the self.should_use with
    True in locations that we should use for the prediction (i.e., high scalar
    gradient) and False in locations we should not use.
    
    Arguments:
    mfq -- instance of MeanFlowQuantities class containing necessary arrays to
              calculate should_use
    threshold -- cut-off for the scalar gradient magnitude; we only make a prediction
                 in points with higher magnitude than this        
    """
    
    # the magnitude of the scalar gradient, shape (n_cells)
    grad_mag = np.sqrt(np.sum(mfq.gradT**2, axis=1))
    
    # Takes the gradient and non-dimensionalizes the length part of it
    grad_mag = grad_mag*(mfq.tke**(1.5)/mfq.epsilon)

    should_use = (grad_mag >= threshold)
    
    return should_use
    

def calculateFeatures(mfq, should_use):
    """
    This function calculates the ML features for this dataset. 
    
    Arguments:
    mfq -- instance of MeanFlowQuantities class containing necessary arrays to
              calculate should_use
    should_use -- boolean array, indicating which cells have large enough
                  gradient to work with. Used as a mask on the full dataset.   
                 
    Returns:
    x_features -- array of shape (n_useful, n_features) that contains the features
                  that are used by the ML model to make a prediction at each point.
    """
    
    # this tells us how many of the total number of elements we use for predictions        
    n_useful = np.sum(should_use)
    
    print("Out of {} total points, ".format(mfq.n_cells)
          + "{} have significant gradient and will be used".format(n_useful))
    print("Extracting features that will be used by ML model...")
    
    # this is the feature vector
    x_features = np.empty((n_useful, constants.N_FEATURES))         
            
    # Non-dimensionalize in bulk and select only the points where should_use is true
    gradU_factor = 0.5*mfq.tke[should_use]/mfq.epsilon[should_use]
    gradT_factor = (mfq.tke[should_use]**(1.5)/mfq.epsilon[should_use])
    gradU_temporary = mfq.gradU[should_use,:,:]*gradU_factor[:,None,None]
    gradT_temporary = mfq.gradT[should_use,:]*gradT_factor[:,None]
    
    # Loop only where should_use is true to extract invariant basis
    # tqdm wraps around the iterable and generates a progress bar
    for i in tqdm(range(n_useful)): 
        x_features[i, 0:constants.N_FEATURES-2] = calcInvariants(gradU_temporary[i,:,:],
                                                                 gradT_temporary[i,:],
                                                                 constants.N_FEATURES)            
    
    # Add last two scalars to the features (distance to wall and nu_t/nu)
    Re_wall = np.sqrt(mfq.tke[should_use])*mfq.d[should_use]*\
                mfq.rho[should_use]/mfq.mu[should_use]
    nut_over_nu = mfq.mut[should_use]/mfq.mu[should_use]    
    x_features[:, constants.N_FEATURES-2] = Re_wall
    x_features[:, constants.N_FEATURES-1] = nut_over_nu            
    
    print("Done!")
            
    return x_features # return the features, only where should_use == true
    

def cleanFeatures(x_features, should_use, verbose=False):
    """
    This function removes outlier points from consideration.
    
    This function utilizes the iterative standard deviation outlier detector (ISDOD) to
    remove outlier datapoints from the dataset used to train/test. It returns
    cleaned versions of x_features and should_use.
    
    Arguments:
    x_features -- numpy array containing the features x (shape: n_useful, N_FEATURES)
    should_use -- boolean array, indicating which cells have large enough
                  gradient to work with. Used as a mask on the full dataset.    
    verbose -- optional argument, parameter for ISDOD. Controls whether the execution is
               verbose or not.

    Returns:
    new_x_features -- new version of x_features with only clean points
    new_should_use -- new version of should_use with only clean points
    """
    
    # Hyperparameters, defined in constants.py    
    n_std=constants.N_STD # controls how many standard deviations around the mean
    max_iter=constants.MAX_ITER # maximum number of iterations allowed
    tol=constants.TOL # tolerance for cleaning; stops iterating if less than tol get cleaned
    max_clean=constants.MAX_CLEAN # maximum percentage of the dataset that can be cleaned  
    
    mask = np.ones(x_features.shape[0], dtype=np.bool_)
    warning_flag = True # if this is true, we have a particularly noisy dataset
    print("Cleaning features with n_std={} and max_iter={}...".format(n_std, max_iter))
    
    # Iterate through max iterations
    cleaned_exs = None
    for i in range(max_iter):
        
        # Calculate mean and std for each feature with only the masked data
        x_mean = np.mean(x_features[mask, :], axis=0, keepdims=True)
        x_std = np.std(x_features[mask, :], axis=0, keepdims=True)
        
        # boolean matrix (shape: n_examples, n_features) where True means that point
        # is an outlier
        new_mask = (x_features < x_mean - n_std*x_std)+(x_features > x_mean + n_std*x_std) 
        
        # eliminate the whole example if any feature is bad (sum is logical or)
        new_mask = np.sum(new_mask, axis = 1).astype(np.bool_)
        
        # mask should contain True when we want to use that example, so we negate it
        mask = np.logical_not(new_mask)        
        
        # Calculate how many examples are cleaned and break if not changing
        new_cleaned_exs = x_features.shape[0] - np.sum(mask)
        if cleaned_exs is not None and (new_cleaned_exs - cleaned_exs)/cleaned_exs < tol:
            cleaned_exs = new_cleaned_exs
            warning_flag = False
            break
        cleaned_exs = new_cleaned_exs
        
        # Break if we already cleaned too many examples
        if cleaned_exs/x_features.shape[0] > max_clean:
            break        
        
        # Print per-iteration status
        if verbose:
            print("\t Iteration {}, examples eliminated: {}".format(i, new_cleaned_exs))
            print("\t Mean/std of feature 5: {:g}/{:g}".format(x_mean[0,5], x_std[0,5]))
    
    # Print messages at the end
    print("Done! {:.2f}% of points cleaned".format(100*(1.0-np.sum(mask)/mask.shape[0])))
    if warning_flag:
        print("Warning! Cleaning algorithm did not fully converge, which indicates"
              + " this dataset is particularly noisy...")
              
    # Modify x_features and should_use and return new values
    new_x_features = x_features[mask, :]
    new_should_use = np.zeros(should_use.shape[0], dtype=np.bool_)
    new_should_use[should_use] = mask
    return new_x_features, new_should_use


def fillPrt(Prt, should_use):
    """
    Takes in a 'skinny' Pr_t field and returns a full one
    
    Arguments:
    Prt -- a numpy array shape (n_useful, ) containing the turbulent Prandtl number
           at each cell where should_use == True. This is the dimensionless
           diffusivity directly predicted by the machine learning model.
    should_use -- boolean array, indicating which cells have large enough
                  gradient to work with. Used as a mask on the full dataset.
               
    Returns:
    Prt_full -- a numpy array of shape (n_cells, ) containing a dimensional
                turbulent diffusivity in every cell of the domain. In cells
                where should_use == False, use fixed Pr_t assumption.
    """
    
    # Sizes from should_use
    n_cells = should_use.shape[0]
    n_useful = np.sum(should_use)   
    
    # make sure Pr_t has right size and is positive
    assert Prt.size == n_useful, "Pr_t has wrong number of entries!"
    assert (Prt >= 0).all(), "Found negative entries for Prt!"
    
    # Use Reynolds analogy everywhere first
    Prt_full = np.ones(n_cells) * constants.PR_T # initial guess everywhere
    
    # Fill in places where should_use is True with the predicted Prt_ML:
    Prt_full[should_use] = Prt       
    
    return Prt_full

    
def calculateGamma(mfq, prt_cap, use_correction):
    """
    This function calculates gamma = 1/Pr_t from the training data
    
    Arguments:
    mfq -- instance of MeanFlowQuantities class containing necessary arrays to
              calculate should_use
    prt_cap -- number that determines max/min values for Pr_t (and gamma)
    use_correction -- boolean, which determines whether correction for gamma
                      should be employed.
                 
    Returns:
    gamma -- array of shape (n_useful, ) that contains the label that will
             be used to train the regression algorithm.
    """
    
    print("Calculating 1/Pr_t from training data...", end="", flush=True)
    
    # Here, calculate gamma = 1.0/log(Pr_t) from u'c' data.
    if use_correction:        
        # F is a factor that weighs all three directions unequally. 
        # It gives higher weight when the mean velocity in that direction is lower.
        # See paper for equation.
        F1 = (np.sqrt(np.sum(mfq.uc**2, axis=1, keepdims=True))
              + np.abs(mfq.U*np.expand_dims(mfq.T, axis=1)))
        F = 1.0/F1 
        top = np.sum(F*mfq.uc*mfq.gradT, axis=1)
        bottom = np.sum(F*mfq.gradT*mfq.gradT, axis=1)
        alpha_t = (-1.0)*top/bottom
        gamma = alpha_t*(mfq.rho/mfq.mut)
    else:
        top = np.sum(mfq.uc*mfq.gradT, axis=1)
        bottom = np.sum(mfq.gradT*mfq.gradT, axis=1)
        alpha_t = (-1.0)*top/bottom            
        gamma = alpha_t*(mfq.rho/mfq.mut)    
    
    # Clip extracted Prt value
    gamma[gamma > prt_cap] = prt_cap
    gamma[gamma < 1.0/prt_cap] = 1.0/prt_cap
    
    print(" Done!")
    
    return gamma
    
    
class MeanFlowQuantities:
    """
    This class holds numpy arrays that correspond to the different mean flow
    quantities of one particular dataset. Its data are used to calculate 
    features, should_use, etc
    """    
    
    def __init__(self, zone, var_names, deltaT0):    
        """
        Constructor for MeanFlowQuantities class.
        
        Arguments:
        zone -- tecplot.data.zone containing the fluid zone, where the variables live
        var_names -- dictionary mapping default names to the actual variable names in the
                     present .plt file
        deltaT0 -- scale to non-dimensionalize the temperature gradient
        """       
        
        print("Loading data for feature calculation...", end="", flush=True)
        self.n_cells = zone.num_elements # number of cell centers
        
        # Velocity Gradients: dudx, dudy, dudz, dvdx, dydy, dydz, dwdx, dwdy, dwdz
        self.gradU = np.empty((self.n_cells, 3, 3)) # this is a 3D array of size Nx3x3
        self.gradU[:, 0, 0] = np.asarray(zone.values(var_names["ddx_U"])[:])
        self.gradU[:, 1, 0] = np.asarray(zone.values(var_names["ddy_U"])[:])
        self.gradU[:, 2, 0] = np.asarray(zone.values(var_names["ddz_U"])[:])
        self.gradU[:, 0, 1] = np.asarray(zone.values(var_names["ddx_V"])[:])
        self.gradU[:, 1, 1] = np.asarray(zone.values(var_names["ddy_V"])[:])
        self.gradU[:, 2, 1] = np.asarray(zone.values(var_names["ddz_V"])[:])
        self.gradU[:, 0, 2] = np.asarray(zone.values(var_names["ddx_W"])[:])
        self.gradU[:, 1, 2] = np.asarray(zone.values(var_names["ddy_W"])[:])
        self.gradU[:, 2, 2] = np.asarray(zone.values(var_names["ddz_W"])[:])
        
        # Temperature Gradients: dTdx, dTdy, dTdz
        self.gradT = np.empty((self.n_cells, 3)) # this is a 2D array of size Nx3
        self.gradT[:, 0] = np.asarray(zone.values(var_names["ddx_T"])[:])
        self.gradT[:, 1] = np.asarray(zone.values(var_names["ddy_T"])[:])
        self.gradT[:, 2] = np.asarray(zone.values(var_names["ddz_T"])[:])
        
        # Other scalars: density, tke, epsilon, mu_t, mu, distance to wall
        self.tke = np.asarray(zone.values(var_names["TKE"])[:])
        self.epsilon = np.asarray(zone.values(var_names["epsilon"])[:])
        self.rho  = np.asarray(zone.values(var_names["Density"])[:])
        self.mu = np.asarray(zone.values(var_names["laminar viscosity"])[:])
        self.mut = np.asarray(zone.values(var_names["turbulent viscosity"])[:])
        self.d = np.asarray(zone.values(var_names["distance to wall"])[:])
        
        # Non-dimensionalization done in different method
        self.nonDimensionalize(deltaT0)
        
        # Check that all variables have the correct size and range
        self.sanityCheck()
        
        print(" Done!")
        
    def nonDimensionalize(self, deltaT0):
        """
        This function is called to non-dimensionalize the temperature gradient.
        """
        
        self.gradT /= deltaT0
        
        
    def sanityCheck(self):
        """
        This function contains assertions to check the current state of the dataset
        """        
         
        assert self.tke.size == self.n_cells, \
               "Wrong number of entries for TKE. Check that it is cell centered."               
        assert self.epsilon.size == self.n_cells, \
               "Wrong number of entries for epsilon. Check that it is cell centered."
        assert self.rho.size == self.n_cells, \
               "Wrong number of entries for rho. Check that it is cell centered."
        assert self.mut.size == self.n_cells, \
               "Wrong number of entries for mu_t. Check that it is cell centered."
        assert self.mu.size == self.n_cells, \
               "Wrong number of entries for mu. Check that it is cell centered."
        assert self.d.size == self.n_cells, \
               "Wrong number of entries for d. Check that it is cell centered."
        
        assert (self.tke >= 0).all(), "Found negative entries for tke!"
        assert (self.epsilon >= 0).all(), "Found negative entries for epsilon!"
        assert (self.rho >= 0).all(), "Found negative entries for rho!"
        assert (self.mut >= 0).all(), "Found negative entries for mut!"
        assert (self.mu >= 0).all(), "Found negative entries for mu!"
        assert (self.d >= 0).all(), "Found negative entries for d!"
        

class MeanFlowQuantities_Prt:
    """
    This class holds numpy arrays that correspond to the different mean flow 
    quantities of the dataset that are needed to calculate the "true" Pr_t at
    training time.
    """    
    
    def __init__(self, zone, var_names, should_use):    
        """
        Constructor for MeanFlowQuantities class.
        
        Arguments:
        zone -- tecplot.data.zone containing the fluid zone, where the variables live
        var_names -- dictionary mapping default names to the actual variable names in the
                     present .plt file
        should_use -- boolean array, indicating which cells have large enough
                      gradient to work with. Used as a mask on the full dataset.   
        """       
        
        print("Loading data for Prt calculation...", end="", flush=True)
        self.n_useful = np.sum(should_use) # number of useful cells
        
        # Mean velocity: U, V, W
        self.U = np.empty((self.n_useful, 3)) # this is a 2D array of size Nx3
        self.U[:, 0] = np.asarray(zone.values(var_names["U"])[:])[should_use]
        self.U[:, 1] = np.asarray(zone.values(var_names["V"])[:])[should_use]
        self.U[:, 2] = np.asarray(zone.values(var_names["W"])[:])[should_use]
        
        # Temperature Gradients: dTdx, dTdy, dTdz
        self.gradT = np.empty((self.n_useful, 3)) # this is a 2D array of size Nx3
        self.gradT[:, 0] = np.asarray(zone.values(var_names["ddx_T"])[:])[should_use]
        self.gradT[:, 1] = np.asarray(zone.values(var_names["ddy_T"])[:])[should_use]
        self.gradT[:, 2] = np.asarray(zone.values(var_names["ddz_T"])[:])[should_use]
        
        # u'c' vector
        self.uc = np.empty((self.n_useful, 3)) # this is a 2D array of size Nx3
        self.uc[:, 0] = np.asarray(zone.values(var_names["uc"])[:])[should_use]
        self.uc[:, 1] = np.asarray(zone.values(var_names["vc"])[:])[should_use]
        self.uc[:, 2] = np.asarray(zone.values(var_names["wc"])[:])[should_use]
        
        # Other scalars: T, rho, mu_t
        self.T  = np.asarray(zone.values(var_names["T"])[:])[should_use]
        self.rho  = np.asarray(zone.values(var_names["Density"])[:])[should_use]
        self.mut = np.asarray(zone.values(var_names["turbulent viscosity"])[:])[should_use]
        
        # Check that all variables have the correct size and range
        self.sanityCheck()
        
        print("Done!")
        
    def sanityCheck(self):
        """
        This function contains assertions to check the current state of the dataset
        """        
        
        assert self.T.size == self.n_useful, \
               "Wrong number of entries for T"
        assert self.rho.size == self.n_useful, \
               "Wrong number of entries for rho"
        assert self.mut.size == self.n_useful, \
               "Wrong number of entries for mu_t"
        
        assert (self.rho >= 0).all(), "Found negative entries for rho!"
        assert (self.mut >= 0).all(), "Found negative entries for mut!"
        