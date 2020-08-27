#----------------------------------- processing.py -------------------------------------#
"""
This file contains utility functions to process a case into useful quantities (stored 
in numpy arrays). These include calculating features, Pr_t, should_use, etc.
"""

# ------------ Import statements
import tecplot
import timeit
import numpy as np
import joblib
from tqdm import tqdm # progress bar
from boreas import constants


def calcInvariants(S, R, gradT, with_tensor_basis=False, reduced=True):
    """
    This function calculates the invariant basis at one point.

    Arguments:
    S -- symmetric part of local velocity gradient (numpy array shape (3,3))
    R -- anti-symmetric part of local velocity gradient (numpy array shape (3,3))
    gradT -- array with local temperature gradient (numpy array shape (3,))    
    with_tensor_basis -- optional, a flag that determines whether to also calculate
                         tensor basis. By default, it is false (so only invariants 
                         are returned)
    reduced -- optional argument, a boolean flag that determines whether the features
               that depend on a vector (lambda 7 thru lambda 13) should be calculated.
               If reduced==True, extra features are NOT calculated. Default value is 
               True.
                     
    Returns:
    invariants -- array of shape (n_features-2,) that contains the invariant basis
                  from the gradient tensors that are used by the ML model to make a
                  prediction at the current point.
    tensor_basis -- array of shape (n_basis,3,3) that contains the form invariant
                    tensor basis that are used by the TBNN to construct the tensorial
                    diffusivity at the current point.
    
    # Taken from the paper of Zheng, 1994, "Theory of representations for tensor 
      functions - A unified invariant approach to constitutive equations"
    """

    # For speed, pre-calculate these
    S2 = np.linalg.multi_dot([S, S])
    R2 = np.linalg.multi_dot([R, R])
    S_R2 = np.linalg.multi_dot([S, R2])    
    
    ### Fill basis 0-12
    if reduced: num_features = constants.NUM_FEATURES_F2-2        
    else: num_features = constants.NUM_FEATURES_F1-2
    invariants = np.empty(num_features)
    
    # Velocity gradient only (0-5)
    invariants[0] = np.trace(S2)
    invariants[1] = np.trace(np.linalg.multi_dot([S2, S]))
    invariants[2] = np.trace(R2)
    invariants[3] = np.trace(S_R2)
    invariants[4] = np.trace(np.linalg.multi_dot([S2, R2]))
    invariants[5] = np.trace(np.linalg.multi_dot([S2, R2, S, R]))
    
    # Velocity + temperature gradients (6-12)
    if not reduced:
        invariants[6] = np.linalg.multi_dot([gradT, gradT])
        invariants[7] = np.linalg.multi_dot([gradT, S, gradT])
        invariants[8] = np.linalg.multi_dot([gradT, S2, gradT])
        invariants[9] = np.linalg.multi_dot([gradT, R2, gradT])
        invariants[10] = np.linalg.multi_dot([gradT, S, R, gradT])
        invariants[11] = np.linalg.multi_dot([gradT, S2, R, gradT])
        invariants[12] = np.linalg.multi_dot([gradT, R, S_R2, gradT])
    
    # Also calculate the tensor basis
    if with_tensor_basis:
        tensor_basis = np.empty((constants.N_BASIS,3,3))    
        tensor_basis[0,:,:] = np.eye(3)
        tensor_basis[1,:,:] = S
        tensor_basis[2,:,:] = R
        tensor_basis[3,:,:] = S2
        tensor_basis[4,:,:] = R2
        tensor_basis[5,:,:] = np.linalg.multi_dot([S, R]) + np.linalg.multi_dot([R, S])        
        return invariants, tensor_basis
    
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
    

def calculateFeatures(mfq, should_use, with_tensor_basis=False, features_type="F2"):
    """
    This function calculates the ML features for this dataset. 
    
    Arguments:
    mfq -- instance of MeanFlowQuantities class containing necessary arrays to
              calculate should_use
    should_use -- boolean array, indicating which cells have large enough
                  gradient to work with. Used as a mask on the full dataset.    
    with_tensor_basis -- optional argument, boolean that determines whether we extract
                         tensor basis together with features or not. Should be True when
                         TBNN-s is employed. Default value is False.
    features_type -- optional argument, string determining the type of features that
                     we are currently extracting. Options are "F1" and "F2". Default
                     value is "F2".
                     
    Returns:
    x_features -- array of shape (n_useful, n_features) that contains the features
                  that are used by the ML model to make a prediction at each point.
    tensor_basis -- array of shape (n_useful, n_basis, 3, 3) that contains the tensor
                    basis calculate for this dataset. If with_tensor_basis is False, this
                    return value is just None.
    """
    
    # this tells us how many of the total number of elements we use for predictions        
    n_useful = np.sum(should_use)
    
    print("Out of {} total points, ".format(should_use.shape[0])
          + "{} have significant gradient and will be used".format(n_useful))    
    print("Extracting features that will be used by ML model...")
    
    # this is the feature vector
    if features_type=="F1": num_features = constants.NUM_FEATURES_F1        
    elif features_type=="F2": num_features = constants.NUM_FEATURES_F2
    x_features = np.empty((n_useful, num_features))
                
    # Non-dimensionalize in bulk and select only the points where should_use is true
    gradU_factor = 0.5*mfq.tke[should_use]/mfq.epsilon[should_use]
    gradU_temporary = mfq.gradU[should_use,:,:]*gradU_factor[:,None,None]
    gradU_temporary_transp = np.transpose(gradU_temporary, (0,2,1))
    S = gradU_temporary + gradU_temporary_transp
    R = gradU_temporary - gradU_temporary_transp
    
    if features_type=="F1":
        gradT_factor = (mfq.tke[should_use]**(1.5)/mfq.epsilon[should_use])    
        gradT_temporary = mfq.gradT[should_use,:]*gradT_factor[:,None]
    elif features_type=="F2":
        gradT_temporary = np.zeros((n_useful, 3))
    
    # Loop only where should_use is true to extract invariant basis
    # tqdm wraps around the iterable and generates a progress bar
    if with_tensor_basis:
        tensor_basis = np.empty((n_useful, constants.N_BASIS, 3, 3))
        for i in tqdm(range(n_useful)): 
            (x_features[i,0:num_features-2],
                    tensor_basis[i,:,:,:]) = calcInvariants(S[i,:,:], R[i,:,:],
                                                            gradT_temporary[i,:], 
                                                            with_tensor_basis,
                                                            features_type=="F2")
    else:
        tensor_basis = None
        for i in tqdm(range(n_useful)): 
            x_features[i,0:num_features-2] = calcInvariants(S[i,:,:], R[i,:,:],
                                                            gradT_temporary[i,:], 
                                                            with_tensor_basis,
                                                            features_type=="F2")
        
    
    # Add last two scalars to the features (distance to wall and nu_t/nu)
    Re_wall = np.sqrt(mfq.tke[should_use])*mfq.d[should_use]*\
                mfq.rho[should_use]/mfq.mu[should_use]
    nut_over_nu = mfq.mut[should_use]/mfq.mu[should_use]    
    x_features[:, num_features-2] = Re_wall
    x_features[:, num_features-1] = nut_over_nu            
    
    print("Done!")
            
    return x_features, tensor_basis
    

def cleanFeatures(x_features, tensor_basis, should_use, verbose=False):
    """
    This function removes outlier points from consideration.
    
    This function utilizes the iterative standard deviation outlier detector (ISDOD) to
    remove outlier datapoints from the dataset used to train/test. It returns
    cleaned versions of x_features and should_use.
    
    Arguments:
    x_features -- numpy array containing the features x (shape: n_useful,N_FEATURES)
    should_use -- boolean array, indicating which cells have large enough
                  gradient to work with. Used as a mask on the full dataset.
    tensor_basis -- numpy array containing the tensor basis. This
                    should be passed when the anisotropic model is employed, but
                    is irrelevant for the RF models (shape: n_useful,N_BASIS,3,3). This
                    can also be None, in case we are using an isotropic model.
    verbose -- optional argument, parameter for ISDOD. Controls whether the execution is
               verbose or not.

    Returns:
    new_x_features -- new version of x_features with only clean points
    new_tensor_basis -- new version of tensor_basis with only clean points
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
        if new_cleaned_exs == 0 or \
         (cleaned_exs is not None and (new_cleaned_exs - cleaned_exs)/cleaned_exs < tol):
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
    
    # Return different number of arguments depending on whether a tensor_basis array
    # was passed as an argument
    if tensor_basis is None:
        return new_x_features, None, new_should_use
    else:
        new_tensor_basis = tensor_basis[mask,:,:,:]
        return new_x_features, new_tensor_basis, new_should_use 
        
        
def fillPrt(Prt, should_use, default_prt):
    """
    Takes in a Pr_t array with entries only in cells of significant gradient
    and fills it up to have entries everywhere in the flow (default values in
    other places of the flow).
    
    Arguments:
    Prt -- a numpy array of shape (n_useful, ) containing the turbulent Prandtl number
           at each cell where should_use = True.
    should_use -- boolean array, indicating which cells have large enough
                  gradient to work with. Used as a mask on the full dataset.
    default_prt -- value for the default Pr_t to use where should_use == False.
                       Can be None, in which case default value is read from constants.py
               
    Returns:
    Prt_full -- a numpy array of shape (n_cells, ) containing the turbulent Prandtl
                number in every cell of the domain. Where should_use == False, use a
                fixed value of Pr_t prescribed in constants.py
    """
    
    # Sizes from should_use
    n_cells = should_use.shape[0]
    n_useful = np.sum(should_use)   
    
    # make sure Pr_t has right size and is positive
    assert Prt.size == n_useful, "Pr_t has wrong number of entries!"
    assert (Prt >= 0).all(), "Found negative entries for Prt!"
    
    # Set default value of Pr_t
    if default_prt is None:
        default_prt = constants.PRT_DEFAULT
    assert default_prt > 0, "default_prt must be a positive floating point number!"
    
    # Use Reynolds analogy everywhere first
    Prt_full = np.ones(n_cells) * default_prt # initial guess everywhere
    
    # Fill in places where should_use is True with the predicted Prt_ML:
    Prt_full[should_use] = Prt       
    
    return Prt_full
    

def fillAlpha(alphaij, should_use, default_prt):
    """
    Takes in an alpha_ij matrix with entries only in cells of significant gradient
    and fills it up to have entries everywhere in the flow (default values in
    other places of the flow).
    
    Arguments:
    alphaij -- a numpy array of shape (num_useful,3,3) containing the dimensionless
               diffusivity tensor at each cell where should_use = True.
    should_use -- boolean array, indicating which cells have large enough
                  gradient to work with. Used as a mask on the full dataset.
    default_prt -- value for the default Pr_t to use where should_use == False.
                   Can be None, in which case default value is read from constants.py
               
    Returns:
    alphaij_full -- a numpy array of shape (num_cells,3,3) containing a dimensionless
                    turbulent diffusivity in every cell of the domain. In cells
                    where should_use == False, use an isotropic diffusivity based on
                    a fixed Pr_t given in constants.py, and zero off-diagonal entries.
                    NOTE: this returns the diffusivity alpha_t/nu_t, and not the 
                    turbulent Prandtl number nu_t/alpha_t
    """
    
    # Sizes from should_use
    n_cells = should_use.shape[0]
    n_useful = np.sum(should_use)   
    
    # make sure alphaij has right size and has non-negative diagonals
    assert alphaij.shape[0] == n_useful, "alphaij has wrong number of entries!"
    assert alphaij.shape[1] == 3 and alphaij.shape[2] == 3, "alphaij is not a 3D tensor!"
    assert (alphaij[:,0,0] >= 0).all(), "Found negative entries for alpha_xx"
    assert (alphaij[:,1,1] >= 0).all(), "Found negative entries for alpha_yy"
    assert (alphaij[:,2,2] >= 0).all(), "Found negative entries for alpha_zz"
    
    # Set default value of Pr_t
    if default_prt is None:
        default_prt = constants.PRT_DEFAULT
    assert default_prt > 0, "default_prt must be a positive floating point number!"
    
    # Use Reynolds analogy everywhere first
    alphaij_full = np.zeros((n_cells,3,3))
    for i in range(3):
        alphaij_full[:,i,i] = 1.0/default_prt
    
    # Fill in places where should_use is True with the predicted Prt_ML:
    alphaij_full[should_use,:,:] = alphaij       
    
    return alphaij_full

    
def fillG(g, should_use, default_prt):
    """
    Analogous to above function. Takes in a g array with entries only in cells of
    significant gradient and fills it up to have entries everywhere in the flow 
    (with default values in places of the flow without significant gradients)
    
    Arguments:
    g -- a numpy array of shape (num_useful,num_basis) containing the dimensionless
               diffusivity tensor at each cell where should_use = True.
    should_use -- boolean array, indicating which cells have large enough
                  gradient to work with. Used as a mask on the full dataset.
    default_prt -- value for the default Pr_t to use where should_use == False.
                   Can be None, in which case default value is read from constants.py
               
    Returns:
    g_full -- a numpy array of shape (num_cells,num_basis) containing g in every cell of
              the domain. In cells where should_use == False, use an isotropic
              diffusivity based on default_prt.
    """
    
    # Sizes from should_use
    n_cells = should_use.shape[0]
    n_useful = np.sum(should_use)   
    
    # make sure alphaij has right size and has non-negative diagonals
    assert g.shape[0] == n_useful, "g has wrong number of entries!"
    assert g.shape[1] == constants.N_BASIS, "g has wrong shape!"    
    
    # Set default value of Pr_t
    if default_prt is None:
        default_prt = constants.PRT_DEFAULT
    assert default_prt > 0, "default_prt must be a positive floating point number!"
    
    # Use Reynolds analogy everywhere first
    g_full = np.zeros((n_cells,constants.N_BASIS))
    g_full[:,0] = 1.0/default_prt  
    
    # Fill in places where should_use is True with the predicted Prt_ML:
    g_full[should_use,:] = g       
    
    return g_full

    
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
              + np.abs(mfq.u*np.expand_dims(mfq.t, axis=1)))
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
    

def downsampleIdx(n_total, downsample):
    """
    Produces a set of indices to index into a numpy array and shuffle/downsample it.
    
    Arguments:
    n_total -- int, total size of the array in that dimensionalization
    downsample -- number that controls how we downsample the data
                  before saving it to disk. If None, it is deactivated.
                  If this number is more than 1,
                  then it represents the number of examples we want to save; if
                  it is less than 1, it represents the ratio of all training 
                  examples we want to save.
                  
    Returns:
    idx -- numpy array of ints of size (n_take), which contains the indices
           that we are supposed to take for downsampling.    
    """
    
    idx_tot = np.arange(n_total)
    np.random.shuffle(idx_tot)
    
    if downsample is None: # if downsample=None, deactivates downsampling
        return idx_tot
    
    assert downsample > 0, "downsample must be greater than 0!"
    
    if int(downsample) > 1:
        n_take = int(downsample)            
        if n_take > n_total:
            print("Warning! This dataset has fewer than {} usable points. "
                  + "All of them will be taken.".format(n_take))
            n_take = n_total                 
    else:
        n_take = int(downsample * n_total)
        if n_take > n_total: n_take = n_total # catches bug where downsample = 1.1            
    
    idx = idx_tot[0:n_take]
    
    return idx 
    
    
def saveTrainingFeatures(training_list, metadata, filename, downsample):
    """
    Saves a .pckl file with the features and labels for training. Downsamples data
    if necessary
    
    Arguments:
    training_list -- a list (of variable length) containing numpy arrays that
                     are required for training. We want to save all of them to
                     disk
    metadata -- dictionary containing information about the saved data. This will be 
                read and asserted when the present data is read for training       
    filename -- the location/name of the file where we save the features
                and labels
    downsample -- number that controls how we downsample the data
                  before saving it to disk. If None, it is deactivated. 
                  If this number is more than 1, then it represents the number of
                  examples we want to save; if it is less than 1, it represents
                  the ratio of all training examples we want to save.
    """        
    
    print("Saving features and labels to disk in file {}...".format(filename),
          end="", flush=True)
    
    # These lines implement downsampling
    n_total = training_list[0].shape[0] # size of first axis of first entry
    idx = downsampleIdx(n_total, downsample)
    
    save_var = [] # Variables that will be saved
    for var in training_list:
        save_var.append(var[idx]) # downsample variable by variable
        
    joblib.dump([metadata, save_var], filename, compress=constants.COMPRESS,
                protocol=constants.PROTOCOL)
    
    print(" Done!")
 

def loadTrainingFeatures(file, model_type, downsample, features_type):
    """
    Counterpart to saveTrainingFeatures, this function loads features and labels for
    training.
    
    Arguments:
    file -- string containing the location of the file that will be read
    model_type -- string containing the model type, like "RF" or "TBNNS"
    downsample -- number that controls how we downsample the data
                  before saving it to disk. If None, it is deactivated. 
                  If this number is more than 1,
                  then it represents the number of examples we want to save; if
                  it is less than 1, it represents the ratio of all training 
                  examples we want to save.
    features_type -- string determining the type of features that
                     we are currently extracting. Options are "F1" and "F2".
                  
    Returns:
    training_list -- a list (of variable length) containing numpy arrays that
                     are required for training. This list differs depending on
                     the model type.    
    """
    
    assert model_type == "RF" or model_type == "TBNNS" or model_type == "TBNNS_hybrid", \
           "Invalid model_type received!"
    
    # Load from disk, check features type, and prepare downsampling indices
    metadata, save_vars = joblib.load(file)
    assert metadata["features_type"] == features_type, "Loaded incorrect feature type!"
    n_points = save_vars[0].shape[0]
    idx = downsampleIdx(n_points, downsample) # downsampling
    
    # Assert correct number of features and that all saved variables have same
    # length (first dimension) 
    if features_type=="F1": 
        assert save_vars[0].shape[1] == constants.NUM_FEATURES_F1, \
            "Wrong number of F1 features!"
    if features_type=="F2": 
        assert save_vars[0].shape[1] == constants.NUM_FEATURES_F2, \
            "Wrong number of F2 features!"        
    for var in save_vars:
        assert var.shape[0] == n_points, "Wrong number of points in file {}".format(file)
    
    # Gather required variables and return
    if model_type=="RF":
        assert metadata["with_gamma"]==True, "To train RF, we need gamma!"
        x = save_vars[0][idx]
        gamma = save_vars[-1][idx]
        return x, gamma    
    elif model_type=="TBNNS":
        assert metadata["with_tensor_basis"]==True, "To train TBNN-s, we need tb!"
        x = save_vars[0][idx]
        tb = save_vars[1][idx]
        uc = save_vars[2][idx]
        gradT = save_vars[3][idx]
        nut = save_vars[4][idx] 
        return x, tb, uc, gradT, nut    
    elif model_type=="TBNNS_hybrid":
        assert metadata["with_gamma"]==True, "To train RF, we need gamma!"
        assert metadata["with_tensor_basis"]==True, "To train TBNN-s, we need tb!"
        x = save_vars[0][idx]
        tb = save_vars[1][idx]
        uc = save_vars[2][idx]
        gradT = save_vars[3][idx]
        nut = save_vars[4][idx]
        gamma = save_vars[5][idx]
        return x, tb, uc, gradT, nut, gamma    
      
  
class MeanFlowQuantities:
    """
    This class holds numpy arrays that correspond to the different mean flow
    quantities of one particular dataset. Its data are used to calculate 
    features, should_use, gamma, u'c', etc
    """    
    
    def __init__(self, zone, var_names, features_type="F2", deltaT0=1, 
                 labels=False, mask=None):
        """
        Constructor for MeanFlowQuantities class.
        
        Arguments:
        zone -- tecplot.data.zone containing the fluid zone, where the variables live
        var_names -- dictionary mapping default names to the actual variable names in the
                     present .plt file
        features_type -- optional argument, string containing the type of features we
                         are going to produce. Options are "F1" or "F2"; "F1" is the
                         standard, "F2" removes the temperature gradient features.
        deltaT0 -- scale to non-dimensionalize the temperature gradient. Optional,
                   when deltaT0=1 then it does not do anything to the dataset
        labels -- optional, boolean flag which indicates if we are using this class to 
                  calculate the features or to calculate the labels. By default, False
                  which means we are calculating features only.
        mask -- optional, array containing a mask to apply to all the arrays
                extracted. If None, then no mask is used and all points are taken.
        """       
        
        # Print what we are doing
        if labels: print("Loading data for label calculation...", end="", flush=True)
        else: print("Loading data for feature calculation...", end="", flush=True)
        
        # Update mask and save labels
        if mask is None: # if no mask is provided, take all points
            mask = np.ones(zone.num_elements, dtype=bool)        
        self.n_points = np.sum(mask) # number of points used  
        self.labels = labels
        
        # Variables I need no matter what
        self.gradT = np.empty((self.n_points, 3)) # this is a 2D array of size Nx3
        self.gradT[:, 0] = zone.values(var_names["ddx_T"]).as_numpy_array()[mask]
        self.gradT[:, 1] = zone.values(var_names["ddy_T"]).as_numpy_array()[mask]
        self.gradT[:, 2] = zone.values(var_names["ddz_T"]).as_numpy_array()[mask]
        self.rho  = zone.values(var_names["Density"]).as_numpy_array()[mask]
        self.mut = zone.values(var_names["turbulent viscosity"]).as_numpy_array()[mask]
        
        # Variables only needed when extracting labels
        if labels:
            # Mean velocity: U, V, W
            self.u = np.empty((self.n_points, 3)) # this is a 2D array of size Nx3
            self.u[:, 0] = (zone.values(var_names["U"]).as_numpy_array())[mask]
            self.u[:, 1] = (zone.values(var_names["V"]).as_numpy_array())[mask]
            self.u[:, 2] = (zone.values(var_names["W"]).as_numpy_array())[mask]
            
            # u'c' vector
            self.uc = np.empty((self.n_points, 3)) # this is a 2D array of size Nx3
            self.uc[:, 0] = (zone.values(var_names["uc"]).as_numpy_array())[mask]
            self.uc[:, 1] = (zone.values(var_names["vc"]).as_numpy_array())[mask]
            self.uc[:, 2] = (zone.values(var_names["wc"]).as_numpy_array())[mask]
            
            self.t  = (zone.values(var_names["T"]).as_numpy_array())[mask]  
        
        # Variables only needed when calculating features
        else:
            # Velocity Gradients: dudx, dudy, dudz, dvdx, dydy, dydz, dwdx, dwdy, dwdz
            self.gradU = np.empty((self.n_points, 3, 3)) # 3D array of size Nx3x3
            self.gradU[:, 0, 0] = zone.values(var_names["ddx_U"]).as_numpy_array()[mask]
            self.gradU[:, 1, 0] = zone.values(var_names["ddy_U"]).as_numpy_array()[mask]
            self.gradU[:, 2, 0] = zone.values(var_names["ddz_U"]).as_numpy_array()[mask]
            self.gradU[:, 0, 1] = zone.values(var_names["ddx_V"]).as_numpy_array()[mask]
            self.gradU[:, 1, 1] = zone.values(var_names["ddy_V"]).as_numpy_array()[mask]
            self.gradU[:, 2, 1] = zone.values(var_names["ddz_V"]).as_numpy_array()[mask]
            self.gradU[:, 0, 2] = zone.values(var_names["ddx_W"]).as_numpy_array()[mask]
            self.gradU[:, 1, 2] = zone.values(var_names["ddy_W"]).as_numpy_array()[mask]
            self.gradU[:, 2, 2] = zone.values(var_names["ddz_W"]).as_numpy_array()[mask]
                
            # Other scalars: density, tke, epsilon, mu_t, mu, distance to wall
            self.tke = zone.values(var_names["TKE"]).as_numpy_array()[mask]
            self.epsilon = zone.values(var_names["epsilon"]).as_numpy_array()[mask]        
            self.mu = zone.values(var_names["laminar viscosity"]).as_numpy_array()[mask]      
            self.d = zone.values(var_names["distance to wall"]).as_numpy_array()[mask]           
        
        # Non-dimensionalization done in different method
        if features_type == "F1" and labels==False:
            self.nonDimensionalize(deltaT0)
        
        # Check that all variables have the correct size and range
        self.check()
        
        print(" Done!")
        
    def nonDimensionalize(self, deltaT0):
        """
        This function is called to non-dimensionalize the temperature gradient.
        """        
        if not self.labels:       
            self.gradT /= deltaT0        
        
    def check(self):
        """
        This function contains assertions to check the current state of the dataset
        """        
        
        assert self.rho.size == self.n_points, \
               "Wrong number of entries for rho. Check that it is cell centered."
        assert self.mut.size == self.n_points, \
               "Wrong number of entries for mu_t. Check that it is cell centered."
        assert (self.rho >= 0).all(), "Found negative entries for rho!"
        assert (self.mut >= 0).all(), "Found negative entries for mut!"
        
        if self.labels:
            assert self.t.size == self.n_points, \
               "Wrong number of entries for T"        
        else:
            assert self.tke.size == self.n_points, \
               "Wrong number of entries for TKE. Check that it is cell centered."               
            assert self.epsilon.size == self.n_points, \
                   "Wrong number of entries for epsilon. Check that it is cell centered."            
            assert self.mu.size == self.n_points, \
                   "Wrong number of entries for mu. Check that it is cell centered."
            assert self.d.size == self.n_points, \
                   "Wrong number of entries for d. Check that it is cell centered."            
            assert (self.tke >= 0).all(), "Found negative entries for tke!"
            assert (self.epsilon >= 0).all(), "Found negative entries for epsilon!"            
            assert (self.mu >= 0).all(), "Found negative entries for mu!"
            assert (self.d >= 0).all(), "Found negative entries for d!"