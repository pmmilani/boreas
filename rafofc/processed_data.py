#--------------------------------- processed_data.py -----------------------------------#
"""
This file contains the class that holds a processed dataset (a bunch of numpy arrays with all
the data from a RANS k-epsilon simulation).
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
    
    # from Smith, same order I have in my notebook page 12
    invariants[0] = np.trace(np.linalg.multi_dot([S, S]))
    invariants[1] = np.trace(np.linalg.multi_dot([S, S, S]))
    invariants[2] = np.trace(np.linalg.multi_dot([R, R]))
    invariants[3] = np.trace(np.linalg.multi_dot([R, R, S]))
    invariants[4] = np.trace(np.linalg.multi_dot([R, R, S, S]))
    invariants[5] = np.trace(np.linalg.multi_dot([R, R, S, R, S, S]))
    invariants[6] = np.linalg.multi_dot([gradT, gradT])
    invariants[7] = np.linalg.multi_dot([gradT, S, gradT])
    invariants[8] = np.linalg.multi_dot([gradT, S, S, gradT])
    invariants[9] = np.linalg.multi_dot([gradT, R, R, gradT])
    invariants[10] = np.linalg.multi_dot([gradT, R, S, gradT])
    invariants[11] = np.linalg.multi_dot([gradT, R, S, S, gradT])
    invariants[12] = np.linalg.multi_dot([gradT, S, R, S, S, gradT])
    invariants[13] = np.linalg.multi_dot([gradT, R, R, S, gradT])
    invariants[14] = np.linalg.multi_dot([gradT, R, R, S, S, gradT])
    invariants[15] = np.linalg.multi_dot([gradT, R, R, S, R, gradT])
    invariants[16] = np.linalg.multi_dot([gradT, R, S, S, R, R, gradT])
    invariants[17] = Re_wall
    invariants[18] = nut/nu
    """

    S = (gradU + np.transpose(gradU)) # symmetric component
    R = (gradU - np.transpose(gradU)) # anti-symmetric component
     
    # For speed, pre-calculate these
    S2 = np.linalg.multi_dot([S, S])
    R2 = np.linalg.multi_dot([R, R])
    R2_S = np.linalg.multi_dot([R2, S])   
    R_S2 = np.linalg.multi_dot([R, S2])  
    
    ### Fill basis 0-16 
    invariants = np.empty(n_features-2)
    
    # Velocity gradient only (0-5)
    invariants[0] = np.trace(S2)
    invariants[1] = np.trace(np.linalg.multi_dot([S2, S]))
    invariants[2] = np.trace(R2)
    invariants[3] = np.trace(R2_S)
    invariants[4] = np.trace(np.linalg.multi_dot([R2, S2]))
    invariants[5] = np.trace(np.linalg.multi_dot([R2_S, R_S2]))
    
    # Velocity + temperature gradients (6-16)
    invariants[6] = np.linalg.multi_dot([gradT, gradT])
    invariants[7] = np.linalg.multi_dot([gradT, S, gradT])
    invariants[8] = np.linalg.multi_dot([gradT, S2, gradT])
    invariants[9] = np.linalg.multi_dot([gradT, R2, gradT])
    invariants[10] = np.linalg.multi_dot([gradT, R, S, gradT])
    invariants[11] = np.linalg.multi_dot([gradT, R_S2, gradT])
    invariants[12] = np.linalg.multi_dot([gradT, S, R_S2, gradT])
    invariants[13] = np.linalg.multi_dot([gradT, R2_S, gradT])
    invariants[14] = np.linalg.multi_dot([gradT, R2, S2, gradT])
    invariants[15] = np.linalg.multi_dot([gradT, R2_S, R, gradT])
    invariants[16] = np.linalg.multi_dot([gradT, R_S2, R2, gradT])    
    
    return invariants
    
    
class ProcessedRANS:
    """
    This class holds numpy arrays that correspond to the different RANS variables that we
    use. Numpy arrays are much easier/faster to operate on, so that is why this is used.
    """    
    
    def __init__(self, n_cells, deltaT):    
        """
        Constructor for ProcessedRANS class.
        
        Arguments:
        n_cells -- number of cells in the fluid zone (the numpy arrays must be that long)
        deltaT -- temperature scale, used for non-dimensionalization (Tmax-Tmin)        
        """
        
        self.N_FEATURES = constants.N_FEATURES # constant, indicates number of features in the ML model
                
        self.n_cells = n_cells # this is the number of elements(cells) in this dataset
        
        # These contain the gradients, which must be initialized with right dimensions
        self.gradU = np.empty((n_cells, 3, 3)) # this is a 3D array of size Nx3x3
        self.gradT = np.empty((n_cells, 3)) # this is a 2D array of size Nx3
        
        # This is a 1D boolean array which indicates which indices have high enough gradient
        self.should_use = np.zeros(n_cells,dtype=bool)
        
        # The temperature scale with which to normalize the dataset
        self.deltaT0 = deltaT
        
        
    def fillAndNonDimensionalize(self, zone, var_names):
        """
        This function is called to fill the numpy arrays with data from Tecplot.

        It takes in a zone of the .plt file and uses that data to fill the numpy
        arrays contained by this class. It also non-dimensionalizes the temperature
        differences.
        
        Arguments:
        zone -- tecplot.data.zone containing the fluid zone, where the variables live
        var_names -- dictionary mapping default names to the actual variable names in the
                     present .plt file
        """
        # Cell-centered locations:
        self.x = np.asarray(zone.values('X_cell')[:])
        self.y = np.asarray(zone.values('Y_cell')[:])
        self.z = np.asarray(zone.values('Z_cell')[:])
        
        # Scalar concentration
        self.T = np.asarray(zone.values(var_names['T'])[:])
        
        # Velocity Gradients: dudx, dudy, dudz, dvdx, dydy, dydz, dwdx, dwdy, dwdz
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
        self.nonDimensionalize()
        
        # Check that all variables have the correct size and range
        self.sanityCheck()
    
   
    def nonDimensionalize(self):
        """
        This function is called to non-dimensionalize the temperature gradient.
        """
        
        self.gradT /= self.deltaT0
        
        
    def sanityCheck(self):
        """
        This function contains assertions to check the current state of the dataset
        """
        
        assert self.x.size == self.n_cells, "Wrong number of entries for x"
        assert self.y.size == self.n_cells, "Wrong number of entries for y"
        assert self.z.size == self.n_cells, "Wrong number of entries for z"
        
        assert self.T.size == self.n_cells, \
               "Wrong number of entries for T. Check that it is cell centered."        
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
        
        
    def determineShouldUse(self, threshold):
        """
        This function is called to determine in which points we should make a prediction.

        We basically take a threshold (typically 1e-4) and fill the self.should_use with
        True in locations that we should use for the prediction (i.e., high scalar
        gradient) and False in locations we should not use.
        
        Arguments:
        threshold -- cut-off for the scalar gradient magnitude; we only make a prediction
                     in points with higher magnitude than this        
        """
        
        # Save the threshold that was used
        self.threhold = threshold
        
        # the magnitude of RANS scalar gradient, shape (n_cells)
        grad_mag = np.sqrt(np.sum(self.gradT**2, axis=1))
        
        # Takes the gradient and non-dimensionalizes the length part of it
        grad_mag = grad_mag*(self.tke**(1.5)/self.epsilon)

        self.should_use = (grad_mag >= threshold)        


    def produceFeatures(self):
        """
        This function calculates the invariant basis (ML features) for this dataset. 
        
        Returns:
        x_features -- array of shape (n_useful, n_features) that contains the features
                      that are used by the ML model to make a prediction at each point.
        """
        
        # this tells us how many of the total number of elements we use for predictions        
        self.n_useful = np.sum(self.should_use)
        
        print("Out of {} total points, ".format(self.n_cells)
              + "{} have significant gradient and will be used".format(self.n_useful))
        print("Extracting features that will be used by ML model...")
        
        # this is the feature vector
        x_features = np.empty((self.n_useful, self.N_FEATURES))         
                
        # Non-dimensionalize in bulk and select only the points where should_use is true
        gradU_factor = 0.5*self.tke[self.should_use]/self.epsilon[self.should_use]
        gradT_factor = (self.tke[self.should_use]**(1.5)/self.epsilon[self.should_use])
        gradU_temporary = self.gradU[self.should_use,:,:]*gradU_factor[:,None,None]
        gradT_temporary = self.gradT[self.should_use,:]*gradT_factor[:,None]
        
        # Loop only where should_use is true to extract invariant basis
        # tqdm wraps around the iterable and generates a progress bar
        for i in tqdm(range(self.n_useful)): 
            x_features[i, 0:self.N_FEATURES-2] = calcInvariants(gradU_temporary[i,:,:],
                                                                gradT_temporary[i,:],
                                                                self.N_FEATURES)            
        
        # Add last two scalars to the features (distance to wall and nu_t/nu)
        Re_wall = np.sqrt(self.tke[self.should_use])*self.d[self.should_use]*\
                    self.rho[self.should_use]/self.mu[self.should_use]
        nut_over_nu = self.mut[self.should_use]/self.mu[self.should_use]
        
        x_features[:, self.N_FEATURES-2] = Re_wall
        x_features[:, self.N_FEATURES-1] = nut_over_nu
                
        self.x_features = x_features
        print("Done!")
                
        return self.x_features # return the features, only where should_use == true
        
    def fillPrt(self, Prt):
        """
        Takes in a 'skinny' Pr_t field and returns a full one
        
        Arguments:
        Prt -- a numpy array shape (n_useful, ) containing the turbulent Prandtl number
               at each cell where should_use == True. This is the dimensionless
               diffusivity directly predicted by the machine learning model.
                   
        Returns:
        Prt_full -- a numpy array of shape (n_cells, ) containing a dimensional
                    turbulent diffusivity in every cell of the domain. In cells
                    where should_use == False, use fixed Pr_t assumption.
        """
        
        # make sure alpha_t has right size
        assert Prt.size == self.n_useful, "Pr_t has wrong number of entries!"
        
        # Use Reynolds analogy everywhere first
        Prt_full = np.ones(self.n_cells) * constants.PR_T # initial guess everywhere
        
        # Fill in places where should_use is true with the predicted Prt_ML:
        Prt_full[self.should_use] = Prt       
        
        return Prt_full    
