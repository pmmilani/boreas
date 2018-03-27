#---------------------------------- rans_dataset.py ------------------------------------#
"""
This file contains the class that holds a rans dataset (a bunch of numpy arrays with all
the data from a RANS k-epsilon simulation).
"""

# ------------ Import statements
import tecplot
import numpy as np
import time # needed by tqdm
from tqdm import tqdm # progress bar
        
        
def calcInvariants(gradU, gradT, tke, epsilon, d, nut, nu, n_features):
    """
    This function calculates the invariant basis at each point.

    Arguments:
    gradU -- 2D tensor with local velocity gradient (numpy array shape (3,3))
    gradT -- array with local temperature gradient (numpy array shape (3,))
    tke -- scalar with the local turbulent kinetic energy
    epsilon -- scalar with the local dissipation rate
    d -- scalar with the local distance to the wall
    nut -- scalar with local eddy viscosity
    nu -- scalar with the local laminar viscosity
    n_features -- number of features for the ML model
    
    Returns:
    invariants -- array of shape (n_features,) that contains the features
                  that are used by the ML model to make a prediction at the current point
    """

    invariants = np.empty(n_features)

    S = 0.5*tke/epsilon*(gradU + np.transpose(gradU)) # symmetric component
    R = 0.5*tke/epsilon*(gradU - np.transpose(gradU)) # anti-symmetric component
    u = (tke**(1.5)/epsilon)*gradT # vector with concentration gradient
    Re_wall = np.sqrt(tke)*d/nu # Reynolds number indicating distance to wall

    # from Smith, same order I have in my notebook page 12
    invariants[0] = np.trace(np.linalg.multi_dot([S, S]))
    invariants[1] = np.trace(np.linalg.multi_dot([S, S, S]))
    invariants[2] = np.trace(np.linalg.multi_dot([R, R]))
    invariants[3] = np.trace(np.linalg.multi_dot([R, R, S]))
    invariants[4] = np.trace(np.linalg.multi_dot([R, R, S, S]))
    invariants[5] = np.trace(np.linalg.multi_dot([R, R, S, R, S, S]))
    invariants[6] = np.linalg.multi_dot([u, u])
    invariants[7] = np.linalg.multi_dot([u, S, u])
    invariants[8] = np.linalg.multi_dot([u, S, S, u])
    invariants[9] = np.linalg.multi_dot([u, R, R, u])
    invariants[10] = np.linalg.multi_dot([u, R, S, u])
    invariants[11] = np.linalg.multi_dot([u, R, S, S, u])
    invariants[12] = np.linalg.multi_dot([u, S, R, S, S, u])
    invariants[13] = np.linalg.multi_dot([u, R, R, S, u])
    invariants[14] = np.linalg.multi_dot([u, R, R, S, S, u])
    invariants[15] = np.linalg.multi_dot([u, R, R, S, R, u])
    invariants[16] = np.linalg.multi_dot([u, R, S, S, R, R, u])
    invariants[17] = Re_wall
    invariants[18] = nut/nu

    return invariants
    
    
class RANSDataset:
    """
    This class holds numpy arrays that correspond to the different RANS variables that we
    use. Numpy arrays are much easier/faster to operate on, so that is why this is used.
    """    
    
    def __init__(self, n_cells, U0, D, rho0, deltaT):    
        """
        Constructor for RANSDataset class.
        
        Arguments:
        n_cells -- number of cells in the fluid zone (the numpy arrays must be that long)
        zone -- tecplot.data.zone containing the fluid zone, where the variables live
        U0 -- velocity scale, used for non-dimensionalization
        D -- length scale, used for non-dimensionalization
        rho0 -- density scale, used for non-dimensionalization
        deltaT -- temperature scale, used for non-dimensionalization (Tmax-Tmin)        
        """
        
        self.N_FEATURES = 19 # constant, indicates number of features in the ML model
                
        self.n_cells = n_cells # this is the number of elements(cells) in this dataset
        
        # These are 1D arrays of size N, containing the x,y,z position of the cell
        self.x = np.empty(n_cells) 
        self.y = np.empty(n_cells) 
        self.z = np.empty(n_cells)        
        
        # This is a 1D array of size N, containing different quantities (one per cell)
        self.rho = np.empty(n_cells) # density
        self.T = np.empty(n_cells) # the temperature (scalar concentration)       
        self.tke = np.empty(n_cells) # RANS turbulent kinetic energy
        self.epsilon = np.empty(n_cells) # RANS dissipation
        self.nut = np.empty(n_cells) # eddy viscosity from RANS
        self.d = np.empty(n_cells) # distance to nearest wall
        
        # These contain the gradients
        self.gradU = np.empty((n_cells, 3, 3)) # this is a 3D array of size Nx3x3
        self.gradT = np.empty((n_cells, 3)) # this is a 2D array of size Nx3
        
        # This is a 1D boolean array which indicates which indices have high enough gradient
        self.should_use = np.zeros(n_cells,dtype=bool)
        
        # The scales with which to normalize the dataset
        self.U0 = U0
        self.D = D 
        self.rho0 = rho0
        self.deltaT = deltaT
        
        
    def fillAndNonDimensionalize(self, zone, var_names):
        """
        This function is called to fill the numpy arrays with data from Tecplot.

        It takes in a zone of the .plt file and uses that data to fill the numpy
        arrays contained by this class. It also non-dimensionalizes all the data
        
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
        self.T = np.asarray(zone.values(var_names['Temperature'])[:])
        
        # Velocity Gradients: dudx, dudy, dudz, dvdx, dydy, dydz, dwdx, dwdy, dwdz
        # non-dimensionalize at the same time
        self.gradU[:, 0, 0] = np.asarray(zone.values("ddx_U")[:])
        self.gradU[:, 1, 0] = np.asarray(zone.values("ddy_U")[:])
        self.gradU[:, 2, 0] = np.asarray(zone.values("ddz_U")[:])
        self.gradU[:, 0, 1] = np.asarray(zone.values("ddx_V")[:])
        self.gradU[:, 1, 1] = np.asarray(zone.values("ddy_V")[:])
        self.gradU[:, 2, 1] = np.asarray(zone.values("ddz_V")[:])
        self.gradU[:, 0, 2] = np.asarray(zone.values("ddx_W")[:])
        self.gradU[:, 1, 2] = np.asarray(zone.values("ddy_W")[:])
        self.gradU[:, 2, 2] = np.asarray(zone.values("ddz_W")[:])
        
        # Temperature Gradients: dTdx, dTdy, dTdz 
        self.gradT[:, 0] = np.asarray(zone.values("ddx_Temperature")[:])
        self.gradT[:, 1] = np.asarray(zone.values("ddy_Temperature")[:])
        self.gradT[:, 2] = np.asarray(zone.values("ddz_Temperature")[:])
        
        # Other scalars: density, tke, epsilon, nu_t, nu, distance to wall
        self.tke = np.asarray(zone.values(var_names["TKE"])[:])
        self.epsilon = np.asarray(zone.values(var_names["epsilon"])[:])
        self.rho  = np.asarray(zone.values(var_names["Density"])[:])
        self.nu = np.asarray(zone.values(var_names["laminar viscosity"])[:])
        self.nut = np.asarray(zone.values(var_names["turbulent viscosity"])[:])
        self.d = np.asarray(zone.values(var_names["distance to wall"])[:])
        
        # non-dimensionalization done in different method
        self.nonDimensionalize() 
    
   
    def nonDimensionalize(self):
        """
        This function is called to non-dimensionalize the numpy arrays in the class.
        """
        
        self.x /= self.D
        self.y /= self.D
        self.z /= self.D
        self.T /= self.deltaT
        self.gradU /= (self.U0/self.D)
        self.gradT /= (self.deltaT/self.D)
        self.tke /= (self.U0**2)
        self.epsilon /= (self.U0**3/self.D)
        self.rho /= self.rho0
        self.nut /= (self.rho0*self.U0*self.D)
        self.nu /= (self.rho0*self.U0*self.D)
        self.d /= self.D    


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
        
        # this is the feature vector
        x_features = np.empty((self.n_useful, self.N_FEATURES)) 
        
        # this index grows sequentially (0, 1, ..., self.n_useful-1) to count which 
        # element of x_features we are in
        index = 0; 
        
        print("Extracting features that will be used by ML model...")
        
        # Loop only through the indices where should_use is true
        # tqdm wraps around the iterable and generates a progress bar
        indices_to_iterate = (np.arange(self.n_cells))[self.should_use]
        for i in tqdm(indices_to_iterate): 
            x_features[index, :] = calcInvariants(self.gradU[i,:,:], self.gradT[i,:], 
                                                  self.tke[i], self.epsilon[i], 
                                                  self.d[i], self.nut[i], self.nu[i],
                                                  self.N_FEATURES)
            index += 1
        
        self.x_features = x_features
        
        return self.x_features # return the features, only where should_use == true
        
    def fillDiffusivity(self, alpha_t):
        """
        Takes in a 'skinny', dimensionless diffusivity and returns a dimensional value 
        at every cell.
        
        Arguments:
        alpha_t -- a numpy array shape (n_useful, ) containing a turbulent diffusivity
                   value at each cell where should_use == True. This is the dimensionless
                   diffusivity directly predicted by the machine learning model.
                   
        Returns:
        alpha_t_full -- a numpy array of shape (n_cells, ) containing a dimensional
                        turbulent diffusivity in every cell of the domain. In cells
                        where should_use == False, use Sc_t=0.85 assumption.
        """
        
        # make sure alpha_t has right size
        assert alpha_t.size == self.n_useful, "alpha_t has wrong number of entries!"
        
        # Use Reynolds analogy everywhere first
        SC_T = 0.85        
        alpha_t_full = self.nut/SC_T # initial guess everywhere
        
        # Fill in places where should_use is true with the predicted diffusivity
        alpha_t_full[self.should_use] = alpha_t
        
        # Re-dimensionalize alpha_t_full
        # rho0, U0, D are just scalars (entered by the user)
        # rho has shape (n_cells,); it has the (dimensionless) density at every cell. 
        #      It is needed for non-incompressible simulations.
        alpha_t_full *= ( (self.rho0*self.rho)*self.U0*self.D )
        
        return alpha_t_full
        
    