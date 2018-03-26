#---------------------------------- tecplot_data.py ------------------------------------#
"""
This file contains the class that holds a tecplot file and defines all the helpers to
load/edit/save a tecplot plot file
"""

# ------------ Import statements
import tecplot
import os
import numpy as np


def getFloatFromUser(message):
    """
    Obtains positive real number from user.
        
    Arguments:
    message -- the message that is printed for the user, describing what is expected
    
    Returns:
    var -- a positive real number that the user entered
    """
    
    var = None # initialize the variable we return
    
    # try to get input from user until it works
    while var is None:
        var = input(message)
        try:
            var = float(var)
            if var < 0.0: raise ValueError('Must be positive!')
        except: # if var is not float or positive, it comes here
            print("\t Please, enter a valid positive real number response...")
            var = None
            
    return var    
    

def getVarNameFromUser(message, dataset, default_name):
    """
    Obtains a string from user, which is a variable name in the Tecplot file.
    
    This function is used to obtain an input string from the user, which must be the
    name of a variable in the Tecplot dataset that we loaded. User can enter the empty
    string to decline entering a name, which goes to the default name.
    
    Arguments:
    message -- the message that is printed for the user, describing what is expected
    dataset -- the tecplot.data.dataset class in which "name" must be a variable name
    default_name -- the name we return by default if the user enters an empty string. We
                    only return it if default_name is, itself, a valid variable name.
    
    Returns:
    name -- a string that represents a valid variable name in the Tecplot file
    """
    
    name = None # initialize the variable we return
    
    # try to get input from user until it works
    while name is None:
        name = input(message)
        
        # if user enters empty string, use default name.
        if name == "":
            name = default_name           
        
        if not isVariable(name, dataset):
            print("\t Please, enter a valid variable name in the Tecplot file." +
                  'Variable named "{}" does not exist.'.format(name))
            name = None                    
            
    return name


    
def isVariable(name, dataset):
    """
    Determines if given string is an existing variable name for a variable in dataset.
    
    This helper function returns True if the given name is a valid name of a variable
    in the dataset, False otherwise.
    
    Arguments:
    name -- the string that we are testing if it corresponds to a variable name or not
    dataset -- the tecplot.data.dataset class in which "name" must be a variable name
    
    Returns:
    True/False depending on whether name is a valid variable name in dataset
    """

    try:
        # if name is not a valid variable name, exception is thrown
        _ = dataset.variable(name) 
        return True
    
    except: 
        return False
    


class TPDataset:
    """
    This class is holds the information of a single Tecplot data file internally.
    """

    def __init__(self, filepath, zone=None):
        """
        Constructor for MLModel class
        
        Arguments:
        filepath -- the path from where to load the Tecplot data. Should correspond to
                    the binary .plt file of a RANS solution using the k-epsilon 
                    turbulence model. 
        zone -- optional, tells us in which zone to look for the solution. Can be either
                an integer (e.g. 0, 1, ...) or a string with the zone name (e.g. "fluid").
                By default, we look into zone 0. Supply a different zone in case the RANS
                solution for the flow field is not in zone 0.        
        """
        
        # Make sure the file exists 
        assert os.path.isfile(filepath), "Tecplot file not found!"
        
        # Load the file. Print number of zones/variables as sanity check
        print("Loading Tecplot dataset...")
        self.__dataset = tecplot.data.load_tecplot(filepath, 
                                read_data_option=tecplot.constant.ReadDataOption.Replace)
        print("Tecplot file loaded successfully! "
               + "It contains {} zones and ".format(self.__dataset.num_zones) 
               + "{} variables".format(self.__dataset.num_variables))
               
        # Extracting zone. Print zone name and number of cells/nodes as sanity check
        if zone is None:
            print("All operations will be performed on zone 0 (default)")
            self.__zone = self.__dataset.zone(0)            
        else:
            print("All operations will be performed on zone {}".format(zone))
            self.__zone = self.__dataset.zone(zone)
            
        print('Zone "{}" has {} cells'.format(self.__zone.name, self.__zone.num_elements) 
              + " and {} nodes".format(self.__zone.num_points))
        
        # Initializes a dictionary containing correct names for this dataset
        self.initializeVarNames()

    
    def initializeVarNames(self):
        """
        Initializes a dictionary containing the string names for variables of interest.
        
        This function is called to initialize, internally, the correct names for each of 
        the variables that we need from the Tecplot file. It will ask for user input. The
        user can enter empty strings to use a default variable name (in ANSYS Fluent).
        """
        
        # this dictionary maps from key to the actual variable name in Tecplot file
        self.__var_names = {} 
        
        # These are the keys for the 8 relevant variables we need
        variables = ["U", "V", "W", "Temperature", "TKE", "epsilon",
                     "turbulent viscosity", "distance to wall"]
                     
        # These are the default names we enter if user refuses to provide one (from ANSYS
        # Fluent solutions)
        default_names = ["X Velocity", "Y Velocity", "Z Velocity", "UDS 0", 
                         "Turbulent Kinetic Energy", "Turbulent Dissipation Rate",
                         "Turbulent Viscosity", "Wall Distribution"]
                         
        # Ask user for variable names and stores them in the dictionary
        for i, key in enumerate(variables):
            self.__var_names[key] = getVarNameFromUser("Enter name for "
                                                       + "{} variable: ".format(key), 
                                                        self.__dataset, default_names[i])  
              
    
    def normalize(self, U=None, D=None, rho=None, miu=None, deltaT=None):
        """
        Collects scales from the user, which are used to non-dimensionalize data.
        
        This function is called to read in the dimensional scales of the dataset, so we
        can appropriately non-dimensionalize everything later. It can also be called with 
        arguments, in which case the user is not prompted to write anything. These scales
        must be given in the same units as the dataset.
        
        Arguments:
        U -- velocity scale
        D -- length scale
        rho -- density
        miu -- laminar dynamic viscosity
        deltaT -- temperature scale
        """
        
        # Read in the five quantities if they are not passed in
        if U is None:
            U = getFloatFromUser("Enter velocity scale (typically hole bulk velocity): ")        
        if D is None:
            D = getFloatFromUser("Enter length scale (typically hole diameter): ")            
        if rho is None:
            rho = getFloatFromUser("Enter density scale: ")            
        if miu is None:
            miu = getFloatFromUser("Enter dynamic viscosity (miu): ")            
        if deltaT is None:
            deltaT = getFloatFromUser("Enter temperature delta (Tmax-Tmin): ")
            
        # Set instance variables to hold them
        self.U = U
        self.D = D
        self.rho = rho
        self.miu = miu
        self.deltaT = deltaT
        
        # Calculate Re and print warning if necessary
        Re = rho*U*D/miu
        print("Reynolds number: Re = {}".format(Re))
        if Re < 1000 or Re > 6000: # print warning
            print("\t Warning: the model was calibrated with data between " 
                  + "Re=3000 and Re=5400.")
            print("\t Extrapolation to different Reynolds number is an "
                  + "open research topic.")                             
                                           
    def calculateDerivatives(self):
        """
        Calculates x,y,z derivatives of U, V, W, and T.
        
        This function is called to calculate x,y,z derivatives of the specified variables
        (using Tecplot's derivative engine). This can take a while for large datasets.
        """
        
        print("Taking x, y, z derivatives. This can take a while...")
        
        variables = ["U", "V", "W", "Temperature"] # variables that we differentiate        
        
        # Go through each of them and create ddx, ddy, ddz
        for var in variables:
        
            # Strings below are Fortran-style equations that the Tecplot engine takes
            ddx_eqn = "{ddx_" + var + "} = ddx({" + self.__var_names[var] + "})"
            ddy_eqn = "{ddy_" + var + "} = ddy({" + self.__var_names[var] + "})"
            ddz_eqn = "{ddz_" + var + "} = ddz({" + self.__var_names[var] + "})"
            
            print("Differentiating {}...    ".format(var), end="", flush=True)
            
            # These lines actually execute the equations            
            tecplot.data.operate.execute_equation(ddx_eqn, 
                              value_location=tecplot.constant.ValueLocation.CellCentered,
                              variable_data_type=tecplot.constant.FieldDataType.Float)
            print("ddx done!", end="", flush=True)
            tecplot.data.operate.execute_equation(ddy_eqn,
                              value_location=tecplot.constant.ValueLocation.CellCentered,
                              variable_data_type=tecplot.constant.FieldDataType.Float)
            print(" --- ddy done!", end="", flush=True)
            tecplot.data.operate.execute_equation(ddz_eqn,
                              value_location=tecplot.constant.ValueLocation.CellCentered,
                              variable_data_type=tecplot.constant.FieldDataType.Float)
            print(" --- ddz done!")

    
    def extractQuantityArrays(self):
        """
        Extract quantities from the tecplot file into numpy arrays.
        
        This method is called to extract all Tecplot quantities into appropriate numpy
        arrays, which will be used in all the subsequent processing (including ML). It
        initializes an instance of the RANSDataset class and stores all numpy arrays in
        that class.
        
        Returns:
        rans_data -- instance of RANSDataset class containing numpy arrays extracted from
                     tecplot dataset.
        """
        # First, calculate locations of the cell centers (X,Y,Z are usually on the node)
        tecplot.data.operate.execute_equation('{X_cell} = {X}',
                              value_location=tecplot.constant.ValueLocation.CellCentered)
        tecplot.data.operate.execute_equation('{Y_cell} = {Y}',
                              value_location=tecplot.constant.ValueLocation.CellCentered)
        tecplot.data.operate.execute_equation('{Z_cell} = {Z}',
                              value_location=tecplot.constant.ValueLocation.CellCentered)
                              
        
        # Initialize the RANS dataset (numpy arrays only) with num_elements.
        rans_data = RANSDataset(self.__zone.num_elements)
        
        ### Extract appropriate quantities below:        
        # Cell-centered locations:
        rans_data.x = np.asarray(self.__zone.values('X_cell')[:])
        rans_data.y = np.asarray(self.__zone.values('Y_cell')[:])
        rans_data.z = np.asarray(self.__zone.values('Z_cell')[:])
        
        # Scalar concentration
        rans_data.T = np.asarray(self.__zone.values(self.__var_names['Temperature'])[:])
        
        # Velocity Gradients: dudx, dudy, dudz, dvdx, dydy, dydz, dwdx, dwdy, dwdz 
        rans_data.gradU[:, 0, 0] = np.asarray(self.__zone.values("ddx_U")[:])
        rans_data.gradU[:, 1, 0] = np.asarray(self.__zone.values("ddy_U")[:])
        rans_data.gradU[:, 2, 0] = np.asarray(self.__zone.values("ddz_U")[:])
        rans_data.gradU[:, 0, 1] = np.asarray(self.__zone.values("ddx_V")[:])
        rans_data.gradU[:, 1, 1] = np.asarray(self.__zone.values("ddy_V")[:])
        rans_data.gradU[:, 2, 1] = np.asarray(self.__zone.values("ddz_V")[:])
        rans_data.gradU[:, 0, 2] = np.asarray(self.__zone.values("ddx_W")[:])
        rans_data.gradU[:, 1, 2] = np.asarray(self.__zone.values("ddy_W")[:])
        rans_data.gradU[:, 2, 2] = np.asarray(self.__zone.values("ddz_W")[:])
        
        # Temperature Gradients: dTdx, dTdy, dTdz 
        rans_data.gradT[:, 0] = np.asarray(self.__zone.values("ddx_Temperature")[:])
        rans_data.gradT[:, 1] = np.asarray(self.__zone.values("ddy_Temperature")[:])
        rans_data.gradT[:, 2] = np.asarray(self.__zone.values("ddz_Temperature")[:]) 
        
        # Other scalar: tke, epsilon, nu_t, distance to wall
        rans_data.tke = np.asarray(self.__zone.values(self.__var_names['TKE'])[:])
        rans_data.epsilon = np.asarray(self.__zone.values(self.__var_names['epsilon'])[:])
        rans_data.nut = np.asarray(self.__zone.values(self.__var_names['turbulent viscosity'])[:])
        rans_data.d = np.asarray(self.__zone.values(self.__var_names['distance to wall'])[:])        
        
        
        # return the extracted arrays
        return rans_data
    
    
    def saveDataset(self, path):
        """
        Saves current state of tecplot dataset as .plt binary file.        
        
        Arguments:
        path -- path where .plt file should be saved.
        """
        tecplot.data.save_tecplot_plt(filename=path, dataset=self.__dataset) 
        
        
        

"""
This class holds numpy arrays that correspond to different RANS variables that we need.
Numpy arrays are much easier/faster to operate on
"""
class RANSDataset:
    def __init__(self, n_cells):
        self.n_cells = n_cells # this is the number of elements(cells) in this dataset
        
        # These are 1D arrays of size N, containing the x,y,z position of the cell
        self.x = np.empty(n_cells) 
        self.y = np.empty(n_cells) 
        self.z = np.empty(n_cells)        
        
        # this is a 1D array of size N, containing different quantities (one per cell)
        self.T = np.empty(n_cells) # the scalar concentration        
        self.tke = np.empty(n_cells) # RANS turbulent kinetic energy
        self.epsilon = np.empty(n_cells) # RANS dissipation
        self.nut = np.empty(n_cells) # eddy viscosity from RANS
        self.d = np.empty(n_cells) # distance to nearest wall
        
        # These contain the gradients
        self.gradU = np.empty((n_cells, 3, 3)) # this is a 3D array of size Nx3x3
        self.gradT = np.empty((n_cells, 3)) # this is a 2D array of size Nx3
        
        # this is a 1D boolean array which indicates which indices have high enough gradient
        self.shouldUse = np.empty(n_cells,dtype=bool)        
        