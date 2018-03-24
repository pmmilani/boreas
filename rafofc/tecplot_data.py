# --------------------------------- tecplot_data.py ------------------------------------#
# This file contains the class that holds a tecplot file and defines all the helpers to
# load/edit/save a tecplot plot file

# ------------ Import statements
import tecplot
import os
import numpy as np



"""
This function is used to obtain an input from the user, which must be a positive floating
point number (e.g. velocity and length scales)
"""
def GetFloatFromUser(message):
    
    var = None # initialize the variable we return
    
    # try to get input from user until it works
    while var is None:
        var = input(message)
        try:
            var = float(var)
            if var < 0.0: raise ValueError('Must be positive!')
        except: # if var is not float or positive, it comes here
            print("\t Please, enter a valid response...")
            var = None
            
    return var
    
    
"""
This function is used to obtain an input string from the user, which must be the
name of a variable in the dataset. User can enter the empty string to decline entering
a name, which goes to the default name.
"""
def GetVarNameFromUser(message, dataset, default_name):
    
    name = None # initialize the variable we return
    
    # try to get input from user until it works
    while name is None:
        name = input(message)
        
        if name == "": 
            return default_name # if user enters empty string, return default name.
        
        if not IsVariable(name, dataset):
            print("\t Please, enter a valid variable name...")
            name = None                    
            
    return name


    
"""
This helper function returns true if the given name is a valid name of a variable
in the dataset, False otherwise.
"""    
def IsVariable(name, dataset):
    try:
        # if name is not a valid variable name, exception is thrown
        _ = dataset.variable(name) 
        return True
    
    except: 
        return False
    


"""
This class is holds the information of a single Tecplot data file internally.
"""
class TPDataset:

    def __init__(self, filepath, zone=None):
        
        # make sure the file exists 
        assert os.path.isfile(filepath), "Tecplot file not found!"
        
        # Load the file
        print("Loading Tecplot dataset...")
        self.__dataset = tecplot.data.load_tecplot(filepath, 
                                read_data_option=tecplot.constant.ReadDataOption.Replace)
        print("Tecplot file loaded successfully! "
               + "It contains {} zones and ".format(self.__dataset.num_zones) 
               + "{} variables".format(self.__dataset.num_variables))
               
        # Extracting zone
        if zone is None:
            print("All operations will be performed on zone 0 (default)")
            self.__zone = self.__dataset.zone(0)            
        else:
            print("All operations will be performed on zone {}".format(zone))
            self.__zone = self.__dataset.zone(zone)
            
        print('Zone "{}" has {} cells'.format(self.__zone.name, self.__zone.num_elements) 
              + " and {} nodes".format(self.__zone.num_points))
        
        # initializes a dictionary containing correct names for this dataset
        self.InitializeVarNames()       
              
    
    """
    This function is called to read in the dimensional scales of the dataset, so we can
    appropriately non-dimensionalize everything later. It can also be called with 
    arguments, in which case the user is not prompted to write anything.
    """
    def Normalize(self, U=None, D=None, rho=None, miu=None, deltaT=None):
        
        # Read in the five quantities if they are not passed in
        if U is None:
            U = GetFloatFromUser("Enter velocity scale (typically hole bulk velocity): ")        
        if D is None:
            D = GetFloatFromUser("Enter length scale (typically hole diameter): ")            
        if rho is None:
            rho = GetFloatFromUser("Enter density scale: ")            
        if miu is None:
            miu = GetFloatFromUser("Enter dynamic viscosity (miu): ")            
        if deltaT is None:
            deltaT = GetFloatFromUser("Enter temperature delta (Tmax-Tmin): ")
            
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
    

    """
    This function is called to initialize, internally, the correct names for each of the
    variables that we need from the file
    """
    def InitializeVarNames(self):
        
        # this dictionary maps from key to the actual variable name in Tecplot file
        self.__var_names = {} 
        
        # These are the keys for the 8 relevant variables we need
        variables = ["U", "V", "W", "Temperature", "TKE", "epsilon",
                     "turbulent viscosity", "distance to wall"]
                     
        # These are the default names we enter if user refuses to provide one
        default_names = ["X Velocity", "Y Velocity", "Z Velocity", "UDS 0", 
                         "Turbulent Kinetic Energy", "Turbulent Dissipation Rate",
                         "Turbulent Viscosity", "Wall Distribution"]
                         
        # Ask user for variable names and stores them in the dictionary
        for i, key in enumerate(variables):
            self.__var_names[key] = GetVarNameFromUser("Enter name for "
                                                       + "{} variable: ".format(key), 
                                                        self.__dataset, default_names[i])                            
                                           
    """
    This function is called to calculate x,y,z derivatives of the specified variables
    (using Tecplot's derivative engine). This can take a while for large datasets
    """
    def CalculateDerivatives(self):
        
        print("Starting to take x, y, z derivatives. This can take a while...")
        
        variables = ["U", "V", "W", "Temperature"] # variables that we differentiate        
        
        # Go through each of them and create ddx, ddy, ddz
        for var in variables:
        
            # Strings below are Fortran-style equations that the Tecplot engine takes
            ddxname = "{ddx_" + var + "} = ddx({" + self.__var_names[var] + "})"
            ddyname = "{ddy_" + var + "} = ddy({" + self.__var_names[var] + "})"
            ddzname = "{ddz_" + var + "} = ddz({" + self.__var_names[var] + "})"
            
            print("Differentiating {} ...".format(var))
            
            # These lines actually execute the equations            
            tecplot.data.operate.execute_equation(ddxname, 
                              value_location=tecplot.constant.ValueLocation.CellCentered,
                              variable_data_type=tecplot.constant.FieldDataType.Float)
            tecplot.data.operate.execute_equation(ddyname,
                              value_location=tecplot.constant.ValueLocation.CellCentered,
                              variable_data_type=tecplot.constant.FieldDataType.Float)
            tecplot.data.operate.execute_equation(ddzname,
                              value_location=tecplot.constant.ValueLocation.CellCentered,
                              variable_data_type=tecplot.constant.FieldDataType.Float)

    
    """
    This method is called to extract all Tecplot quantities into appropriate numpy
    arrays, which will be used in all the subsequent processing (including ML). 
    """
    def ExtractQuantityArrays(self):
    
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
        rans_data.gradU[:, 0, 0] = np.asarray(self.__zone.values("ddx_" + self.__var_names['U'])[:])
        rans_data.gradU[:, 1, 0] = np.asarray(self.__zone.values("ddy_" + self.__var_names['U'])[:])
        rans_data.gradU[:, 2, 0] = np.asarray(self.__zone.values("ddz_" + self.__var_names['U'])[:])
        rans_data.gradU[:, 0, 1] = np.asarray(self.__zone.values("ddx_" + self.__var_names['V'])[:])
        rans_data.gradU[:, 1, 1] = np.asarray(self.__zone.values("ddy_" + self.__var_names['V'])[:])
        rans_data.gradU[:, 2, 1] = np.asarray(self.__zone.values("ddz_" + self.__var_names['V'])[:])
        rans_data.gradU[:, 0, 2] = np.asarray(self.__zone.values("ddx_" + self.__var_names['W'])[:])
        rans_data.gradU[:, 1, 2] = np.asarray(self.__zone.values("ddy_" + self.__var_names['W'])[:])
        rans_data.gradU[:, 2, 2] = np.asarray(self.__zone.values("ddz_" + self.__var_names['W'])[:])
        
        # Temperature Gradients: dTdx, dTdy, dTdz 
        rans_data.gradT[:, 0] = np.asarray(self.__zone.values("ddx_" + self.__var_names['Temperature'])[:])
        rans_data.gradT[:, 1] = np.asarray(self.__zone.values("ddy_" + self.__var_names['Temperature'])[:])
        rans_data.gradT[:, 2] = np.asarray(self.__zone.values("ddz_" + self.__var_names['Temperature'])[:]) 
        
        # Other scalar: tke, epsilon, nu_t, distance to wall
        rans_data.tke = np.asarray(self.__zone.values(self.__var_names['TKE'])[:])
        rans_data.epsilon = np.asarray(self.__zone.values(self.__var_names['epsilon'])[:])
        rans_data.nut = np.asarray(self.__zone.values(self.__var_names['turbulent viscosity'])[:])
        rans_data.d = np.asarray(self.__zone.values(self.__var_names['distance to wall'])[:])
        
        
        
        # return the extracted arrays
        return rans_data
        
    
    """
    This function is called to save the current state of the Tecplot dataset as a .plt
    tecplot file.
    """
    def SaveDataset(self, path):
        tecplot.data.save_tecplot_plt(filename=path, dataset=self.__dataset) 
        
        
        

"""
This class holds numpy arrays that correspond to different RANS variables that we need.
Numpy arrays are much easier/faster to operate on
"""
class RANSDataset:
    def __init__(self, N_cells):
        self.N_cells = N_cells # this is the number of elements(cells) in this dataset
        
        # These are 1D arrays of size N, containing the x,y,z position of the cell
        self.x = np.empty(N_cells) 
        self.y = np.empty(N_cells) 
        self.z = np.empty(N_cells)        
        
        # this is a 1D array of size N, containing different quantities (one per cell)
        self.T = np.empty(N_cells) # the scalar concentration        
        self.tke = np.empty(N_cells) # RANS turbulent kinetic energy
        self.epsilon = np.empty(N_cells) # RANS dissipation
        self.nut = np.empty(N_cells) # eddy viscosity from RANS
        self.d = np.empty(N_cells) # distance to nearest wall
        
        # These contain the gradients
        self.gradU = np.empty((N_cells, 3, 3)) # this is a 3D array of size Nx3x3
        self.gradT = np.empty((N_cells, 3)) # this is a 2D array of size Nx3
        
        # this is a 1D boolean array which indicates which indices have high enough gradient
        self.shouldUse = np.empty(N_cells,dtype=bool)        
        