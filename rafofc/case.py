#-------------------------------------- case.py ----------------------------------------#
"""
This file contains the class that holds a tecplot file and defines all the helpers to
load/edit/save a tecplot plot file
"""

# ------------ Import statements
from sklearn.externals import joblib # joblib is used to load files from disk
import timeit # for timing the derivative
import tecplot
import os
import numpy as np
from rafofc.processed_data import ProcessedRANS


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
            print("\t Please, enter a positive real number...")
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
            print('\t Variable "{}" not found.'.format(name)
                  + " Please, enter a valid variable name in the Tecplot file.")
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
        
    
def writeValues(file, variable):
    """
    Writes a single variable (one entry per line) in the Fluent interpolation file
      
    Arguments:
    file -- the file object to which we are writing the current variable
    variable -- any iterable (e.g. numpy array or list) with one entry per location where
                we write    
    """
    
    file.write("(") # must start with a parenthesis
    
    for x in variable:
        file.write("{:.6e}\n".format(x))
        
    file.write(")\n") # must end with parenthesis and new line
    

class TestCase:
    """
    This class is holds the information of a single test case (just RANS simulation).
    """

    def __init__(self, filepath, zone=None, use_default_names=False):
        """
        Constructor for TestCase class
        
        Arguments:
        filepath -- the path from where to load the Tecplot data. Should correspond to
                    the binary .plt file of a RANS solution using the k-epsilon 
                    turbulence model. 
        zone -- optional, tells us in which zone to look for the solution. Can be either
                an integer (e.g. 0, 1, ...) or a string with the zone name (e.g. "fluid").
                By default, we look into zone 0. Supply a different zone in case the RANS
                solution for the flow field is not in zone 0.
        use_default_names -- optional, whether to use default names (from Fluent) to 
                             fetch variables in the .plt file. Must be false unless all 
                             variables of interest in tecplot have the default name.
        """
        
        # Make sure the file exists 
        assert os.path.isfile(filepath), "Tecplot file {} not found!".format(filepath)
        
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
        self.initializeVarNames(use_default_names)
        
        # Initialize proc_data as None
        self.__proc_data = None

    
    def initializeVarNames(self, use_default_names=False):
        """
        Initializes a dictionary containing the string names for variables of interest.
        
        This function is called to initialize, internally, the correct names for each of 
        the variables that we need from the Tecplot file. It will ask for user input. The
        user can enter empty strings to use a default variable name (in ANSYS Fluent).
        
        use_default_names -- optional, whether to use default names (from Fluent) to 
                             fetch variables in the .plt file. Must be false unless all 
                             variables of interest in Tecplot have the default name.
        """
        
        # this dictionary maps from key to the actual variable name in Tecplot file
        self.__var_names = {} 
        
        # These are the keys for the 9 relevant variables we need
        variables = ["U", "V", "W", "Density", "T", "TKE", "epsilon",
                     "turbulent viscosity", "distance to wall", "laminar viscosity"]
                     
        # These are the default names we enter if user refuses to provide one (from ANSYS
        # Fluent solutions)
        default_names = ["X Velocity", "Y Velocity", "Z Velocity", "Density", "UDS 0", 
                         "Turbulent Kinetic Energy", "Turbulent Dissipation Rate",
                         "Turbulent Viscosity", "Wall Distribution", "Laminar Viscosity"]
                         
        # Loop through all the variables that we must add
        for i, key in enumerate(variables):
        
            # If I want to use default names for the variables
            if use_default_names:                
                
                # Check default names are valid, otherwise crash
                assert isVariable(default_names[i], self.__dataset), \
                      "Default name {} is not a valid variable!".format(default_names[i])
                
                # Add default name to dictionary
                self.__var_names[key] = default_names[i]
            
            # Here, just get a name from the user
            else:
                self.__var_names[key] = getVarNameFromUser("Enter name for "
                                                           + "{} variable: ".format(key), 
                                                            self.__dataset, 
                                                            default_names[i])
                                                            
        # Finally, assert X, Y, Z are valid variables:
        assert isVariable("X", self.__dataset), 'Variable "X" not found in the dataset!'
        assert isVariable("Y", self.__dataset), 'Variable "Y" not found in the dataset!'
        assert isVariable("Z", self.__dataset), 'Variable "Z" not found in the dataset!'
            
    
    def normalize(self, U0=None, D=None, rho0=None, miu=None, deltaT=None):
        """
        Collects scales from the user, which are used to non-dimensionalize data.
        
        This function is called to read in the dimensional scales of the dataset, so we
        can appropriately non-dimensionalize everything later. It can also be called with 
        arguments, in which case the user is not prompted to write anything. These scales
        must be given in the same units as the dataset.
        
        Arguments:
        U0 -- velocity scale
        D -- length scale
        rho0 -- density
        miu -- laminar dynamic viscosity
        deltaT -- temperature scale
        """
        
        # Read in the five quantities if they are not passed in
        if U0 is None:
            U0 = getFloatFromUser("Enter velocity scale (typically hole bulk velocity): ")        
        if D is None:
            D = getFloatFromUser("Enter length scale (typically hole diameter): ")            
        if rho0 is None:
            rho0 = getFloatFromUser("Enter density scale: ")            
        if miu is None:
            miu = getFloatFromUser("Enter dynamic viscosity (miu): ")            
        if deltaT is None:
            deltaT = getFloatFromUser("Enter temperature delta (Tmax-Tmin): ")
            
        # Set instance variables to hold them
        self.U0 = U0
        self.D = D
        self.rho0 = rho0
        self.miu = miu
        self.deltaT = deltaT
        
        # Calculate Re and print warning if necessary
        Re = rho0*U0*D/miu
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
        
        variables = ["U", "V", "W", "T"] # variables that we differentiate        
        
        tic=timeit.default_timer() # timing
        
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
        
        # timing
        toc=timeit.default_timer()
        print("Taking derivatives took {:.1f} minutes".format((toc - tic)/60.0))
        
        # calls function below to input correct names for the derivatives
        self.addDerivativeNames(use_default_derivative_names=True)

    
    def addDerivativeNames(self, use_default_derivative_names):
        """
        Adds the names of derivative variables to the name dictionary.
        
        This function is called to add the correct names for each of the Tecplot  
        variables holding derivatives of interest. It will ask for user input. The
        user can enter empty strings to use a default variable name. The default
        names are "ddx_U", "ddy_U", ..., "ddz_T"
        
        use_default_derivative_names -- optional, whether to use default names to 
                             fetch variables in the .plt file. Should only be true if ALL 
                             derivatives of interest in tecplot have the default name.
        """    
        
        # Variables whose names we want to add (i.e., all derivatives)
        variables = ["ddx_U", "ddy_U", "ddz_U", 
                     "ddx_V", "ddy_V", "ddz_V",
                     "ddx_W", "ddy_W", "ddz_W", 
                     "ddx_T", "ddy_T", "ddz_T"]
        
        # Default names for variables above        
        default_names = ["ddx_U", "ddy_U", "ddz_U", 
                         "ddx_V", "ddy_V", "ddz_V",
                         "ddx_W", "ddy_W", "ddz_W", 
                         "ddx_T", "ddy_T", "ddz_T"]    
        
        # Add derivative names to the dictionary of names
        for i, var in enumerate(variables):
            
            if use_default_derivative_names:
            
                # Check default names are valid, otherwise crash
                assert isVariable(default_names[i], self.__dataset), \
                      "Default name {} is not a valid variable!".format(default_names[i])
                                
                # Add default names to dictionary
                self.__var_names[var] = default_names[i]                
            
            else: 
                
                # Here, we ask the user for each derivative's name
                self.__var_names[var] = getVarNameFromUser("Enter name for " 
                                                           + "{} variable: ".format(var),
                                                           self.__dataset,
                                                           default_names[i]) 
                                                                  
                                                                  
    def extractMLFeatures(self, threshold=1e-4, processed_load_path=None, 
                          processed_dump_path=None):
        """
        Extract quantities from the tecplot file into numpy arrays.
        
        This method is called to extract all Tecplot quantities into appropriate numpy
        arrays, which will be used in all the subsequent processing (including ML). It
        initializes an instance of the ProcessedRANS class and stores all numpy arrays in
        that class. It then returns x, which contains the ML features.
        
        Arguments:
        threshold -- magnitude cut-off for the temperature scalar gradient: only use 
                     points with (dimensionless) magnitude higher than this. Default is 
                     1e-4. Only change this if you know what you are doing.
        processed_load_path -- optional keyword argument. If this is not None, this 
                               function will attempt to read rans_data from disk and
                               restore it instead of recalculating everything. The path
                               where rans_data is located is given by rans_data_path.
        processed_dump_path -- optional keyword argument. If this is not None, and the 
                               above argument is None, then this method will calculate
                               rans_data for the current dataset and then save it to
                               disk (to path specified by processed_dump_path). Use it
                               to avoid recalculating features (since it can be an 
                               expensive operation)
        
        Returns:
        x -- a numpy array containing the features for prediction in the ML step. The 
             shape should be (n_cells, n_features)
        """
        # First, calculate locations of the cell centers (X,Y,Z are usually on the node)
        tecplot.data.operate.execute_equation('{X_cell} = {X}',
                              value_location=tecplot.constant.ValueLocation.CellCentered)
        tecplot.data.operate.execute_equation('{Y_cell} = {Y}',
                              value_location=tecplot.constant.ValueLocation.CellCentered)
        tecplot.data.operate.execute_equation('{Z_cell} = {Z}',
                              value_location=tecplot.constant.ValueLocation.CellCentered)                              
        
        # If we were passed a valid rans_data_path, restore it from disk and return its 
        # value of x_features as x 
        if processed_load_path:
            if os.path.isfile(processed_load_path):
                print("A valid path for rans_data was supplied. "
                      + "It will be read from disk...", end="", flush=True)
                self.__proc_rans = joblib.load(processed_load_path)
                x = self.__proc_rans.x_features
                print(" Done")
                return x 
        
        #---- These next four commands initialize the ProcessedRANS and do all the necessary
        #---- processing to obtain features.
        # Initialize the processed RANS dataset (numpy arrays only) with num_elements.
        self.__proc_rans = ProcessedRANS(self.__zone.num_elements, self.U0, self.D, 
                                         self.rho0, self.deltaT)
        # Fill the ProcessedRANS instance with relevant quantities from this zone
        self.__proc_rans.fillAndNonDimensionalize(self.__zone, self.__var_names)
        # Determine which points should be used
        self.__proc_rans.determineShouldUse(threshold) 
        # Produces features used for ML prediction        
        x = self.__proc_rans.produceFeatures()       
        #---- Finished calculating rans_data       
        
        # If a dump path is supplied, then save self.__proc_rans to disk
        if processed_dump_path:
            print("Saving rans_data to disk...")
            joblib.dump(self.__proc_rans, processed_dump_path, protocol=2)           
        
        # return the extracted arrays
        return x
        
        
    def addMLDiffusivity(self, alpha_t):
        """
        Adds alpha_t_ML and should_use as variables in the Tecplot file.
        
        This method takes in a diffusivity array alpha_t that was predicted by the 
        machine learning model and adds that as a variable to the Tecplot file. It
        also adds should_use, which is very useful for visualization. should_use is 1
        where we use the ML model and 0 where we use the default Reynolds analogy.
        
        Arguments:
        alpha_t -- numpy array shape (num_useful, ) with the dimensionless turbulent
                   diffusivity predicted at each cell. 
        """
    
        # First, create a diffusivity that is dimensional and available at every cell
        alpha_t_full = self.__proc_rans.fillDiffusivity(alpha_t)
        
        # Creates a variable called "alpha_t_ML" and "should_use" everywhere
        self.__dataset.add_variable(name="alpha_t_ML",
                                    dtypes=tecplot.constant.FieldDataType.Float,
                                    locations=tecplot.constant.ValueLocation.CellCentered)
        self.__dataset.add_variable(name="should_use",
                                    dtypes=tecplot.constant.FieldDataType.Int16,
                                    locations=tecplot.constant.ValueLocation.CellCentered)
        
        # Add alpha_t_full to the zone
        assert self.__zone.num_elements == alpha_t_full.size, \
                                "alpha_t_full has wrong number of entries"
        self.__zone.values("alpha_t_ML")[:] = alpha_t_full.tolist()

        # Add should_use to the zone                
        self.__zone.values("should_use")[:] = self.__proc_rans.should_use.tolist()           
    
    
    def fetchVariablesToWrite(self, variable_list):
        """
        Returns positions of each cell and a list of variables to write in each cell.
        
        Arguments:
        variable_list -- a list of strings containing names of variables present in the
                         tecplot .plt file that should be present in the 
                         interpolation/csv file.
                         
        Returns:
        x,y,z -- lists containing x,y,z positions of each cell. len(x) is the number 
                 of cells in the dataset
        vars -- list of lists, containing different variables to be written in csv or
                interpolation file. len(vars) is the number of variables to be written
                and len(vars[0]) is the number of cells in the dataset.
        """
    
        # number of points that will be written
        N = self.__zone.num_elements
        
        # First, get x,y,z from the cell center
        x = self.__zone.values("X_cell")[:]
        y = self.__zone.values("Y_cell")[:]
        z = self.__zone.values("Z_cell")[:]
        
        assert (len(x) == N and len(y) == N and len(z) == N), \
                                "x,y,z variables have wrong number of entries!"
        
        # Now, get a list of variables (as numpy arrays) that will be written
        vars = []
        for var_name in variable_list:
        
            # Check names are valid, otherwise crash
            assert isVariable(var_name, self.__dataset), \
               "{} is not a valid variable to write in interp/csv file!".format(var_name)
                      
            var = self.__zone.values(var_name)[:]
            
            # Check variable has correct length
            assert len(var) == N, \
                  ("Variable to write {} has incorrect number of".format(var_name)
                   + " elements! Check node vs cell.")
            
            vars.append(var)

        return x,y,z,vars
    
        
    def createInterpFile(self, path, variable_list, outname_list):
        """
        Creates a Fluent interpolation file with the variables specified in variable_list
        
        Arguments:
        path -- path where .ip file should be saved.
        variable_list -- a list of strings containing names of variables present in the
                         tecplot .plt file that should be present in the interpolation
                         file.
        outname_list -- a list of strings indicating the desired name for the variables
                        in the interpolation file. Typically, we use user defined scalars
                        to input diffusivity, so name should be uds-0, uds-1, etc.                        
        """
        
        print("Writing interpolation file to read results in Fluent...", 
                end="", flush=True)
        
        # Check to make sure the number of names is the same as the number of variables,
        # and is greater than one.
        assert len(variable_list) == len(outname_list), \
                 ("The number of names (outnames_to_write) must match the number"
                  + "of variables (variables_to_write)!")                  
        assert len(variable_list) >= 1, "Must write at least one variable!"
        
        # Call helper to get appropriate lists.
        x,y,z,vars = self.fetchVariablesToWrite(variable_list)
        
        # Open the file and write the variables with the correct format
        with open(path, "w") as interp_file:
            
            # Write header
            interp_file.write("3\n") # version. Must be 3
            interp_file.write("3\n") # dimensionality. must be 3
            interp_file.write("{}\n".format(len(x))) # number of points
            interp_file.write("{}\n".format(len(variable_list))) # number of variables
            for name in outname_list:
                interp_file.write("{}\n".format(name))
            
            # Here, write all the x,y,z positions. writeValues is a helper that writes
            # all entries of one variable.
            writeValues(interp_file, x)
            writeValues(interp_file, y)
            writeValues(interp_file, z)
            
            # Finally, write the variables of interest
            for var in vars: 
                writeValues(interp_file, var)

        print(" Done")
        
        
    def createCsvFile(self, path, variable_list, outname_list):
        """
        Creates a csv file with the variables specified in variable_list
        
        Arguments:
        path -- path where .csv file should be saved.
        variable_list -- a list of strings containing names of variables present in the
                         tecplot .plt file that should be present in the interpolation
                         file.
        outname_list -- a list of strings indicating the desired name for the variables
                        in the interpolation file. Typically, we use user defined scalars
                        to input diffusivity, so name should be uds-0, uds-1, etc.                        
        """
        
        print("Writing csv file with results...", 
                end="", flush=True)
                
        # Check to make sure the number of names is the same as the number of variables,
        # and is greater than one.
        assert len(variable_list) == len(outname_list), \
                 ("The number of names (outnames_to_write) must match the number"
                  + "of variables (variables_to_write)!")                  
        assert len(variable_list) >= 1, "Must write at least one variable!"
        
        # Call helper to get appropriate lists.
        x,y,z,vars = self.fetchVariablesToWrite(variable_list)
        
        with open(path, "w") as csv_file:
            
            # Write header (i.e., all variable names) 
            csv_file.write("X, Y, Z")
            for var_name in variable_list:
                csv_file.write(", {}".format(var_name))
            csv_file.write("\n")
            
            # Write the variables
            for i in range(len(x)):
                # Here, write the x,y,z positions
                csv_file.write("{:.6e}, {:.6e}, {:.6e}".format(x[i], y[i], z[i]))
                
                # Now, write the variables
                for var in vars:
                    csv_file.write(", {:.6e}".format(var[i]))
                    
                csv_file.write("\n")                

        print(" Done")
        
        
    def saveDataset(self, path):
        """
        Saves current state of tecplot dataset as .plt binary file.        
        
        Arguments:
        path -- path where .plt file should be saved.
        """
        
        print("Saving .plt file to {}...".format(path), end="", flush=True)
        tecplot.data.save_tecplot_plt(filename=path, dataset=self.__dataset)
        print(" Done")
        
        