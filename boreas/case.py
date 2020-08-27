#-------------------------------------- case.py ----------------------------------------#
"""
This file contains the class that holds a tecplot file and defines all the helpers to
load/edit/save a tecplot plot file
"""

# ------------ Import statements
import joblib # joblib is used to load files from disk
import timeit # for timing the derivative
import tecplot
import os
import numpy as np
from boreas import process
from boreas import constants
from tbnns.utils import cleanDiffusivity


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
    in the Tecplot file, False otherwise.
    
    Arguments:
    name -- the string that we are testing if it corresponds to a variable name or not
    dataset -- the tecplot.data.dataset class in which "name" must be a variable name
    
    Returns:
    True/False depending on whether name is a valid variable name in dataset
    """

    try:
        # if name is not a valid variable name, exception is thrown (or it returns None)
        var = dataset.variable(name)
        
        if var is None:
            return False
        
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
    

def collectCellSpatialVariables(tpdataset, zone):
    """
    Collects cell-centered spatial variables for a dataset
    
    Arguments:
    tpdataset -- the Tecplot dataset
    zone -- the zone of the dataset that we consider
    
    Returns:
    x, y, z -- numpy arrays containing cell centered values of Cartesian coordinates    
    """
    
    # First, interpolate x,y,z to cell centers        
    if not isVariable("X_cell", tpdataset):
        tecplot.data.operate.execute_equation('{X_cell} = {X}',
                          value_location=tecplot.constant.ValueLocation.CellCentered)
    if not isVariable("Y_cell", tpdataset):
        tecplot.data.operate.execute_equation('{Y_cell} = {Y}',
                          value_location=tecplot.constant.ValueLocation.CellCentered)
    if not isVariable("Z_cell", tpdataset):
        tecplot.data.operate.execute_equation('{Z_cell} = {Z}',
                          value_location=tecplot.constant.ValueLocation.CellCentered)
    
    # Collect as lists    
    x = zone.values("X_cell").as_numpy_array()
    y = zone.values("Y_cell").as_numpy_array()
    z = zone.values("Z_cell").as_numpy_array()
    
    # Remove X_cell, Y_cell, Z_cell in case they exist
    xvars = ['X_cell', 'Y_cell', 'Z_cell']
    for var in xvars:
        tpdataset.delete_variables(tpdataset.variable(var))
    
    return x, y, z   
    
    
class Case:
    """
    This class is holds the information of a single case (either training or testing).
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
        self._tpdataset = tecplot.data.load_tecplot(filepath, 
                                read_data_option=tecplot.constant.ReadDataOption.Replace)
        print("Tecplot file loaded successfully! "
               + "It contains {} zones and ".format(self._tpdataset.num_zones) 
               + "{} variables".format(self._tpdataset.num_variables))
               
        # Extracting zone. Print zone name and number of cells/nodes as sanity check
        if zone is None:
            print("All operations will be performed on zone 0 (default)")
            self._zone = self._tpdataset.zone(0)            
        else:
            print("All operations will be performed on zone {}".format(zone))
            self._zone = self._tpdataset.zone(zone)
            
        print('Zone "{}" has {} cells'.format(self._zone.name, self._zone.num_elements) 
              + " and {} nodes".format(self._zone.num_points))
        
        # Initializes a dictionary containing correct names for this dataset
        self.initializeVarNames(use_default_names)        

    
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
        self.var_names = {} 
        
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
                assert isVariable(default_names[i], self._tpdataset), \
                      "Default name {} is not a valid variable!".format(default_names[i])
                
                # Add default name to dictionary
                self.var_names[key] = default_names[i]
            
            # Here, just get a name from the user
            else:
                self.var_names[key] = getVarNameFromUser("Enter name for "
                                                           + "{} variable: ".format(key), 
                                                            self._tpdataset, 
                                                            default_names[i])
                                                            
        # Finally, assert X, Y, Z are valid variables:
        assert isVariable("X", self._tpdataset), 'Variable "X" not found in the dataset!'
        assert isVariable("Y", self._tpdataset), 'Variable "Y" not found in the dataset!'
        assert isVariable("Z", self._tpdataset), 'Variable "Z" not found in the dataset!'
            
    
    def normalize(self, deltaT0=None):
        """
        Collects scales from the user, which are used to non-dimensionalize data.
        
        This function is called to read in the dimensional scales of the dataset, so we
        can appropriately non-dimensionalize the temperature later. It can also be called 
        with an argument, in which case the user is not prompted to write anything. The
        temperature scale must be given in the same units as the dataset - it will be
        used to re-scale the temperature gradient.
        
        Arguments:
        deltaT -- temperature scale
        """
        
        # Read in the five quantities if they are not passed in                
        if deltaT0 is None:
            deltaT0 = getFloatFromUser("Enter temperature delta (Tmax-Tmin): ")
            
        # Set instance variables to hold them
        self.deltaT0 = deltaT0
        
    
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
            ddx_eqn = "{ddx_" + var + "} = ddx({" + self.var_names[var] + "})"
            ddy_eqn = "{ddy_" + var + "} = ddy({" + self.var_names[var] + "})"
            ddz_eqn = "{ddz_" + var + "} = ddz({" + self.var_names[var] + "})"
            
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
        print("Taking derivatives took {:.1f} min".format((toc - tic)/60.0))
        
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
                assert isVariable(default_names[i], self._tpdataset), \
                      "Default name {} is not a valid variable!".format(default_names[i])
                                
                # Add default names to dictionary
                self.var_names[var] = default_names[i]                
            
            else: 
                
                # Here, we ask the user for each derivative's name
                self.var_names[var] = getVarNameFromUser("Enter name for " 
                                                           + "{} variable: ".format(var),
                                                           self._tpdataset,
                                                           default_names[i]) 
                                                                  
                                                                  
    def extractFeatures(self, with_tensor_basis=False, features_type="F2",
                        threshold=None, clean_features=True, 
                        features_load_path=None, features_dump_path=None):
        """
        Extract quantities from the tecplot file and calculates features.
        
        This method is called to extract all Tecplot quantities into appropriate numpy
        arrays and then calculating the features for ML prediction.
        
        Arguments:
        with_tensor_basis -- optional argument, boolean that determines whether we extract
                         tensor basis together with features or not. Should be True when
                         TBNN-s is employed. Default value is False.
        features_type -- optional argument, string determining the type of features that
                         we are currently extracting. Options are "F1" and "F2". Default
                         value is "F2".
        threshold -- magnitude cut-off for the temperature scalar gradient: only use 
                     points with (dimensionless) magnitude higher than this. Default is
                     None (which means the value in constants.py is employed).
        clean_features -- optional keyword argument. If this is True, we clean the
                          features that are returned, and the should_use array is
                          modified accordingly.
        features_load_path -- optional keyword argument. If this is not None, this 
                              function will attempt to read rans_data from disk and
                              restore it instead of recalculating everything. The path
                              where rans_data is located is given by rans_data_path.
        features_dump_path -- optional keyword argument. If this is not None, and the 
                              above argument is None, then this method will calculate
                              rans_data for the current dataset and then save it to
                              disk (to path specified by features_dump_path). Use it
                              to avoid recalculating features (since it can be an 
                              expensive operation)        
        
        Returns:
        x_features -- a numpy array containing the features for prediction in the ML
                      step. The shape should be (n_useful, n_features)
        """
        
        # If we were passed a valid features_load_path, restore features from disk and
        # return them. Joblib .pckl file containing features is a list containing 
        # [metadata, [x_features, tensor_basis, should_use]]
        print("Feature settings: features_type={} ".format(features_type)
              + "and with_tensor_basis={}".format(with_tensor_basis))
        if features_load_path:
            if os.path.isfile(features_load_path):
                print("A valid path for the features was provided "
                      + "(path provided: {})".format(features_load_path))
                print("They will be read from disk...", end="", flush=True)
                
                metadata, data = joblib.load(features_load_path)
                self.x_features, self.tensor_basis, self.should_use = data
                
                # Make sure the metadata matches what we need.
                if with_tensor_basis is True:
                    assert metadata['with_tensor_basis'], "No tensor basis!"
                    assert self.tensor_basis is not None, "No tensor basis!"
                if metadata['features_type']=="F1":
                    msg = "Incompatible features_type or features.shape! " \
                             + "metadata['features_type']=F1"
                    assert features_type=="F1" or features_type=="F2", msg
                    assert self.x_features.shape[1] == constants.NUM_FEATURES_F1, msg
                elif metadata['features_type']=="F2":
                    msg = "Incompatible features_type or features.shape! " \
                             + "metadata['features_type']=F2"
                    assert features_type=="F2", msg
                    assert self.x_features.shape[1] == constants.NUM_FEATURES_F2, msg               
                                
                print(" Done!")
                
                if features_type=="F2" and metadata['features_type']!="F2":
                    self.x_features = np.concatenate((self.x_features[:, 0:6], 
                                                      self.x_features[:, 13:15]), axis=1)
                return self.x_features, self.tensor_basis                
                
            else:
                print("Invalid path provided for the features! "
                      + "(path provided: {})".format(features_load_path))
                print("Features will be calculated instead.")
                       
        #---- This section performs all necessary commands to obtain features
        # Initialize the class containing appropriate mean flow quantities
        mean_qts = process.MeanFlowQuantities(self._zone, self.var_names, 
                                              features_type, self.deltaT0)
        
        # Determine which points should be used, shape (n_cells)
        if threshold is None:
            threshold = constants.THRESHOLD
        self.should_use = process.calculateShouldUse(mean_qts, threshold)
        
        # Calculate features, shape (n_useful, N_FEATURES)
        self.x_features, self.tensor_basis = \
                process.calculateFeatures(mean_qts, self.should_use, 
                                          with_tensor_basis, features_type)

        # Remove outlier features if necessary
        if clean_features:
            (self.x_features, self.tensor_basis, 
                    self.should_use) = process.cleanFeatures(self.x_features,
                                                             self.tensor_basis,
                                                             self.should_use)
        #---- Finished calculating features       
        
        # If a dump path is supplied, then save features to disk to disk
        if features_dump_path:
            print("Saving features to disk...", end="", flush=True)
            metadata = {"with_tensor_basis": with_tensor_basis,
                        "features_type": features_type}
            joblib.dump([metadata, [self.x_features, self.tensor_basis, self.should_use]],
                        features_dump_path, 
                        compress=constants.COMPRESS, protocol=constants.PROTOCOL)
            print(" Done!")
        
        # return the extracted feature array
        return self.x_features, self.tensor_basis
        
    
    def addPrt(self, prt, varname, default_prt):
        """
        Adds Prt and should_use as variables in the Tecplot file.
        
        This method takes in a diffusivity array alpha_t that was predicted by the 
        machine learning model and adds that as a variable to the Tecplot file. It
        also adds should_use, which is very useful for visualization. should_use is 1
        where we use the ML model and 0 where we use the default Reynolds analogy.
        
        Arguments:
        prt -- numpy array shape (num_useful, ) with the turbulent Prandtl number
               (Pr_t) at each cell.
        varname -- string with the name for the new Prt variable in the dataset.
        default_prt -- value for the default Pr_t to use where should_use == False.
                       Can be None, in which case default value is read from constants.py
        """
    
        # First, create a Prandtl number that is available at every cell
        Prt_full = process.fillPrt(prt, self.should_use, default_prt)
        
        # Creates a variable called "Prt_ML" and "should_use" everywhere
        self._tpdataset.add_variable(name=varname,
                                   dtypes=tecplot.constant.FieldDataType.Float,
                                   locations=tecplot.constant.ValueLocation.CellCentered)
        self._tpdataset.add_variable(name="should_use",
                                   dtypes=tecplot.constant.FieldDataType.Int16,
                                   locations=tecplot.constant.ValueLocation.CellCentered)
        
        # Add Pr_t_full to the zone
        assert self._zone.num_elements == Prt_full.size, \
                                "Prt_full has wrong number of entries"
        self._zone.values(varname)[:] = Prt_full

        # Add should_use to the zone                
        self._zone.values("should_use")[:] = self.should_use
        
        
    def saveDataset(self, path):
        """
        Saves current state of tecplot dataset as .plt binary file.        
        
        Arguments:
        path -- path where .plt file should be saved.
        """        
                
        print("Saving .plt file to {}...".format(path), end="", flush=True)
        tecplot.data.save_tecplot_plt(filename=path, dataset=self._tpdataset)
        print(" Done!")
        
        
class TestCase(Case):
    """
    This class is holds the information of a single test case (just RANS simulation).
    """

    def enforcePrt(self, alphaij, prt):
        """
        Combines the direction from tensorial diffusivity with magnitude from Pr_t
        
        Arguments:
        alphaij -- numpy array shape (num_useful, 3, 3) with the dimensionless turbulent
                   diffusivity matrix at each significant point of the domain.
        prt -- numpy array shape (num_useful, ) with the value of turbulent Prandtl 
               number predicted by an isotropic model (RF)

        Returns:
        alphaij_mod -- numpy array shape (num_useful, 3, 3) with the modified turbulent
                       diffusivity, that takes Pr_t into account by scaling the tensor
                       accordingly.        
        """
        
        print("Modifying diffusivity tensor to enforce turbulent Prandtl number...", 
              end="", flush=True)
        
        # Get gradc vector to determine appropriate induced norm
        dcdx = self._zone.values(self.var_names["ddx_T"]).as_numpy_array()[self.should_use]
        dcdy = self._zone.values(self.var_names["ddy_T"]).as_numpy_array()[self.should_use]
        dcdz = self._zone.values(self.var_names["ddz_T"]).as_numpy_array()[self.should_use]
        gradmag = dcdx**2 + dcdy**2 + dcdz**2
        
        # Spell out calculation of the induced two-norm of A given vector gradc
        uc_predicted = (alphaij[:,0,0]*dcdx + alphaij[:,0,1]*dcdy + alphaij[:,0,2]*dcdz)
        vc_predicted = (alphaij[:,1,0]*dcdx + alphaij[:,1,1]*dcdy + alphaij[:,1,2]*dcdz)
        wc_predicted = (alphaij[:,2,0]*dcdx + alphaij[:,2,1]*dcdy + alphaij[:,2,2]*dcdz)
        gamma_now = (uc_predicted*dcdx + vc_predicted*dcdy + wc_predicted*dcdz)/gradmag

        # Gamma required is given by prt
        gamma_desired = 1.0/prt
        
        # Clip both values of gamma (might be redundant, but just to be safe)
        gamma_now = np.minimum(gamma_now, constants.PRT_CAP)
        gamma_now = np.maximum(gamma_now, 1.0/constants.PRT_CAP)
        gamma_desired = np.minimum(gamma_desired, constants.PRT_CAP)
        gamma_desired = np.maximum(gamma_desired, 1.0/constants.PRT_CAP)
        
        # Calculate modified diffusivity
        factor = np.reshape(gamma_desired/gamma_now, [-1, 1, 1])
        alphaij_mod = alphaij * factor
        print(" Done!")
        
        # Clean diffusivity, from tbnns.utils
        alphaij_mod = cleanDiffusivity(alphaij_mod, prt_default=constants.PRT_DEFAULT,
                                       gamma_min=1.0/constants.PRT_CAP, 
                                       clip_elements=True)    
    
        return alphaij_mod
    
    def addTensorDiff(self, alphaij, varnames, default_prt):
        """
        Adds a tensorial diffusivity to the Tecplot file
        
        Arguments:
        alphaij -- numpy array shape (num_useful, 3, 3) with the dimensionless turbulent
                   diffusivity matrix at each significant point of the domain.
        varnames -- list of strings with the names of each tensor entry for the dataset.
        default_prt -- value for the default Pr_t to use where should_use == False.
                       Can be None, in which case default value is read from constants.py
        """
        
        assert alphaij[0].size == len(varnames), "Wrong number of names for alphaij!"
        
        # Create full tensor, at every point of the domain
        alphaij_full = process.fillAlpha(alphaij, self.should_use, default_prt)
        
        # Creates all necessary variables in the .plt file
        for name in varnames:
            self._tpdataset.add_variable(name=name,
                                   dtypes=tecplot.constant.FieldDataType.Float,
                                   locations=tecplot.constant.ValueLocation.CellCentered)
        self._tpdataset.add_variable(name="should_use",
                                   dtypes=tecplot.constant.FieldDataType.Int16,
                                   locations=tecplot.constant.ValueLocation.CellCentered)
        
        # Add alphaij to the zone
        assert self._zone.num_elements == alphaij_full.shape[0], \
                                       "alphaij_full has wrong number of entries"
        for i in range(3):
            for j in range(3):
                name = varnames[3*i+j]
                self._zone.values(name)[:] = alphaij_full[:,i,j]

        # Add should_use to the zone                
        self._zone.values("should_use")[:] = self.should_use
    
    
    def addG(self, g, varnames, default_prt):
        """
        Adds the coefficient g to the Tecplot file
        
        Arguments:
        g -- numpy array shape (num_useful, num_basis) with the coefficient g that
             multiplies the equivalent tensor basis
        varnames -- list of strings with the names of each coefficient for the dataset.
        default_prt -- value for the default Pr_t to use where should_use == False.
                       Can be None, in which case default value is read from constants.py
        """
        
        assert g.shape[1] == len(varnames), "Wrong number of names for g!"
        
        # Create full tensor, at every point of the domain
        g_full = process.fillG(g, self.should_use, default_prt)
        
        # Creates all necessary variables in the .plt file
        for name in varnames:
            self._tpdataset.add_variable(name=name,
                                   dtypes=tecplot.constant.FieldDataType.Float,
                                   locations=tecplot.constant.ValueLocation.CellCentered)        
        
        # Add g to the zone
        assert self._zone.num_elements == g_full.shape[0], \
                                          "g_full has wrong number of entries"
        for i in range(constants.N_BASIS):
            name = varnames[i]
            self._zone.values(name)[:] = g_full[:,i]
        
    
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
        num_cells = self._zone.num_elements
        
        x,y,z = collectCellSpatialVariables(self._tpdataset, self._zone)       
        
        assert (x.size == num_cells and y.size == num_cells and z.size == num_cells), \
                                "x,y,z variables have wrong number of entries!"
        
        # Now, get a list of variables (as numpy arrays) that will be written
        vars = []
        for var_name in variable_list:
        
            # Check names are valid, otherwise crash
            assert isVariable(var_name, self._tpdataset), \
               "{} is not a valid variable to write in interp/csv file!".format(var_name)
                      
            var = self._zone.values(var_name).as_numpy_array()
            
            # Check variable has correct length
            assert var.size == num_cells, \
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
        
        print("Writing interpolation Fluent file {}...".format(path), end="", flush=True)
        
        # Check to make sure the number of names is the same as the number of variables,
        # and is greater than one.
        assert len(variable_list) == len(outname_list), \
                 ("The number of names (outnames_to_write) must match the number"
                  + "of variables (variables_to_write)!")                  
        assert len(variable_list) >= 1, "Must write at least one variable!"
        
        # Call helper to get appropriate numpy arrays, also checks the sizes
        x,y,z,vars = self.fetchVariablesToWrite(variable_list)
        
        # Open the file and write the variables with the correct format
        with open(path, "w") as interp_file:
            
            # Write header
            interp_file.write("3\n") # version. Must be 3
            interp_file.write("3\n") # dimensionality. must be 3
            interp_file.write("{}\n".format(x.size)) # number of points
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

        print(" Done!")
        
        
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
        
        print("Writing csv file {}...".format(path), end="", flush=True)
                
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
            for i in range(x.size):
                # Here, write the x,y,z positions
                csv_file.write("{:.6e}, {:.6e}, {:.6e}".format(x[i], y[i], z[i]))
                
                # Now, write the variables
                for var in vars:
                    csv_file.write(", {:.6e}".format(var[i]))
                    
                csv_file.write("\n")                

        print(" Done!")



class TrainingCase(Case):
    """
    This class is holds the information of a single training case (LES information, including u'c')
    """
    
    def __init__(self, filepath, zone=None, use_default_names=False):
        """
        This initializes a TrainingCase, where the names for u'c' vector are also collected.
        """
        super().__init__(filepath, zone, use_default_names) # performs regular initialization
        
        self.initializeUcNames(use_default_names); # adds extra variable names
    
        
    def initializeUcNames(self, use_default_names=False):
        """
        Completes the dictionary containing the string names with u'c' variables.
        
        This function adds extra names to the existing dictionary of variables->names
        related to the training set, where we need u'c' information
        
        Arguments:
        use_default_names -- optional, whether to use default names (from Fluent) to 
                             fetch variables in the .plt file. Must be false unless all 
                             variables of interest in Tecplot have the default name.
        """
                
        # These are the keys for the 3 extra variables we need for training
        variables = ["uc", "vc", "wc"]
                     
        # These are the default names we enter if user refuses to provide one
        default_names = ["uc", "vc", "wc"]
                         
        # Loop through all the variables that we must add
        for i, key in enumerate(variables):
        
            # If I want to use default names for the variables
            if use_default_names:                
                
                # Check default names are valid, otherwise crash
                assert isVariable(default_names[i], self._tpdataset), \
                      "Default name {} is not a valid variable!".format(default_names[i])
                
                # Add default name to dictionary
                self.var_names[key] = default_names[i]
            
            # Here, just get a name from the user
            else:
                self.var_names[key] = getVarNameFromUser("Enter name for "
                                                           + "{} variable: ".format(key), 
                                                            self._tpdataset, 
                                                            default_names[i])
    
    
    def extractGamma(self, prt_cap=None, use_correction=False):
        """
        Extracts the value of 1.0/Pr_t that will be used as a label for regression
        
        Arguments:
        prt_cap -- optional, contains the (symmetric) cap on the value of Pr_t. If None,
                   then use the value in constants.py. If this value is 100, for example,
                   then 0.01 < Pr_t < 100, and values outside of this range are capped.
        use_correction -- optional. If True, use the correction defined in 
                          Milani, Ling, Eaton (JTM 2020). That correction only makes
                          sense if training data is on a fixed reference frame (it breaks
                          Galilean invariance), so it is turned off by default. However,
                          it can improve results in some cases.
                          
        Returns:
        gamma -- numpy array containing gamma = 1.0/Pr_t at each cell, with 
                 shape (n_useful,)
        """
        
        # This contains all info we need as arrays. The arrays are already
        # filtered according to should_use, so they have shape (n_useful,...)
        # This means that this function can only run after extractFeatures
        mean_qts = process.MeanFlowQuantities(self._zone, self.var_names, labels=True,
                                              mask=self.should_use)
        
        # set to constants.py value if None is provided
        if prt_cap is None:
            prt_cap = constants.PRT_CAP
        
        gamma = process.calculateGamma(mean_qts, prt_cap, use_correction)
        
        return gamma

    
    def extractUc(self):
        """
        Extracts the values of uc, gradT, and nut which are used to train the TBNN-s
        
                          
        Returns:
        uc -- numpy array containing u'c' vector of shape (n_useful,3)
        gradT -- numpy array containing gradient of temperature of shape (n_useful,3)
        nut -- numpy array containing eddy kinematic viscosity of shape (n_useful,)
        """
        
        # This contains all info we need as arrays. The arrays are already
        # filtered according to should_use, so they have shape (n_useful,...)
        # This means that this function can only run after extractFeatures
        mean_qts = process.MeanFlowQuantities(self._zone, self.var_names, labels=True,
                                              mask=self.should_use)
                                              
        uc = mean_qts.uc
        gradT = mean_qts.gradT
        nut = mean_qts.mut/mean_qts.rho
        
        return uc, gradT, nut
        