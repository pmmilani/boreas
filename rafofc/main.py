#-------------------------------- main.py file -----------------------------------------#
"""
Main file - entry point to the code. This file coordinates all other files and 
implements all the functionality directly available to the user.
"""

# import statements
import numpy as np
from pkg_resources import get_distribution
from rafofc.models import MLModel
from rafofc.case import TestCase, TrainingCase
from rafofc import constants

def printInfo():
    """
    Makes sure everything is properly installed.
    
    We print a welcome message, the version of the package, and attempt to load
    the pre-trained model to make sure the data file is there. Return 1 at the end
    if no exceptions were raised.
    """
    
    print('Welcome to RaFoFC - Random Forest for Film Cooling package!')
    
    # Get distribution version
    dist = get_distribution('rafofc')
    print('Version: {}'.format(dist.version))
    
    # Try to load the model and print information about it
    print('Attempting to load the default model...')
    rafo = MLModel()
    print('Default model was found and can be loaded properly.')
    rafo.printDescription()
    
    return 1 # return this if everything went ok

    
def applyMLModel(tecplot_in_path, tecplot_out_path, *,  
                 zone = None, 
                 deltaT0 = None, 
                 use_default_var_names = False, use_default_derivative_names = True,
                 calc_derivatives = True, write_derivatives = True, 
                 threshold = 1e-3, clean_features = True, 
                 features_load_path = None, features_dump_path = None,
                 ip_file_path = None, csv_file_path = None,
                 variables_to_write = ["Prt_ML"], outnames_to_write = ["uds-1"],                 
                 ml_model_path = None):
    """
    Main function of package. Call this to take in a Tecplot file, process it, apply
    the machine learning model, and save results to disk. All optional arguments may
    only be used with the keyword (that's what * means)
    
    Arguments:
    tecplot_in_path -- string containing the path of the input tecplot file. It must be
                       a binary .plt file, resulting from a k-epsilon simulation.
    tecplot_out_path -- string containing the path to which the final tecplot dataset
                        will be saved.    
    zone -- optional argument. The zone where the flow field solution is saved in 
            Tecplot. By default, it is zone 0. This can be either a string (with the 
            zone name) or an integer with the zone index.    
    deltaT0 -- optional argument. Temperature scale (Tmax - Tmin) that will be used to 
               non-dimensionalize the dataset. If it is not provided (default behavior),
               the user will be prompted to enter an appropriate number.    
    use_default_var_names -- optional argument. Boolean flag (True/False) that determines
                             whether default Fluent names will be used to fetch variables
                             in the Tecplot dataset. If the flag is False (default 
                             behavior), the user will be prompted to enter names for each
                             variable that is needed.
    use_default_derivative_names -- optional argument. Boolean flag (True/False) that 
                                    determine if the user will pick the names for the
                                    derivative quantities in the Tecplot file or whether
                                    default names are used. This flag is only used if the
                                    next flag is False (i.e., if derivatives are 
                                    already pre-calculated, then setting this flag to 
                                    False allows the user to input the names of each
                                    derivative in the input .plt file). It defaults to
                                    True.
    calc_derivatives -- optional argument. Boolean flag (True/False) that determines 
                        whether derivatives need to be calculated in the Tecplot file.
                        Note we need derivatives of U, V, W, and Temperature, with names
                        ddx_{}, ddy_{}, ddz_{}. If such variables were already calculated
                        and exist in the dataset, set this flag to False to speed up the 
                        process. By default (True), derivatives are calculated and a new
                        file with derivatives called "derivatives_{}" will be saved to 
                        disk.
    write_derivatives -- optional argument. Boolean flag (True/False) that determines 
                         whether to write a binary Tecplot file to disk with the newly
                         calculated derivatives. The file will have the same name as the
                         input, except followed by "_derivatives". This is useful because
                         calculating derivatives takes a long time, so you might want to
                         save results to disk as soon as they are calculated.    
    threshold -- optional argument. This variable determines the threshold for 
                 (non-dimensional) temperature gradient below which we throw away a 
                 point. Default value is 1e-3. For temperature gradient less than that,
                 we use the Reynolds analagoy (with Pr_t=0.85). For gradients larger than
                 that, we use the ML model.
    clean_features -- optional argument. This determines whether we should remove outlier
                      points from the dataset before applying the model. This is measured
                      by the standard deviation of points around the mean.
    features_load_path -- optional argument. If this is supplied, then the function will
                           try to load the features from disk instead of 
                           processing the tecplot file all over again. Since calculating
                           the features can take a while for large datasets, this can be 
                           useful to speed up repetitions.                           
    features_dump_path -- optional argument. If this is provided and we processed the
                           tecplot data from scratch (i.e. we calculated the features), 
                           then the function will save the features to disk, so it is 
                           much faster to perform the same computations again later.
    ip_file_path -- optional argument. String containing the path to which the
                    interpolation file (which is read by ANSYS Fluent) will be saved. If
                    this argument is None (by default), then no interpolation file is
                    written.
    csv_file_path -- optional argument. String containing the path to which the csv file
                     (which can be read by StarCCM+) will be saved. If this is None 
                     (default), then no csv file is written.    
    variables_to_write -- optional argument. This is a list of strings containing names 
                          of variables in the Tecplot file that we want to write in the 
                          Fluent interpolation file/CSV file. By default, it contains
                          only "Prt_ML", which is the machine learning turbulent
                          Prandtl number we just calculated.   
    outnames_to_write -- optional argument. This is a list of strings that must have the 
                        same length as the previous argument. It contains the names that
                        each of the variables written in the interpolation file will 
                        have. By default, this calls the turbulent Prandtl number "uds-1" 
                        because in Fluent, an easy way to solve the Reynolds-averaged 
                        scalar transport equation with a custom diffusivity is to use a 
                        user-defined scalar as containing the custom Prandtl number.
    ml_model_path -- optional argument. This is the path where the function will look for
                     a pre-trained machine learning model. The file must be a pickled
                     instance of a random forest regressor class, saved to disk using 
                     joblib. By default, the default machine learning model (which is
                     already pre-trained with LES/DNS of 4 cases) is loaded, which comes
                     together with the package.    
    """
    
    
    # Initialize dataset and get scales for non-dimensionalization. The default behavior
    # is to ask the user for the names and the scales. Passing keyword arguments to this
    # function can be done to go around this behavior
    dataset = TestCase(tecplot_in_path, zone=zone, 
                        use_default_names=use_default_var_names)
    dataset.normalize(deltaT0=deltaT0)
    
    # If this flag is True (default) calculate the derivatives and save the result to
    # disk (since it takes a while to do that...)
    if calc_derivatives:
        dataset.calculateDerivatives()
        if write_derivatives: # write new Tecplot file to disk
            dataset.saveDataset(tecplot_in_path[0:-4] + "_derivatives.plt")
    else:
        print("Derivatives already calculated!")
        dataset.addDerivativeNames(use_default_derivative_names)
    
    # This line processes the dataset, create rans_data (a class holding all variables
    # as numpy arrays), and extracts features for the ML step (the latter can take a
    # time). features_load_path and features_dump_path can be not None to make the 
    # method load/save the processed quantities from disk.
    x = dataset.extractMLFeatures(threshold=threshold, 
                                  features_load_path=features_load_path,
                                  features_dump_path=features_dump_path,
                                  clean_features=clean_features)
    
    # Initialize the ML model and use it for prediction. If ml_model_path is None, just
    # load the default model from disk. 
    rf = MLModel(ml_model_path)
    Prt_ML = rf.predict(x)
    
    # Add Prt_ML as a variable in tecplot, create interp/csv files, and save new
    # tecplot file with all new variables.
    dataset.addPrtML(Prt_ML)    
    if ip_file_path is not None:
        dataset.createInterpFile(ip_file_path, variables_to_write, outnames_to_write)    
    if csv_file_path is not None:
        dataset.createCsvFile(csv_file_path, variables_to_write, outnames_to_write)        
    dataset.saveDataset(tecplot_out_path)


def trainMLModel():

    # Initialize dataset and get scales for non-dimensionalization. The default behavior
    # is to ask the user for the names and the scales. Passing keyword arguments to this
    # function can be done to go around this behavior
    dataset = TrainingCase(tecplot_in_path, zone=zone, 
                        use_default_names=use_default_var_names)
    dataset.normalize(deltaT=deltaT)
    
    # If this flag is True (default) calculate the derivatives and save the result to
    # disk (since it takes a while to do that...)
    if calc_derivatives:
        dataset.calculateDerivatives()
        if write_derivatives: # write new Tecplot file to disk
            dataset.saveDataset(tecplot_in_path[0:-4] + "_derivatives.plt")
    else:
        print("Derivatives already calculated!")
        dataset.addDerivativeNames(use_default_derivative_names)
