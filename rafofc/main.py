#-------------------------------- main.py file -----------------------------------------#
"""
Main file - entry point to the code. This file coordinates all other files and 
implements all the functionality directly available to the user.
"""

# import statements
import numpy as np
from pkg_resources import get_distribution
from rafofc.models import MLModel
from rafofc.tecplot_data import TPDataset

def printInfo():
    """
    Makes sure everything is properly installed.
    
    We print a welcome message, the version of the package, and attempt to load
    the pre-trained model to make sure the data file is there.
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

    
def applyMLModel(tecplot_in_path, tecplot_out_path, ip_file_path, zone=None, U0=None,
                 D=None, rho0=None, miu=None, deltaT=None, use_default_names=False, 
                 calc_derivatives=True, threshold=1e-4, rans_data_load_path=None,
                 rans_data_dump_path=None, variables_to_ip = ["alpha_t_ML"], 
                 outname_ip = ["uds-1"], ml_model_path=None):
    """
    Main function of package. Call this to take in a Tecplot file, process it, apply
    the machine learning model, and save results to disk.
    
    Arguments:
    tecplot_in_path -- string containing the path of the input tecplot file. It must be
                       a binary .plt file, resulting from a k-epsilon simulation.
    tecplot_out_path -- string containing the path to which the final tecplot dataset
                        will be saved.
    ip_file_path -- string containing the path to which the interpolation file (which 
                    is read by ANSYS Fluent) will be saved.
    zone -- optional argument. The zone where the flow field solution is saved in 
            Tecplot. By default, it is zone 0. This can be either a string (with the 
            zone name) or an integer with the zone index.
    U0 -- optional argument. Velocity scale that will be used to non-dimensionalize the
          dataset. If it is not provided (default behavior), then the user will be 
          prompted to enter an appropriate number.
    D -- optional argument. Length scale that will be used to non-dimensionalize the
         dataset. If it is not provided (default behavior), then the user will be 
         prompted to enter an appropriate number.
    rho0 -- optional argument. Density scale that will be used to non-dimensionalize the
            dataset. If it is not provided (default behavior), then the user will be 
            prompted to enter an appropriate number.
    miu -- optional argument. Dynamic viscosity scale of the fluid, which will be used to
           non-dimensionalize the dataset and calculate the Reynolds number. If it is not
           provided (default behavior), then the user will be prompted to enter an 
           appropriate number.
    deltaT -- optional argument. Temperature scale (Tmax - Tmin) that will be used to 
              non-dimensionalize the dataset. If it is not provided (default behavior),
              the user will be prompted to enter an appropriate number.    
    use_default_names -- optional argument. Boolean flag (True/False) that determines
                         whether default Fluent names will be used to fetch variables
                         in the Tecplot dataset. If the flag is False (default behavior),
                         the user will be prompted to enter names for each variable that
                         is needed.
    calc_derivatives -- optional argument. Boolean flag (True/False) that determines 
                        whether derivatives need to be calculated in the Tecplot file.
                        Note we need derivatives of U, V, W, and Temperature, with names
                        ddx_{}, ddy_{}, ddz_{}. If such variables were already calculated
                        and exist in the dataset, set this flag to False to speed up the 
                        process. By default (True), derivatives are calculated and a new
                        file with derivatives called "derivatives_{}" will be saved to 
                        disk.
    threshold -- optional argument. This variable determines the threshold for 
                 (non-dimensional) temperature gradient below which we throw away a 
                 point. Default value is 1e-4. For temperature gradient less than that,
                 we use the Reynolds analagoy (with Pr_t=0.85). For gradients larger than
                 that, we use the ML model.                 
    rans_data_load_path -- optional argument. If this is supplied, then the function will
                           try to load the rans_data class from disk instead of 
                           processing the tecplot file all over again. Since calculating
                           the features can take a while for large datasets, this can be 
                           useful to speed up repetitions.                           
    rans_data_dump_path -- optional argument. If this is provided and we processed the
                           tecplot data from scratch (i.e. we calculated the features), 
                           then the function will save the processed data (in rans_data)
                           to disk, so it is much faster to perform the same computations
                           again later.
    variables_to_ip -- optional argument. This is a list of strings containing names of
                       variables in the Tecplot file that we want to write in the Fluent
                       interpolation file. By default, it contains only "alpha_t_ML", 
                       which is the machine learning turbulent diffusivity we just 
                       calculated.
    outname_ip -- optional argument. This is a list of strings that must have the same
                  length as the previous argument. It contains the names that each of the
                  variables written in the interpolation file will have. By default, this
                  calls the turbulent diffusivity "uds-1" because in Fluent, an easy way
                  to solve the Reynolds-averaged scalar transport equation with a custom
                  diffusivity is to use a user-defined scalar as containing that custom
                  diffusivity.
    ml_model_path -- optional argument. This is the path where the function will look for
                     a pre-trained machine learning model. The file must be a pickled
                     instance of a random forest regressor class, saved to disk using 
                     joblib. By default, the default machine learning model (which is
                     already pre-trained with LES/DNS of 3 cases) is loaded, which comes
                     together with the package.  
    """
    
    
    # Initialize dataset and get scales for non-dimensionalization. The default behavior
    # is to ask the user for the names and the scales. Passing keyword arguments to this
    # function can be done to go around this behavior
    dataset = TPDataset(tecplot_in_path, zone=zone, use_default_names=use_default_names)
    dataset.normalize(U0=U0, D=D, rho0=rho0, miu=miu, deltaT=deltaT)
    
    # If this flag is True (default) calculate the derivatives and save the result to
    # disk (since it takes a while to do that...)
    if calc_derivatives:
        dataset.calculateDerivatives()
        dataset.saveDataset("derivatives_" + tecplot_in_path)
    
    # This line processes the dataset, create rans_data (a class holding all variables
    # as numpy arrays), and extracts features for the ML step (the latter can take a
    # time). rans_data_load_path and rans_data_dump_path can be not None to make the 
    # method load/save the processed quantities from disk.
    x = dataset.extractMLFeatures(threshold=threshold, 
                                  rans_data_load_path=rans_data_load_path,
                                  rans_data_dump_path=rans_data_dump_path)
    
    # Initialize the ML model and use it for prediction. If ml_model_path is None, jus
    # load the default model from disk. 
    rf = MLModel(ml_model_path)
    alpha_t_ML = rf.predict(x)
    
    # Add alpha_t_ML as a variable in tecplot, create Fluent interp file, and save new
    # tecplot file with all new variables.
    dataset.addMLDiffusivity(alpha_t_ML) 
    dataset.createInterpFile(ip_file_path, variables_to_ip, outname_ip) 
    dataset.saveDataset(tecplot_out_path)   

