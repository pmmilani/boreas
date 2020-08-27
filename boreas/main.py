#-------------------------------- main.py file -----------------------------------------#
"""
Main file - entry point to the code. This file coordinates all other files and 
implements all the functionality directly available to the user.
"""

# import statements
import numpy as np
import joblib
from pkg_resources import get_distribution
from boreas.models import makePrediction, RFModelIsotropic, TBNNSModelAnisotropic
from boreas.case import TestCase, TrainingCase
from boreas import process
from boreas import constants


def printInfo():
    """
    Makes sure everything is properly installed.
    
    We print a welcome message, the version of the package, and attempt to load
    the pre-trained models to make sure the data file is there. Return 1 at the end
    if no exceptions were raised.
    """
    
    print('Welcome to Boreas - a package for industrial deployment of machine-learned '
          + 'turbulent mixing models for film cooling (formerly known as RaFoFC)!')
    
    # Get distribution version
    dist = get_distribution('boreas')
    print('Version: {}'.format(dist.version))
    
    # Try to load the default RF model and print information about it
    print('Attempting to load the default RF model...')
    rf = RFModelIsotropic()
    rf.loadFromDisk()
    print('Default model was found and can be loaded properly.')
    print('\t Description: ', end="", flush=True)
    rf.printDescription()
    
    # Try to load the default TBNNS model and print information about it
    print('Attempting to load the default TBNN-s model...')
    nn = TBNNSModelAnisotropic()
    nn.loadFromDisk()
    print('Default model was found and can be loaded properly.')
    print('\t Description: ', end="", flush=True)
    nn.printDescription()
    
    return 1 # return this if everything went ok

    
def applyMLModel(tecplot_in_path, tecplot_out_path, *,  
                 zone = None, deltaT0 = None, 
                 use_default_var_names = False, use_default_derivative_names = True,
                 calc_derivatives = True, write_derivatives = True, 
                 threshold = None, default_prt = None, clean_features = True, 
                 features_load_path = None, features_dump_path = None,
                 ip_file_path = None, csv_file_path = None,
                 variables_to_write = None, outnames_to_write = None,                 
                 model_path = None, secondary_model_path = None, 
                 model_type = "RF", features_type="F2",
                 ensemble_of_models = False, std_ensemble = False):
    """
    Applies ML model on a single test case, given in a Tecplot file.
    
    Main function of package. Call this to take in a Tecplot file, process it, apply
    the machine learning model, and save results to disk. All optional arguments must
    be used with the identifying keyword (that's what * means)
    
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
                 point. If None, use the value in constants.py (default value is 1e-3).
                 For temperature gradient less than that, we use the Reynolds analogy
                 (with fixed Pr_t). For gradients larger than that, we use the 
                 model.
    default_prt -- optional argument, this variable contains the default value of Pr_t to
                   use in regions where gradients are low or features have been cleaned.
                   If this is None (default), then use the value from constants.py.
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
                          Fluent interpolation file/CSV file. By default, it is None, 
                          which leads the program to pick only the diffusivity variables
                          just calculated.
    outnames_to_write -- optional argument. This is a list of strings that must have the 
                        same length as the previous argument. It contains the names that
                        each of the variables written in the interpolation/csv files will 
                        have. By default, this is None, which leads to code to name all
                        variables being written sequentially, starting at "uds-2". Naming
                        them as "user defined scalars x" (uds-x) is an easy way to read
                        them in Fluent.
    model_path -- optional argument. This is the path where the function will look for
                  a pre-trained machine learning model. The file must be a pickled
                  instance of a random forest regressor class or a pickled instance of
                  the TBNN-s class, saved to disk using joblib. If None, the default
                  machine learning model that comes with the package(which is already
                  pre-trained with LES/DNS) is employed.
    secondary_model_path -- optional argument. This is the path where the function will
                            look for a pre-trained random forest model to support the 
                            TBNN-s model in the hybrid formulation. The file must be a
                            pickled instance of a random forest regressor class, saved to
                            disk using joblib. By default, the default RF is loaded. This
                            argument is only relevant when model_type = "TBNNS_hybrid".
    model_type -- optional argument. This tells us which type of model we are loading.
                  It must be a string, and the currently supported options are "RF",
                  "TBNNS", and "TBNNS_hybrid". The default option is "RF".
    features_type -- optional argument, string determining the type of features that
                     we are currently extracting. Options are "F1" and "F2". Default
                     value is "F2".
    ensemble_of_models -- optional argument. This is a boolean flag that tells us whether
                          to use a model ensemble instead of a single model instance. If
                          this is true, the model_path parameter must be a list of paths
                          instead of a single path. The default option is "False"
    std_ensemble -- optional argument. This is a boolean flag that instructs the solver
                    to return the standard deviation across the ensemble of models. This
                    can only be True if ensemble_of_models=True; in which case, we only
                    return the standard deviation and not the actual diffusivity. This
                    option only makes sense for the TBNN-s model (since the RF is already
                    an ensemble) and is not supported for the hybrid model. The default
                    option is "False"
    """
    
    assert model_type == "RF" or model_type == "TBNNS" or model_type == "TBNNS_hybrid", \
            "Invalid model_type received!"
    assert features_type == "F1" or features_type == "F2", \
            "Invalid features_type received!"
            
    if ensemble_of_models: # check whether model_path is a list if model ensemble
        assert type(model_path) is list, \
            "Error! For a model ensemble, model_path must be a list"
        assert len(model_path) > 0, "Error! model_path is an empty list!"
    
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
    
    # Here, run the code for applying random forest model ("RF")
    if model_type == "RF":
        # This line processes the dataset and extracts features for the ML step which
        # can take a long time. features_load_path and features_dump_path can be
        # set to make the method load/save the processed quantities from disk.
        x, _ = dataset.extractFeatures(with_tensor_basis=False, 
                                       features_type=features_type, threshold=threshold, 
                                       features_load_path=features_load_path,
                                       features_dump_path=features_dump_path,
                                       clean_features=clean_features)        
        prt_ML = makePrediction("RF", model_path, x, features_type)
        
        # Adds result to tecplot and sets the default variable names to output
        varname = "Prt_ML"
        dataset.addPrt(prt_ML, varname, default_prt)
        if variables_to_write is None: 
            variables_to_write = [varname]
        if outnames_to_write is None: 
            outnames_to_write = ["uds-2"]
    
    # Here, run the code for applying TBNN model ("TBNNS")
    elif model_type == "TBNNS":
        # This line processes the dataset and returns the features and tensor basis
        # at each point in the dataset where gradients are significant.
        x, tb = dataset.extractFeatures(with_tensor_basis=True, 
                                        features_type=features_type, threshold=threshold, 
                                        features_load_path=features_load_path,
                                        features_dump_path=features_dump_path,
                                        clean_features=clean_features)        
        alphaij_ML, g_ML = makePrediction("TBNNS", model_path, x, features_type, tb, 
                                          ensemble=ensemble_of_models, 
                                          std_flag=std_ensemble)
                
        # Adds result to tecplot and sets the default variable names to output
        varname = ["Dxx", "Dxy", "Dxz", "Dyx", "Dyy", "Dyz", "Dzx", "Dzy", "Dzz"]
        dataset.addTensorDiff(alphaij_ML, varname, default_prt)
        g_name = ["g1", "g2", "g3", "g4", "g5", "g6"]
        dataset.addG(g_ML, g_name, default_prt)
        
        if variables_to_write is None: 
            variables_to_write = varname
        if outnames_to_write is None: 
            outnames_to_write = ["uds-2", "uds-3", "uds-4", "uds-5", "uds-6", "uds-7",
                                 "uds-8", "uds-9", "uds-10"]
    
    # Here, run the code for applying TBNN-s + random forest model ("TBNNS_hybrid")
    elif model_type == "TBNNS_hybrid":
        # This line processes the dataset and returns the features and tensor basis
        # at each point in the dataset where gradients are significant.
        x, tb = dataset.extractFeatures(with_tensor_basis=True, 
                                        features_type=features_type, threshold=threshold, 
                                        features_load_path=features_load_path,
                                        features_dump_path=features_dump_path,
                                        clean_features=clean_features)        
        alphaij_ML, g_ML = makePrediction("TBNNS", model_path, x, features_type, tb,
                                          ensemble=ensemble_of_models, 
                                          std_flag=std_ensemble)

        # Now, get a random forest prediction for the turbulent Prandtl number
        prt_ML = makePrediction("RF", secondary_model_path, x, features_type)        
        
        # Combine alphaij_ML and prt_ML into a single diffusivity tensor
        alphaij_mod = dataset.enforcePrt(alphaij_ML, prt_ML)
        
        # Adds result to tecplot and sets the default variable names to output
        varname = ["Dxx", "Dxy", "Dxz", "Dyx", "Dyy", "Dyz", "Dzx", "Dzy", "Dzz"]
        dataset.addTensorDiff(alphaij_mod, varname, default_prt)
        g_name = ["g1", "g2", "g3", "g4", "g5", "g6"]
        dataset.addG(g_ML, g_name, default_prt)
        
        if variables_to_write is None: 
            variables_to_write = varname
        if outnames_to_write is None: 
            outnames_to_write = ["uds-2", "uds-3", "uds-4", "uds-5", "uds-6", "uds-7",
                                 "uds-8", "uds-9", "uds-10"]
        
    # Write output: create interp/csv files and produce tecplot file
    if ip_file_path is not None:
        dataset.createInterpFile(ip_file_path, variables_to_write, outnames_to_write)    
    if csv_file_path is not None:
        dataset.createCsvFile(csv_file_path, variables_to_write, outnames_to_write)        
    dataset.saveDataset(tecplot_out_path)


def produceTrainingFeatures(tecplot_in_path, *, data_path = None,  
                            zone = None, deltaT0 = None, 
                            use_default_var_names = False, 
                            use_default_derivative_names = True,
                            calc_derivatives = True, write_derivatives = True, 
                            threshold = None, clean_features = True, 
                            features_load_path = None, features_dump_path = None,
                            prt_cap = None, gamma_correction = False,
                            downsample = None, tecplot_out_path = None,
                            model_type = "RF", features_type="F2"):
                            
    """
    Produces features and labels from a single Tecplot file, used for training.
    
    This function is useful for training your own models. Call it on a single Tecplot
    file (.plt) that contains all mean data including u'c' values, and it will process
    it to generate the features and labels used for training. All optional arguments may
    only be used with the keyword (that's what * means)
    
    Arguments:
    tecplot_in_path -- string containing the path of the input tecplot file. It must be
                       a binary .plt file, resulting from a k-epsilon simulation.
    data_path -- optional argument. A string containing the path where a joblib file is
                 saved, containing features and labels for training ML models. If None
                 (default), a default name is employed.
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
                 point. If None, use the value in constants.py (default value is 1e-3).
                 For temperature gradient less than that, we use the Reynolds analogy
                 (with fixed Pr_t). For gradients larger than that, we use the 
                 model.
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
    prt_cap -- optional, contains the (symmetric) cap on the value of Pr_t. If None,
               then use the value in constants.py. If this value is 100, for example,
               then 0.01 < Pr_t < 100, and values outside of this range are capped.
    gamma_correction -- optional. If True, use the correction defined in 
                        Milani, Ling, Eaton (JTM 2020). That correction only makes
                        sense if training data is on a fixed reference frame (it breaks
                        Galilean invariance), so it is turned off by default. However,
                        it can improve results in some cases.
    downsample -- optional, number that controls how we downsample the data before
                  saving it to disk. If None (default), it will read the number from 
                  constants.py. If this number is more than 1, then it represents the
                  number of examples we want to save; if it is less than 1, it represents
                  the ratio of all training examples we want to save.
    tecplot_out_path -- optional, a string containing the path to which the final tecplot
                        dataset will be saved. Useful for sanity checking the results. By
                        default it is None (no .plt file saved)
    model_type -- optional argument. This tells us which type of model we are loading.
                  It must be a string, and the currently supported options are "RF".
                  The default option is "RF".
    features_type -- optional argument, string determining the type of features that
                     we are currently extracting. Options are "F1" and "F2". Default
                     value is "F2".
    """
    
    assert model_type == "RF" or model_type == "TBNNS" or model_type == "TBNNS_hybrid", \
           "Invalid model_type received!"
    
    # Initialize dataset and get scales for non-dimensionalization. The default behavior
    # is to ask the user for the names and the scales. Passing keyword arguments to this
    # function can be done to go around this behavior
    dataset = TrainingCase(tecplot_in_path, zone=zone, 
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
    
    metadata = {}
    metadata["features_type"] = features_type
    if model_type == "RF":
        # This line processes the dataset and extracts features for the ML step which
        # can take a long time. features_load_path and features_dump_path can be
        # set to make the method load/save the processed quantities from disk.
        x, _ = dataset.extractFeatures(with_tensor_basis=False, 
                                       features_type=features_type, threshold=threshold, 
                                       features_load_path=features_load_path,
                                       features_dump_path=features_dump_path,
                                       clean_features=clean_features)
        gamma = dataset.extractGamma(prt_cap, gamma_correction) # gamma = 1/Prt
        
        training_list = [x, gamma]  # what is used for training
        metadata["with_tensor_basis"]=False
        metadata["with_gamma"]=True
        
        # Write the Tecplot data to disk with the extracted Prt_LES for sanity check
        if tecplot_out_path is not None:
            dataset.addPrt(1.0/gamma, "Prt_LES")
            dataset.saveDataset(tecplot_out_path)  
    
    elif model_type == "TBNNS":
        # This line processes the dataset and returns the features and tensor basis
        # at each point in the dataset where gradients are significant.
        x, tb = dataset.extractFeatures(with_tensor_basis=True, 
                                        features_type=features_type, threshold=threshold, 
                                        features_load_path=features_load_path,
                                        features_dump_path=features_dump_path,
                                        clean_features=clean_features)                                        
        uc, gradT, nut = dataset.extractUc()
        
        training_list = [x, tb, uc, gradT, nut]  # what is used for training
        metadata["with_tensor_basis"]=True
        metadata["with_gamma"]=False    
        
    elif model_type == "TBNNS_hybrid":    
        # This line processes the dataset and returns the features and tensor basis
        # at each point in the dataset where gradients are significant.
        x, tb = dataset.extractFeatures(with_tensor_basis=True, 
                                        features_type=features_type, threshold=threshold, 
                                        features_load_path=features_load_path,
                                        features_dump_path=features_dump_path,
                                        clean_features=clean_features)
                                        
        uc, gradT, nut = dataset.extractUc()
        gamma = dataset.extractGamma(prt_cap, gamma_correction)
        
        training_list = [x, tb, uc, gradT, nut, gamma]  # what is used for training
        metadata["with_tensor_basis"]=True
        metadata["with_gamma"]=True    

    # Saves joblib file to disk with features/labels for this dataset
    # If data_path is None, use default name (appending _trainingdata to the end)
    if data_path is None:
        data_path = tecplot_in_path[0:-4] + "_trainingdata.pckl" # default name
    
    # Save training features to disk
    process.saveTrainingFeatures(training_list, metadata, data_path, downsample)


def trainRFModel(features_list, description, model_path, *, features_type="F1", 
                 downsample=None, n_trees = None, max_depth = None, 
                 min_samples_split = None, n_jobs = None):
    """
    Trains a random forest model and saves it to disk.
    
    Trains a random forest, isotropic model, with features and labels previously 
    calculated. Multiple files can be used at the same time (each file in the list
    comes from a given dataset. All optional arguments may only be used with the 
    keyword (that's what * means)
    
    Arguments:
    features_list -- list containing paths to files saved to disk with features
                     and labels for training. These files are produced by the function
                     above (produceTrainingFeatures) from Tecplot files.
    description -- A short, written description of the model being trained. It will be
                   saved to disk together with the model itself.
    model_path -- The path where the trained model will be saved in disk.
    features_type -- optional argument, string determining the type of features that
                     we are currently extracting. Options are "F1" and "F2". Default
                     value is "F1".
    downsample -- optional, number that controls how we downsample the data before
                  using it to train. If None (default), it will read the number from 
                  constants.py. If this number is more than 1, then it represents the
                  number of examples we want to save; if it is less than 1, it represents
                  the ratio of all training examples we want to save. Can also be a list
                  of numbers, in which case each number is applied to an element of
                  features_list
    n_trees -- optional. Hyperparameter of the random forest, contains number of
                   trees to use. If None (default), reads value from constants.py
    max_depth -- optional. Hyperparameter of the random forest, contains maximum
                 depth of each tree to use. If None (default), reads value from
                 constants.py 
    min_samples_split -- optional. Hyperparameter of the random forest, contains
                         minimum number of samples at a node required to split. Can
                         either be an int (number itself) or a float (ratio of total
                         examples). If None (default), reads value from constants.py
    n_jobs -- optional. Number of processors to use when training the RF (notice that
              training is embarassingly parallel). If None (default behavior), then
              the value is read from constants.py. See manual for 
              RandomForestRegressor class; if this is -1, all processors are used.
    """
    
    # Reads the list of files provided for features/labels
    print("{} file(s) were provided and will be used".format(len(features_list)))
    x_list = []
    y_list = []

    if isinstance(downsample, list): # make sure list is the right size
        assert len(downsample) == len(features_list), \
           "downsample is a list, but it has the wrong number of entries!"
           
    for i, file in enumerate(features_list):
        if isinstance(downsample, list): # if list, take each element sequentially
            x_features, gamma = process.loadTrainingFeatures(file, "RF", downsample[i],
                                                             features_type)
        else:
            x_features, gamma = process.loadTrainingFeatures(file, "RF", downsample,
                                                             features_type)
            
        x_list.append(x_features)
        y_list.append(gamma)             
    
    x_total = np.concatenate(x_list, axis=0)
    y_total = np.concatenate(y_list, axis=0)
    
    # Here, we train and save the model
    rf = RFModelIsotropic()
    rf.train(x_total, y_total, n_trees, max_depth, min_samples_split, n_jobs)
    rf.save(description, model_path)
    
    
def trainTBNNSModel(features_list_train, features_list_dev, description, model_path, 
                    path_to_saver, *, FLAGS={}, features_type="F2",
                    downsample_train=None, downsample_dev=None,):
    """
    Trains a TBNN-s and saves it to disk.
    
    Trains a TBNN-s, anisotropic model, with features and labels previously 
    calculated. Multiple files can be used at the same time (each file in the list
    comes from a given dataset. All optional arguments may only be used with the 
    keyword (that's what * means)
    
    Arguments:
    features_list_train -- list containing paths to files saved to disk with features
                     and labels for training. These files are produced by the function
                     above (produceTrainingFeatures) from Tecplot files.
    features_list_dev -- list containing paths to files saved to disk with features
                     and labels for the validation set. These files are produced by 
                     the function above (produceTrainingFeatures) from Tecplot files.
    description -- A short, written description of the model being trained. It will be
                   saved to disk together with the model itself.
    model_path -- The path where the trained model will be saved in disk.
    path_to_saver -- The path where the model parameters are saved to disk, through the
                    tf.Saver class. Usually, we want to put that in a folder called
                    checkpoints.
    FLAGS -- optional argument, dictionary that controls training parameters for 
             the TBNNS model. Check the tbnns package to see what settings can be used.
    features_type -- optional argument, string determining the type of features that
                     we are currently extracting. Options are "F1" and "F2". Default
                     value is "F2".
    downsample_train -- optional, number that controls how we downsample the data before
                  using it to train. If None (default), it will read the number from 
                  constants.py. If this number is more than 1, then it represents the
                  number of examples we want to save; if it is less than 1, it represents
                  the ratio of all training examples we want to save. Can also be a list
                  of numbers, in which case each number is applied to an element of
                  features_list.
    downsample_dev -- optional, same as above for the dev set.        
    """
    
    # Reads the list of files provided for features/labels
    print("{} file(s) were provided and will be used for training"\
            .format(len(features_list_train)))
    print("{} file(s) were provided and will be used for validation"\
            .format(len(features_list_dev)))
    
    # Makes sure downsample list is the right size
    if isinstance(downsample_train, list): 
        assert len(downsample_train) == len(features_list_train), \
           "downsample is a list, but it has the wrong number of entries!"
    if isinstance(downsample_dev, list): # make sure list is the right size
        assert len(downsample_dev) == len(features_list_dev), \
           "downsample is a list, but it has the wrong number of entries!"    
    
    # Load training files
    x_list_train=[]; tb_list_train=[]; uc_list_train=[]; 
    gradT_list_train=[]; nut_list_train=[]
    for i, file in enumerate(features_list_train):
        if isinstance(downsample_train, list): # if list, take each element sequentially
            x, tb, uc, gradT, nut = process.loadTrainingFeatures(file, "TBNNS", 
                                                                 downsample_train[i], 
                                                                 features_type)
        else:
            x, tb, uc, gradT, nut = process.loadTrainingFeatures(file, "TBNNS", 
                                                                 downsample_train,
                                                                 features_type)            
        x_list_train.append(x); tb_list_train.append(tb); uc_list_train.append(uc)
        gradT_list_train.append(gradT); nut_list_train.append(nut)   
    x_train = np.concatenate(x_list_train, axis=0)
    tb_train = np.concatenate(tb_list_train, axis=0)
    uc_train = np.concatenate(uc_list_train, axis=0)
    gradT_train = np.concatenate(gradT_list_train, axis=0)
    nut_train = np.concatenate(nut_list_train, axis=0)    
    
    # Load dev files
    x_list_dev=[]; tb_list_dev=[]; uc_list_dev=[]; 
    gradT_list_dev=[]; nut_list_dev=[]
    for i, file in enumerate(features_list_dev):
        if isinstance(downsample_dev, list): # if list, take each element sequentially
            x, tb, uc, gradT, nut = process.loadTrainingFeatures(file, "TBNNS", 
                                                                 downsample_dev[i],
                                                                 features_type)
        else:
            x, tb, uc, gradT, nut = process.loadTrainingFeatures(file, "TBNNS", 
                                                                 downsample_dev,
                                                                 features_type)            
        x_list_dev.append(x); tb_list_dev.append(tb); uc_list_dev.append(uc)
        gradT_list_dev.append(gradT); nut_list_dev.append(nut)   
    x_dev = np.concatenate(x_list_dev, axis=0)
    tb_dev = np.concatenate(tb_list_dev, axis=0)
    uc_dev = np.concatenate(uc_list_dev, axis=0)
    gradT_dev = np.concatenate(gradT_list_dev, axis=0)
    nut_dev = np.concatenate(nut_list_dev, axis=0)    

    # Edit FLAGS if necessary:
    if features_type=="F1": FLAGS['num_features'] = constants.NUM_FEATURES_F1
    elif features_type=="F2": FLAGS['num_features'] = constants.NUM_FEATURES_F2
        
    # Here, we train and save the model
    nn = TBNNSModelAnisotropic()
    nn.train(FLAGS, path_to_saver,
             x_train, tb_train, uc_train, gradT_train, nut_train,
             x_dev, tb_dev, uc_dev, gradT_dev, nut_dev)
    nn.save(description, model_path)