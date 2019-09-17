#--------------------------------- test_all.py -----------------------------------------#
"""
Test script containing several functions that will be invoked in unit testing. They are
built so the user can just call 'pytest'
"""

import os
import numpy as np
from boreas.main import printInfo, applyMLModel, produceTrainingFeatures, trainRFModel
from boreas import constants
from boreas.models import RFModel_Isotropic


def test_print_info():
    """
    This function calls the printInfo helper from main to make sure that it does not
    crash and returns 1 as expected.
    """
    
    assert printInfo() == 1
    
    
def test_loading_default_rf_model():
    """
    This function checks to see whether we can load the default RF model.
    """            
        
    # Test loading
    rafo_default = RFModel_Isotropic()
    rafo_default.loadFromDisk()    
    rafo_default.printDescription()
    
    # Test predicting
    n_test = 1000    
    x = -100.0 + 100*np.random.rand(n_test, constants.N_FEATURES)
    y = rafo_default.predict(x)
    assert y.shape == (n_test, )
    assert (y >= 1.0/constants.PRT_CAP).all() and (y <= constants.PRT_CAP).all()

    
def test_full_cycle_rf():
    """
    This function runs the full boreas procedure on the sample coarse RANS simulation
    and makes sure the results are the same that we got previously.
    """
    
    # Make the path relative to the location of the present script
    dirname = os.path.dirname(__file__)        
    
    # All relevant file names --
    input_filename_r1 = os.path.join(dirname, "JICF_BR1_supercoarse.plt")
    output_features_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_trainingdata.pckl")
    derivatives_file = os.path.join(dirname, "JICF_BR1_supercoarse_derivatives.plt")
    feature_dump_r1 = os.path.join(dirname, "features_r1.pckl")
    tecplot_filename_out_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_out.plt") #output tecplot file
    csv_output_name_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_out.csv") #output csv file
    savepath = os.path.join(dirname, "RFtest.pckl") # the model saved to disk
    # -------------------------
    
    #--------------- This first part trains a model using the BR=1 dataset
    print("\n")
    print("(1) -------------- Extracting features from BR=1 case")
    produceTrainingFeatures(input_filename_r1, data_path=output_features_r1,
                            deltaT0 = 1, # Tmax-Tmin = 1 in this case
                            use_default_var_names = True, # variable names are default
                            write_derivatives = True, # no need to cache derivatives
                            features_dump_path = feature_dump_r1,
                            model_type = "RF")   
    
    # Use the features extracted before to train the RF model
    print("\n")
    print("(2) -------------- Training model on extracted features")
    feature_list = [output_features_r1,]
    description = "Example RF model trained with supercoarse BR=1 LES"        
    trainRFModel(feature_list, description, savepath)    
    
    #------------- This second part applies the model trained previously to the BR=1 case
    print("\n")
    print("(3) -------------- Applying trained model on BR=1 case")   
    applyMLModel(derivatives_file, tecplot_filename_out_r1,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case
                 use_default_var_names = True, # variable names are default
                 calc_derivatives = False, # we are reading the derivatives file we just saved
                 write_derivatives = False,
                 features_load_path = feature_dump_r1, # load what we processed before
                 csv_file_path = csv_output_name_r1,
                 model_path = savepath, # path is the same we saved to previously
                 model_type = "RF") # here we choose the model type    
    
    # Remove all files I just wrote to disk
    os.remove(output_features_r1)
    os.remove(derivatives_file)
    os.remove(feature_dump_r1)
    os.remove(tecplot_filename_out_r1)
    os.remove(csv_output_name_r1)
    os.remove(savepath)