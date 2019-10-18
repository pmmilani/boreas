#--------------------------------- test_all.py -----------------------------------------#
"""
Test script containing several functions that will be invoked in unit testing. They are
built so the user can just call 'pytest'
"""

import os
import joblib
import numpy as np
from boreas.main import printInfo, applyMLModel, produceTrainingFeatures, trainRFModel
from boreas import constants

def test_print_info():
    """
    This function calls the printInfo helper from main to make sure that it does not
    crash and returns 1 as expected.
    """
    
    assert printInfo() == 1
    
 
def test_full_cycle_rf():
    """
    This function runs the full boreas procedure on the sample coarse RANS simulation
    with the isotropic model (RF)
    """
    
    # Make the path relative to the location of the present script
    dirname = os.path.dirname(__file__)        
    
    # All relevant file names -------
    input_filename_r2 = os.path.join(dirname, "JICF_BR2_supercoarse.plt")
    output_features_r2 = os.path.join(dirname, "JICF_BR2_supercoarse_trainingdata.pckl")
    derivatives_file = os.path.join(dirname, "JICF_BR2_supercoarse_derivatives.plt")
    feature_dump_r2 = os.path.join(dirname, "features_r2.pckl")
    tecplot_filename_out_r2 = os.path.join(dirname, "JICF_BR2_supercoarse_out.plt") #output tecplot file
    ip_output_name_r2 = os.path.join(dirname, "JICF_BR2_supercoarse_out.ip") #output ip file   
    savepath = os.path.join(dirname, "RFtest.pckl") # the model saved to disk
    # -------------------------------
    
    #--------------- This first part trains a model using the BR=2 dataset
    print("\n")
    print("(1) -------------- Extracting features from BR=2 case")
    produceTrainingFeatures(input_filename_r2, data_path=output_features_r2,
                            deltaT0 = 1, # Tmax-Tmin = 1 in this case
                            use_default_var_names = True, # variable names are default
                            write_derivatives = True, # cache derivatives
                            features_dump_path = feature_dump_r2, # save features to disk
                            model_type = "RF")   
    
    # Use the features extracted before to train the RF model
    print("\n")
    print("(2) -------------- Training model on extracted features")
    feature_list = [output_features_r2,]
    description = "Example RF model trained with supercoarse BR=2 LES"        
    trainRFModel(feature_list, description, savepath)    
    
    #------------- This second part applies the model trained previously to the BR=2 case
    print("\n")
    print("(3) -------------- Applying trained model on BR=2 case")   
    applyMLModel(derivatives_file, tecplot_filename_out_r2,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case
                 use_default_var_names = True, # variable names are default
                 calc_derivatives = False, # we are reading the derivatives file we just saved
                 write_derivatives = False,
                 features_load_path = feature_dump_r2, # load what we processed before
                 ip_file_path = ip_output_name_r2,
                 model_path = savepath, # path is the same we saved to previously
                 model_type = "RF") # here we choose the model type    
    
    # Remove all files I just wrote to disk
    os.remove(output_features_r2)
    os.remove(derivatives_file)
    os.remove(feature_dump_r2)
    os.remove(tecplot_filename_out_r2)
    os.remove(ip_output_name_r2)
    os.remove(savepath)
    

def test_full_cycle_tbnns():
    """
    This function runs the full boreas procedure on the sample coarse RANS simulation
    with the anisotropic model (TBNNS)
    """
    
    # Make the path relative to the location of the present script
    dirname = os.path.dirname(__file__)        
    
    # All relevant file names --    
    input_file = os.path.join(dirname, "JICF_BR1_supercoarse_derivatives.plt")    
    tecplot_filename_out_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_out.plt") #output tecplot file
    feature_dump_r1 = os.path.join(dirname, "features_r1_tbnns.pckl")
    csv_output_name_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_out.csv") #output csv file
    ip_output_name_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_out.ip") #output ip file
    test_model_path = os.path.join(dirname, "nn_test.pckl") # the custom model saved to disk
    test_model_path_modified = os.path.join(dirname, "nn_test_2.pckl") # the custom model saved to disk
    # -------------------------    
    
    # (1) ------------- This part applies the default model to the BR=1 case
    print("\n")
    print("Applying default anisotropic model on BR=1 case")   
    applyMLModel(input_file, tecplot_filename_out_r1,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case
                 use_default_var_names = True, # variable names are default
                 calc_derivatives = False, # we are reading the derivatives file we just saved                 
                 features_dump_path = feature_dump_r1, # dump for future use
                 csv_file_path = csv_output_name_r1,                
                 model_type = "TBNNS") # here we choose the model type

    # (2) ------------- This part applies the custom model to the BR=1 case
    
    # These lines are a little hack to make the path of checkpoints/ relative to the current location.
    # We load the actual file and save a modified version of it, with the correct path.
    description, model_list = joblib.load(test_model_path)        
    FLAGS, saved_path, feat_mean, feat_std = model_list
    saved_path = os.path.join(dirname, saved_path)
    joblib.dump([description, [FLAGS, saved_path, feat_mean, feat_std]], test_model_path_modified)
    
    print("\n")
    print("Applying a custom anisotropic model on BR=1 case")   
    applyMLModel(input_file, tecplot_filename_out_r1,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case
                 use_default_var_names = True, # variable names are default
                 calc_derivatives = False, # we are reading the derivatives file we just saved                 
                 features_load_path = feature_dump_r1, # load what we processed before
                 ip_file_path = ip_output_name_r1,
                 model_path = test_model_path_modified,
                 model_type = "TBNNS") # here we choose the model type 
    
    # Remove all files I just wrote to disk
    os.remove(tecplot_filename_out_r1)
    os.remove(feature_dump_r1)
    os.remove(csv_output_name_r1)
    os.remove(ip_output_name_r1)
    os.remove(test_model_path_modified)