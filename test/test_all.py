#--------------------------------- test_all.py -----------------------------------------#
"""
Test script containing several functions that will be invoked in unit testing. They are
built so the user can just call 'pytest'
"""

import os
import joblib
import numpy as np
from boreas.main import printInfo, applyMLModel, produceTrainingFeatures, trainRFModel, trainTBNNSModel
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
                            model_type = "RF", features_type="F1")   
    
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
                 model_type = "RF", features_type="F1") # here we choose the model type    
    
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
    output_features_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_trainingdata.pckl")    
    tecplot_filename_out_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_out.plt") #output tecplot file
    feature_dump_r1 = os.path.join(dirname, "features_r1_tbnns.pckl")
    csv_output_name_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_out.csv") #output csv file
    ip_output_name_r1 = os.path.join(dirname, "JICF_BR1_supercoarse_out.ip") #output ip file
    savepath = os.path.join(dirname, "nn_test.pckl") # the custom model saved to disk    
    path_to_saver = os.path.join(dirname, "checkpoints/nn_test")
    # -------------------------    
    
    # (1) ------------- This part applies the default model to the BR=1 case
    print("\n")
    print("(1) -------------- Applying default anisotropic model on BR=1 case")   
    applyMLModel(input_file, tecplot_filename_out_r1,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case
                 use_default_var_names = True, # variable names are default
                 calc_derivatives = False, # we are reading the derivatives file we just saved                 
                 features_dump_path = feature_dump_r1, # dump for future use
                 csv_file_path = csv_output_name_r1,                
                 model_type = "TBNNS", features_type="F2") # here we choose the model type

    # (2) ------------- This part applies the custom model to the BR=1 case
    print("\n")
    print("(2) -------------- Produce training features for TBNN-s")
    produceTrainingFeatures(input_file, data_path=output_features_r1,
                            deltaT0 = 1, # Tmax-Tmin = 1 in this case
                            use_default_var_names = True, # variable names are default
                            calc_derivatives = False,
                            write_derivatives = False, # cache derivatives
                            features_load_path = feature_dump_r1, # save features to disk
                            model_type = "TBNNS", features_type="F2")

    # (3) ------------- Use the features extracted before to train the TBNN-s model
    print("\n")
    print("(3) -------------- Training model on extracted features")
    feature_list = [output_features_r1,]
    description = "Example TBNN-s model trained with supercoarse BR=1 LES"        
    trainTBNNSModel(feature_list, feature_list, description, savepath, path_to_saver)    
        
    # (4) ------------- Use the features extracted before to train the TBNN-s model
    print("\n")
    print("(4) -------------- Applying a custom anisotropic model on BR=1 case")   
    applyMLModel(input_file, tecplot_filename_out_r1,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case
                 use_default_var_names = True, # variable names are default
                 calc_derivatives = False, # we are reading the derivatives file we just saved                 
                 features_load_path = feature_dump_r1, # load what we processed before
                 ip_file_path = ip_output_name_r1,
                 model_path = savepath,
                 model_type = "TBNNS", features_type="F2") # here we choose the model type 
    
    # Remove all files I just wrote to disk
    os.remove(output_features_r1)
    os.remove(tecplot_filename_out_r1)
    os.remove(feature_dump_r1)
    os.remove(csv_output_name_r1)
    os.remove(ip_output_name_r1)    
    os.remove(savepath)     