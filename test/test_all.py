#--------------------------------- test_all.py -----------------------------------------#
"""
Test script containing several functions that will be invoked in unit testing. They are
built so the user can just call 'pytest'
"""

import os
import numpy as np
from rafofc.main import printInfo, applyMLModel
from rafofc.models import MLModel
from rafofc.processed_data import ProcessedRANS
from sklearn.externals import joblib


def test_print_info():
    """
    This function calls the printInfo helper from main to make sure that it does not
    crash and returns 1 as expected.
    """
    
    assert printInfo() == 1
    
    
def test_loading_default_ml_model():
    """
    This function checks to see whether we can load the default model.
    """            
        
    # Test loading
    rafo_custom = MLModel() 
    rafo_custom.printDescription()
    
    # Test predicting
    n_test = 100
    N_FEATURES = 19
    x = -100.0 + 100*np.random.rand(n_test, N_FEATURES)
    y = rafo_custom.predict(x)
    assert y.shape == (n_test, )

    
def test_loading_custom_ml_model():
    """
    This function checks to see whether we can load the (embedded) custom model.
    """            
    
    # Make the path relative to the location of the present script
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "ML_2.pckl")
    
    # Test loading
    rafo_custom = MLModel(filepath=filename) 
    rafo_custom.printDescription()
    
    # Test predicting
    n_test = 100
    N_FEATURES = 19
    x = -100.0 + 100*np.random.rand(n_test, N_FEATURES)
    y = rafo_custom.predict(x)
    assert y.shape == (n_test, )
    
    
def test_full_cycle():
    """
    This function runs the full rafofc procedure on the sample coarse RANS simulation
    and makes sure the results are the same that we got previously.
    """
    
    # Make the path relative to the location of the present script
    dirname = os.path.dirname(__file__)
    
    # Relevant names of files    
    tecplot_file_name = os.path.join(dirname, "JICF_coarse_rans.plt")    
    tecplot_file_output_name = os.path.join(dirname, "JICF_coarse_rans_out.plt")    
    csv_output_name = os.path.join(dirname, "JICF_coarse_out.csv")        
    processed_rans_name = os.path.join(dirname, "data_rans_JICF_test.pckl")    
    gold_file_name = os.path.join(dirname, "gold.pckl") # file containing correct results
    
    # Threshold for comparing two real numbers 
    threshold = 1e-6
    
    # Calls the main function on the sample .plt file.
    # Sample file is a coarse jet in crossflow.    
    applyMLModel(tecplot_file_name, tecplot_file_output_name,
                 deltaT=1, 
                 use_default_var_names = True,
                 write_derivatives = False,
                 csv_file_path = csv_output_name,
                 processed_dump_path = processed_rans_name,
                 variables_to_write = ["ddx_U", "ddy_U", "ddz_U", "ddx_V", "ddy_V",
                                       "ddz_V", "ddx_W", "ddy_W", "ddz_W", "ddx_T", 
                                       "ddy_T", "ddz_T", "should_use", "Prt_ML"], 
                 outnames_to_write = ["ddx_U", "ddy_U", "ddz_U", "ddx_V", "ddy_V", 
                                      "ddz_V", "ddx_W", "ddy_W", "ddz_W", "ddx_T", 
                                      "ddy_T", "ddz_T", "should_use", "Prt_ML"])
    
    # Load from disk the pre-saved gold data
    csv_array_gold, processed_rans_gold = joblib.load(gold_file_name)
    
    # Load from disk the files we just dumped
    csv_array, processed_rans = loadNewlyWrittenData(csv_output_name, 
                                                     processed_rans_name)
                                                     
    # Remove files I wrote to disk
    os.remove(tecplot_file_output_name)
    os.remove(csv_output_name)
    os.remove(processed_rans_name)    
    
    # Perform all assertions to guarantee we got the same results as before
    comparePositions(csv_array[:, 0:3], csv_array_gold[:, 0:3], threshold)
    compareDerivatives(csv_array[:, 3:15], csv_array_gold[:, 3:15], threshold)
    compareProcessedRANS(processed_rans, processed_rans_gold, threshold)
    compareMLOutput(csv_array[:, 15:17], csv_array_gold[:, 15:17], threshold)       
                                                     
                                                     
def loadNewlyWrittenData(csv_output_name, processed_rans_name):
    """
    This helper loads data saved to disk as arrays so we can look at it
    
    Arguments:
    csv_output_name -- string, name of the .csv file with the Tecplot variables 
    processed_rans_name -- string, name of the .pckl file
    
    Returns:
    csv_array -- numpy array, with all the data read from the current run on the function
    processed_rans -- instance of ProcessedRANS class, with the post-processing results.
    """
    
    #--------Load processed rans
    processed_rans = joblib.load(processed_rans_name)
    
    #--------Load csv_array
    
    # First, count number of cells and variables
    with open(csv_output_name, "r") as csv_file:
        line = csv_file.readline()
        entries = line.split(", ")
        num_vars = len(entries) # number of variables
        N = sum(1 for line in csv_file) # number of cells        
    
    # Initialize numpy array
    csv_array = np.zeros((N, num_vars))
    
    # Open file again, and now record files
    with open(csv_output_name, "r") as csv_file:
        
        # Deal with header first
        line = csv_file.readline()
        entries = line.split(", ")
        variables_present = ["X", "Y", "Z", "ddx_U", "ddy_U", "ddz_U", "ddx_V", "ddy_V", 
                             "ddz_V", "ddx_W", "ddy_W", "ddz_W", "ddx_T", "ddy_T", 
                             "ddz_T", "should_use", "Prt_ML"]
        assert len(entries) == len(variables_present)
        for i in range(len(entries)):
            assert entries[i].strip() == variables_present[i].strip()
        
        # Now, read variable values
        for i, line in enumerate(csv_file):
            entries = line.split(", ")
            for j in range(num_vars):
                csv_array[i,j] = float(entries[j].strip())
    
    # Return both
    return csv_array, processed_rans
    

def comparePositions(X, X_gold, threshold):
    """
    This function compares positions of the cells to make sure they match
    
    Arguments:
    X -- numpy array containing positions of cells
    X_gold -- numpy array containing correct positions of cells
    threshold -- threshold for comparing real numbers
    """
    compareNumpyArrays(X, X_gold, threshold, "Comparing csv positions...")

    
def compareDerivatives(grad, grad_gold, threshold):
    """
    This function compares derivatives in all cells to make sure they match
    
    Arguments:
    grad -- numpy array containing gradients at the cells
    grad_gold -- numpy array containing correct gradients of the cells
    threshold -- threshold for comparing real numbers
    """
    compareNumpyArrays(grad, grad_gold, threshold, "Comparing csv derivatives...")

    
def compareMLOutput(out, out_gold, threshold):
    """
    This function compares positions of the cells to make sure they match
    
    Arguments:
    out -- numpy array containing ML outputs at the cells
    out_gold -- numpy array containing correct ML outputs of the cells
    threshold -- threshold for comparing real numbers
    """
    compareNumpyArrays(out, out_gold, threshold, "Comparing csv ML output...")

    
def compareProcessedRANS(processed_rans, processed_rans_gold, threshold):
    """
    This function compares the processed RANS class to make sure they match
    
    Arguments:
    processed_rans -- RANSDataset class containing processed data
    processed_rans_gold -- RANSDataset class containing correct processed data
    threshold -- threshold for comparing real numbers
    """
    
    # Compare x positions in processed data
    compareNumpyArrays(processed_rans.x, processed_rans_gold.x, threshold, 
                       "Comparing x position in processed data...")                       
    # Compare y positions in processed data
    compareNumpyArrays(processed_rans.y, processed_rans_gold.y, threshold, 
                       "Comparing y position in processed data...")
    # Compare z positions in processed data
    compareNumpyArrays(processed_rans.z, processed_rans_gold.z, threshold, 
                       "Comparing z position in processed data...")    
    # Compare T in processed data
    compareNumpyArrays(processed_rans.T, processed_rans_gold.T, threshold, 
                       "Comparing T in processed data...")                       
    # Compare gradU in processed data
    compareNumpyArrays(processed_rans.gradU, processed_rans_gold.gradU, threshold, 
                       "Comparing gradU in processed data...")
    # Compare gradT in processed data
    compareNumpyArrays(processed_rans.gradT, processed_rans_gold.gradT, threshold, 
                       "Comparing gradT in processed data...")
    # Compare TKE in processed data
    compareNumpyArrays(processed_rans.tke, processed_rans_gold.tke, threshold, 
                       "Comparing TKE in processed data...")
    # Compare epsilon in processed data
    compareNumpyArrays(processed_rans.epsilon, processed_rans_gold.epsilon, threshold, 
                       "Comparing epsilon in processed data...")
    # Compare rho in processed data
    compareNumpyArrays(processed_rans.rho, processed_rans_gold.rho, threshold, 
                       "Comparing rho in processed data...")
    # Compare nu_t in processed data
    compareNumpyArrays(processed_rans.mut, processed_rans_gold.mut, threshold, 
                       "Comparing nu_t in processed data...")
    # Compare nu in processed data
    compareNumpyArrays(processed_rans.mu, processed_rans_gold.mu, threshold, 
                       "Comparing nu in processed data...")
    # Compare wall distance in processed data
    compareNumpyArrays(processed_rans.d, processed_rans_gold.d, threshold, 
                       "Comparing wall distance in processed data...")
    # Compare should_use in processed data
    compareNumpyArrays(processed_rans.should_use, processed_rans_gold.should_use, 
                       threshold, "Comparing should_use in processed data...", is_boolean=True)
    # Compare x_features in processed data
    compareNumpyArrays(processed_rans.x_features, processed_rans_gold.x_features, 
                       threshold, "Comparing x_features in processed data...")    


def compareNumpyArrays(array, array_gold, threshold, description, is_boolean=False):
    """
    This function compares two numpy arrays to make sure they are the same
    
    Arguments:
    array -- numpy array that we are testing
    array_gold -- correct numpy array
    threshold -- threshold for comparing real numbers
    description -- string containing the description of current arrays
    is_boolean -- used when comparing two boolean arrays (such as should_use)
    """
    
    print(description)
    assert array.shape == array_gold.shape
    if is_boolean:
        diff = np.sqrt((array.astype(float) - array_gold.astype(float))**2)
    else:
        diff = np.sqrt((array - array_gold)**2)
    assert np.amax(diff) <= threshold
    
    
def writeGoldFiles(tecplot_file):
    """
    This function writes new gold.pckl files which serve to test the module
    
    Arguments:
    tecplot_file -- string with name of tecplot file used to test    
    """
    
    # Make the path relative to the location of the present script
    dirname = os.path.dirname(__file__)
    
    # Relevant names of files    
    tecplot_file_name = os.path.join(dirname, tecplot_file)  
    tecplot_file_output_name = os.path.join(dirname, "test_out.plt")    
    csv_output_name = os.path.join(dirname, "JICF_coarse_out.csv")        
    processed_rans_name = os.path.join(dirname, "data_rans_JICF_test.pckl")
    gold_file_name = os.path.join(dirname, "gold.pckl") # file containing correct results    
    
    # Calls the main function on the sample .plt file.
    # Sample file is a coarse jet in crossflow.    
    applyMLModel(tecplot_file_name, tecplot_file_output_name,
                 deltaT=1, 
                 use_default_var_names = True,
                 write_derivatives = False,
                 csv_file_path = csv_output_name,
                 processed_dump_path = processed_rans_name,
                 variables_to_write = ["ddx_U", "ddy_U", "ddz_U", "ddx_V", "ddy_V",
                                       "ddz_V", "ddx_W", "ddy_W", "ddz_W", "ddx_T", 
                                       "ddy_T", "ddz_T", "should_use", "Prt_ML"], 
                 outnames_to_write = ["ddx_U", "ddy_U", "ddz_U", "ddx_V", "ddy_V", 
                                      "ddz_V", "ddx_W", "ddy_W", "ddz_W", "ddx_T", 
                                      "ddy_T", "ddz_T", "should_use", "Prt_ML"])
    
    # Load from disk the files we just dumped
    csv_array, processed_rans = loadNewlyWrittenData(csv_output_name, 
                                                     processed_rans_name)
    
    # Writes list as the gold.pckl file
    gold_list = [csv_array, processed_rans]
    joblib.dump(gold_list, gold_file_name)
                                                     
    # Remove files I wrote to disk
    os.remove(tecplot_file_output_name)
    os.remove(csv_output_name)
    os.remove(processed_rans_name)    
    
