#------------------------------ example_usage.py ---------------------------------------#
"""
Quick script showing how to import and use the Boreas package on two example datasets
"""

# Import statement: only four functions the user needs, which are defined in main.py
from boreas.main import printInfo, applyMLModel, produceTrainingFeatures, trainRFModel


def main():

    # Call this to print information about the package and make sure everything is in
    # order, including testing to see whether we can load the default model
    printInfo()    
    
    #--------------- This part trains a model using the BR=2 dataset
    print("\n")
    print("-------------- Extracting features from BR=2 case")
    # This line produces a set of features/labels and saves it to disk with name 
    # "JICF_BR2_supercoarse_trainingdata.pckl (default name)"
    input_file_name = "JICF_BR2_supercoarse.plt" # name of input tecplot binary file
    data_path = "tmp/JICF_BR2_supercoarse_trainingdata.pckl" # location where training data is saved
    produceTrainingFeatures(input_file_name,
                            data_path = data_path,
                            deltaT0 = 1, # Tmax-Tmin = 1 in this case. If this is None, you are prompted to enter it manually
                            use_default_var_names = True, # variable names are default. If this is False, you are prompted to enter them manually                            
                            calc_derivatives = True, # set to True because we need to calculate derivatives here
                            write_derivatives = False, # whether to save derivatives for faster calculation next time 
                            model_type = "RF") # type of model we produce features for  
    
    # Use the features extracted before to train the RF model
    print("\n")
    print("-------------- Training model on extracted features")
    feature_list = [data_path,] # list of all set of features we use for training
    description = "Example RF model trained with supercoarse BR=2 LES" # description attached to the trained model
    savepath = "tmp/exampleRF_br2.pckl" # location where the trained model is saved   
    trainRFModel(feature_list, description, savepath)
    
    
    #------------- This part applies the model trained previously to the BR=1 case
    print("\n")
    print("-------------- Applying trained RF model on BR=1 case")
    tecplot_file_name = "JICF_BR1_supercoarse_derivatives.plt" # names of input tecplot binary file
    tecplot_file_output_name = "tmp/JICF_BR1_supercoarse_out.plt" #output tecplot file
    fluent_interp_output_name = "tmp/JICF_BR1_supercoarse_Prt.ip" #output Fluent .ip file
    csv_output_name = "tmp/JICF_BR1_supercoarse_Prt.csv" #output csv file   
    applyMLModel(tecplot_file_name, tecplot_file_output_name,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case. If this is None, you are prompted to enter it manually
                 use_default_var_names = True, # variable names are default. If this is False, you are prompted to enter them manually
                 use_default_derivative_names = True, # derivative names are default. If this is False, you are prompted to enter them manually
                 calc_derivatives = False, # False, because the input file already has derivatives calculated                 
                 ip_file_path = fluent_interp_output_name,
                 csv_file_path = csv_output_name,
                 model_path = savepath, # path is the same we saved to previously
                 model_type = "RF") # here we choose the model type. Either "RF" or "TBNNS"    
    
    
    #------------- This part applies the model trained previously to the BR=1 case
    print("\n")
    print("-------------- Applying standard RF model on BR=1 case")
    tecplot_file_name = "JICF_BR1_supercoarse_derivatives.plt" # names of input tecplot binary file
    tecplot_file_output_name = "tmp/JICF_BR1_supercoarse_out.plt" #output tecplot file
    fluent_interp_output_name = "tmp/JICF_BR1_supercoarse_Prt.ip" #output Fluent ip file
    csv_output_name = "tmp/JICF_BR1_supercoarse_Prt.csv" #output csv file     
    applyMLModel(tecplot_file_name, tecplot_file_output_name,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case. If this is None, you are prompted to enter it manually
                 use_default_var_names = True, # variable names are default. If this is False, you are prompted to enter them manually
                 use_default_derivative_names = True, # derivative names are default
                 calc_derivatives = False, # set to False if derivatives are already calculated in the file, so we save time
                 ip_file_path = fluent_interp_output_name,
                 csv_file_path = csv_output_name,
                 model_path = None, # None means that the default model is used
                 model_type = "RF") # here we choose the model type                
                 
    
    #------------- This part applies the model trained previously to the BR=1 case
    print("\n")
    print("-------------- Applying standard anisotropic model on BR=1 case")
    tecplot_file_name = "JICF_BR1_supercoarse_derivatives.plt" # names of input tecplot binary file
    tecplot_file_output_name = "tmp/JICF_BR1_supercoarse_out_anisotropic.plt" #output tecplot file
    fluent_interp_output_name = "tmp/JICF_BR1_supercoarse_Aij.ip" #output Fluent ip file
    csv_output_name = "tmp/JICF_BR1_supercoarse_Aij.csv" # output csv file    
    applyMLModel(tecplot_file_name, tecplot_file_output_name,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case. If this is None, you are prompted to enter it manually
                 use_default_var_names = True, # variable names are default. If this is False, you are prompted to enter them manually
                 use_default_derivative_names = True, # derivative names are default, so no need to enter manually
                 calc_derivatives = False, # set to False if derivatives are already calculated in the file, so we save time
                 ip_file_path = fluent_interp_output_name,
                 csv_file_path = csv_output_name,
                 model_path = None, # None means that the default model is used
                 model_type = "TBNNS") # here we choose the model type. Use "TBNNS" for anisotropic model
    
    
if __name__ == "__main__":
    main()