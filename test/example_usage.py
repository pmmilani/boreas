#------------------------------ example_usage.py ---------------------------------------#
"""
Quick script showing how to import and use the Boreas package
"""

# Import statement: only four functions the user will need, which are defined in main.py
from boreas.main import printInfo, applyMLModel, produceTrainingFeatures, trainRFModel


def main():

    # Call this to print information about the package and make sure everything is in
    # order, including testing to see whether we can load the default model
    printInfo()    
    
    #--------------- This first part trains a model using the BR=2 dataset
    print("\n")
    print("(1) -------------- Extracting features from BR=2 case")
    # This line produces a set of features/labels and saves it to disk with name 
    # "JICF_BR2_supercoarse_trainingdata.pckl (default name)"
    input_file_name = "JICF_BR2_supercoarse.plt" # name of input tecplot binary file
    produceTrainingFeatures(input_file_name, 
                            data_path = "temp/JICF_BR2_supercoarse_trainingdata.pckl",
                            deltaT0 = 1, # Tmax-Tmin = 1 in this case
                            use_default_var_names = True, # variable names are default
                            calc_derivatives = True, # set to False if derivatives are already calculated in the file
                            write_derivatives = False, # no need to cache derivatives 
                            model_type = "RF")   
    
    # Use the features extracted before to train the RF model
    print("\n")
    print("(2) -------------- Training model on extracted features")
    feature_list = ["temp/JICF_BR2_supercoarse_trainingdata.pckl",]
    description = "Example RF model trained with supercoarse BR=2 LES"
    savepath = "temp/exampleRF.pckl"    
    trainRFModel(feature_list, description, savepath)
    
    
    #------------- This second part applies the model trained previously to the BR=1 case
    print("\n")
    print("(3) -------------- Applying trained model on BR=1 case")
    tecplot_file_name = "JICF_BR1_supercoarse.plt" # names of input tecplot binary file
    tecplot_file_output_name = "temp/JICF_BR1_supercoarse_out.plt" #output tecplot file
    fluent_interp_output_name = "temp/JICF_BR1_supercoarse_Prt.ip" #output Fluent ip file
    csv_output_name = None # output csv file    
    applyMLModel(tecplot_file_name, tecplot_file_output_name,
                 deltaT0 = 1, # Tmax-Tmin = 1 in this case
                 use_default_var_names = True, # variable names are default
                 calc_derivatives = True, # set to False if derivatives are already calculated in the file
                 write_derivatives = False,
                 ip_file_path=fluent_interp_output_name,
                 csv_file_path=csv_output_name,
                 model_path = savepath, # path is the same we saved to previously
                 model_type = "RF") # here we choose the model type
    
        
if __name__ == "__main__":
    main()