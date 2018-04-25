#------------------------------ example_usage.py ---------------------------------------#
"""
Quick script showing how to import and use the RaFoFC package
"""

# Import statement: only two function the user will need
# applyModel is defined in main.py. Look there for documentation and different arguments
from rafofc.main import printInfo, applyMLModel


def main():

    # Call this to print information about the package and make sure everything is in
    # order
    printInfo()
    
    tecplot_file_name = "derivatives_cube.plt" # names of input tecplot binary file
    tecplot_file_output_name = "cube_OUT.plt" # name of output tecplot binary file
    fluent_interp_output_name = "cube_OUT.ip" # name of output Fluent interpolation file
    csv_output_name = "cube_OUT.csv" # name of output csv file
    
    # Bare-bones, default invocation. Supply only the 2 required input arguments. Note
    # that this won't write interpolation or csv files.
    """
    applyMLModel(tecplot_file_name, tecplot_file_output_name)
    """
    
    
    # Bare-bones invocation that actually writes .ip and .csv files with results. Use
    # this to perform full cycle.
    
    applyMLModel(tecplot_file_name, tecplot_file_output_name,
                 ip_file_path=fluent_interp_output_name,
                 csv_file_path=csv_output_name)
            
    
    
    # With some more useful flags:
    """
    applyMLModel(tecplot_file_name, tecplot_file_output_name, 
                 U0=1, D=1, rho0=1, miu=0.0002, deltaT=1,
                 use_default_var_names=True,
                 rans_data_load_path="data_rans_cube.pckl", 
                 rans_data_dump_path="data_rans_cube.pckl",
                 ip_file_path=fluent_interp_output_name,
                 csv_file_path=csv_output_name)
    """
        
    
    # If the derivatives have already been calculated:
    """
    applyMLModel(tecplot_file_name, tecplot_file_output_name, 
                 U0=1, D=1, rho0=1, miu=0.0002, deltaT=1,
                 use_default_var_names=True, 
                 rans_data_load_path="data_rans_cube.pckl", 
                 rans_data_dump_path="data_rans_cube.pckl",
                 calc_derivatives=False,
                 ip_file_path=fluent_interp_output_name,
                 csv_file_path=csv_output_name)
    """
    
        
if __name__ == "__main__":
    main()