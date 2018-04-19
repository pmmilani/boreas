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
    
    tecplot_file_name = "cube.plt" # names of input tecplot binary file   
    tecplot_file_output_name = "cube_OUT.plt" # name of output tecplot binary file    
    fluent_interp_output_name = "cube.ip" # name of output Fluent interpolation file
    
    # Bare-bones, default invocation. Supply only the 3 required input arguments
    applyMLModel(tecplot_file_name, tecplot_file_output_name, fluent_interp_output_name)    
    
    
    # With some more useful flags:
    """
    applyMLModel(tecplot_file_name, tecplot_file_output_name, fluent_interp_output_name, 
                 use_default_names=True, U0=1, D=1, rho0=1, miu=0.0002, deltaT=1,
                 rans_data_load_path="data_rans_cube.pckl", 
                 rans_data_dump_path="data_rans_cube.pckl")
    """
    

    
    # If the derivatives have already been calculated:
    """
    applyMLModel(tecplot_file_name, tecplot_file_output_name, fluent_interp_output_name, 
                 use_default_names=True, U0=1, D=1, rho0=1, miu=0.0002, deltaT=1,
                 rans_data_load_path="data_rans_cube.pckl", 
                 rans_data_dump_path="data_rans_cube.pckl",
                 calc_derivatives=False)
    """
    
        
if __name__ == "__main__":
    main()