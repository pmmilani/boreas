#----------------------------------- test.py -------------------------------------------#
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
    
    # Bare-bones, default invocation
    applyMLModel("cube.plt", "cube_OUT.plt", "cube.ip")    
    
    
    # With some more useful flags:
    """
    applyMLModel("cube.plt", "cube_OUT.plt", "cube.ip", 
                 use_default_names=True, U0=1, D=1, rho0=1, miu=0.0002, deltaT=1,
                 rans_data_load_path="data_rans_cube.pckl", 
                 rans_data_dump_path="data_rans_cube.pckl")
    """
    

    
    # If the derivatives have already been calculated:
    """
    applyMLModel("derivatives_cube.plt", "cube_OUT.plt", "cube.ip", 
                 use_default_names=True, U0=1, D=1, rho0=1, miu=0.0002, deltaT=1,
                 rans_data_load_path="data_rans_cube.pckl", 
                 rans_data_dump_path="data_rans_cube.pckl",
                 calc_derivatives=False)
    """
    
        
if __name__ == "__main__":
    main()