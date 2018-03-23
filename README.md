## RaFoFC v1.0 - Random Forest for Film Cooling Package
Author: Pedro Milani (email: pmmilani@stanford.edu)

### Installation
To install, run the following command from this directory: 

    python setup.py install [--user]
    
To test the program while it is being developed, run:

    python setup.py develop [--user] [--uninstall]
    
You will need the flag --user in case you don't have 
administrator privileges in the machine to which you are 
installing the package. The flag --uninstall can be added 
to uninstall the package. The commands above will also install
some dependencies (included in the file "requirements.txt")
needed for this package. 

### Usage
After installation, you can just use the following import 
statement in any Python program:

    import rafofc
    
To test everything was installed properly, run the following
command in a Python script after the import statement above:

    rafofc.PrintInfo()
