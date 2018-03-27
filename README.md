## RaFoFC v1.0.0 - Random Forest for Film Cooling Package
Author: Pedro Milani (email: pmmilani@stanford.edu)
Developed and tested in Python 3.6

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

    from rafofc.main import printInfo, applyMLModel
    
To test everything was installed properly, run the following
command in a Python script after the import statement above:

    printInfo()
    
The function applyMLModel contains all the functionality that a
user should need; please review the comments in the main.py file
to understand all the possible arguments. The simplest usage looks
as follows, assuming there is a Tecplot file called "jicf_rans.plt"
from a k-epsilon simulation:

    applyMLModel("jicf_rans.plt", "jicf_rans_out.plt", "jicf_rans.ip")
    
The folder test contains a sample Tecplot file and a script that
tests functionality. After installation, just run the following from
within the test folder:

    python test.py
