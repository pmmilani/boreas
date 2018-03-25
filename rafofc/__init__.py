#-------------------------- init.py file for RaFoFC package ----------------------------#

# Only import statement. main.py contains the entry point of the code
from . import main

"""
This simple function can be called by the user to check that everything was installed 
properly.
"""


"""
JL comment: I dont think any of the following needs to be in the init.py.  You can just call
"from rafofc.main import PrintInfo in your testingPackage.py"

JL comment: This file is named very liberally, given what it actually does.  It should really just be called
"checkInstallation.py"
"""

def PrintInfo():
    main.PrintInfo()



""" Test """   
def Test(path, zone=None):
    data = main.TestTecplot(path, zone)
    return data