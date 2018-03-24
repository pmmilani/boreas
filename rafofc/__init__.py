#-------------------------- init.py file for RaFoFC package ----------------------------#

# Only import statement. main.py contains the entry point of the code
from . import main

"""
This simple function can be called by the user to check that everything was installed 
properly.
"""
def PrintInfo():
    main.PrintInfo()



""" Test """   
def Test(path, zone=None):
    data = main.TestTecplot(path, zone)
    return data