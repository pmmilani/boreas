#----------------------------------- test.py -------------------------------------------#
"""
Test script containing several functions that will be invoked in unit testing (e.g. by
using pytest)
"""

# Import statement: only two function the user will need
# applyModel is defined in main.py. Look there for documentation and different arguments
from rafofc.main import printInfo, applyMLModel

def test_1():
    assert printInfo() == 1
    
