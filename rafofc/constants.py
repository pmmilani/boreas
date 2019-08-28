#--------------------------------- processed_data.py -----------------------------------#
"""
This file contains all global constants, for easy access
"""

# default turbulent Prandtl number, which is used whenever the ML model cannot
PR_T = 0.7

# number of features that the ML uses for regression
N_FEATURES = 19

# parameters for the feature cleaning algorithm
N_STD = 30
MAX_ITER = 100
TOL = 0.005
MAX_CLEAN = 0.05