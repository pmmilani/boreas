#---------------------------------- constants.py ---------------------------------------#
"""
This file contains all global constants, for easy access
"""

# default turbulent Prandtl number, which is used whenever the ML model cannot
PR_T = 0.7

# number of features that the ML uses for regression
N_FEATURES = 19

# Threshold for calculating should_use (based on temperature gradient magnitude)
THRESHOLD = 1e-3

# parameters for the feature cleaning algorithm
N_STD = 30
MAX_ITER = 100
TOL = 0.005
MAX_CLEAN = 0.05

# Cap for the value of Prt (symmetric, so the smallest value is 1.0/PRT_CAP)
PRT_CAP = 100

# Downsample
DOWNSAMPLE = 1.0

# Joblib parameters for saving files to disc. If compatibility with Python 2
# is required, set PROTOCOL=2.
COMPRESS = True
PROTOCOL = -1