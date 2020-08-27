#---------------------------------- constants.py ---------------------------------------#
"""
This file contains all global constants, for easy access
"""

# default turbulent Prandtl number, which is used whenever the ML model cannot
PRT_DEFAULT = 0.85

# number of features that the ML uses for regression
NUM_FEATURES_F1 = 15
NUM_FEATURES_F2 = 8

# number of form invariant basis used for the tensorial diffusivity (TBNN)
N_BASIS = 6

# Threshold for calculating should_use (based on temperature gradient magnitude)
THRESHOLD = 1e-2

# parameters for the feature cleaning algorithm
N_STD = 25
MAX_ITER = 100
TOL = 0.005
MAX_CLEAN = 0.05

# Cap for the value of Prt (symmetric, so the smallest value is 1.0/PRT_CAP)
PRT_CAP = 50

# Joblib parameters for saving files to disc. If compatibility with Python 2
# is required, set PROTOCOL=2.
COMPRESS = True
PROTOCOL = -1

# Default parameters for training random forests
N_TREES = 250 # number of trees to use in the random forest
MAX_DEPTH = 30 # max depth of tree
MIN_SPLIT = 0.0001 # only split if the node has more than MIN_SPLIT * N points
N_PROCESSORS = -1 # how many processors used when training the ML algorithm.