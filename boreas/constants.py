#---------------------------------- constants.py ---------------------------------------#
"""
This file contains all global constants, for easy access
"""

# default turbulent Prandtl number, which is used whenever the ML model cannot
PR_T = 0.85

# number of features that the ML uses for regression
N_FEATURES = 15

# number of form invariant basis used for the tensorial diffusivity (TBNN)
N_BASIS = 6

# Threshold for calculating should_use (based on temperature gradient magnitude)
THRESHOLD = 1e-3

# parameters for the feature cleaning algorithm
N_STD = 50
MAX_ITER = 100
TOL = 0.005
MAX_CLEAN = 0.05

# Cap for the value of Prt (symmetric, so the smallest value is 1.0/PRT_CAP)
PRT_CAP = 100

# Downsample, applied when producing the features/labels to be used at training time
DOWNSAMPLE = 1.0

# Joblib parameters for saving files to disc. If compatibility with Python 2
# is required, set PROTOCOL=2.
COMPRESS = True
PROTOCOL = -1

# Default parameters for training random forests
N_TREES = 200 # number of trees to use in the random forest
MAX_DEPTH = 20 # max depth of tree
MIN_SPLIT = 0.001 # only split if the node has more than MIN_SPLIT * N points
N_PROCESSORS = -1 # how many processors used when training the ML algorithm.