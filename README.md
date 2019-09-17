## Boreas v2.0 - a package for industrial deployment of machine-learned turbulent mixing models for film cooling (formerly RaFoFC)
Author: Pedro M. Milani (email: pmmilani@stanford.edu)

Last modified: 09/17/2019

Developed and tested in Python 3.7

### Installation
To install, run the following command from this directory: 

    pip install -e .

To uninstall, run the following from anywhere:

    pip uninstall boreas
    
The commands above will also install some dependencies 
(included in the file "requirements.txt") needed for this
package. Make sure pip is installed (it comes with most
Python distributions).

Note that the package requires pytecplot
to interface with Tecplot files. To install pytecplot, the user must
have Tecplot 360 2017 R1 or later installed. Follow the instructions
in https://www.tecplot.com/docs/pytecplot/install.html

### Usage
After installation, you can just use the following import 
statement in any Python program:

    from boreas.main import printInfo, applyMLModel, produceTrainingFeatures, trainRFModel
    
To test everything was installed properly, run the following
command in a Python script after the import statement above:

    printInfo()
    
The function applyMLModel is called to apply a given trained machine learning model to 
a Tecplot file containing the solution of the RANS equations with the k-epsilon model.
produceTrainingFeatures and trainRFModel are used, in this order, to train a machine
learning model given LES/DNS data. See the file test/example_usage.py for a complete
example of training and applying a model.

### Examples and Testing
The folder test contains a sample Tecplot file and a script that
shows how the functions can be called (example_usage.py). After 
installation, just run the following from within the test folder:

    python example_usage.py
    
The test folder also contains a script with unit tests, meant to run
with the pytest library (test_all.py). To run all the tests, just type 
the following from the root directory:

    pytest