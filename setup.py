#------------------------- setup.py file for RaFoFC package ----------------------------#
# https://stackoverflow.com/questions/1471994/what-is-setup-py/23998536

# Note: this installs the requirements needed to the package, but this needs setuptools
# to be installed to work. Make sure it is installed; you might potentially need to run
# "pip install setuptools" or an equivalent command.
from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='rafofc',
   version='1.2.1',
   description='RaFoFC v1.2.1 - Random Forests for Film Cooling Package',
   license='Apache',
   long_description=long_description,
   author='Pedro M. Milani',
   author_email='pmmilani@stanford.edu',
   packages=['rafofc'],  # same as name
   install_requires=['pytecplot>=0.10.0', 'scikit-learn=0.21.2', 'tqdm', 'pytest'], # dependencies
   include_package_data=True      
)
