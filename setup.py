#------------------------- setup.py file for BOREAS package ----------------------------#
# https://stackoverflow.com/questions/1471994/what-is-setup-py/23998536

# Note: this installs the requirements needed to the package, but this needs setuptools
# to be installed to work. Make sure it is installed; you might potentially need to run
# "pip install setuptools" or an equivalent command.
from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='boreas',
   version='1.3.0',
   description='Boreas v1.3.0 - a package for automated deployment of machine-learned turbulent mixing models for film cooling',
   license='Apache',
   long_description=long_description,
   author='Pedro M. Milani',
   author_email='pmmilani@stanford.edu',
   packages=['boreas'],  # same as name
   install_requires=['pytecplot>=1.0.0', 'scikit-learn==0.21.3', 'tbnns==0.5.0', 'tqdm', 'pytest'], # dependencies
   include_package_data=True
)
