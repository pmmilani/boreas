#------------------------------- produce_gold.py ---------------------------------------#
"""
This script can be called directly to write new gold standar files with which to 
compare results when pytest is called
"""

import test_all


def yes_or_no(question):
    """ 
    Function to get a yes/no answer from the user
    
    Credit: https://gist.github.com/garrettdreyfus/8153571
    """
    
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... please enter ")
        

def main():
    """
    This function calls the printInfo helper from main to make sure that it does not
    crash and returns 1 as expected.
    """
    
    tecplot_file = "JICF_coarse_rans.plt"
    
    if yes_or_no("This writes a new gold.pckl file. Are you sure you want to proceed?"):    
        test_all.writeGoldFiles(tecplot_file)
    else:
        return

    
if __name__ == "__main__":
    main()   
    
