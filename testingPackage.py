## Script to test the package

from rafofc.main import printInfo, testTecplot

def main():
    printInfo()
    data = testTecplot('cube.plt')
        
if __name__ == "__main__":
    main()