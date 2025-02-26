# main.py
from gen_eulerian import gen_eulerian_data

def main():
    # Define file paths and constants
    eulerdir = '/home/divyaprakash/Downloads/pythonscripts/python_code_11Feb25'
    fname = '/home/divyaprakash/Downloads/gendata_python_dec2/Eulerian_003000.nc'
    outdir = '.'
    tstep = 3000
    nx, ny, nz = 32, 32, 32

    # Call the processing function
    gen_eulerian_data(eulerdir, fname, outdir, tstep, nx, ny, nz)

if __name__ == "__main__":
    main()
