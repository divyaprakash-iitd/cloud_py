# main.py
from process_eulerian import process_eulerian_data
from process_lagrangian import process_lagrangian_data
from process_training import process_training_data
import os

def main():
    # Eulerian parameters and paths
    eulerdir_euler = '/home/divyaprakash/Downloads/pythonscripts/python_code_11Feb25'
    fname_euler = '/home/divyaprakash/Downloads/gendata_python_dec2/Eulerian_003000.nc'
    #outdir_euler = '.'
    outdir_euler = "outdir/003000"
    tstep_euler = 3000
    nx_euler, ny_euler, nz_euler = 32, 32, 32

    # Lagrangian parameters and paths
    lagdir = "../Lag_Data_ASCII"
    eulerdir_lag = "../Eulerian"
    outdir_lag = "outdir/003000"
    fname_eul_lag = "/home/divyaprakash/Downloads/gendata_python_dec2/Eulerian_003000.nc"
    fname_lag = "/home/divyaprakash/Downloads/gendata_python_dec2/Ascii_xyz_rqt.003000.dat"
    tstep_lag = 3000
    iskip = 1
    lx = 51.21
    ly = 51.21
    lz = 51.21
    nx_lag = 32
    ny_lag = 32
    nz_lag = 32
    nm = 300
    rminh = 0
    rmaxh = 20
    nh = 20

    # Output directory for the time step
    #outdir = f"{outputdir_lag}/{tstep:06d}"
    os.makedirs(outdir_lag, exist_ok=True)

    # Training parameters
    outdir_training = "outdir/003000"  # Matches Lagrangian output as it uses its data
    tstep_training = 3000
    nc = 3  # Stencil size

    # Call Eulerian processing
    print("Starting Eulerian processing...")
    process_eulerian_data(eulerdir_euler, fname_euler, outdir_euler, 
                         tstep_euler, nx_euler, ny_euler, nz_euler)
    print("Eulerian processing completed.\n")

    # Call Lagrangian processing
    print("Starting Lagrangian processing...")
    process_lagrangian_data(lagdir, eulerdir_lag, outdir_lag, tstep_lag, 
                          fname_eul_lag, fname_lag, iskip, lx, ly, lz, 
                          nx_lag, ny_lag, nz_lag, nm, rminh, rmaxh, nh)
    print("Lagrangian processing completed.\n")

    # Call Training data processing
    print("Starting Training data processing...")
    process_training_data(outdir_training, tstep_training, nc)
    print("Training data processing completed.")

if __name__ == "__main__":
    main()
