# main.py
from process_eulerian import process_eulerian_data
from process_lagrangian import process_lagrangian_data

def main():
    # Eulerian parameters and paths
    eulerdir_euler = '/home/divyaprakash/Downloads/pythonscripts/python_code_11Feb25'
    fname_euler = '/home/divyaprakash/Downloads/gendata_python_dec2/Eulerian_003000.nc'
    outdir_euler = '.'
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
    print("Lagrangian processing completed.")

if __name__ == "__main__":
    main()
