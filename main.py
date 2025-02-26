# main.py
from process_eulerian import process_eulerian_data
from process_lagrangian import process_lagrangian_data
from process_training import process_training_data
import os
import config
import path

def main():
    # Eulerian parameters and paths
    eulerdir = path.eulerdir

    # Lagrangian parameters and paths
    lagdir = path.lagdir
    iskip = 1
    lx = 51.2
    ly = 51.2
    lz = 51.2
    nx = 32
    ny = 32
    nz = 32
    nm = 300
    rminh = 0
    rmaxh = 20
    nh = 20
    nc = 3  # Stencil size

    tstep_list = [6000]

    for tstep in tstep_list:
        fname_lag = f"{path.lagdir}/Ascii_xyz_rqt.{tstep:06d}.dat"
        fname_euler = f"{path.eulerdir}/Eulerian_{tstep:06d}.nc"
        
        # Output directory for the time step
        outdir = f"outdir_sk{iskip}_nm{nm}_nx{nx}"
        os.makedirs(outdir, exist_ok=True)


        # Call Eulerian processing
        print("Starting Eulerian processing...")
        process_eulerian_data(eulerdir, fname_euler, outdir, 
                             tstep, nx, ny, nz)
        print("Eulerian processing completed.\n")

        # Call Lagrangian processing
        print("Starting Lagrangian processing...")
        process_lagrangian_data(lagdir, eulerdir, outdir, tstep, 
                              fname_euler, fname_lag, iskip, lx, ly, lz, 
                              nx, ny, nz, nm, rminh, rmaxh, nh)
        print("Lagrangian processing completed.\n")

        # Call Training data processing
        print("Starting Training data processing...")
        process_training_data(outdir, tstep, nc)
        print("Training data processing completed.")

if __name__ == "__main__":
    main()
