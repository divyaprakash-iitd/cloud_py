# main.py
import os
import time
from process_eulerian import process_eulerian_data
from process_lagrangian import process_lagrangian_data
from process_training import process_training_data

def main():
    """
    Main function to process Eulerian, Lagrangian, and Training data for multiple timesteps.
    Measures execution time and provides feedback on completion.
    """
    # Base directories (assuming these are correctly defined in a separate path module)
    try:
        import path
        eulerdir = path.eulerdir
        lagdir = path.lagdir
        outputdir = path.outputdir
    except ImportError:
        print("Error: 'path' module not found. Please ensure it exists and defines 'eulerdir' and 'lagdir'.")
        return


    # Fixed parameters
    iskip = 1
    lx, ly, lz = 51.2, 51.2, 51.2  # Domain sizes
    nx, ny, nz = 32, 32, 32       # Grid resolutions
    nm = 300                      # Number of superdroplets
    rminh, rmaxh = 0, 20          # Radius bounds
    nh = 20                       # Number of histogram bins
    nc = 3                        # Stencil size for training

    # List of timesteps to process
    tstep_list = [1000*i for i in range(4,10)]

    # Process each timestep
    for tstep in tstep_list:
        print(f"\nProcessing timestep {tstep:06d}...")
        start_time_total = time.time()

        # Define file paths and output directory
        fname_lag = f"{lagdir}/Ascii_xyz_rqt.{tstep:06d}.dat"
        fname_euler = f"{eulerdir}/Eulerian_{tstep:06d}.nc"
        outdirname = f"outdir_sk{iskip}_nm{nm}_nx{nx}"
        outdir = f"{outputdir}/{outdirname}/{tstep:06d}"

        # Create output directory if it doesn't exist
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create output directory '{outdir}': {e}")
            continue

        # Check if input files exist
        if not os.path.exists(fname_euler):
            print(f"Error: Eulerian input file '{fname_euler}' not found.")
            continue
        if not os.path.exists(fname_lag):
            print(f"Error: Lagrangian input file '{fname_lag}' not found.")
            continue

        # Eulerian processing
        print("Starting Eulerian processing...")
        start_time = time.time()
        process_eulerian_data(eulerdir, fname_euler, outdir, tstep, nx, ny, nz)
        euler_time = time.time() - start_time
        print(f"Eulerian processing completed in {euler_time:.2f} seconds.\n")

        # Lagrangian processing
        print("Starting Lagrangian processing...")
        start_time = time.time()
        process_lagrangian_data(lagdir, eulerdir, outdir, tstep, fname_euler, fname_lag,
                               iskip, lx, ly, lz, nx, ny, nz, nm, rminh, rmaxh, nh)
        lag_time = time.time() - start_time
        print(f"Lagrangian processing completed in {lag_time:.2f} seconds.\n")

        # Training data processing
        print("Starting Training data processing...")
        start_time = time.time()
        process_training_data(outdir, tstep, nc)
        train_time = time.time() - start_time
        print(f"Training data processing completed in {train_time:.2f} seconds.")

        # Total time for this timestep
        total_time = time.time() - start_time_total
        print(f"\nData generation completed for timestep {tstep:06d}.")
        print(f"Total processing time: {total_time:.2f} seconds "
              f"(Eulerian: {euler_time:.2f}s, Lagrangian: {lag_time:.2f}s, Training: {train_time:.2f}s)")

if __name__ == "__main__":
    main()
