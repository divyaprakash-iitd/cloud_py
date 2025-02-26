import numpy as np
import matplotlib.pyplot as plt
import config
import path

def read_in_chunks(filename, chunk_size=1000000):
    """Read large file in chunks"""
    with open(filename, 'r') as f:
        while True:
            lines = []
            # Read chunk_size lines at a time
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            
            if not lines:
                break
                
            # Convert chunk to numpy array
            chunk = np.array([list(map(float, line.split())) for line in lines])
            yield chunk

def plot_droplet_slice_efficient(tstep, z_center, z_thickness):
    # Define z-slice range
    z_min = z_center - z_thickness/2
    z_max = z_center + z_thickness/2
    
    # Initialize arrays for storing filtered data
    actual_x_filtered = []
    actual_y_filtered = []
    
    # Read and filter actual droplet data in chunks
    fname_lag = "/home/divyaprakash/Downloads/gendata_python_dec2/Ascii_xyz_rqt.003000.dat"
    chunk_count = 0
    total_points = 0
    filtered_points = 0
    
    print("Reading actual droplet data in chunks...")
    for chunk in read_in_chunks(fname_lag):
        chunk_count += 1
        total_points += len(chunk)
        
        # Filter points in this chunk that fall within our z-slice
        mask = (chunk[:, 2] >= z_min) & (chunk[:, 2] <= z_max)
        points_in_slice = np.sum(mask)
        filtered_points += points_in_slice
        
        if points_in_slice > 0:
            actual_x_filtered.extend(chunk[mask, 0])
            actual_y_filtered.extend(chunk[mask, 1])
        
        # Print progress
        if chunk_count % 10 == 0:
            print(f"Processed {total_points:,} points, found {filtered_points:,} in slice...")
    
    actual_x_filtered = np.array(actual_x_filtered)
    actual_y_filtered = np.array(actual_y_filtered)
    
    print("\nReading super droplet data...")
    # Read super droplet data (this should be much smaller)
    outdirname = f"outdir_sk{config.iskip}_nm{config.nm}_nx{config.nx}"
    outdir = f"{outdirname}/{tstep:06d}"
    super_data = np.loadtxt("lag_super.txt", skiprows=1)
    
    # Filter super droplets
    super_mask = (super_data[:, 2] >= z_min) & (super_data[:, 2] <= z_max)
    super_x = super_data[super_mask, 0]
    super_y = super_data[super_mask, 1]
    
    print("\nCreating plot...")
    # Create the plot
    plt.figure(figsize=(10, 10))
    
    plt.scatter(actual_x_filtered, actual_y_filtered, 
                   c='blue', alpha=0.5, s=1, label='Actual Droplets')
    # Plot super droplets
    plt.scatter(super_x, super_y, 
               c='red', alpha=0.8, s=50, marker='*', label='Super Droplets')
    
    # Set axis limits based on domain size
    plt.xlim(0, config.lx)
    plt.ylim(0, config.ly)
    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Droplet Distribution at Z = {z_center:.2f} ± {z_thickness/2:.2f}')
    plt.legend()
    
    # Show the plot
    plt.grid(True)
    
    # Save the plot
    plot_filename =f"droplet_slice_z{z_center:.1f}_thick{z_thickness:.1f}_{tstep}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    
    print(f"\nFinal Statistics:")
    print(f"Z-slice range: {z_min:.2f} to {z_max:.2f}")
    print(f"Total points processed: {total_points:,}")
    print(f"Points in slice: {len(actual_x_filtered):,}")
    print(f"Super droplets in slice: {len(super_x):,}")

if __name__ == "__main__":
	istart = config.tstep
	iend=config.tstep2
	for tstep in range(istart,iend,1000):
		z_center = config.lz / 3  
		z_thickness = 1.0  # Thickness of the slice
    
		plot_droplet_slice_efficient(tstep, z_center, z_thickness)
