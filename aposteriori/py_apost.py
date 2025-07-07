import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from numba import njit
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import joblib
from tensorflow import keras

def batched_interpolation(points_array, interp_func, batch_size=10000):
    """Process interpolation in batches to avoid memory issues and improve speed."""
    num_points = points_array.shape[0]
    results = []
    
    for i in range(0, num_points, batch_size):
        batch = points_array[i:i + batch_size]
        results.append(interp_func(batch).flatten())
    
    return np.concatenate(results)


os.makedirs('output', exist_ok=True)
files = sorted(glob.glob('../extracted_euler_filter_data_files/e*'))
outdir = '.'
nc = 3

#-----------------------------------------------------#
# 		Initialize parameters
#-----------------------------------------------------#
# Stencil size for contiguous Eulerian grid cells
nch = (nc - 1) // 2  # Integer division for Python

print("Reading parameters from params.txt...")
with open(f'{outdir}/params.txt', 'r') as f:
	headers = f.readline().strip().split()
	values = f.readline().strip().split()

params = {headers[i]: float(values[i]) for i in range(len(headers))}

# Extract parameters
lx, ly, lz = params['lx'], params['ly'], params['lz']
nx, ny, nz, nh = int(params['nx']), int(params['ny']), int(params['nz']), int(params['nh'])
rmin, rmax = params['rmin'], params['rmax']

dx, dy, dz = lx / nx, ly / ny, lz / nz

# Total number of superdroplets
ns = 67772

# Time step
dt = 0.04

Kr=5.07e-11;

#-----------------------------------------------------#
# Generate uniformly random distributed superdroplets
#-----------------------------------------------------#
# Create coordinate arrays
xcoor = (np.arange(0, nx) * lx/nx) + 0.5 * lx/nx
ycoor = xcoor.copy()
zcoor = xcoor.copy()

# Create random source points
xs = np.random.uniform(lx/3.0, 2*lx/3.0, ns)
ys = np.random.uniform(0, lx, ns)
zs = np.random.uniform(0, lx, ns)

# Initialize superdroplet's radius
rs = np.full_like(xs, 16e-6)
ss = np.full_like(xs, 0.0)

rs2=rs*rs;

## Create figure and 3D axis
#fig = plt.figure(figsize=(8, 6))
#ax = fig.add_subplot(111, projection='3d')
#
## Create scatter plot (equivalent to scatter3)
#scatter = ax.scatter(xs, ys, zs, 
#                    c='b',    # color
#                    marker='o', # marker style
#                    s=50,     # marker size
#                    alpha=0.8) # transparency
#plt.show()
    
#-----------------------------------------------------#
#    Start the time loop and read files one by one
#-----------------------------------------------------#
nfiles = len(files)
for it in range(nfiles):
    #-----------------------------------------------------#
    # 		Load Eulerian data fields
    #-----------------------------------------------------#
    print("Loading Eulerian data...")
    fdata = np.loadtxt(files[it])
    
    sf = np.reshape(fdata[:, 0], (nz, ny, nx))
    Tf = np.reshape(fdata[:, 1], (nz, ny, nx))
    uf = np.reshape(fdata[:, 2], (nz, ny, nx))
    vf = np.reshape(fdata[:, 3], (nz, ny, nx))
    wf = np.reshape(fdata[:, 4], (nz, ny, nx))
    
    #-----------------------------------------------------#
    #  Interpolate properties at superdroplets' locations
    #-----------------------------------------------------#

    # Construct interpolators
    x3 = np.linspace(0.5*lx/nx, lx - 0.5*lx/nx, nx)
    y3 = np.linspace(0.5*ly/ny, ly - 0.5*ly/ny, ny)
    z3 = np.linspace(0.5*lz/nz, lz - 0.5*lz/nz, nz)
    
    fill_value = None #2
    method = 'linear'
    # There is not much clarity if the order of indices in (x3,y3,z3) match with those of sf. 
    sf_interp = RegularGridInterpolator((z3, y3, x3), sf, method=method, bounds_error=False, fill_value=fill_value)
    Tf_interp = RegularGridInterpolator((z3, y3, x3), Tf, method=method, bounds_error=False, fill_value=fill_value)
    uf_interp = RegularGridInterpolator((z3, y3, x3), uf, method=method, bounds_error=False, fill_value=fill_value)
    vf_interp = RegularGridInterpolator((z3, y3, x3), vf, method=method, bounds_error=False, fill_value=fill_value)
    wf_interp = RegularGridInterpolator((z3, y3, x3), wf, method=method, bounds_error=False, fill_value=fill_value)

    
    # Create a dictionary to hold the superdroplet properties
    tab = {
        'x': xs,  # x coordinates
        'y': ys,  # y coordinates
        'z': zs,  # z coordinates
        'r': rs,  # radius values
        's': ss   # supersaturation values
    }
    
    # The f_inter means that it's the interpolated filtered values at the superdroplet locations
    sf_inter = np.zeros(ns)
    Tf_inter = np.zeros(ns)
    uf_inter = np.zeros(ns)
    vf_inter = np.zeros(ns)
    wf_inter = np.zeros(ns)

    print("Calculating filtered data at droplet locations...")
    points_array = np.zeros((ns,3))
    for i in range(ns):
        # Create a 3D point for the i-th superdroplet using its (x, y, z) coordinates
        # Result is a NumPy array in the form [x, y, z]
        # points_array[i,:] = np.array([tab['x'][i], tab['y'][i], tab['z'][i]])
        points_array[i,:] = np.array([tab['z'][i], tab['y'][i], tab['x'][i]])

    # Interpolate in batches
    batch_size = 5000  # Adjust based on your system's memory
    
    sf_inter = batched_interpolation(points_array, sf_interp, batch_size)
    Tf_inter = batched_interpolation(points_array, Tf_interp, batch_size)
    uf_inter = batched_interpolation(points_array, uf_interp, batch_size)
    vf_inter = batched_interpolation(points_array, vf_interp, batch_size)
    wf_inter = batched_interpolation(points_array, wf_interp, batch_size)    
    
    #-----------------------------------------------------#
    #            Construct feature vector
    #-----------------------------------------------------#
    # Initialize an empty list to store all rows
    features_array = []
    
    nfeatures = 144
    features_array = np.zeros((ns,nfeatures))    
    print("Creating feature vector array")
    for i in tqdm(range(ns)):
        ix0 = int(np.floor(tab['x'][i] / dx))
        iy0 = int(np.floor(tab['y'][i] / dy))
        iz0 = int(np.floor(tab['z'][i] / dz))
    
        delx0 = tab['x'][i] - ix0 * dx
        dely0 = tab['y'][i] - iy0 * dy
        delz0 = tab['z'][i] - iz0 * dz
    
        # Create a list for this row
        ifeat = 0
        features_array[i,ifeat] = delx0
        ifeat = ifeat + 1 
        features_array[i,ifeat] = dely0 
        ifeat = ifeat + 1 
        features_array[i,ifeat] = delz0 
        ifeat = ifeat + 1 
        features_array[i,ifeat] = tab["r"][i]/1e-6 

        ifeat = ifeat + 1 
        features_array[i,ifeat] = sf_inter[i] 
        ifeat = ifeat + 1 
        features_array[i,ifeat] = Tf_inter[i]  
        ifeat = ifeat + 1 
        features_array[i,ifeat] = uf_inter[i]  
        ifeat = ifeat + 1 
        features_array[i,ifeat] = vf_inter[i]  
        ifeat = ifeat + 1 
        features_array[i,ifeat] = wf_inter[i]  
        
        # Add all neighborhood values to the row
        for dix in range(-nch, nch + 1):
            for diy in range(-nch, nch + 1):
                for diz in range(-nch, nch + 1):
                    ix = (ix0 + dix + nx) % nx
                    iy = (iy0 + diy + ny) % ny
                    iz = (iz0 + diz + nz) % nz

                    ifeat = ifeat + 1 
                    features_array[i,ifeat] = sf[iz, iy, ix] 
                    ifeat = ifeat + 1 
                    features_array[i,ifeat] = Tf[iz, iy, ix]  
                    ifeat = ifeat + 1 
                    features_array[i,ifeat] = uf[iz, iy, ix]  
                    ifeat = ifeat + 1 
                    features_array[i,ifeat] = vf[iz, iy, ix]  
                    ifeat = ifeat + 1 
        
                    features_array[i,ifeat] = wf[iz, iy, ix]  
    print(f"Shape of feature array: {features_array.shape}")

    #-----------------------------------------------------#
    #     Predict supersaturation and calculate radius
    #-----------------------------------------------------#
    # Scale the feature vectors according to the saved scaler 
    # Load the scaler
    scaler = joblib.load('../training/scaler.pkl')
    
    # Transform the concatenated features
    features_scaled = scaler.transform(features_array)
    print(f"Shape of scaled array: {features_scaled.shape}")
    # Feed the feature array to the ml model to get predictions for each superdroplet.
    # 1. Load the trained model
    model = keras.models.load_model('../training/output/model_ml_1.keras')  # Load your Keras model
    
    # 2. Ensure X_new_scaled is correct
    print("Shape of new data:", features_scaled.shape)  # Should be (N_samples, N_features)
    
    # 3. Make predictions
    #predictions = model.predict(features_scaled)
    print("Making predictions...")
    predictions = model.predict(features_scaled, batch_size=1024)  # Adjust batch size
    print(f"Predicitons' shape: {predictions.shape}")

    # Placeholder prediction
    ss = sf_inter.copy()

    seff = np.squeeze(predictions)    
    rs2 = np.maximum(rs2 + 2 * dt * Kr * seff, 0.0)
    rs = np.sqrt(rs2)

    #-----------------------------------------------------#
    # 		Advect the superdroplets
    #-----------------------------------------------------#
    # Update the locations
    xs += uf_inter * dt
    ys += vf_inter * dt
    zs += wf_inter * dt
    
    # Periodic boundary conditions (using modulo operator)
    xs = np.mod(xs, lx)
    ys = np.mod(ys, ly)
    zs = np.mod(zs, lz)

    #-----------------------------------------------------#
    # Make histogram and contour plots
    #-----------------------------------------------------#
    if it % 12 == 0:
        # Compute 2D average along y-axis
        s2d = np.mean(sf, axis=1)  # Shape will be (32, 32)
        
        tme = it * dt
        
        # Create figure with two subplots
        plt.figure(figsize=(12, 5))
        
        # First subplot: Histogram
        plt.subplot(1, 3, 1)
        edges= np.arange(0,18)
        h = plt.hist(rs/1e-6, bins=edges, density=True)
        #h = plt.hist(rs/1e-6, density=True)
        plt.xlim([0,17])
        plt.ylabel('PDF')
        plt.xlabel('Radius (µm)')
        
        # Second subplot: Contour plot with droplets
        plt.subplot(1, 3, 2)
        cont = plt.contourf(xcoor, zcoor, s2d)
        plt.scatter(xs, zs, color='k', s=5)  # Plot droplets in the slice
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'xz-Time = {tme:.2f}')
        
        plt.subplot(1, 3, 3)
        cont = plt.contourf(xcoor, zcoor, s2d)
        plt.scatter(xs, ys, color='k', s=5)  # Plot droplets in the slice
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'xy-Time = {tme:.2f}')
        
        # Super title
        plt.suptitle('From a posteriori simulation')
        outfname = f'output/LES_dt_0p04_t{tme:.2f}.jpg'
        plt.savefig(outfname, dpi=300, format='jpeg', bbox_inches='tight')
        plt.close()
        #plt.pause(0.01)  # Keep this after savefig to ensure plot is updated
   
