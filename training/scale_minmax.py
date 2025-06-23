import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

time_counter = 0

def generate_features_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]

    lines_per_sdrop = 83
    nsdrops = len(lines) // lines_per_sdrop
    sdrops = np.array_split(lines, nsdrops)

    # Superdroplet data
    nsdropprops = 5
    sdropdata = np.zeros((nsdrops, nsdropprops))
    for i, drop in enumerate(sdrops):
        sdropdata[i] = np.array([float(item) for item in drop[0].split()])

    # Supersaturation and other data
    ssdata = np.zeros((nsdrops, 4))
    for i, drop in enumerate(sdrops):
        ssdata[i, 0] = [float(item) for item in drop[0].split()][-1]  # Actual/Effective Supersaturation
        ssdata[i, 1] = [float(item) for item in drop[1].split()][0]   # Filtered supersaturation
        ssdata[i, 2] = [float(item) for item in drop[0].split()][3]   # Superdroplet radius
        ssdata[i, 3] = time_counter                                  # Time counter

    # LES cell data
    lesdata = [sdrop[2:-1:3] for sdrop in sdrops]
    nlesprops = 135  # 3*3*3*5
    sdropdatales = np.zeros((nsdrops, nlesprops))
    for i, drop in enumerate(lesdata):
        sdropdatales[i] = np.array([drop[j].split() for j in range(len(drop))]).flatten().astype('float32')

    # Histogram data
    histdata = [sdrop[4::3] for sdrop in sdrops]
    nhistbins = 540  # 3*3*3*20
    histdatales = np.zeros((nsdrops, nhistbins))
    for i, drop in enumerate(histdata):
        histdatales[i] = np.array([drop[j].split() for j in range(len(drop))]).flatten().astype('float32')

    # Number of actual droplets
    nactdrops = [np.array(drop[3::3]).astype('int') for drop in sdrops]

    # Extract sdrop_1 and sdrop_2
    sdrop_1 = sdropdata[:, :-1]  # Remove the label (supersaturation value)
    sdrop_2 = np.zeros((nsdrops, nsdropprops))
    for i, drop in enumerate(sdrops):
        sdrop_2[i] = np.array([float(item) for item in drop[1].split()])

    return sdrop_1, sdrop_2, sdropdatales, nactdrops, histdatales, ssdata, sdropdata[:, -1]

# File paths
base_path = "/media/divyaprakash/data/cloud_py/post-processing/outdir_sk1_nm300_nx32"
file_path = [f"{base_path}/{i:06d}/training.txt" for i in range(1000, 2000, 1000)]
# file_path = [f"/home/divyaprakash/cloud_tf_cuda/nikita/10_files/training_{i:06d}.txt" for i in np.arange(1000, 10001, 1000)]

# Initialize lists to collect data
all_sdrop_1 = []
all_sdrop_2 = []
all_sdropdatales = []
all_nactdrops = []
all_histdatales = []
all_ssdata = []
all_labels = []

# Load all data
for fname in file_path:
    sdrop_1, sdrop_2, sdropdatales, nactdrops, histdatales, ssdata, labels = generate_features_labels(fname)
    all_sdrop_1.append(sdrop_1)
    all_sdrop_2.append(sdrop_2)
    all_sdropdatales.append(sdropdatales)
    all_nactdrops.append(nactdrops)
    all_histdatales.append(histdatales)
    all_ssdata.append(ssdata)
    all_labels.append(labels)
    time_counter += 1

# Concatenate all data
all_sdrop_1 = np.concatenate(all_sdrop_1, axis=0)
all_sdrop_2 = np.concatenate(all_sdrop_2, axis=0)
all_sdropdatales = np.concatenate(all_sdropdatales, axis=0)
all_nactdrops = np.concatenate(all_nactdrops, axis=0)
all_histdatales = np.concatenate(all_histdatales, axis=0)
all_ssdata = np.concatenate(all_ssdata, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Combine features for obtaing min-max scalers
original_shape = all_sdropdatales.shape 
all_sdrop_2_reshaped = all_sdrop_2.reshape(-1, 1, 5)
all_sdropdatales_reshaped = all_sdropdatales.reshape(-1, 27, 5)
combined = np.concatenate([all_sdrop_2_reshaped, all_sdropdatales_reshaped], axis=1)
min_max_scalers = []
for i in range(5):
    s = combined[:, :, i].flatten()
    min_val = s.min()
    max_val = s.max()
    min_max_scalers.append((min_val, max_val))
# min_max_scalers is now a list of tuples: [(min0, max0), (min1, max1), ...]
np.save('min_max_scalers.npy', min_max_scalers)

# Factors for sdrop_1
cellsize = 51.2 / 32
maxrad = 15
sdrop_1_factors = np.array([cellsize, cellsize, cellsize, maxrad])

# Scale sdrop_1
scaled_all_sdrop_1 = all_sdrop_1 / sdrop_1_factors

# Scale sdrop_2 and sdropdatales using min-max scaling
scaled_all_sdrop_2 = np.zeros_like(all_sdrop_2)
for i in range(5):
    min_val, max_val = min_max_scalers[i]
    scaled_all_sdrop_2[:, i] = (all_sdrop_2[:, i] - min_val) / (max_val - min_val)

scaled_all_sdropdatales_reshaped = np.zeros_like(all_sdropdatales_reshaped)
for i in range(all_sdropdatales_reshaped.shape[0]):
    for j in range(5):
        min_val, max_val = min_max_scalers[j]
        scaled_all_sdropdatales_reshaped[i, :, j] = (all_sdropdatales_reshaped[i, :, j] - min_val) / (max_val - min_val)

scaled_all_sdropdatales = scaled_all_sdropdatales_reshaped.reshape(original_shape)
# Concatenate features for scaling
# all_features = np.concatenate((all_sdrop_1, all_sdrop_2, all_sdropdatales), axis=1)
all_scaled_features = np.concatenate((scaled_all_sdrop_1, scaled_all_sdrop_2, scaled_all_sdropdatales), axis=1)

# # Initialize and fit a single scaler
# scaler = StandardScaler()
# scaler.fit(all_features)

# # Save the fitted scaler
# joblib.dump(scaler, 'scaler.pkl')

# # Transform the concatenated features
# all_features_scaled = scaler.transform(all_features)

# # Split the scaled features back into subsets (optional, for clarity)
# n_sdrop_1 = all_sdrop_1.shape[1]  # 4 features
# n_sdrop_2 = all_sdrop_2.shape[1]  # 5 features
# n_sdropdatales = all_sdropdatales.shape[1]  # 135 features

# sdrop_1_scaled = all_features_scaled[:, :n_sdrop_1]
# sdrop_2_scaled = all_features_scaled[:, n_sdrop_1:n_sdrop_1 + n_sdrop_2]
# sdropdatales_scaled = all_features_scaled[:, n_sdrop_1 + n_sdrop_2:]

# Combine features for final feature matrix
# X = np.concatenate((sdrop_1_scaled, sdrop_2_scaled, sdropdatales_scaled), axis=1)
X = all_scaled_features
y = all_labels
ss = all_ssdata

# Save the processed data
np.save('ssdata.npy', ss)
np.save('features_hist.npy', X)
np.save('labels.npy', y)
