# Cloud Data Processing

A comprehensive framework for processing and analyzing cloud microphysics data, focusing on the conversion between Eulerian and Lagrangian representations with support for superdroplet generation.

## Overview

This repository contains a set of Python scripts for processing cloud microphysics simulation data. The framework handles both Eulerian grid-based data and Lagrangian particle-based data, with tools for generating training datasets for machine learning applications.

Key features include:
- Conversion between high-resolution and coarse-resolution Eulerian fields
- Generation of superdroplets from detailed Lagrangian droplet data
- Efficient neighbor searching with periodic boundary conditions
- Creation of training datasets that combine Eulerian and Lagrangian information

## Components

### Main Processing Scripts

- **`main.py`**: Orchestrates the overall data processing workflow
- **`process_eulerian.py`**: Handles Eulerian grid data filtering and processing
- **`process_lagrangian.py`**: Processes Lagrangian droplet data and generates superdroplets
- **`process_training.py`**: Creates training datasets by combining Eulerian and Lagrangian data

### Supporting Modules

- **`periodic_droplet_manager.py`**: Efficient management of droplet locations in periodic domains
- **`path.py`**: Configuration for input/output directories

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/cloud-microphysics-processing.git
cd cloud-microphysics-processing
```

### Dependencies

This project uses conda for environment management. An `environment.yml` file is provided to set up all necessary dependencies:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate sdrop
```

The environment includes the following packages:
- Python 3.9
- numpy (≥1.23.0)
- scikit-learn (≥1.2.0)
- joblib (≥1.2.0)
- matplotlib (≥3.6.0)
- netcdf4 (≥1.6.0)
- numba (≥0.58.0) - for performance optimization
- pandas
- tqdm (≥4.66.0) - for progress tracking
- h5py - for HDF5 file handling

## Data Structure

### Input Data

The workflow expects two main types of input data:

1. **Eulerian Data**: NetCDF files with grid-based fields
   - Location: `./Eulerian/Eulerian_XXXXXX.nc`
   - Variables: mixing_ratio, Temp, Uvel, Vvel, Wvel
   
2. **Lagrangian Data**: ASCII files with droplet information
   - Location: `./Lag_Data_ASCII/Ascii_xyz_rqt.XXXXXX.dat`
   - Columns: x, y, z, r, q, t

### Output Data

The processing generates several output files:

1. **Filtered Eulerian Data**: 
   - `eul.txt`: Contains filtered fields on a coarse grid

2. **Lagrangian Processing Results**:
   - `params.txt`: Parameters used for processing
   - `lag_grid.txt`: Grid-based droplet statistics
   - `lag_super.txt`: Generated superdroplet information
   - `superdroplet_neighbors.npy`: Detailed information about superdroplets and their neighbors

3. **Training Data**:
   - `training.txt`: Combined Eulerian and Lagrangian information for machine learning

## Usage

1. Set the correct paths in `path.py`:
   ```python
   lagdir = './Lag_Data_ASCII'  # Path to Lagrangian data
   eulerdir = './Eulerian'      # Path to Eulerian data
   outputdir = '.'              # Output directory
   ```

2. Run the main processing script:
   ```bash
   python main.py
   ```

## Configuration

The main processing parameters are set in `main.py`:

```python
# Domain configuration
lx, ly, lz = 51.2, 51.2, 51.2  # Domain sizes
nx, ny, nz = 32, 32, 32        # Grid resolutions

# Lagrangian processing parameters
nm = 300                       # Number of superdroplets
rminh, rmaxh = 0, 20           # Radius bounds
nh = 20                        # Number of histogram bins

# Training data parameters
nc = 3                         # Stencil size for training
```

## Examples

### Visualization

The `PeriodicDropletManager` class provides visualization capabilities:

```python
from periodic_droplet_manager import PeriodicDropletManager
import numpy as np

# Create manager with droplet locations and box size
locations = np.array(your_droplet_xyz_coordinates)
box_size = np.array([lx, ly, lz])
pdm = PeriodicDropletManager(locations, box_size)

# Visualize droplets and their neighbors
query_points = np.array([[25, 25, 25]])  # Example query point
pdm.visualize(query_points, k=10)  # Find 10 nearest neighbors
```

## Methodology

### Eulerian Processing

The Eulerian processing (`process_eulerian.py`) performs:
1. Reading of high-resolution data (512³)
2. Filtering to a coarser grid (default 32³)
3. Calculation of derived quantities (supersaturation, etc.)
4. Generation of comparison plots

### Lagrangian Processing

The Lagrangian processing (`process_lagrangian.py`) includes:
1. Reading droplet data and binning into grid cells
2. Creation of size distribution histograms
3. Generation of superdroplets with appropriate spatial distribution
4. Efficient nearest-neighbor searches with periodic boundaries
5. Calculation of superdroplet properties from their constituent real droplets

### Training Data Generation

The training data generation (`process_training.py`) combines:
1. Eulerian field data at superdroplet locations
2. Local Eulerian field stencils around each superdroplet
3. Droplet size distribution information

