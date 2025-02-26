# process_training.py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from numba import njit

def process_training_data(outdir, tstep=3000, nc=3):
    # Stencil size for contiguous Eulerian grid cells
    nch = (nc - 1) // 2  # Integer division for Python

    print("Reading parameters from params.txt...")
    with open(f'{outdir}/params.txt', 'r') as f:
        headers = f.readline().strip().split()
        values = f.readline().strip().split()

    params = {headers[i]: float(values[i]) for i in range(len(headers))}

    # Extract parameters
    lx = params['lx']
    ly = params['ly']
    lz = params['lz']
    nx = int(params['nx'])
    ny = int(params['ny'])
    nz = int(params['nz'])
    nh = int(params['nh'])
    rmin = params['rmin']
    rmax = params['rmax']
    rvec = (np.arange(nh) + 0.5) * (rmax - rmin) / nh

    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    xcoor = np.arange(nx) * dx + 0.5 * dx
    ycoor = np.arange(ny) * dy + 0.5 * dy
    zcoor = np.arange(nz) * dz + 0.5 * dz

    print("Loading grid data...")
    grid = np.loadtxt(f'{outdir}/lag_grid.txt')
    n3d = np.zeros((nx, ny, nz), dtype=int)
    hist3d = np.zeros((nx, ny, nz, nh), dtype=int)

    @njit
    def process_grid_data(nx, ny, nz, nh, grid, n3d, hist3d):
        cnt = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cnt += 1
                    n3d[ix, iy, iz] = grid[cnt-1, 0]
                    for ih in range(nh):
                        hist3d[ix, iy, iz, ih] = grid[cnt-1, ih+1]
        return n3d, hist3d

    print("Processing grid data...")
    total_cells = nx * ny * nz
    with tqdm(total=total_cells) as pbar:
        n3d, hist3d = process_grid_data(nx, ny, nz, nh, grid, n3d, hist3d)
        pbar.update(total_cells)

    print("Loading Eulerian data...")
    fdata = np.loadtxt(f'{outdir}/eul.txt')
    sf = np.zeros((nx, ny, nz))
    Tf = np.zeros((nx, ny, nz))
    uf = np.zeros((nx, ny, nz))
    vf = np.zeros((nx, ny, nz))
    wf = np.zeros((nx, ny, nz))

    @njit
    def process_eulerian_data(nx, ny, nz, fdata, sf, Tf, uf, vf, wf):
        cnt = 0
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    cnt += 1
                    sf[iz, iy, ix] = fdata[cnt-1, 0]
                    Tf[iz, iy, ix] = fdata[cnt-1, 1]
                    uf[iz, iy, ix] = fdata[cnt-1, 2]
                    vf[iz, iy, ix] = fdata[cnt-1, 3]
                    wf[iz, iy, ix] = fdata[cnt-1, 4]
        return sf, Tf, uf, vf, wf

    print("Processing Eulerian data...")
    with tqdm(total=nz*ny*nx) as pbar:
        sf, Tf, uf, vf, wf = process_eulerian_data(nx, ny, nz, fdata, sf, Tf, uf, vf, wf)
        pbar.update(nz*ny*nx)

    print("Reading superdroplet data...")
    with open(f'{outdir}/lag_super.txt', 'r') as f:
        first_line = f.readline().strip()
        try:
            [float(val) for val in first_line.split()]
            headers = ['x', 'y', 'z', 'r', 's']
            data_start = 0
        except ValueError:
            headers = first_line.split()
            data_start = 1

    super_data = np.loadtxt(f'{outdir}/lag_super.txt', skiprows=data_start)
    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    tab = DotDict()
    for i, col in enumerate(headers):
        if i < super_data.shape[1]:
            tab[col] = super_data[:, i]

    ns = len(tab['x'])

    sf_inter = np.zeros(ns)
    Tf_inter = np.zeros(ns)
    uf_inter = np.zeros(ns)
    vf_inter = np.zeros(ns)
    wf_inter = np.zeros(ns)

    print("Calculating filtered data at droplet locations...")
    for i in tqdm(range(ns)):
        ix0 = int(np.floor(tab['x'][i] / dx))
        iy0 = int(np.floor(tab['y'][i] / dy))
        iz0 = int(np.floor(tab['z'][i] / dz))

        sf_loc = np.zeros((3, 3, 3))
        Tf_loc = np.zeros((3, 3, 3))
        uf_loc = np.zeros((3, 3, 3))
        vf_loc = np.zeros((3, 3, 3))
        wf_loc = np.zeros((3, 3, 3))

        for dix in range(-1, 2):
            for diy in range(-1, 2):
                for diz in range(-1, 2):
                    ix = (ix0 + dix + nx) % nx
                    iy = (iy0 + diy + ny) % ny
                    iz = (iz0 + diz + nz) % nz
                    sf_loc[dix+1, diy+1, diz+1] = sf[ix, iy, iz]
                    Tf_loc[dix+1, diy+1, diz+1] = Tf[ix, iy, iz]
                    uf_loc[dix+1, diy+1, diz+1] = uf[ix, iy, iz]
                    vf_loc[dix+1, diy+1, diz+1] = vf[ix, iy, iz]
                    wf_loc[dix+1, diy+1, diz+1] = wf[ix, iy, iz]

        x3 = (ix0 + np.arange(-1, 2) + 0.5) * dx
        y3 = (iy0 + np.arange(-1, 2) + 0.5) * dy
        z3 = (iz0 + np.arange(-1, 2) + 0.5) * dz

        sf_interp = RegularGridInterpolator((x3, y3, z3), sf_loc, method='linear', bounds_error=False, fill_value=2)
        Tf_interp = RegularGridInterpolator((x3, y3, z3), Tf_loc, method='linear', bounds_error=False, fill_value=2)
        uf_interp = RegularGridInterpolator((x3, y3, z3), uf_loc, method='linear', bounds_error=False, fill_value=2)
        vf_interp = RegularGridInterpolator((x3, y3, z3), vf_loc, method='linear', bounds_error=False, fill_value=2)
        wf_interp = RegularGridInterpolator((x3, y3, z3), wf_loc, method='linear', bounds_error=False, fill_value=2)

        point = np.array([tab['x'][i], tab['y'][i], tab['z'][i]])
        sf_inter[i] = float(sf_interp(point)[0])
        Tf_inter[i] = float(Tf_interp(point)[0])
        uf_inter[i] = float(uf_interp(point)[0])
        vf_inter[i] = float(vf_interp(point)[0])
        wf_inter[i] = float(wf_interp(point)[0])

    print("Writing training data to file...")
    with open(f'{outdir}/training.txt', 'w') as fid:
        for i in tqdm(range(ns)):
            ix0 = int(np.floor(tab['x'][i] / dx))
            iy0 = int(np.floor(tab['y'][i] / dy))
            iz0 = int(np.floor(tab['z'][i] / dz))

            delx0 = tab['x'][i] - ix0 * dx
            dely0 = tab['y'][i] - iy0 * dy
            delz0 = tab['z'][i] - iz0 * dz

            fid.write(f'{delx0:.10e} {dely0:.10e} {delz0:.10e} {tab["r"][i]:.10e} {tab["s"][i]:.10e}\n')
            fid.write(f'{sf_inter[i]:.10e} {Tf_inter[i]:.10e} {uf_inter[i]:.10e} {vf_inter[i]:.10e} {wf_inter[i]:.10e}\n')

            for dix in range(-nch, nch + 1):
                for diy in range(-nch, nch + 1):
                    for diz in range(-nch, nch + 1):
                        ix = (ix0 + dix + nx) % nx
                        iy = (iy0 + diy + ny) % ny
                        iz = (iz0 + diz + nz) % nz
                        fid.write(f'{sf[iz, iy, ix]:.10e} {Tf[iz, iy, ix]:.10e} {uf[iz, iy, ix]:.10e} '
                                f'{vf[iz, iy, ix]:.10e} {wf[iz, iy, ix]:.10e}\n')
                        fid.write(f'{n3d[iz, iy, ix]:d}\n')
                        fid.write(' '.join(f'{hist3d[iz, iy, ix, ih]:d}' for ih in range(nh)) + '\n')

    print("Training data generation complete!")
