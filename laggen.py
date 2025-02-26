import numpy as np
from scipy.interpolate import interp1d
from periodic_droplet_manager import PeriodicDropletManager
import os
from tqdm import tqdm

# Function to read parameters from a text file
def read_params(filename):
    params = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                params[key] = value
    return params

# Load parameters
params = read_params('params.txt')

# Extract parameters (convert types as needed)
lagdir = params['lagdir']
eulerdir = params['eulerdir']
outdir = params['outdir']
tstep = int(params['tstep'])
fname_eul = params['fname_eul']
fname_lag = params['fname_lag']
iskip = int(params['iskip'])
lx = float(params['lx'])
ly = float(params['ly'])
lz = float(params['lz'])
nx = int(params['nx'])
ny = int(params['ny'])
nz = int(params['nz'])
nm = int(params['nm'])
rminh = float(params['rminh'])
rmaxh = float(params['rmaxh'])
nh = int(params['nh'])

# Create output directory if it doesn't exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Load data
data = np.loadtxt(fname_lag)
x, y, z, r, q, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]
np_total = len(r)

# Wrap coordinates into the periodic box
x = np.mod(x, lx)
y = np.mod(y, ly)
z = np.mod(z, lz)

# Generate grid-based data
dx, dy, dz = lx / nx, ly / ny, lz / nz
np3d = np.zeros((nx, ny, nz))
hist3d = np.zeros((nx, ny, nz, nh))
drh = (rmaxh - rminh) / nh

print("Generating grid-based data...")
for ip in tqdm(range(np_total), desc="Processing droplets"):
    ix = min(int(np.floor(x[ip] / dx)), nx - 1)
    iy = min(int(np.floor(y[ip] / dy)), ny - 1)
    iz = min(int(np.floor(z[ip] / dz)), nz - 1)
    ih = min(int(np.floor(r[ip] / drh)), nh - 1)
    np3d[ix, iy, iz] += 1
    hist3d[ix, iy, iz, ih] += 1

# Write parameters to file
with open(os.path.join(outdir, 'params.txt'), 'w') as fid:
    fid.write('lx ly lz nx ny nz nh nm rmin rmax\n')
    fid.write(f'{lx:.10e} {ly:.10e} {lz:.10e} ')
    fid.write(f'{nx} {ny} {nz} {nh} ')
    fid.write(f'{nm} ')
    fid.write(f'{rminh:.10e} {rmaxh:.10e}\n')

# Write grid data to file
with open(os.path.join(outdir, 'lag_grid.txt'), 'w') as fid:
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                fid.write(f'{int(np3d[ix, iy, iz])} ')
                for ih in range(nh):
                    fid.write(f'{int(hist3d[ix, iy, iz, ih])} ')
                fid.write('\n')

# Histogram and CDF along r
nr = 1000
npr = np.zeros(nr)
rmin, rmax = np.min(r), np.max(r)
dr = (rmax - rmin) / nr
rvec = rmin + np.arange(nr + 1) * dr

print("Computing histogram along r...")
for ip in tqdm(range(np_total), desc="Building histogram"):
    ir = min(int(np.floor((r[ip] - rmin) / dr)), nr - 1)
    npr[ir] += 1

cdfr = np.zeros(nr + 1)
for ir in range(1, nr + 1):
    cdfr[ir] = cdfr[ir - 1] + npr[ir - 1]
cdfr /= np_total

cdfru, idx = np.unique(cdfr, return_index=True)
rvecu = rvec[idx]

# Number of CDF intervals
ncdfr = 10
sdrop = [None] * ncdfr
superdroplet_data = []  # List to store superdroplet info with neighbors

# Box size remains constant
box_size = np.array([lx, ly, lz])
k = nm  # Number of nearest neighbors

for icdfr in tqdm(range(ncdfr), desc="Processing CDF intervals"):
    print(f'icdfr = {icdfr + 1}')
    cdf0 = (1.0 / ncdfr) * icdfr
    cdf1 = (1.0 / ncdfr) * (icdfr + 1)

    r0 = np.interp(cdf0, cdfru, rvecu)
    r1 = np.interp(cdf1, cdfru, rvecu)

    mask = (r >= r0) & (r <= r1)
    npdr = np.sum(mask)
    xdr, ydr, zdr = x[mask], y[mask], z[mask]
    rdr, qdr = r[mask], q[mask]

    # Wrap subset coordinates into the periodic box
    xdr = np.mod(xdr, lx)
    ydr = np.mod(ydr, ly)
    zdr = np.mod(zdr, lz)

    # Reinitialize PeriodicDropletManager with the subset
    droplet_locations_subset = np.column_stack((xdr, ydr, zdr))
    pdm = PeriodicDropletManager(droplet_locations_subset, box_size)

    np3d.fill(0)
    for ipdr in range(npdr):
        ix = min(int(np.floor(xdr[ipdr] / dx)), nx - 1)
        iy = min(int(np.floor(ydr[ipdr] / dy)), ny - 1)
        iz = min(int(np.floor(zdr[ipdr] / dz)), nz - 1)
        np3d[ix, iy, iz] += 1

    ns = int(np.round(npdr / nm))
    nscnt = 0
    super = {}

    for iy in range(ny):
        for iz in range(nz):
            npencil = int(np.sum(np3d[:, iy, iz]))
            pdf = np3d[:, iy, iz] / npencil if npencil > 0 else np.zeros(nx)
            nsp = int(np.round(ns * npencil / npdr)) if npdr > 0 else 0
            nscnt += nsp

            if nsp > 0:
                ixu = np.nonzero(pdf)[0]
                pdfu = pdf[ixu]
                nxu = len(pdfu)
                cdfxu = np.zeros(nxu + 1)
                for ix in range(1, nxu + 1):
                    cdfxu[ix] = cdfxu[ix - 1] + pdfu[ix - 1]

                xs0 = np.random.rand(nsp)
                ys0 = np.random.rand(nsp)
                zs0 = np.random.rand(nsp)
                ixs = interp1d(cdfxu, np.arange(nxu + 1), kind='linear')(xs0)
                dixs = ixs - np.floor(ixs)
                xs = (ixu[np.floor(ixs).astype(int)] + dixs) * dx
                ys = (iy + ys0) * dy
                zs = (iz + zs0) * dz

                super[(iy, iz)] = {'nsp': nsp, 'xs': xs, 'ys': ys, 'zs': zs}
            else:
                super[(iy, iz)] = {'nsp': 0}

    xsdr = np.zeros(nscnt)
    ysdr = np.zeros(nscnt)
    zsdr = np.zeros(nscnt)
    iscnt = 0
    for iy in range(ny):
        for iz in range(nz):
            nsp = super[(iy, iz)]['nsp']
            if nsp > 0:
                xsdr[iscnt:iscnt + nsp] = super[(iy, iz)]['xs']
                ysdr[iscnt:iscnt + nsp] = super[(iy, iz)]['ys']
                zsdr[iscnt:iscnt + nsp] = super[(iy, iz)]['zs']
                iscnt += nsp

    # Wrap superdroplet coordinates into the periodic box
    xsdr = np.mod(xsdr, lx)
    ysdr = np.mod(ysdr, ly)
    zsdr = np.mod(zsdr, lz)
    xs3d = np.column_stack((xsdr, ysdr, zsdr))

    sdrop[icdfr] = {
        'ns': nscnt,
        'xs': np.zeros(nscnt),
        'ys': np.zeros(nscnt),
        'zs': np.zeros(nscnt),
        'xc': np.zeros(nscnt),
        'yc': np.zeros(nscnt),
        'zc': np.zeros(nscnt),
        'r': np.zeros(nscnt),
        'seff': np.zeros(nscnt)
    }

    superdroplet_info = pdm.generate_superdroplet_info(xs3d, k)

    for iscnt in tqdm(range(0, nscnt, iskip), desc=f"Superdroplets (icdfr={icdfr+1})"):
        info = superdroplet_info[iscnt]
        sdrop[icdfr]['xs'][iscnt] = info['query_point'][0]
        sdrop[icdfr]['ys'][iscnt] = info['query_point'][1]
        sdrop[icdfr]['zs'][iscnt] = info['query_point'][2]
        neighbor_indices = info['neighbor_indices']
        xnbr, ynbr, znbr = droplet_locations_subset[neighbor_indices, 0], droplet_locations_subset[neighbor_indices, 1], droplet_locations_subset[neighbor_indices, 2]
        rnbr, qnbr = rdr[neighbor_indices], qdr[neighbor_indices]
        sdrop[icdfr]['xc'][iscnt] = info['centroid'][0]
        sdrop[icdfr]['yc'][iscnt] = info['centroid'][1]
        sdrop[icdfr]['zc'][iscnt] = info['centroid'][2]
        rseff = np.mean(rnbr ** 3) ** (1/3)
        sdrop[icdfr]['r'][iscnt] = rseff
        seff = np.mean(rnbr * qnbr) / rseff
        sdrop[icdfr]['seff'][iscnt] = seff

        # Store superdroplet and neighbor info for binary file
        superdroplet_data.append({
            'icdfr': icdfr,
            'super_x': info['query_point'][0],
            'super_y': info['query_point'][1],
            'super_z': info['query_point'][2],
            'centroid_x': info['centroid'][0],
            'centroid_y': info['centroid'][1],
            'centroid_z': info['centroid'][2],
            'neighbor_x': xnbr,
            'neighbor_y': ynbr,
            'neighbor_z': znbr,
            'neighbor_r': rnbr,
            'neighbor_q': qnbr
        })

# Write superdroplet data to text file
with open(os.path.join(outdir, 'lag_super.txt'), 'w') as fid:
    fid.write('x y z r s\n')
    total_writes = sum(sdrop[icdfr]['ns'] // iskip for icdfr in range(ncdfr))
    with tqdm(total=total_writes, desc="Writing superdroplet data") as pbar:
        for icdfr in range(ncdfr):
            nscnt = sdrop[icdfr]['ns']
            for iscnt in range(0, nscnt, iskip):
                fid.write(f"{sdrop[icdfr]['xs'][iscnt]:.10e} ")
                fid.write(f"{sdrop[icdfr]['ys'][iscnt]:.10e} ")
                fid.write(f"{sdrop[icdfr]['zs'][iscnt]:.10e} ")
                fid.write(f"{sdrop[icdfr]['r'][iscnt]:.10e} ")
                fid.write(f"{sdrop[icdfr]['seff'][iscnt]:.10e}\n")
                pbar.update(1)

# Save superdroplet and neighbor info to a binary file
np.save(os.path.join(outdir, 'superdroplet_neighbors.npy'), superdroplet_data)

print("Conversion and processing completed.")
