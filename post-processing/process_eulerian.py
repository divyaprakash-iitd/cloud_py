# process_eulerian.py
import numpy as np
from numba import njit
import netCDF4 as nc
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_eulerian_data(eulerdir, fname, outdir, tstep=3000, nx=32, ny=32, nz=32):
    # Constants
    rho0 = 1.06
    rhovs = 0.0039

    # Load mean data
    meandata = np.loadtxt(f'{eulerdir}/v_max_min.dat')
    Tavg = meandata[tstep-1, 6]  # Adjusting index since Python is 0-based
    rvavg = meandata[tstep-1, 5]

    # Variable names
    vnames = ['mixing_ratio', 'Temp', 'Uvel', 'Vvel', 'Wvel']

    @njit
    def filter_field(vdns, nx, ny, nz, di):
        vfilt = np.zeros((nx, ny, nz))
        for iz in range(512):
            for iy in range(512):
                for ix in range(512):
                    ixf = int(ix / di)
                    iyf = int(iy / di)
                    izf = int(iz / di)
                    vfilt[ixf, iyf, izf] += vdns[iz, iy, ix]
        return vfilt / (di ** 3)

    # Read and process data
    ds = nc.Dataset(fname)
    di = 512 / nx
    vfilt_dict = {}

    print("Processing variables...")
    for vname in tqdm(vnames):
        vdns = ds.variables[vname][:]
        masked_count = 0
        if np.ma.is_masked(vdns):
            masked_count = np.ma.count_masked(vdns)
            vdns = vdns.filled(fill_value=0.0)
            print(f"{vname}: Filled {masked_count} masked values with 0.0")
        else:
            vdns = np.asarray(vdns)
        vfilt_dict[vname] = filter_field(vdns, nx, ny, nz, di)

    # Calculate derived quantities
    sf = (vfilt_dict['mixing_ratio'] + rvavg) * rho0 / rhovs - 1
    Tf = vfilt_dict['Temp']
    uf = vfilt_dict['Uvel']
    vf = vfilt_dict['Vvel']
    wf = vfilt_dict['Wvel']

    # Create contour plots
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    mixing_full = ds.variables['mixing_ratio'][0,:,:]
    if np.ma.is_masked(mixing_full):
        masked_count = np.ma.count_masked(mixing_full)
        mixing_full = mixing_full.filled(fill_value=0.0)
        print(f"Full resolution plot: Filled {masked_count} masked values with 0.0")
    plt.contourf(mixing_full, cmap='viridis')
    plt.colorbar(label='Mixing Ratio (Full Resolution)')
    plt.title('Full Resolution (512x512)')

    plt.subplot(122)
    plt.contourf(vfilt_dict['mixing_ratio'][:,:,0].T, cmap='viridis')
    plt.colorbar(label='Mixing Ratio (Filtered)')
    plt.title(f'Filtered Resolution ({nx}x{ny})')

    plt.tight_layout()
    plt.savefig(f'contour_comparison_{tstep}.png',dpi=150)
    plt.close()

    # Write output file
    print("Writing output file...")
    with open(f'{outdir}/eul.txt', 'w') as fid:
        for iz in tqdm(range(nz)):
            for iy in range(ny):
                for ix in range(nx):
                    fid.write(f'{sf[ix,iy,iz]:10e} ')
                    fid.write(f'{Tf[ix,iy,iz]:10e} ')
                    fid.write(f'{uf[ix,iy,iz]:10e} ')
                    fid.write(f'{vf[ix,iy,iz]:10e} ')
                    fid.write(f'{wf[ix,iy,iz]:10e}\n')

    print(f"Output saved to {outdir}/eul.txt")
    print("Contour plots saved as contour_comparison.png")
