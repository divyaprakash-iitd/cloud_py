import gc
import numpy as np
import path
#startime step
tstep = 11000
#end time step
tstep2 = 20001
#skip super droplets for faster run
iskip = 1
#size of domain
lx = 51.2
ly = lx
lz = lx
#number of LES grid cells along x,y,z
nx = 32
ny = 32
nz = 32
#define multiplicity
nm = 300
#define minimum and maximum radius for local histogram
rminh = 0
rmaxh = 16
nh = 20
drh = (rmaxh-rminh)/nh
rhvec = [(i*drh+0.5*drh) for i in range(nh)]
rho0 = 1.06
rhovs = 0.0039
##Load mean data from file
#meandata = np.loadtxt(f"{path.eulerdir}/v_max_min.dat")
##Extract necessary values
#Tavg = meandata[tstep,6]
#rvavg = meandata[tstep,5]
#print(f"Tavg={Tavg} and rvavg={rvavg}")
#del meandata
gc.collect()

Acons=5.17e8
Bcons=5420.0

