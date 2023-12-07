import pathlib
import os, fnmatch, sys
from os.path import expanduser
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib import cm
import matplotlib.pyplot as pl
import scipy
import copy
import scipy.stats as stats
import csv
from herbie import Herbie, Herbie_latest, FastHerbie

####Authors: Shawn Corvec and Leo Separovic
####This is a very rudimentary example script to compute power spectra of NWP kinetic energy (wind) fields.
####This is used extensively in NWP model research and development to evaluate and compare
####the "effective" resolution of models. I'm aware there may be a scaling factor missing somewhere.

###Download some native coordinate wind data from the HRRR usign Herbie
###(note that we don't use the pressure level data here as it is likely smoothed!)
H = Herbie("2023-12-07", searchString="40 hybrid level:",
 model="hrrr",product="nat",freq="1H",member=1,fxx=6,verbose=True,ovewrite=True)

uu=H.xarray(":UGRD:40 hybrid level:", remove_grib=False).u.values
vv=H.xarray(":VGRD:40 hybrid level:", remove_grib=False).v.values

resolution=3.0 # approx resolution in KM

# Calculate DCT 2D spectral variance #
dct=np.abs((scipy.fft.dctn(uu,norm='ortho')/np.sqrt(uu.shape[0]*uu.shape[1]))**2)
shape0=dct.shape[0]
shape1=dct.shape[1]
dct1=np.abs((scipy.fft.dctn(vv,norm='ortho')/np.sqrt(vv.shape[0]*uu.shape[1]))**2)
uu_pspec2_py=0.5*(dct+dct1) #Add u and v power spectra together


# Begin calculations of the 1D spectral variance #
nbwv = min(shape0,shape1)
nbwx = max(shape0,shape1)

# initialize
uu_pspec1_py=np.empty([nbwv])
uu_pspec1_py *= 0.

# Generate ALPHAS #
FreqCompRows = np.arange(0, int(shape0), 1)
FreqCompCols = np.arange(0, int(shape1), 1)

knrm=np.empty([shape0,shape1])
i=0
while i < len(FreqCompRows):
 j=0
 while j < len(FreqCompCols):
     alpha=np.sqrt((FreqCompRows[i]**2)/(shape0**2)+(FreqCompCols[j]**2)/(shape1**2))*nbwv
     knrm[i,j]=alpha
     j += 1
 i += 1
knrm = knrm.flatten()

##############################################################################################
# Epsilon is needed to deal with the fact that stats.binned_statistic 
# works with bin intervals that are closed on the left and open on the right
# whereas we need the opposite in order to follow Bertrand's paper.
# A rule of thumb for the largest epsilon that could still work can be derived from the grid dimensions. 
# The larger the grid the smaller the epsilon needed. 
# This prevents the possibility that the program fails to work as specified for large grids.
eps0 = 1.e-1
eps  = eps0*np.sqrt(1/nbwx)
##############################################################################################

# Sum over elliptical rings # 
kbins = np.arange(-1.+eps,nbwv+eps,1.)
Abins, _, _ = stats.binned_statistic(knrm, uu_pspec2_py.flatten(),
                                   statistic = np.nansum,
                                   bins = kbins)

# Remove zeroth element (the mean) from power spectra for the purpose of plotting #
uu_pspec1_py = Abins[1:]

A1bins = uu_pspec1_py[0::2]
A2bins = uu_pspec1_py[1::2]
uu_pspec1_py = A1bins[0:len(A1bins)-1] + A2bins
wavenumbers = np.arange(0.,nbwv/2, 1)

print(wavenumbers)
wavelengths = resolution*nbwv/wavenumbers[1:]

pl.loglog((resolution*uu.shape[1]/wavenumbers[10:600]), 11.5*10**-5*(resolution*uu.shape[1]/wavenumbers[10:600])**(5/3), label="-5/3 slope")
pl.loglog(wavelengths, uu_pspec1_py, color='r', label='python c='+str(0) ,linewidth=0.75)
pl.xlabel("wavelength (km)")
pl.ylabel("KE power spectrum [m^2/s^2]")
pl.tight_layout()
#pl.legend(['test'])
pl.legend(loc="lower left")
plt.title('test')
plt.gca().invert_xaxis()
pl.savefig("power_spec", dpi = 300, bbox_inches = "tight")
pl.close()
print("plotting complete")
