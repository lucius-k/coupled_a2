
# coding: utf-8

# # Heat Diffusion in Soils
#
# This Jupyter Notebook gives an example how to implement a 1D heat diffusion model in Python.
#
# First we need to import the packages which we will be using:
#

# In[1]:


import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

import MyTicToc as mt

# plot figures inline
#matplotlib inline

# plot figures as interactive graphs...
#%matplotlib qt

# ## Definition of functions
# Then we need to define the functions which we will be using:
# BndTTop for calculating the top temperature as a function of time;
# HeatFlux for calculating all heat fluxes in the domain;
# DivHeatFlux for calculating the divergence of the heat flux across the cells in the domain.

sns.set()



def BndTTop(t, bPar):
    bndT = bPar.avgT - bPar.rangeT * np.cos(2 * np.pi
                                        * (t - bPar.tMin) / 365.25)
    return bndT

def zetaBulk(t, T, sPar, mDim, bPar):
    zetaBN = sPar.zetaBN
    return zetaBN

def HeatFlux(t, T, sPar, mDim, bPar):
    nr,nc = T.shape
    nIN = mDim.nIN
    nN = mDim.nN
    dzN = mDim.dzN
    lambdaIN = sPar.lambdaIN
    
    q = np.zeros((nIN, nc),dtype=T.dtype)

    # Temperature at top boundary
    bndT = BndTTop(t, bPar)
    
    # Calculate heat flux in domain
    # Bottom layer Robin condition

    q[0] = -bPar.lambdaRobBot * (T[0] - bPar.TBndBot)

    # Flux in all intermediate nodes
    ii = np.arange(1, nIN-1)
    q[ii] = -lambdaIN[ii] * ((T[ii] - T[ii-1])
                                   / dzN[ii-1])
    # Top layer
    # Robin condition
    q[nIN-1] = -bPar.lambdaRobTop * (bndT - T[nN-1])
        
    return q


def DivHeatFlux(t, T, sPar, mDim, bPar):
    nr,nc = T.shape
    nN = mDim.nN
    dzIN = mDim.dzIN
    zetaBN = zetaBulk(t, T, sPar, mDim, bPar )
    
   # Calculate heat fluxes accross all internodes
    qH = HeatFlux(t, T, sPar, mDim, bPar)

    divqH = np.zeros([nN, nc],dtype=T.dtype)
    # Calculate divergence of flux for all nodes
    ii = np.arange(0, nN)
    divqH[ii] = -(qH[ii + 1] - qH[ii]) \
                   / (dzIN[ii] * zetaBN[ii])

    return divqH


def IntegrateHF(tRange, iniSt, sPar, mDim, bPar):
    
    def dYdt(t, T):
        if len(T.shape)==1:
            T = T.reshape(mDim.nN,1)
        rates = DivHeatFlux(t, T, sPar, mDim, bPar)
        return rates
    
    def jacFun(t,y):
        if len(y.shape)==1:
           y = y.reshape(mDim.nN,1)
    
        nr, nc = y.shape
        dh = np.sqrt(np.finfo(float).eps)
        ycmplx = y.copy().astype(complex)
        ycmplx = np.repeat(ycmplx,nr,axis=1)
        c_ex = np.ones([nr,1])* 1j*dh
        ycmplx = ycmplx + np.diagflat(c_ex,0)
        dfdy = dYdt(t, ycmplx).imag/dh
        return sp.coo_matrix(dfdy)
    
    # solve rate equatio
    t_span = [tRange[0],tRange[-1]]
    int_result = spi.solve_ivp(dYdt, t_span, iniSt.squeeze(), 
                                t_eval=tRange, 
                                method='BDF', vectorized=True, #jac=jacFun, 
                                rtol=1e-8)
    
    return int_result


## Main 

# Domain
nIN = 151
# soil profile until 15 meters depth
zIN = np.linspace(-15, 0, num=nIN).reshape(nIN, 1)
# nIN = np.shape(zIN)[0]
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
zN[0, 0] = zIN[0, 0]
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
zN[nIN - 2, 0] = zIN[nIN - 1]
nN = np.shape(zN)[0]

ii = np.arange(0, nN - 1)
dzN = (zN[ii + 1, 0] - zN[ii, 0]).reshape(nN - 1, 1)
dzIN = (zIN[1:, 0] - zIN[0:-1, 0]).reshape(nIN - 1, 1)

# collect model dimensions in a pandas series: mDim
mDim = {'zN' : zN,
        'zIN' : zIN,
        'dzN' : dzN,
        'dzIN' : dzIN,
        'nN' : nN,
        'nIN' : nIN
        }
mDim = pd.Series(mDim)

# ## Definition of material properties
# In this section of the code we define the material properties

# Soil Properties
# [J/(m3 K)] volumetric heat capacity of soil solids
zetaSol = 2.235e6
# [J/(m3 K)] volumetric heat capacity of water (Fredlund 2006)
zetaWat = 4.154e6

# rhoW = 1000  # [kg/m3] density of water
rhoS = 2650  # [kg/m3] density of solid phase
rhoB = 1700  # %[kg/m3] dry bulk density of soil
n = 1 - rhoB / rhoS  # [-] porosity of soil = saturated water content.
q = 0.75  # quartz content

# [W/(mK)] thermal conductivity of water (Remember W = J/s)
lambdaWat = 0.58
lambdaQuartz = 6.5  # [W/(mK)] thermal conductivity of quartz
lambdaOther = 2.0  # [W/(mK)] thermal conductivity of other minerals

lambdaSolids = lambdaQuartz ** q * lambdaOther ** (1 - q)
lambdaBulk = lambdaWat ** n * lambdaSolids ** (1 - n)

# collect soil parameters in a namedtuple: soilPar

sPar = {'zetaBN': np.ones(np.shape(zN)) * ((1 - n) * zetaSol
                                                + n * zetaWat),
        'lambdaIN': np.ones(np.shape(zIN)) * lambdaBulk * (24 * 3600)
        }
sPar = pd.Series(sPar)

# ## Definition of the Boundary Parameters
# boundary parameters
# collect boundary parameters in a named tuple boundpar...
bPar = {'avgT': 273.15 + 10,
        'rangeT': 20,
        'tMin': 46,
        'topCond': 'Robin',
        'lambdaRobTop': 1e7,
        'lambdaRobBot': 1e6,
        'TBndBot': 273.15 + 10
        }

bPar = pd.Series(bPar)
# ## Initial Conditions
# Initial Conditions
TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
# Time Discretization
tOut = np.linspace(0, 365.25 * 10, 365)  # time
nOut = np.shape(tOut)[0]


mt.tic()
int_result = IntegrateHF(tOut, TIni, sPar, mDim, bPar)

mt.toc()

# Dirichlet boundary condition: write boundary temperature to output.
if int_result.success:
    print('Integration has been successful')

qH = HeatFlux(tOut, int_result.y, sPar, mDim, bPar)

plt.close('all')
fig1, ax1 = plt.subplots(figsize=(7, 4))
for ii in np.arange(0, nN, 10):
    ax1.plot(tOut, int_result.y[ii, :], '-')
ax1.set_title('Temperature (ODE)')
ax1.set_xlabel('time (days)')
ax1.set_ylabel('temperature [K]')

fig2, ax2 = plt.subplots(figsize=(4, 7))
for ii in np.arange(0, nOut, 10):
    ax2.plot(int_result.y[:, ii], zN, '-')

ax2.set_title('Temperature vs. depth (ODE)')
ax2.set_ylabel('depth [m]')
ax2.set_xlabel('temperature [K]')


fig3, ax3 = plt.subplots(figsize=(4, 7))
# plot fluxes after 2nd output time (initial rate is extreme due to initial conditions)
for ii in np.arange(2, nOut, 10):
    ax3.plot(qH[:, ii], zIN, '-')

ax3.set_title('Heat Flux vs. depth (ODE)')
ax3.set_ylabel('depth [m]')
ax3.set_xlabel('temperature [J/m2]')


# plt.show()

plt.show()
# plt.savefig('myfig.png')

# if __name__ == "__main__":
# main()
