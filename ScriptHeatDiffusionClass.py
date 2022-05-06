#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:46:11 2021

@author: theimovaara
"""

import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.sparse as sp
import MyTicToc as mt
import matplotlib.pyplot as plt
import seaborn as sns

# Version of code based on classes
import HeatDiffusionClass as HD

"""
Script for running Heat Diffusion problem. The code is given in the HeatDiffusionClass.
This script, initializes the problem and then calls the solver in the class and it
generates some plots"""

# In[1]: Initialize
sns.set()
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

# collect model dimensions in a namedtuple: modDim
mDim = {'zN' : zN,
        'zIN' : zIN,
        'dzN' : dzN,
        'dzIN' : dzIN,
        'nN' : nN,
        'nIN' : nIN
        }
mDim = pd.Series(mDim)

# ## Definition of material properties
# Currently this model is based on a homogeneous soil with constant
# water content

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
# no Dirichlet condition (top and bottom are both Robin)

bPar = {'avgT': 273.15 + 10,
        'rangeT': 20,
        'tMin': 46,
        'topCond': 'RObin',
        'lambdaRobTop': 1e7,
        'lambdaRobBot': 1e6,
        'TBndBot': 273.15 + 10
        }

bPar = pd.Series(bPar)
# ## Initial Conditions
# Initial Conditions
TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K

MyHD = HD.HeatDiffusion(sPar, mDim, bPar)

# In[2]: Solve IVP over time range

# Time Discretization
tOut = np.linspace(0, 365.25 * 10, 365)  # time
nOut = np.shape(tOut)[0]

mt.tic()
int_result = MyHD.IntegrateHF(tOut, TIni.squeeze())

mt.toc()

# Dirichlet boundary condition: write boundary temperature to output.
if int_result.success:
    print('Integration has been successful')

qH = MyHD.HeatFlux(tOut, int_result.y)

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
