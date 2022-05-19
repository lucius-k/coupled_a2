# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:42:36 2022

@author: Tom
"""

import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
# import MyTicToc as mt

# Import class
from UnsaturatedFlowClass import FlowDiffusion

# =============================================================================
# =================================== Domain ==================================
# =============================================================================
sns.set()

nIN = 101
# soil profile until 1 meters depth
zIN = np.linspace(-1, 0, num=nIN).reshape(nIN, 1)
# nIN = np.shape(zIN)[0]
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
zN[0, 0] = zIN[0, 0]
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
zN[nIN - 2, 0] = zIN[nIN - 1]
nN = np.shape(zN)[0]

ii = np.arange(0, nN-1)
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

tOut = np.linspace(1, 200, 200)
nOut = np.shape(tOut)[0]
# =============================================================================
# ============================== Soil Properties ==============================
# =============================================================================

rhoW = 1000                         # [kg/m3] density of water
rhoS = 2650                         # [kg/m3] density of solid phase
rhoB = 1700                         # [kg/m3] dry bulk density of soil
por = 1 - rhoB / rhoS               # [-] porosity of soil = saturated water content
beta = 4.5e-6                       # Compressibility of water

# Soil properties match those of a silt                          
theta_r = 0.02                      # [-] Residual water content
theta_s = por                       # [-] Saturated water content
k_sat = 1                           # [m/day] Saturated hydaulic conductivity
a = 3                               # [/m] van Genuchten parameter
n = 1                               # [-] van Genuchten parameter
cv = 10**-4                         # [/m] Compressibility
sPar = {'theta_r': theta_r * np.ones(np.shape(zN)),
        'theta_s': theta_s * np.ones(np.shape(zN)),
        'k_sat': k_sat * np.ones(np.shape(zN)), 
        'a': a * np.ones(np.shape(zN)),
        'n': n * np.ones(np.shape(zN)),
        'cv': cv * np.ones(np.shape(zN)),
        'rhoW': rhoW * np.ones(np.shape(zN))
        }
sPar = pd.Series(sPar)
UF = FlowDiffusion(theta_r, theta_s, k_sat, a, n, cv)

# =============================================================================
# ============================ Boundary Conditions ============================
# =============================================================================
Robin = [1, 0.005]
bndB = 'gravity'

def BndTTop(t):           # Head top as function of time
### FIX
        bndT = -0.001 #* (t > 25)  m/day
        return bndT
#bndT = UF.BndTTop(tOut)

bPar = {'bndB' : bndB,
        'robin': 1,
        'hrobin': -0.25,
        'TopBndFunc': BndTTop 
        }
bPar = pd.Series(bPar)


# Initial conditions
WL = -0.25
hw_initial = -zN + WL
Seff_initial = UF.Seff(hw_initial, sPar)
theta_w_initial = UF.theta_w(hw_initial, sPar, Seff_initial)
C_initial = UF.C(hw_initial, theta_w_initial)
par = {'rhoW': rhoW * np.ones(np.shape(zN)),
       'hw': hw_initial * np.ones(np.shape(zN)),
       'Seff': Seff_initial * np.ones(np.shape(zN)), 
       'theta_w': theta_w_initial * np.ones(np.shape(zN)),
       'C': C_initial * np.ones(np.shape(zN)),
       'beta': beta * np.ones(np.shape(zN))
        }
par = pd.Series(par)
int_result = UF.IntegrateFF(tOut, hw_initial.squeeze(), sPar, mDim, par, bPar)