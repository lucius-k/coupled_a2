# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:48:53 2022

@author: Mariska

"""
import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.sparse as sp
#import MyTicToc as mt
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
#    Initialize
# =============================================================================
import UnsaturatedFlowClass as UFC

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

# =============================================================================
# ============================== Soil Properties ==============================
# =============================================================================

rhoW = 1000                                 # [kg/m3] Density of water
rhoS = 2650                                 # [kg/m3] Density of solid phase
rhoB = 1700                                 # [kg/m3] Dry bulk density of soil
por = 1 - rhoB / rhoS                       # [-] Porosity of soil = saturated water content
beta = 4.5 * 10 ** -6                       # [/m] Compressibility of water
g = 9.81                        

# Soil properties match those of a silt
theta_r = np.random.uniform(0.05, 0.07)    # [-] Residual water content
theta_s = por                              # [-] Saturated water content
k_sat = np.random.uniform(1, 3)*(10**-5)   # [m/day] Saturated hydaulic conductivity
a = np.random.uniform(1.5, 1.1)            # [/m] van Genuchten parameter
n = np.random.uniform(1.3, 1.5)            # [-] van Genuchten parameter
cv = 10**-4                                # [/m] Compressibility of soil
sPar = {'theta_r': theta_r, 'theta_s': theta_s, 'k_sat': k_sat, 'a': a, 'n': n, 'cv': cv}
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

bPar = pd.Series(bPar)           # one dimensional array with all boundary parameters
# ## Initial Conditions
# Initial Conditions
WL = -0.25  #initial water level
hIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K

MyHD = UFC.FlowDiffusion(sPar, mDim, bPar)



# In[2]: Solve IVP over time range

# Time Discretization
tOut = np.linspace(0, 365.25 * 10, 365)  # time
nOut = np.shape(tOut)[0]

#mt.tic()
int_result = MyHD.IntegrateFF(tOut, hIni.squeeze())

#mt.toc()

# Dirichlet boundary condition: write boundary temperature to output.
if int_result.success:
    print('Integration has been successful')

qW = MyHD.FlowFlux(tOut, int_result.y)  #Flux at the internode

def BndHTop(self, t):           # Head top as function of time
    bndH = -0.001 * (t > 25)  # m/day       
    return bndH

#Effective saturation function 
def Seff (hw, sPar):
    a = sPar.a
    n = sPar.n
    hc = -hw
    Seff = ((1 + (a * (hc > 0).hc)**n)**(1 - 1 // n))
    return Seff




#!!! INCOMPLETE!!! Differential water capacity function

#Differential water capacity function
<<<<<<< HEAD
>>>>>>> aed3bfc3e7db2ff29e053491a2dfd6483d420879
def C(hw, theta_w):
=======
def C (hw, theta_w):
>>>>>>> 6961d17ccf393aa32d6c4bbc96f47a73dbe401b6
    dh = np.spacing()
    hw = hw + 1j * dh
    C = theta_w // dh
    return C

#Mass Matrix for Richards equation
def Mass_Matrix(h_w):
    S = theta_w / theta_s  
    S_s = rhoW * g *(cv + theta_s * beta)
    MMatrix = C + S_s * S
    return MMatrix

<<<<<<< HEAD
   
def K_int(h_w):
    k_sat = sPar.k_sat
    nIn = mDim.nIn
    K_rw = Seff**3    #Relative permiability
    K_node = k_sat * K_rw
    k_int[1] = k_node[1]
    ii = np.arange(2, nIN-1)
    K_int[ii] = min(K_node[ii], K_node[ii-1])
    K_int[nIn] = K_node[-1]
    return K_int

    
#Flux at the internodes
=======
#Flux at the internodes
def Flux(h_w, t):
    K = np.transpose("return Kfunction")
    ii = np.arange(2, nIN-1)
    FLux[ii] = - K [ii - 1] * (h_w[ii] - h_w[ii - 1]) / (dzN[ii - 1] + 1)
    return Flux
=======
#Net flux at the nodes
>>>>>>> 6961d17ccf393aa32d6c4bbc96f47a73dbe401b6
def NF (t, hw, sPar, mDim, Bnd):
    nIN = mDim.nIN
    dzIN = mDim.dzIN
    MM = Mass_Matrix(hw)
    F = Flux(h_w, t)
    ii = np.arange(2, nIN-1)
    NF = - (F [ii + 1, 1] - F [ii ,1]) // (dzIN [ii, 1] * MM [ii, 1])
    return NF
<<<<<<< HEAD



>>>>>>> aed3bfc3e7db2ff29e053491a2dfd6483d420879
=======
>>>>>>> 6961d17ccf393aa32d6c4bbc96f47a73dbe401b6

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
    ax3.plot(qW[:, ii], zIN, '-')

ax3.set_title('Heat Flux vs. depth (ODE)')
ax3.set_ylabel('depth [m]')
ax3.set_xlabel('temperature [J/m2]')


# plt.show()

plt.show()
# plt.savefig('myfig.png')

# if __name__ == "__main__":
# main()
