# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:48:53 2022

@author: Mariska

"""
import numpy as np
import pandas as pd
#import scipy.integrate as spi
import scipy.sparse as sp
#import MyTicToc as mt
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
#    Initialize
# =============================================================================
#import UnsaturatedFlowClass as UFC

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
theta_r = 0.02#np.random.uniform(0.05, 0.07)    # [-] Residual water content
theta_s = por                              # [-] Saturated water content
k_sat = 1 #np.random.uniform(1, 3)*(10**-5)   # [m/day] Saturated hydaulic conductivity
a = 10 #np.random.uniform(1.5, 1.1)            # [/m] van Genuchten parameter
n = 3 #np.random.uniform(1.3, 1.5)            # [-] van Genuchten parameter
cv = 10**-4                                # [/m] Compressibility of soil
sPar = {'theta_r': theta_r * np.ones(np.shape(zN)), 
        'theta_s': theta_s * np.ones(np.shape(zN)), 
        'k_sat': k_sat * np.ones(np.shape(zN)),
        'a': a * np.ones(np.shape(zN)),
        'n': n * np.ones(np.shape(zN)),
        'cv': cv * np.ones(np.shape(zN))}
sPar = pd.Series(sPar)

# ## Definition of the Boundary Parameters
# boundary parameters
# no Dirichlet condition (top and bottom are both Robin)

# bPar = {'avgT': 273.15 + 10,
#         'rangeT': 20,
#         'tMin': 46,
#         'topCond': 'RObin',
#         'lambdaRobTop': 1e7,
#         'lambdaRobBot': 1e6,
#         'TBndBot': 273.15 + 10
#         }

#bPar = pd.Series(bPar)           # one dimensional array with all boundary parameters
# ## Initial Conditions
# Initial Conditions
WL = -0.25  #initial water level
hIni = -0.75 - zN #np.ones(np.shape(zN)) * (10.0 + 273.15)  # K




# In[2]: Solve IVP over time range

# Time Discretization
tOut = np.linspace(0, 365.25 * 10, 365)  # time
nOut = np.shape(tOut)[0]

#mt.tic()


def BndHTop(self, t):           # Head top as function of time
    bndH = -0.001 * (t > 25)  # m/day       
    return bndH

#Effective saturation function 
def Seff (hw, sPar):
    a = sPar.a
    n = sPar.n
    hc = -hw
    Seff = ((1 + (a * (hc > 0) * hc)**n)**-(1 - 1 / n))
    return Seff

#Volumetric water content function
def theta_w(hw, sPar):
    theta_r = sPar.theta_r
    theta_s = sPar.theta_s
    theta_w = theta_r + (theta_s - theta_r) * Seff
    return theta_w

#Differential water capacity function
def C (hw, theta_w):
    dh = 1e-8
    hw = hw + 1j * dh
    C = theta_w / dh
    return C

#Mass Matrix for Richards equation
def Mass_Matrix(h_w):
    S = theta_w / theta_s  
    S_s = rhoW * g *(cv + theta_s * beta)
    MMatrix = C + S_s * S
    return MMatrix

#Hydraulic conductivities at internodes
def K_int(h_w, sPar, mDim):
    k_sat = sPar.k_sat
    nIN = mDim.nIN
    K_rw = Seff**3    #Relative permiability
    K_node = k_sat * K_rw
    K_int[1] = K_node[1]
    ii = np.arange(2, nIN-1)
    K_int[ii] = min(K_node[ii], K_node[ii-1])
    K_int[nIN] = K_node[-1]
    return K_int

#Flux at the internodes
def FlowFlux(self, t, F, mDim, hw, k, bndH, bndB, robin):  
    nr,nc = F.shape                     
    nIN = self.mDim.nIN             
    nN = self.mDim.nN                   
    dzN = self.mDim.dzN
    q = np.zeros((nIN, nc),dtype=F.dtype)  
    
    # Flux at all the intermediate nodes
    ii = np.arange(1, nIN-1)
    q[ii] = -k[ii-1]*((hw[ii]-hw[ii-1])/(dzN[ii-1])+1)
    
    # Flux at top
    if t > 25:
        q[nIN-1] = bndH
    else:
        q[nIN-1] = 0
        return
    
    if bndB == 'gravity':
        q[0] = 0
    else:
        q[0] = -robin[1]*(hw[0]-robin[1])
        return
    return q

#Net flux at the nodes
def NF (t, hw, sPar, mDim):
    nIN = mDim.nIN
    dzIN = mDim.dzIN
    MM = Mass_Matrix(hw)
    F = FlowFlux(hw, t)
    ii = np.arange(2, nIN-1)
    NF = - (F [ii + 1, 1] - F [ii ,1]) / (dzIN [ii, 1] * MM [ii, 1])
    return NF

def IntegrateFF(self, tRange, iniSt):
    
    def dYdt(t, F):
        if len(F.shape)==1:
            F = F.reshape(self.mDim.nN,1)
        rates = self.DivHeatFlux(t, F)
        return rates
    
    def jacFun(t, y):
        if len(y.shape)==1:
            y = y.reshape(self.mDim.nN,1)
    
        nr, nc = y.shape
        dh = 1e-8
        ycmplx = y.copy().astype(complex)
        ycmplx = np.repeat(ycmplx,nr,axis=1)
        c_ex = np.ones([nr,1])* 1j*dh
        ycmplx = ycmplx + np.diagflat(c_ex,0)
        dfdy = dYdt(t, ycmplx).imag/dh
        return sp.coo_matrix(dfdy)

# def J(t, h_w, sPar, mDim, Bnd):
#     dh = np.spacing()
#     n = h_w.shape
#     Jac = np.zeros(n(1))
    
#     for ii in range(len(h_w)):
#         h_w_component = h_w
#         h_w_component[ii] = h_w_component[ii] + 1j * dh
#         dFdy =NF(t, h_w, sPar, mDim, Bnd).imag / dh
#         Jac[:, ii] = dFdy[:]
#     Jac = sp.sparse.csr_matrix(Jac)    
#     return Jac

MyHD = NF(t, hw, sPar, mDim)

int_result = MyHD.IntegrateFF(tOut, hIni.squeeze())


# Dirichlet boundary condition: write boundary temperature to output.
if int_result.success:
    print('Integration has been successful')

qW = MyHD.FlowFlux(tOut, int_result.y)  #Flux at the internode

plt.close('all')
fig1, ax1 = plt.subplots(figsize=(7, 4))
for ii in np.arange(0, nN, 10):
    ax1.plot(tOut, int_result.y[ii, :], '-')
ax1.set_title('Hydralic head over time')
ax1.set_xlabel('time (days)')
ax1.set_ylabel('hydralyc head [m]')

fig2, ax2 = plt.subplots(figsize=(4, 7))
for ii in np.arange(0, nOut, 10):
    ax2.plot(int_result.y[:, ii], zN, '-')

ax2.set_title('Hydralic head over depth')
ax2.set_ylabel('depth [m]')
ax2.set_xlabel('Hydralic head [K]')


fig3, ax3 = plt.subplots(figsize=(4, 7))
# plot fluxes after 2nd output time (initial rate is extreme due to initial conditions)
for ii in np.arange(2, nOut, 10):
    ax3.plot(qW[:, ii], zIN, '-')

ax3.set_title('Flux vs. depth')
ax3.set_ylabel('depth [m]')
ax3.set_xlabel('hydralic head [J/m2]')

plt.show()

# plt.savefig('A2.png')
# if __name__ == "__main__":
# main()
