
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
import MyTicToc as mt
import matplotlib.pyplot as plt
import seaborn as sns


class HeatDiffusion:
# ## Definition of functions
# Then we need to define the functions which we will be using:
# BndTTop for calculating the top temperature as a function of time;
# HeatFlux for calculating all heat fluxes in the domain;
# DivHeatFlux for calculating the divergence of the heat flux across the cells in the domain.
    def __init__(self, sPar, mDim, bPar):
        self.sPar = sPar
        self.mDim = mDim
        self.bPar = bPar
        
        

    def BndTTop(self, t):
        #bndT = self.bPar.avgT - self.bPar.rangeT * np.cos(2 * np.pi
                                            #* (t - self.bPar.tMin) / 365.25)
        bndT = 280
        return bndT
    
    def zetaBulk(self):
        # currently a dummy function, but here for making this
        # function of the states in the future
        zetaBN = self.sPar.zetaBN
        return zetaBN
    
    def HeatFlux(self, t, T):
        nr,nc = T.shape
        nIN = self.mDim.nIN
        nN = self.mDim.nN
        dzN = self.mDim.dzN
        lambdaIN = self.sPar.lambdaIN
        
        q = np.zeros((nIN, nc),dtype=T.dtype)
    
        # Temperature at top boundary
        bndT = self.BndTTop(t)
        
        # Calculate heat flux in domain
        # Bottom layer Robin condition
    
        q[0] = -self.bPar.lambdaRobBot * (T[0] - self.bPar.TBndBot)
    
        # Flux in all intermediate nodes
        ii = np.arange(1, nIN-1)
        q[ii] = -lambdaIN[ii] * ((T[ii] - T[ii-1])
                                       / dzN[ii-1])
        # Top layer
        # Robin condition
        q[nIN-1] = -self.bPar.lambdaRobTop * (bndT - T[nN-1])
            
        return q
    
    
    def DivHeatFlux(self, t, T):
        nr,nc = T.shape
        nN = self.mDim.nN
        dzIN = self.mDim.dzIN
        zetaBN = self.zetaBulk( )
        
       # Calculate heat fluxes accross all internodes
        qH = self.HeatFlux(t, T)
    
        divqH = np.zeros([nN, nc],dtype=T.dtype)
        # Calculate divergence of flux for all nodes
        ii = np.arange(0, nN)
        divqH[ii] = -(qH[ii + 1] - qH[ii]) \
                       / (dzIN[ii] * zetaBN[ii])
    
        divqHRet = divqH # .reshape(T.shape)
        return divqHRet
    
    
    def IntegrateHF(self, tRange, iniSt):
        
        def dYdt(t, T):
            if len(T.shape)==1:
                T = T.reshape(self.mDim.nN,1)
            rates = self.DivHeatFlux(t, T)
            return rates
        
        def jacFun(t, y):
            if len(y.shape)==1:
                y = y.reshape(self.mDim.nN,1)
        
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
                                    method='BDF', vectorized=True, jac=jacFun, 
                                    rtol=1e-8)
        
        return int_result


