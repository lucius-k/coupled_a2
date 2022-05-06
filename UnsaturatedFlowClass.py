# -*- coding: utf-8 -*-
"""
Created on Fri May  6 13:21:50 2022

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
# Creating class for Unsaturated flow wiht Richard's equation
# =============================================================================

class FlowDiffusion:
    def __init__(self, sPar, mDim, bPar):                  
        self.sPar = sPar            
        self.mDim = mDim            
        self.bPar = bPar            

    def BndHTop(self, t):           # Head top as function of time
       # if 
        bndH = -0.001 # m/day       # For now, assume that top BC flux constant
        return bndH
    
    def zetaBulk(self):             
        zetaBN = self.sPar.zetaBN
        return zetaBN
        
    def FlowFlux(self, t, H):               
        nr,nc = H.shape                     
        nIN = self.mDim.nIN                 
        nN = self.mDim.nN                   
        dzN = self.mDim.dzN
        lambdaIN = self.sPar.lambdaIN       
        q = np.zeros((nIN, nc),dtype=H.dtype)       
        bndH = self.BndHTop(t)                  
        
        q[0] = -self.bPar.lambdaRobBot * (H[0] - self.bPar.TBndBot)         
        ii = np.arange(1, nIN-1)
        q[ii] = -lambdaIN[ii] * ((H[ii] - H[ii-1])
                                       / dzN[ii-1])
        q[nIN-1] = -self.bPar.lambdaRobTop * (bndH - H[nN-1])
        return q
    
    
    def DivFlowFlux(self, t, H):
        nr,nc = H.shape                     
        nN = self.mDim.nN                   
        dzIN = self.mDim.dzIN               
        zetaBN = self.zetaBulk()
        
        qH = self.FlowFlux(t, H)               
        divqH = np.zeros([nN, nc],dtype=H.dtype)
        ii = np.arange(0, nN)
        divqH[ii] = -(qH[ii + 1] - qH[ii]) \
                       / (dzIN[ii] * zetaBN[ii])
    
        divqHRet = divqH # .reshape(H.shape)
        return divqHRet
    
    
    def IntegrateFF(self, tRange, iniSt):           
        
        def dYdt(t, H):                             
            if len(H.shape)==1:
                H = H.reshape(self.mDim.nN,1)
            rates = self.DivFlowFlux(t, H)          
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
        
        # solve rate equation
        t_span = [tRange[0],tRange[-1]]
        int_result = spi.solve_ivp(dYdt, t_span, iniSt.squeeze(), 
                                    t_eval=tRange, 
                                    method='BDF', vectorized=True, jac=jacFun, 
                                    rtol=1e-8)
        
        return int_result
