# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:22:56 2022

@author: tdenh
"""

import numpy as np
# import pandas as pd
import scipy.integrate as spi
import scipy.sparse as sp
#import MyTicToc as mt
# import matplotlib.pyplot as plt
# import seaborn as sns

# =============================================================================
# Creating class for Unsaturated flow wiht Richard's equation
# =============================================================================

class FlowDiffusion:
    def __init__(self, theta_r, theta_s, k_sat, a, n, cv):                  
        self.theta_r = theta_r           
        self.theta_s = theta_s           
        self.k_sat = k_sat  
        self.a = a    
        self.n = n
        self.cv = cv
    
    def BndHTop(t):           # Head top as function of time
        bndH = -0.001 * (t > 25)  # m/day       
        return bndH
    
    #Effective saturation function 
    def Seff (hw, sPar):
        a = sPar.a
        n = sPar.n
        hc = -hw
        Seff = ((1 + (a * (hc > 0) * hc)**n)**-(1 - 1 / n))
        return Seff
 
    #Differential water capacity function    
    def theta_w(hw, sPar, Seff):
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
    def Mass_Matrix(sPar, rhoW, h_w, theta_w, beta, C):
        theta_s = sPar.theta_s
        cv = sPar.cv        
        g = 9.81
        S = theta_w / theta_s  
        S_s = rhoW * g *(cv + theta_s * beta)
        MMatrix = C + S_s * S
        return MMatrix

    def K_int(h_w, sPar, mDim, Seff, H):
        nr,nc = H.shape 
        k_sat = sPar.k_sat
        nIN = mDim.nIN
        K_rw = Seff**3    #Relative permiability
        K_node = k_sat * K_rw
        K_int = np.zeros((nIN, nc))  
        K_int[0] = K_node[0]
        ii = np.arange(2, nIN-1)
        K_int[ii] = min(K_node[ii], K_node[ii-1])
        K_int[nIN] = K_node[-1]
        return 

    #Flux at the internodes
    def FlowFlux(self, t, H, mDim, hw, k, bndH, bndB, robin):  
        nr,nc = H.shape                     
        nIN = self.mDim.nIN             
        nN = self.mDim.nN                   
        dzN = self.mDim.dzN
        q = np.zeros((nIN, nc),dtype=H.dtype)  
        
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
    def NF (self, t, hw, sPar, mDim):
        nIN = mDim.nIN
        dzIN = mDim.dzIN
        MM = self.Mass_Matrix(hw)
        F = self.FlowFlux(hw, t)
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
        
        # solve rate equation
        t_span = [tRange[0],tRange[-1]]
        int_result = spi.solve_ivp(dYdt, t_span, iniSt.squeeze(), 
                                    t_eval=tRange, 
                                    method='BDF', vectorized=True, jac=jacFun, 
                                    rtol=1e-8)
        
        return int_result
    
    
