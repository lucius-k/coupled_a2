# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:22:56 2022

@author: tdenh
"""

import numpy as np
import pandas as pd
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
    
    
    
    #Effective saturation function 
    def Seff (self, hw, sPar):
        a = sPar.a
        n = sPar.n
        hc = -hw
        Seff = ((1 + (a * (hc > 0) * hc)**n)**-(1 - 1 / n))
        return Seff
 
    #Differential water capacity function    
    def theta_w(self, hw, sPar, Seff):
        theta_r = sPar.theta_r
        theta_s = sPar.theta_s
        theta_w = theta_r + (theta_s - theta_r) * Seff
        return theta_w   

    #Differential water capacity function 
    def C (self, hw, theta_w):
        dh = 1e-8
        hw = hw + 1j * dh
        C = theta_w / dh
        return C    

    #Mass Matrix for Richards equation
    def Mass_Matrix(self, sPar, rhoW, hw, theta_w, beta, C):
        theta_s = sPar.theta_s
        cv = sPar.cv        
        g = 9.81
        S = theta_w / theta_s  
        S_s = rhoW * g *(cv + theta_s * beta)
        MMatrix = C + S_s * S
        return MMatrix

    def K_int(self, sPar, mDim, Seff, H):
        nc = H.shape 
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
    def FlowFlux(self, t, hw, mDim, bPar, sPar, par):  
        nr, nc = hw.shape                     
        nIN = mDim.nIN             
        dzN = mDim.dzN
        q = np.zeros((nIN, nc),dtype=hw.dtype)  
        hw = par.hw
        k = sPar.k_sat
        #bndT = bPar.bndT
        bndB = bPar.bndB
        
        # Flux at all the intermediate nodes
        ii = np.arange(1, nIN-1)
        print(f"ii = {ii}")
        print(f"k = {k}")
        print(f"hw = {hw}")
        print(f"dzN = {dzN}")
        q[ii] = -k[ii]*((hw[ii]-hw[ii-1])/(dzN[ii-1])+1)
        
        print(q)
        # Flux at top
        
        qTop = bPar.TopBndFunc(t)
        
        q[nIN-1] = qTop # bndT *(t > 25)
        
        if bndB == 'gravity':
            q[0] = -k[0]
        else:
            q[0] = -bPar.robin*(hw[0]-bPar.hrobin)
            return
        print(q)
        return q
    
    #Net flux at the nodes
    def DivFlowFlux (self, t, T, hw, sPar, mDim, par, bPar):
        nr, nc = hw.shape
        rhoW = par.rhoW
        nN = mDim.nN
        theta_w = par.theta_w
        C = par.C
        nIN = mDim.nIN
        dzIN = mDim.dzIN
        beta = par.beta
        T = par.hw
        MM = self.Mass_Matrix(sPar, rhoW, hw, theta_w, beta, C)
        test = self.FlowFlux(t, T, mDim, bPar, sPar, par)
        DivFlowFlux = np.zeros([nN, nc],dtype=hw.dtype)
        ii = np.arange(2, nIN-1)
        print(test)
        DivFlowFlux[ii] = -(test [ii + 1] - test [ii]) / (dzIN [ii] * MM [ii])
        return DivFlowFlux 
    
    def IntegrateFF(self, tRange, iniSt, sPar, mDim, par, bPar):
        
        def dYdt(t, T):
            if len(T.shape)==1:
                T = T.reshape(self.mDim.nN,1)
            rates = self.DivFlowFlux(self, t, T, sPar, mDim, par, bPar)
            return rates
        
#        def jacFun(t, y):
#            if len(y.shape)==1:
#                y = y.reshape(self.mDim.nN,1)
#        
#            nr, nc = y.shape
#            dh = 1e-8
#            ycmplx = y.copy().astype(complex)
#            ycmplx = np.repeat(ycmplx,nr,axis=1)
#            c_ex = np.ones([nr,1])* 1j*dh
#            ycmplx = ycmplx + np.diagflat(c_ex,0)
#            dfdy = dYdt(t, ycmplx).imag/dh
#            return sp.coo_matrix(dfdy)
        
        # solve rate equation
        t_span = [tRange[0],tRange[-1]]
        int_result = spi.solve_ivp(dYdt, t_span, iniSt, 
                                    t_eval=tRange, 
                                    method='BDF', vectorized=True, 
                                    rtol=1e-8)
        
        return int_result
    
    
