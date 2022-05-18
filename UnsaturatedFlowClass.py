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
    
    def BndTTop(self, t):           # Head top as function of time
        bndT = -0.001 * (t > 25)  # m/day       
        return bndT
    
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

    def K_int(self, hw, sPar, mDim, Seff, H):
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
    def FlowFlux(self, t, T, mDim, bPar, sPar, par):  
        nr,nc = T.shape                     
        nIN = mDim.nIN             
        nN = mDim.nN                   
        dzN = mDim.dzN
        q = np.zeros((nIN, nc),dtype=T.dtype)  
        hw = par.hw
        k = sPar.k_sat
        bndT = bPar.bndT
        bndB = bPar.bndB
        robin = bPar.robin
        
        # Flux at all the intermediate nodes
        ii = np.arange(1, nIN-1)
        print(f"ii = {ii}")
        print(f"k = {k}")
        print(f"hw = {hw}")
        print(f"dzN = {dzN}")
        q[ii] = -k[ii]*((hw[ii]-hw[ii-1])/(dzN[ii-1])+1)
        
        print(q)
        # Flux at top
        # if t > 25:
        #     q[nIN-1] = bndT
        # else:
        #     q[nIN-1] = 0
        #     return
        
        # if bndB == 'gravity':
        #     q[0] = 0
        # else:
        #     q[0] = -robin[1]*(hw[0]-robin[1])
        #     return
        print(q)
        return q
    
    #Net flux at the nodes
    def DivFlowFlux (self, t, hw, sPar, mDim, par, bPar):
        rhoW = par.rhoW
        theta_w = par.theta_w
        C = par.C
        nIN = mDim.nIN
        dzIN = mDim.dzIN
        beta = par.beta
        k = sPar.k_sat
        bndT = bPar.bndT
        bndB = bPar.bndB
        robin = bPar.robin
        T = par.hw
        MM = self.Mass_Matrix(sPar, rhoW, hw, theta_w, beta, C)
        test = self.FlowFlux(t, T, mDim, bPar, sPar, par)
        ii = np.arange(2, nIN-1)
        print(test)
        DivFlowFlux = -(test [ii + 1] - test [ii]) / (dzIN [ii] * MM [ii])
        return DivFlowFlux, test 
    
    def IntegrateFF(self, tRange, iniSt, sPar, mDim, par, bPar):
        
        def dYdt(t, T):
            if len(T.shape)==1:
                T = T.reshape(self.mDim.nN,1)
            rates = self.DivFlowFlux(t, T, sPar, mDim, par, bPar)
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
    
    
