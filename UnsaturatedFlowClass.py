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
        Seff = (1 + (a * (hc * (hc>0))) ** n) ** -(1 - 1 / n)  
        return Seff
 
    #Differential water capacity function    
    def theta_w(self, hw, sPar):
        theta_r = sPar.theta_r
        theta_s = sPar.theta_s
        Seff = self.Seff(hw, sPar)
        theta_w = Seff * (theta_s - theta_r) + theta_r
        return theta_w   

    #Differential water capacity function 
    def C (self, hw, sPar):
        dh = np.sqrt(np.finfo(float).eps)
        h = hw + 1j * dh
        theta_w = self.theta_w(h, sPar)
        C = theta_w.imag / dh
        return C    

    #Mass Matrix for Richards equation
    def Mass_Matrix(self, sPar, hw, beta, par):
        theta_s = sPar.theta_s
        rhoW = par.rhoW
        cv = sPar.cv        
        g = 9.81
        S_s = rhoW * g *(cv + theta_s * beta)
        theta_w = self.theta_w(hw, sPar)
        S = theta_w / theta_s  
        C = self.C(hw, sPar)
        MMatrix = C + S_s * S
        return MMatrix

    # def K_int(self, sPar, mDim, hw):
    #     nr, nc = hw.shape 
    #     k_sat = sPar.k_sat
    #     nIN = mDim.nIN
    #     Seff = self.Seff(hw, sPar)
    #     K_rw = Seff**3    #Relative permiability
    #     K_node = k_sat * K_rw
    #     K_int = np.zeros((nIN, nc),dtype=hw.dtype)  
    #     K_int[0] = K_node[0]
    #     ii = np.arange(2, nIN-1)
    #     K_int[ii] = K_node[ii]
    #     K_int[-1] = K_node[-1]
    #     return  K_int

    #Flux at the internodes
    def FlowFlux(self, t, hw, mDim, bPar, sPar):  
        nr, nc = hw.shape                     
        nIN = mDim.nIN             
        dzN = mDim.dzN
        q = np.zeros((nIN, nc),dtype=hw.dtype)  
        bndB = bPar.bndB    
        k = sPar.k_sat
        S = self.Seff(hw, sPar)
        k_rw = S**3
        K = k*k_rw
        
        i = np.arange(1, nIN-1)
        
        # Flux at bottom
        if bndB == 'gravity':
            q[0] = -K[0]
        else:
            q[0] = -bPar.robin*(hw[0]-bPar.hrobin)       
        
        # Flux at all the intermediate nodes
        q[i] = (-K[i])*((hw[i]-hw[i-1])/(dzN[i-1])+1)
        
        # Flux at top
        qTop = bPar.TopBndFunc(t)
        q[nIN-1] = qTop # bndT *(t > 25)
        return q
    
    #Net flux at the nodes
    def DivFlowFlux (self, t, hw, sPar, mDim, par, bPar):
        nr, nc = hw.shape
        nN = mDim.nN
        dzIN = mDim.dzIN
        beta = par.beta
        MM = self.Mass_Matrix(sPar, hw, beta, par)
        flow = self.FlowFlux(t, hw, mDim, bPar, sPar)
        DivFlowFlux = np.zeros([nN, nc],dtype=hw.dtype)
        ii = np.arange(0, nN)
        DivFlowFlux[ii] = -(flow[ii+1]-flow[ii])/(dzIN[ii]*MM[ii])
        return DivFlowFlux 
    
    def IntegrateFF(self, tRange, iniSt, sPar, mDim, par, bPar):
        
        def dYdt(t, hw):
            if len(hw.shape)==1:
                hw = hw.reshape(self.mDim.nN,1)
            rates = self.DivFlowFlux(t, hw, sPar, mDim, par, bPar)
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
        int_result = spi.solve_ivp(dYdt, t_span, iniSt, 
                                    t_eval=tRange, 
                                    method='BDF', vectorized=True, 
                                    rtol=1e-8)
        
        return int_result
    
    
