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
    def Seff (self, T, sPar):
        a = sPar.a
        n = sPar.n
        hc = -T
        Seff = (1 + (a * (hc * (hc>0))) ** n) ** -(1 - 1 / n)  
        return Seff
 
    #Differential water capacity function    
    def theta_w(self, T, sPar):
        theta_r = sPar.theta_r
        theta_s = sPar.theta_s
        Seff = self.Seff(T, sPar)
        theta_w = Seff * (theta_s - theta_r) + theta_r
        return theta_w   

    #Differential water capacity function 
    def C (self, T, sPar):
        dh = np.sqrt(np.finfo(float).eps)
        hw = T + 1j * dh
        theta_w = self.theta_w(hw, sPar)
        C = theta_w.imag / dh
        return C    

    #Mass Matrix for Richards equation
    def Mass_Matrix(self, sPar, T, beta, par):
        theta_s = sPar.theta_s
        rhoW = par.rhoW
        cv = sPar.cv        
        g = 9.81
        S_s = rhoW * g *(cv + theta_s * beta)
        theta_w = self.theta_w(T, sPar)
        S = theta_w / theta_s  
        C = self.C(T, sPar)
        MMatrix = C + S_s * S
        return MMatrix

    def K_int(self, sPar, mDim, T):
        nr, nc = T.shape 
        k_sat = sPar.k_sat
        nIN = mDim.nIN
        Seff = self.Seff(T, sPar)
        K_rw = Seff**3    #Relative permiability
        K_node = k_sat * K_rw
        K_int = np.zeros((nIN, nc),dtype=T.dtype)  
        K_int[0] = K_node[0]
        ii = np.arange(2, nIN-1)
        K_int[ii] = K_node[ii]
        K_int[-1] = K_node[-1]
        return  K_int

    #Flux at the internodes
    def FlowFlux(self, t, T, mDim, bPar, sPar):  
        nr, nc = T.shape                     
        nIN = mDim.nIN             
        dzN = mDim.dzN
        q = np.zeros((nIN, nc),dtype=T.dtype)  
        hw = T
        bndB = bPar.bndB    
        
        # Flux at top
        qTop = bPar.TopBndFunc(t)
        q[nIN-1] = qTop # bndT *(t > 25)
        
        S = self.Seff(T, sPar)
        S = S**3
        # Conductivity
        # K = self.K_int(sPar, mDim, T)
        k = sPar.k_sat
        # Flux at all the intermediate nodes
        ii = np.arange(1, nIN-1)
        q[ii] = (-k[ii]*S[ii])*((T[ii]-T[ii-1])/(dzN[ii-1])+1)
        
        # Flux at bottom
        if bndB == 'gravity':
            q[0] = -k[0]*S[0]
        else:
            q[0] = -bPar.robin*(T[0]-bPar.hrobin)
            return
        return q
    
    #Net flux at the nodes
    def DivFlowFlux (self, t, T, sPar, mDim, par, bPar):
        nr, nc = T.shape
        rhoW = par.rhoW
        nN = mDim.nN
        # theta_w = self.theta_w(T, sPar)
        # C = self.C(T, sPar)
        nIN = mDim.nIN
        dzIN = mDim.dzIN
        beta = par.beta
        MM = self.Mass_Matrix(sPar, T, beta, par)
        test = self.FlowFlux(t, T, mDim, bPar, sPar)
        DivFlowFlux = np.zeros([nN, nc],dtype=T.dtype)
        ii = np.arange(2, nIN-1)
        DivFlowFlux[ii] = -(test [ii + 1] - test [ii]) / (dzIN [ii] * MM[ii])
        return DivFlowFlux 
    
    def IntegrateFF(self, tRange, iniSt, sPar, mDim, par, bPar):
        
        def dYdt(t, T):
            if len(T.shape)==1:
                T = T.reshape(self.mDim.nN,1)
            rates = self.DivFlowFlux(t, T, sPar, mDim, par, bPar)
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
    
    
