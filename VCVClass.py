# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:03:34 2019

@author: jrilla
"""
# =============================================================================
# Packages
# =============================================================================
import numpy as np 
import numpy.linalg as la
import pandas as pd
# =============================================================================
# VCV class 
# =============================================================================
class VCV(object):
    def __init__(self, residuals, factors):
        self.residuals = pd.DataFrame(residuals)
        self.factors = pd.DataFrame(factors)
        
        self.T, self.total_N_inc_K = self.residuals.shape
        self.K = self.factors.shape[1]
        
        
    def _dpart1(self):
        d_part1 = np.ones((self.K + 1, self.K + 1))
        
        for i in range(self.K):
            d_part1[0][i + 1] = self.factors.iloc[:, i].mean()
            d_part1[i + 1][0] = d_part1[0][i + 1]
            
            for j in range(i, self.K):
                inner = np.inner(self.factors.iloc[:, i], 
                                 self.factors.iloc[:, j])
                d_part1[i + 1][j + 1] = inner / self.T
                
                if i != j:
                    d_part1[j + 1][i + 1] = d_part1[i + 1][j + 1]
                    
        return d_part1
        
    
    def d(self):
        """
        Cochrane (2005) p234
        """
        I_N = np.identity(self.total_N_inc_K)
        
#        self._d_part1 = np.ones((self.K + 1, self.K + 1))
#        
#        for i in range(self.K):
#            self._d_part1[0][i + 1] = self.factors.iloc[:, i].mean()
#            self._d_part1[i + 1][0] = self._d_part1[0][i + 1]
#            
#            for j in range(i, self.K):
#                inner = np.inner(self.factors.iloc[:, i], 
#                                 self.factors.iloc[:, j])
#                self._d_part1[i + 1][j + 1] = inner / self.T
#                
#                if i != j:
#                    self._d_part1[j + 1][i + 1] = self._d_part1[i + 1][j + 1]
        d_part1 = self._dpart1()
                    
                
        return - np.kron(d_part1, I_N)
    
    def Sigma(self):
#        Sig = np.zeros((self.total_N_inc_K, self.total_N_inc_K))
#        
#        for t in range(self.T):
#            Sig += np.outer(self.residuals.iloc[t], self.residuals.iloc[t])
#        
#        # Both have been extensively tested, minor differences
#        Sig = 1 / (self.T - self.K - 1) * Sig
#        Sig = 1/self.T * Sig 
        
        # Following gives exact same results as whole for loop
        Sig = self.T / (self.T - self.K - 1) * self.residuals.cov() * ((self.T - 1) / self.T) 
        return Sig
        
    
    def __call__(self):
        d = self.d()
        d_inv = la.inv(d)
        
        S = self.Spectral_density()
        
#        # possible finite sample solution
        N = self.total_N_inc_K - self.K #excluding factors from testing assets
        df = self.T - N - self.K
#        
     
#        return 1/self.T * d_inv @ S @ d_inv   
        return N / df *  d_inv @ S @ d_inv   
    

class Standard(VCV):
    def Spectral_density(self):
        
        Sig = self.Sigma()
        
        S = np.kron(self._dpart1(), Sig)       
        
        return S
    
class Bartlett(VCV):
    def Spectral_density(self, bandwith=None): 
        if bandwith is not None:
            b = bandwith
        
        else:
            b = int(self.T / 10)
        N = self.total_N_inc_K
        K = self.K
        T = self.T
        res = self.residuals
        f = self.factors
        
        S = np.zeros((N * (1 + K), N * (1 + K))) 
        
        for j in range(-b + 1, b):
            av_points = T - np.abs(j)
            for t in range(av_points):
                weight = (b - np.abs(j)) / b * 1/av_points
#                weight = (b - np.abs(j)) / b * 1/T
                if j < 0:
                    S[:N, :N] +=  weight * np.outer(res.iloc[t], res.iloc[t - j])
                elif j >= 0:
                    S[:N, :N] +=  weight * np.outer(res.iloc[t + j], res.iloc[t])
        
        for k in range(1, K + 1):
            for j in range(-b + 1, b):
                av_points = T - np.abs(j)
                for t in range(av_points):
                    weight = (b - np.abs(j)) / b * 1/av_points
#                    weight = (b - np.abs(j)) / b * 1/T
                    if j < 0:
                        S[k*N: k*N + N, :N] +=  weight * np.outer(f.iloc[t, k - 1] * res.iloc[t], res.iloc[t - j])
                    elif j >= 0:
                        S[k*N: k*N + N, :N] +=  weight * np.outer(f.iloc[t + j, k - 1] * res.iloc[t + j], res.iloc[t])
            
            # mirror image 
            S[:N, k*N: k*N + N] =  S[k*N: k*N + N, :N].T  
        
        for k in range(1, K + 1):
            for k2 in range(k, K + 1):
                for j in range(-b + 1, b):
                    av_points = T - np.abs(j)
                    for t in range(av_points):
                        weight = (b - np.abs(j)) / b * 1/av_points
#                        weight = (b - np.abs(j)) / b * 1/T
                        if j < 0:
                            S[k2*N:k2*N + N,k*N: k*N + N ]  +=  weight * np.outer(f.iloc[t, k - 1] * res.iloc[t], res.iloc[t - j] * f.iloc[t - j, k2 - 1])
                        elif j >= 0:
                            S[k2*N:k2*N + N,k*N: k*N + N ]  +=  weight * np.outer(f.iloc[t + j, k - 1] * res.iloc[t + j], res.iloc[t] * f.iloc[t, k2 - 1])
          
            # mirror image - right upper 
                S[k*N: k*N + N, k2*N:k2*N + N] =  S[k2*N:k2*N + N,k*N: k*N + N ].T 
                
                
        return S
    
    
    
    
    
    
    
    