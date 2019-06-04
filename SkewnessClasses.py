"""
Created on Wed Apr 10 12:06:02 2019

@author: jrilla
"""
# =============================================================================
#%% Package importing
# =============================================================================
import pandas as pd
import numpy as np

#import os
#
### Change local directory - Windows: change C:\ into C:/
## directory work desktop
#directory = ("C:/Users\jrilla\OneDrive - bf.uzh.ch"
#             "\Courses\Asset pricing\Project")
#
## directory home laptop
##directory = ("D:/OneDrive - bf.uzh.ch\Courses"
##             "\Asset pricing\Project")
#
#os.chdir(directory)

# Own files
#import latexclasses as ltx
import functions_support as sf
# =============================================================================
# %% Skewness measures
# =============================================================================
class Coskewness(object):
    """
    Base class for different measures of Coskewness
    """
    def __init__(self, factor_series, returns_df):
        self.factor = factor_series
        self.returns = returns_df
        
                 
    def _regressions(self):
        self._reg_results = {}
        for i in self.returns:
            self._reg_results[i] = sf.OLS_regression(self.returns[i], 
                                                     self.factor)

### 1) beta_skd
class BetaSKD(Coskewness):
    """
    Harvey, Siddique (2000) - eq (11):
        
    beta_skd = E[eps_{i,t+1}eps_{M,t+1}^2] / 
               sqrt{E[eps_{i,t+1}}^2]E[eps_{M,t+1}^2] 
                            
    """
    def __init__(self, returns_df, factor_series):
        Coskewness.__init__(self, factor_series, returns_df)
        
        try:
            self._regressions()
            
        except ValueError:
            print('Factor and Return series do not have the same time-length')
            if len(self.factor) > self.returns.shape[0]:
                print('Factor series has been shortened')
                self.factor = self.factor.loc[self.returns.index]
            
            else:
                print('Returns series have been shortened')
                self.returns = self.returns.loc[self.factor.index]
                
            self._regressions()
        
    def __call__(self):
        
        # prepare df for residuals atfor each asset and time
        eps = pd.DataFrame(index=self.returns.index, 
                           columns=self.returns.columns) 
        
        # fill up residual df
        for i in self.returns:
            eps[i] = self._reg_results[i].resid
            
        factor_eps = self.factor - self.factor.mean()
        
        #numerator of (11)
        num = eps.multiply(factor_eps**2, axis=0).mean() 
        
        #denominator of (11)
        den = np.sqrt((eps**2).mean()) * (factor_eps**2).mean() 
        
        beta_skd = num / den  
        
        return beta_skd

### 2) beta_squared
class BetaSquared(Coskewness):
    """
    Beta of regressing return on squared factor
    """
    def __init__(self, returns_df, factor_series):
        Coskewness.__init__(self, factor_series, returns_df)
        
        self.factor = self.factor**2
        
        try:
            self._regressions()
            
        except ValueError:
            print('Factor and Return series do not have the same time-length')
            if len(self.factor) > self.returns.shape[0]:
                print('Factor series has been shortened')
                self.factor = self.factor.loc[self.returns.index]
            
            else:
                print('Returns series have been shortened')
                self.returns = self.returns.loc[self.factor.index]
                
            self._regressions()
            
    def __call__(self):
        # prepare df for betas at for each asset
        betas = pd.Series(index=self.returns.columns)
        
        #fill up df
        for i in self.returns:
             betas[i] = self._reg_results[i].params[1]
        
        return betas


#%%
### 3) 
class BetaLongShortPort(Coskewness):
    def __init__(self, returns_df, long_short_hedgeportfolio):
        Coskewness.__init__(self, long_short_hedgeportfolio, returns_df)
        
        try:
            self._regressions()
            
        except ValueError:
            print('Factor and Return series do not have the same time-length')
            if len(self.factor) > self.returns.shape[0]:
                print('Factor series has been shortened')
                self.factor = self.factor.loc[self.returns.index]
            
            else:
                print('Returns series have been shortened')
                self.returns = self.returns.loc[self.factor.index]
                
            self._regressions()
        
    def __call__(self):
        # prepare df for betas at for each asset
        betas = pd.Series(index=self.returns.columns)
        
        #fill up df
        for i in self.returns:
             betas[i] = self._reg_results[i].params[1]
        
        return betas

### 3) 
class BetaNegSkewPort(Coskewness):
    def __init__(self, returns_df, negative_skew_port):
        Coskewness.__init__(self, negative_skew_port, returns_df)
        
        try:
            self._regressions()
            
        except ValueError:
            print('Factor and Return series do not have the same time-length')
            if len(self.factor) > self.returns.shape[0]:
                print('Factor series has been shortened')
                self.factor = self.factor.loc[self.returns.index]
            
            else:
                print('Returns series have been shortened')
                self.returns = self.returns.loc[self.factor.index]
                
            self._regressions()
    
    def __call__(self):
        # prepare df for betas at for each asset 
        betas = pd.Series(index=self.returns.columns)
        
        #fill up df
        for i in self.returns:
             betas[i] = self._reg_results[i].params[1]
        
        return betas