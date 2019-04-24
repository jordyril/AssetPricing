"""
Created on Mon Apr 15 19:06:13 2019

@author: jrilla
"""
# =============================================================================
#%% 
# =============================================================================
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm

from functions_support import OLS_regression
from VCVClass import Standard
# =============================================================================
#%% CLASSES
# =============================================================================
class TSRegResults(object):
    def __init__(self):
        self.alphas = None
        self.betas = None
        pass
    

# =============================================================================
# BASE
# =============================================================================
class FactorModel(object):
    """
    Baseclass for factor models
    """
    def __init__(self, assets_excess_return=None, factors=None):
        """
        Initialises a FactorModel object
        
        Keyword arguments:
            assets_excess_returns -- pd.Dataframe where index are the dates
                                     and the columns are the different asset. 
                                     Values are the excess returns of that 
                                     particular asset (default None)
            factors -- pd.Dataframe, pd.Series or listlike object to give
                       factor excess return series or just columnames within 
                       main returns dataframe (default None)
        """
#        self.K = 1
        self._factor_superscript()
        
        if assets_excess_return is not None:
            self.set_assets_factors(assets_excess_return, factors)
        
#        self.K = 1
#        self._factor_superscript()
        self._ts_coefficient_names()
        self._cs_coefficient_names()
        self.ts_results = TSRegResults()
                    
    def _factor_superscript(self):
        self._factor_superscript_list = ['f' + str(i)  for i in 
                                         range(1, 1 + self.K)]
        return None
    
    def _ts_coefficient_names(self):
        self._betahat_str = ['$\\hat{\\beta}^{' 
                              + i
                              + '}$' for i in self._factor_superscript_list]
    
        self._alphahat_str = '$\\hat{\\alpha}$'
        
        return None 
    
    def _cs_coefficient_names(self):
        self._lambdahat_str = ['$\\hat{\\lambda}^{' 
                              + i
                              + '}$' for i in self._factor_superscript_list]
    
        self._alphahat_str = '$\\hat{\\alpha}$'
        
        return None 
    
    
    def set_assets_factors(self, assets_excess_return, factors=None):
        """
        Set df (index=date, columns=assets) with excess returns
        
        keyword arguments:
            assets_excess_returns -- pd.Dataframe where index are the dates
                                     and the columns are the different asset. 
                                     Values are the excess returns of that 
                                     particular asset
            factors -- pd.Dataframe, pd.Series or listlike object to give
                       factor excess return series or just columnames within 
                       main returns dataframe (default None)
            
        """
        # No factors are given
        if factors is None:
            self.assets = assets_excess_return
            
        # factors are given as separate df or series
        elif (isinstance(factors, pd.DataFrame) or 
              isinstance(factors, pd.Series)):
            self.factors = pd.DataFrame(factors)
            self.assets = pd.concat([assets_excess_return, factors], axis=1)
        
        # factors are identified with columnnames in main returns df
        elif (isinstance(factors, list) or 
              isinstance(factors, str)):
            self.factors = pd.DataFrame(assets_excess_return[factors])
            self.assets = assets_excess_return
            
        self.assetnames = list(self.assets.columns)
        self.N = len(self.assetnames) - self.K
        self.T = len(self.assets)
      
        return None
    
#    def timeseries_regression(self, return_df=False):
#        """
#        Perform a time series regression:
#            R^e_{it} = \alpha_{i} + \beta_{i}'R_{ft}^e
#        and returns the estimated alphas and betas for all assets
#        
#        keyword arguments:
#            return_df -- return either a df with results or just arrays 
#                         (default False)
#        """
#        # prepare result arrays
#        self.betas = np.zeros((self.assets.shape[1], self.K))
#        self.alphas_ts = np.zeros(self.assets.shape[1])
#        
#        # save regression results
#        self.ts_results = {}
#        # save residuals
#        self.residuals = pd.DataFrame(index=self.assets.index,
#                                      columns=self.assets.columns)
#        
#        # perform OLS regression for each asset
#        for i, ast in zip(range(len(self.betas)), self.assets):
#            reg = OLS_regression(self.assets[ast], self.factors)
#            
#            self.ts_results[ast] = reg        
#            self.residuals[ast] = reg.resid
#            
#            self.alphas_ts[i] = reg.params[0]
#            
#            for b in range(self.K):
#                self.betas[i][b] = reg.params[1 + b]
#        
#        # Prepare df (optional)
#        if return_df:
#            df = pd.DataFrame(columns=[self._alphahat_str],
#                              index=self.assets.columns)
#            df[self._alphahat_str] = self.alphas_ts
#            for b in range(self.K):
#                df[self._betahat_str[b]] = self.betas[:, b]
#            
#            return df.T
#        
#        else:            
#            return self.alphas_ts, self.betas
    
    def timeseries_regression(self, return_df=False):
        """
        Perform a time series regression:
            R^e_{it} = \alpha_{i} + \beta_{i}'R_{ft}^e
        and returns the estimated alphas and betas for all assets
        
        keyword arguments:
            return_df -- return either a df with results or just arrays 
                         (default False)
        """
        # perform OLS regression for each asset
        f = self.factors.copy()
        f['C'] = 1.
        
        param = la.lstsq(f, self.assets, rcond=None)[0]
        self.betas = param[:-1, :]
        self.alphas_ts = param[-1, :]
        
        # fill resultsclass (try out)
        self.ts_results.alphas = self.alphas_ts
        self.ts_results.betas = self.betas
        
        assets_hat = f @ param
        
        self.residuals = self.assets - assets_hat.values
        
        # Prepare df (optional)
        if return_df:
            df = pd.DataFrame(columns=[self._alphahat_str],
                              index=self.assetnames)
            
            df[self._alphahat_str] = self.alphas_ts
            
            for b in range(self.K):
                df[self._betahat_str[b]] = self.betas[b, :]
            
            return df.T
        
        else:            
            return self.alphas_ts, self.betas
   
    def _check_ts_regression(self):
        try:
            self.betas
        except AttributeError:
            _, _ = self.timeseries_regression()
        
        return None
    
    def VCV_param(self, VCVclass=Standard, return_df=False):
        self._check_ts_regression()
        vcv_instance = VCVclass(self.residuals, self.factors)
        vcv = vcv_instance()
        
        if return_df:
            cols = ['$\\alpha_{' + str(i + 1) + '}' 
                               for i in range(self.N  + self.K)]
            
            for i in range(self.K):
                for j in range(self.N + self.K):
                    cols.append('$\\beta_{' 
                                + str(i + 1) 
                                + ',' 
                                + str(j + 1) 
                                + '}')
            vcv_df = pd.DataFrame(vcv, index=cols, columns=cols)
        
            return vcv_df
        else:
            return vcv
        
    def standard_errrors(self, VCVclass=Standard, return_df=False):
        self._check_ts_regression()
        vcv_matrix = self.VCV_param(VCVclass, False)
        
        se = np.sqrt(np.diagonal(vcv_matrix))
        
        if return_df:
            indexes = []
            indexes.append('SE(' + self._alphahat_str + ')')
            for i in self._betahat_str:
                indexes.append('SE(' + i + ')')
                
            se_df = pd.DataFrame(se.reshape((self.K + 1, -1)),
                                 columns=self.assetnames,
                                 index=indexes)
            
            return se_df

        else:
            return se.reshape((self.K + 1, -1))
        
    def estimates_df(self):
        df = pd.DataFrame(columns=[self._alphahat_str],
                          index=self.assetnames)
        
        df[self._alphahat_str] = self.alphas_ts
        
        for b in range(self.K):
            df[self._betahat_str[b]] = self.betas[b, :]
        
        return df.T
        
    def t_stats(self, return_df=True):
        est = self.estimates_df()
        se = self.standard_errrors(return_df=True)
        
        t_stats = est.values / se.values
        if return_df:
            return pd.DataFrame(t_stats, columns=self.assetnames,
                                index=est.index)
        else:
            return t_stats
        
        
    def Gibbons_Ross_Shanken(self, VCVclass=Standard, q=0.95):
#        """
#        Cochrane (2005) (12.6) p233
#        """
        """
        WALD test - THIS SHOULD BE FINITE SAMPLE TEST?
        """
        self._check_ts_regression()
        
        alpha_hat = self.alphas_ts[:self.N]
        
        df1 = self.N
        df2 = self.T - self.N - self.K 

#       VCV class incorporates the fact that S ( or Sigma) is estimated
#       from a finite sample
        Sigma_alpha = self.VCV_param(VCVclass)[:self.N, :self.N]
        Sigma_alpha_inv = la.inv(Sigma_alpha)

        GRS = alpha_hat @ Sigma_alpha_inv  @ alpha_hat
        
        F = scs.f.ppf(q, df1, df2)
        p_value = scs.f.sf(GRS, df1, df2)

        return GRS, F, GRS > F, p_value
    
    def Fama_Mcbeth(self, intercept=False, return_df=True):
        self._check_ts_regression()
        
        a = self.betas[:, :-self.K].T
        
        b = self.assets.iloc[:, :-self.K].T
        
        if intercept:
            a = sm.add_constant(a)
            
        lambdas = la.lstsq(a, b, rcond=None)[0]
        
        self.alphas_fm = b - a @ lambdas
        
        self.alphas_fm = self.alphas_fm.T
        
        self.lambdas_fm = lambdas.T
        
        if return_df:
            columns = self._lambdahat_str.copy()
            if intercept:
                columns.insert(0, '$\\hat{\\lambda}_0$')
                
            self.lambdas_fm = pd.DataFrame(self.lambdas_fm, 
                                           index=self.assets.index,
                                           columns=columns)
        
        return self.lambdas_fm
        
        
            
        
        
       
        
# =============================================================================  
# Single-factor model class
# =============================================================================
class SingleFactorModel(FactorModel):
    """
    One Factor model:
        E[R^e_i] = \alpha^i + \beta^i_{f}*E[R_{f}^e]
    """
    def __init__(self, assets_excess_return=None, factors=None):
        self.K = 1
        FactorModel.__init__(self, assets_excess_return, factors)
        
#        self.K = 1
#        self._ts_coefficient_names()
        
    def plot_expected_returns_vs_betas(self):
        er = self.assets.mean()

        self._check_ts_regression()

        x = np.linspace(-5, 5.)
        plt.plot(x, self.factors.mean()[self._factor_superscript_list[0]] * x, 
                 label='SML')
        
        for i in range(len(er)):
            plt.scatter(self.betas[0][i], er.iloc[i], 
                        label=er.index[i])
#            plt.scatter(self.betas[i], er.iloc[i], 
#                        label=er.index[i])
            
        plt.xlabel(self._betahat_str[0])
        plt.ylabel('E$\\left[R^e_i\\right] (\%)$', rotation='vertical')
        return None
    
#    def Gibbons_Ross_Shanken(self, VCVclass=Standard, q=0.95):
#        """
#        Cochrane (2005) (12.4) p231
#        """
#        self._check_ts_regression()
#        
#        df1 = self.N
#        df2 = self.T - self.N - self.K
#        part_1 = df2 / df1
#        
#        E_f = np.mean(self.factors).values[0]
#        sigma_f = np.std(self.factors).values[0]
#        part_2 = 1 + (E_f / sigma_f)**2
#        
#        alpha_hat = self.alphas_ts[:self.N]         
#        Sigma = VCVclass(self.residuals, self.factors).Sigma()
#        Sigma = Sigma[:self.N, :self.N] #Only for testing assets
#        Sigma_inv = la.inv(Sigma)   
#        part_3 = alpha_hat @ Sigma_inv @ alpha_hat
#        
##        GRS = part_1 / part_2 * part_3
#        GRS = part_3 / part_2
#        
#        F = scs.f.ppf(q, df1, df2)
#        
#        return GRS, F, part_3, part_1, 1/self.T
    

# =============================================================================
# CAPM 
# =============================================================================
class CAPM(SingleFactorModel):
    """
    Capital asset pricing model:
        E[R^e_i] = \alpha^i + \beta^i_{mkt}*E[R_{mkt}^e]
    """
    def _factor_superscript(self):
        self._factor_superscript_list = ['mkt']
        return None


# =============================================================================
# TWOFactor
# =============================================================================
class TwoFactorModel(FactorModel):
    def __init__(self, assets_excess_return=None, factors=None):
        
        self.K = 2        
        FactorModel.__init__(self, assets_excess_return, factors)
        
#        self.K = 3
        self._factor_superscript()
        self._ts_coefficient_names()

# =============================================================================
# FAMAFRENCH        
# =============================================================================
class CAPMF1(TwoFactorModel):    
    def _factor_superscript(self):
        self._factor_superscript_list = ['Mkt', 'f']
        return None
    
    def set_name_other_factor(self, name):
        self._factor_superscript_list[-1] = name
        self._ts_coefficient_names()
        
# =============================================================================
# THREEFactor
# =============================================================================
class ThreeFactorModel(FactorModel):
    def __init__(self, assets_excess_return=None, factors=None):
        
        self.K = 3        
        FactorModel.__init__(self, assets_excess_return, factors)
        
#        self.K = 3
        self._factor_superscript()
        self._ts_coefficient_names()

# =============================================================================
# FAMAFRENCH        
# =============================================================================
class FamaFrench3(ThreeFactorModel):    
    def _factor_superscript(self):
        self._factor_superscript_list = ['Mkt', 'SMB', 'HML']
        return None
 
# =============================================================================
# Four Factor  
# =============================================================================
class FourFactorModel(FactorModel):
    def __init__(self, assets_excess_return=None, factors=None):
        
        self.K = 4
        
        FactorModel.__init__(self, assets_excess_return, factors)
       
        self._factor_superscript()
        self._ts_coefficient_names()

# =============================================================================
# CARHART      
# =============================================================================
class Carhart(FourFactorModel):    
    def _factor_superscript(self):
        self._factor_superscript_list = ['Mkt', 'SMB', 'HML', 'UMD']
        return None

# =============================================================================
# FF3 + 1 other factor
# =============================================================================
class FF3F1(FourFactorModel):    
    def _factor_superscript(self):
        self._factor_superscript_list = ['Mkt', 'SMB', 'HML', 'f']
        return None
    
    def set_name_other_factor(self, name):
        self._factor_superscript_list[-1] = name
        self._ts_coefficient_names()
    
   

