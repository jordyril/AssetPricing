"""
Created on Tue Apr 16 15:06:37 2019

@author: jrilla
"""
# =============================================================================
#%% Package importing
# =============================================================================
import pandas as pd
import numpy as np
import numpy.linalg as la
import statsmodels.api as sm

import matplotlib.pyplot as plt

import os

## Change local directory - Windows: change C:\ into C:/
# directory work desktop
directory = ("C:/Users\jrilla\OneDrive - bf.uzh.ch"
             "\Courses\Asset pricing\Project")

# directory home laptop
#directory = ("D:/OneDrive - bf.uzh.ch\Courses"
#             "\Asset pricing\Project")

os.chdir(directory)


# Own files
import latexclasses as ltx
import functions_support as sf

from Hedgeportfolio_construction import hedgeportfolios
from SkewnessClasses import (BetaSKD, BetaLongShortPort, 
                             BetaNegSkewPort, BetaSquared)

from DataImport import (factors, portfolio_dic, returns)

from FactorModel import FamaFrench3, FF3F1, CAPM, CAPMF1

from functions_support import title

#from VCVClass import Bartlett
# =============================================================================
#%% General
# =============================================================================
S_pl = hedgeportfolios['High']                  
S_min = hedgeportfolios['Low']
LS_S = hedgeportfolios['LH-LS']

# =============================================================================
#%% TIME SERIES REGRESSIONS - III.A
# =============================================================================
portfolio_names = pd.Series(list(portfolio_dic.keys()), name='Portfolio group')

ts_results_df = pd.DataFrame(index=portfolio_names, 
                             columns=['# Portfolios', 'F-test 3F',
                                      'F-test 4F'])
TS_regression_results = {}
#ts_res_dic = {}
# 3 FAMA-FRENCH factors
ff3 = factors[['mkt','smb', 'hml']]

# 3 FF + factor mimicking portfolio for skewness
ff3s1 = ff3.copy()
ff3s1['S_min'] = S_min

# Loop for all portfolios
title('TIME-SERIES REGRESSIONS')
for port in portfolio_dic:
    print(port)
    FF3 = FamaFrench3(portfolio_dic[port], ff3)
    FF3S1 = FF3F1(portfolio_dic[port], ff3s1)
    
    TS_regression_results[port] = {}
    TS_regression_results[port]['model'] = {}
    # TIME SERIES REGRESSIONS
    TS_regression_results[port]['ts'] = {}
#    ts_res_dic[port] = {}
    
    ts_results_df.loc[port]['# Portfolios'] = FF3S1.N
    
    for fname, i in zip(['3F', '4F'], [FF3, FF3S1]):
        print(fname)
        TS_regression_results[port]['ts'][fname] = {}
        TS_regression_results[port]['model'][fname] = i
#        ts_res_dic[port][fname] = {}
#        ts_res_dic[port][fname]['model'] = i
        
        grs, F, _, p = i.Gibbons_Ross_Shanken()
        
        for v, n in zip([grs, F, p], ['F-test', 'F-stat', 'p-value']):
            TS_regression_results[port]['ts'][fname][n] = v
#            ts_res_dic[port][fname][n] = v
        
        # for individual assets do not bother
        
        print('F', TS_regression_results[port]['ts'][fname]['F-test'])
        print('p', TS_regression_results[port]['ts'][fname]['p-value'])
    print()
            
    # FAMA-MCBETH PROCEDURE - SHOULD ALTER BETAS OVER TIME HERE
#    TS_regression_results[port]['fm'] = {}
#    lambda_0 = FF3.Fama_Mcbeth(intercept=True).iloc[:, 0]
#    corr = np.correlate(lambda_0, LS_S)
#   
#    TS_regression_results[port]['fm']['corr'] = corr
    

#TODO
# FILL TABLES  
# =============================================================================
#%% FAMA-MCBETH PROCEDURE III.A - correlations (table II)
# =============================================================================
title('FAMA-MCBETH - correlations')
fm_window_is = 60
fm_window_os = len(factors) - fm_window_is

fm_res = {}
fm_res['corr'] = pd.DataFrame(columns=['corr'], index=portfolio_names, 
                              dtype=float)
for port in portfolio_dic:
    print(port)
    fm_res[port] = {}
    fm_res[port]['lambdas'] = pd.DataFrame(index=portfolio_dic[port]
                                                  .iloc[fm_window_is:].index,
                                                   columns=list(range(4)))
    for t in range(fm_window_os):
        ret = portfolio_dic[port].iloc[t: t + fm_window_is]
        f = ff3.iloc[t: t + fm_window_is]
        fs = ff3s1.iloc[t: t + fm_window_is]
    
        m = FamaFrench3(ret, f) 
        
        m.timeseries_regression(True)   
            
        a = m.betas[:, :-m.K].T

        a1 = sm.add_constant(a)

        b = pd.DataFrame(portfolio_dic[port].iloc[fm_window_is + t])

        param = la.lstsq(a1, b, rcond=None)[0]
    
        fm_res[port]['lambdas'].iloc[t] = param[:, 0]
        
        corr = np.correlate(fm_res[port]['lambdas'].iloc[:, 0], 
                            S_min.iloc[fm_window_is:])[0]
    print(corr, '\n')
    
        
    fm_res['corr'].loc[port] = corr
    
        
#print(fm_res['corr'])
   
        
#for port in portfolio_dic: 
#    fm_res[port] = {}
#    
#    for m, k in zip(['3F', '4F'], [4, 5]):
#        fm_res[port][m] = {}
#        fm_res[port][m]['Lambdas'] = pd.DataFrame(index=portfolio_dic[port]
#                                                  .iloc[fm_window_is:].index,
#                                                   columns=list(range(k)))
#    
#    for t in range(fm_window_os):
#        ret = portfolio_dic[port].iloc[t: t + fm_window_is]
#        f = ff3.iloc[t: t + fm_window_is]
#        fs = ff3s1.iloc[t: t + fm_window_is]
#    
#        model1 = FamaFrench3(ret, f)  
#        model2 = FF3F1(ret, fs)
#        
#        for m, n in zip([model1, model2], ['3F', '4F']):
#            m.timeseries_regression(True)   
#            
#            a = m.betas[:, :-m.K].T
#            a.shape
#            a1 = sm.add_constant(a)
#            a1.shape
#    
#            b = pd.DataFrame(portfolio_dic[port].iloc[fm_window_is + t])
#            b.shape
#    
#            param = la.lstsq(a1, b, rcond=None)[0]
#            param.shape
#    
#            fm_res[port][n]['Lambdas'].iloc[t] = param[:, 0]
#
#    fm_res[port]['3F']['corr'] = np.correlate(fm_res[port]['3F']['Lambdas']
#                                               .iloc[:, 0], 
#                                               S_min
#                                               .iloc[fm_window_is:])[0]
#    print(port, '3F', 'corr',  fm_res[port]['3F']['corr'])
#
#

# =============================================================================
#%% CROSS SECTION - III.B
# =============================================================================
ind = portfolio_dic['Industry']
#TODO Full-information maximum likelihood (FIML)











#%%   

#re = returns.dropna(axis=1)
#test = FamaFrench3(re, ff3)
#
#
#c = test.Fama_Mcbeth(intercept=True).iloc[:, 0]
#np.correlate(c, S_min)

               
#ts_res_dic                     
    
#FF3.betas
#FF3S1.betas
#a = FF3.estimates_df()
#b = FF3.standard_errrors(return_df=True)
#pd.DataFrame(a.values / b.values)
#
#FF3.t_stats(True)
#FF3
#FF3S1.t_stats(True).iloc[-4, :-4]


# =============================================================================
# Cross sectional
# =============================================================================
#import statsmodels.api as sm
#
#FF3.Fama_Mcbeth(intercept=True).iloc[:, 0]
#FF3._lambdahat_str
#
#a = FF3.betas[:, :-FF3.K].T
#a.shape
#a1 = sm.add_constant(a)
#a1.shape
#
#b = FF3.assets.iloc[:, :-FF3.K].T
#b.shape
#
#param = la.lstsq(a, b, rcond=None)[0]
#param.shape
#
#alpha = b.T - param.T @ a.T 


