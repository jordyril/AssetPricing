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

#import os
#
### Change local directory - Windows: change C:\ into C:/
## directory work desktop
#directory = ("C:/Users\jrilla\OneDrive - bf.uzh.ch"
#             "\Courses\Asset pricing\Project\AssetPricing")
#
### directory home laptop
###directory = ("D:/OneDrive - bf.uzh.ch\Courses"
###             "\Asset pricing\Project")
##
#os.chdir(directory)


# Own files
import latexclasses as ltx

from Hedgeportfolio_construction import hedgeportfolios
from SkewnessClasses import (BetaSKD, BetaLongShortPort, 
                             BetaNegSkewPort, BetaSquared)

from DataImport import (factors, portfolio_dic, returns)
from DataProcessor import DataProcessor

from FactorModel import FamaFrench3, FF3F1, CAPM, CAPMF1

from functions_support import title, OLS, subtitle

#from VCVClass import Bartlett
# =============================================================================
#%% General
# =============================================================================
S_pl = hedgeportfolios['High']                  
S_min = hedgeportfolios['Low']
LS_S = hedgeportfolios['LH-LS']

dataprocessor = DataProcessor()

# =============================================================================
#%% TIME SERIES REGRESSIONS - III.A
# =============================================================================
title('Time-Series')
portfolio_names = pd.Series(list(portfolio_dic.keys()), name='Portfolio group')

ts_results_df = pd.DataFrame(index=portfolio_names, 
                             columns=['# Portfolios', 'F-test 3F',
                                      'F-test 4F'])
TS_regression_results = {}
#ts_res_dic = {}
# 3 FAMA-FRENCH factors
ff3_list = ['mkt','smb', 'hml']
ff3 = factors[ff3_list]

# 3 FF + factor mimicking portfolio for skewness
ff3s1 = ff3.copy()
ff3s1['S_min'] = S_min

ff3s = ff3.copy()
ff3s['s_min'] = S_min
ff3s['sks'] = LS_S


# Loop for all portfolios
title('TIME-SERIES REGRESSIONS')
for port in portfolio_names[:-1]:
    print(port)
    FF3 = FamaFrench3(portfolio_dic[port], ff3).timeseries_regression()
    FF3S1 = FF3F1(portfolio_dic[port], ff3s1).timeseries_regression()
    
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
    
# =============================================================================
#%% FAMA-MCBETH PROCEDURE III.A - correlations (table II)
# =============================================================================
title('FAMA-MCBETH - correlations')
fm_window_is = 60
fm_window_os = len(factors) - fm_window_is

fm_res = {}
fm_res['corr'] = pd.DataFrame(columns=['corr'], index=portfolio_names, 
                              dtype=float)
for port in portfolio_names[:-1]:
    print(port)
    fm_res[port] = {}
    fm_res[port]['lambdas'] = pd.DataFrame(index=portfolio_dic[port]
                                                  .iloc[fm_window_is:].index,
                                                   columns=list(range(4)))
    for t in range(fm_window_os):
        ret = portfolio_dic[port].iloc[t: t + fm_window_is]
        f = ff3.iloc[t: t + fm_window_is]
    
        m = FamaFrench3(ret, f) 
        
        m.timeseries_regression()   
            
        a = m.betas[:, :-m.K].T
        
        b = pd.DataFrame(portfolio_dic[port].iloc[fm_window_is + t])

        reg_res = OLS(b, a)
        param = reg_res.param()
    
        fm_res[port]['lambdas'].iloc[t] = param[:, 0]
        
        corr = np.correlate(fm_res[port]['lambdas'].iloc[:, 0], 
                            S_min.iloc[fm_window_is:])[0]
    print(corr, '\n')

        
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
title('FAMA-MCBETH - R')
fm_window_is = 60
fm_window_os = len(factors) - fm_window_is
res_dic = {}

model_list_names = ['CAPM' , 'CAPM, S-', 'CAPM, sks', 'FF3', 'FF3, S-', 'FF3, sks']

#R_df = pd.DataFrame(index=portfolio_dic[port].iloc[fm_window_is:].index, columns=model_list_names)
for port in portfolio_names[:-1]:
    res_dic[port] = pd.DataFrame(index=portfolio_dic[port].iloc[fm_window_is:].index, columns=model_list_names)
    for t in range(fm_window_os):
        ret = portfolio_dic[port].iloc[t: t + fm_window_is]
        fac = ff3s.iloc[t: t + fm_window_is]
        
        model_list = [CAPM(ret, fac[['mkt']]), 
                      CAPMF1(ret, fac[['mkt' , 's_min']]),
                      CAPMF1(ret, fac[['mkt' , 'sks']]), 
                      FamaFrench3(ret, fac[['mkt' , 'smb', 'hml']]),
                      FF3F1(ret, fac[['mkt' , 'smb', 'hml', 's_min']]),
                      FF3F1(ret, fac[['mkt' , 'smb', 'hml', 'sks']])
                      ]
        # R
        for model, name in zip(model_list, model_list_names) : 
    
            ts_res = model.timeseries_regression()
        
            a = ts_res.betas[:, :-ts_res.K].T
            b = pd.DataFrame(portfolio_dic[port].iloc[fm_window_is + t])
            
            # mine
            reg_res = OLS(b, a, True)
            adjR2 = reg_res.adj_R_squared()
            res_dic[port][name].iloc[t] = adjR2[0]
            
    print(port)
    print(res_dic[port].mean(), '\n')



    
# =============================================================================
#%% full information maximum likelihood (1-14) - table III (1)
#TODO
# =============================================================================
title('FIML')
fiml_cr_res = {}
fiml_r_port = pd.DataFrame(index=portfolio_names[:-1], columns=model_list_names)
fiml_port_rmse = {}
for port in portfolio_names[:-1]:
    fiml_port_rmse[port] = pd.DataFrame(columns=model_list_names, index=portfolio_dic[port].columns)
    fiml_cr_res[port] = {}
    # estimate betas and mus using all returns
    mu_hat = portfolio_dic[port].mean()
    ret = portfolio_dic[port]
    model_list = [CAPM(ret, ff3s[['mkt']]), 
                  CAPMF1(ret, ff3s[['mkt' , 's_min']]),
                  CAPMF1(ret, ff3s[['mkt' , 'sks']]), 
                  FamaFrench3(ret, ff3s[['mkt' , 'smb', 'hml']]),
                  FF3F1(ret, ff3s[['mkt' , 'smb', 'hml', 's_min']]),
                  FF3F1(ret, ff3s[['mkt' , 'smb', 'hml', 'sks']])]
     # R

    for model, name in zip(model_list, model_list_names):
        reg_res = model.timeseries_regression()
        betas = reg_res.betas[:, :-reg_res.K].T
    
        res = OLS(mu_hat, betas)
        fiml_cr_res[port][name] = res
        fiml_port_rmse[port][name] = res.residuals()
        fiml_r_port[name][port] = res.adj_R_squared()[0]


print('Constant beta estimates \n\n', fiml_r_port.T, '\n')
print('RMSE \n')
for port in portfolio_names[:-1]:
    print(port)
    print(np.sqrt(np.mean(fiml_port_rmse[port]**2)), ' \n')
    
from linearmodels.asset_pricing import TradedFactorModel
for port in portfolio_names[:-1]:
    for fac in [ff3s[['mkt']], ff3s[['mkt' , 's_min']], ff3s[['mkt' , 'sks']],
                ff3s[['mkt' , 'smb', 'hml']]
                , ff3s[['mkt' , 'smb', 'hml', 's_min']], ff3s[['mkt' , 'smb', 'hml', 'sks']]]:
        mod = TradedFactorModel(portfolio_dic[port], fac)
        res = mod.fit(cov_type='kernel')
        print(res)
        print('betas \n', res.betas)



# =============================================================================
#%% Individual assets
# =============================================================================
title('Individual assets')
ts_estimates_df = pd.DataFrame(columns=returns.columns)
samples = {}
#samples['full'] = returns

ts_betas = {}
ts_stdv = {}

returns

# Sample division
for l, u, name in zip([0, 24, 60, 90, 0], [24, 60, 90, len(returns) + 1, len(returns) + 1],
                      ['T_24', '24_T_60', '60_T_90', '90_T', 'full']):
    print(name)
    test = ((returns.notna().sum()  >= l) & (returns.notna().sum()  < u))
    test2 = test[test == True]
    permnos = list(test2.index)
    samples[name] = returns[permnos]
    print('number of assets:', len(permnos))
    ts_betas[name] = {}
    ts_stdv[name] = {}
    for mname in ['FF3', 'FF3, S-', 'FF3, sks']:
        ts_betas[name][mname] = pd.DataFrame(columns=permnos)
        ts_stdv[name][mname] = pd.DataFrame(columns=permnos, index=['stdv'], dtype=float)
   
    print()

for s in  ['T_24', '24_T_60', '60_T_90', '90_T', 'full']:
    title(s)
    print('Total:', samples[s].shape[1])
    i = 1
    for r in samples[s]:
        ret = samples[s][r].dropna()
        fac = ff3s.loc[ret.index]
        
        for model, mname in zip([FamaFrench3(ret, fac[ff3_list]),
                                 FF3F1(ret, fac[ff3_list + ['s_min']]),
                                 FF3F1(ret, fac[ff3_list + ['sks']])],
                                ['FF3', 'FF3, S-', 'FF3, sks']):
            reg_res = model.timeseries_regression()
            ts_betas[s][mname][r] = reg_res.estimates_df().loc[r]
            ts_stdv[s][mname][r]['stdv'] = reg_res.residuals[r].std()
        print(i, end="")
        i = i + 1
        
        
dataprocessor.save_to_pickle('ts_betasV2', ts_betas)
dataprocessor.save_to_pickle('ts_stdvV2', ts_stdv)
  
ts_betas1 = dataprocessor.open_from_pickle('ts_betasV2')
ts_stdv1 = dataprocessor.open_from_pickle('ts_stdvV2')

wls_res = {}

for model in ['FF3', 'FF3, S-', 'FF3, sks']:
    wls_res[model] = {}
    for s in  ['T_24', '24_T_60', '60_T_90', '90_T', 'full']:
        X = ts_betas1[s][model].iloc[1:, :].T
        y = samples[s].mean() * 100

        w = pd.DataFrame(1 / ts_stdv1[s][model].T, dtype=float)

        test = sm.WLS(y, X, weights=w.values.T[0])
        res = test.fit()
        wls_res[model][s] = res

for model in ['FF3', 'FF3, S-', 'FF3, sks']:
    print(model)
    for s in  ['T_24', '24_T_60', '60_T_90', '90_T', 'full']:
        print(s)
#        print(wls_res[model][s].params)
#        print(wls_res[model][s].bse)
#        print()
        print(wls_res[model][s].summary())
        wls_res[model][s].summary()



#
#for i in returns:
#    ret = returns[i].dropna()
#    fac = ff3s.loc[ret.index]
#    
#    model = FF3F1(ret, fac[ff3_list + ['sks']])
#    reg_res = model.timeseries_regression()
#    ts_estimates_df[i] = reg_res.estimates_df().loc[i]
##dataprocessor.save_to_pickle('ts_indv_estimates', ts_estimates_df)
##ts_estimates_df = dataprocessor.open_from_pickle('ts_indv_estimates')
##print('Time-series-results \n', ts_estimates_df.T.mean())
##print('Correlation\n', ts_estimates_df.T.corr())
#
#full_sample = returns.iloc[:, :10].copy()
#full_sample
#
#for i in full_sample:
#    ret = returns[i].dropna()
#    fac = ff3s.loc[ret.index]
#    
#    model = FF3F1(ret, fac[ff3_list + ['sks']])
#    reg_res = model.timeseries_regression()
#    ts_stdv['full'][i]['stdv'] = reg_res.residuals[i].std()
#
#












   

#from linearmodels.asset_pricing import LinearFactorModel
#LinearFactorModel(full_sample, ff3s[ff3_list])


# =============================================================================
#%% test previous results 
# =============================================================================
"""
from linearmodels.datasets import french
data = french.load()

data = data.set_index('dates')['07-1963':'12-1993']

# Time series test
from linearmodels.asset_pricing import TradedFactorModel
portfolios = data[['S1V1','S1V3','S1V5','S3V1','S3V3','S3V5','S5V1','S5V3','S5V5']]
factors = data[['MktRF']]
mod = TradedFactorModel(portfolios)
res = mod.fit(cov_type='kernel')
print(res)
print('betas \n', res.betas)

res.full_summary


mine = CAPM(portfolios, factors)
mine_res = mine.timeseries_regression()
print('my betas', mine_res.betas)


# Cross-section (Fama-Mcbeth)
factors = data[['MktRF', 'SMB', 'HML']]
portfolios = data[['S1V1','S1V3','S1V5','S5V1','S5V3','S5V5']].copy()
portfolios.loc[:,:] = portfolios.values - data[['RF']].values

from linearmodels.asset_pricing import LinearFactorModel
mod = LinearFactorModel(portfolios, factors)
res = mod.fit(cov_type='kernel')
print(res, '\n')

test = FamaFrench3(portfolios, factors)

print('my est', test.Fama_Mcbeth().mean())


# GMM
from linearmodels.asset_pricing import LinearFactorModelGMM
mod = LinearFactorModelGMM(portfolios, factors)
res = mod.fit()
print(res)

"""







