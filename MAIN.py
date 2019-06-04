"""
Created on Wed Apr 24 11:02:41 2019

@author: jrilla

3 papers trying to be replicated 

(1) Harvey, C. R., & Siddique, A. (2000). 
    Conditional skewness in asset pricing tests. 
    The Journal of Finance, 55(3), 1263-1295.
(2) 
(3) 

"""

# =============================================================================
# PACKAGES
# =============================================================================
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as scs

import numpy as np
import pandas as pd
import sys
import os

## Change local directory - Windows: change C:\ into C:/
# directory work desktop
directory = ("C:/Users\jrilla\OneDrive - bf.uzh.ch"
             "\Courses\Asset pricing\Project\AssetPricing")

# directory home laptop
#directory = ("D:/OneDrive - bf.uzh.ch\Courses"
#             "\Asset pricing\Project\AssetPricing")

os.chdir(directory)
sys.path.insert(0, directory)

#%%
from DataImport import (size_bm_port, size_port, mom_port, industry_port,
                        factors, rf, returns)
from statsmodels.stats.sandwich_covariance import cov_hac
from Settings import general_settings

start = general_settings['start']
end = general_settings['end']

from functions_support import title, subtitle
from SkewnessClasses import (BetaSKD, BetaSquared, 
                             BetaNegSkewPort, BetaLongShortPort)

from Hedgeportfolio_construction import hedgeportfolios
from FactorModel import CAPM

from latexclasses import LatexTable, LatexValue
latextable = LatexTable()
latexvalue = LatexValue()

# =============================================================================
# Functions
# =============================================================================
def Newey_West_se(OLS_results, lags=12):
    se = np.sqrt(np.diag(cov_hac(OLS_results, nlags=lags)))
    return se

def standardise(series):
    standardised_series = (series - series.mean()) / series.std()
    return standardised_series

def brace(a):
    return '(' + a + ')' 

def curlybrace(a):
    return '{' + a + '}' 

def latex_thead(*args):
    s =  "\\thead{" 
    for i in args[:-1]:
        s = s + str(i) + " \\\ "
    s = s + str(args[-1]) + '}'
    
    return s


def latex_thead_se(coef, se, rounding=3):
    r = str(rounding)
    fmt = "{0:." + r + "f}"
    
    coef_str = fmt.format(coef)
    se_str = fmt.format(se)
    se_str = brace(se_str)
    
    p =  latex_thead(coef_str, se_str)
    
    return p

def latex_significance(coef, se, df_t, rounding=3):
    r = str(rounding)
    fmt = "{0:." + r + "f}"
    
    t_stat = coef / se
    t_90, t_95, t_99 = scs.t.ppf([0.90, 0.95, 0.99], df_t)
    
    if np.abs(t_stat) > t_90:
        coef_str = fmt.format(coef) + '$^{*}$'
        se_str = fmt.format(se)
        se_str = brace(se_str)
        se_str = se_str 
        
    if np.abs(t_stat) > t_95:
        coef_str = fmt.format(coef) + '$^{**}$'
        se_str = fmt.format(se)
        se_str = brace(se_str)
        se_str = se_str 
        
    if np.abs(t_stat) > t_99:
        coef_str = fmt.format(coef) + '$^{***}$'
        se_str = fmt.format(se)
        se_str = brace(se_str)
        se_str = se_str 
    
    else:
        coef_str = fmt.format(coef)   
        
        se_str = fmt.format(se)
        se_str = brace(se_str)
            
    p =  latex_thead(coef_str, se_str)
    
    return p
# =============================================================================
# DATA
# =============================================================================
S_pl = hedgeportfolios['High']                  
S_min = hedgeportfolios['Low']
LS_S = hedgeportfolios['LH-LS']

pp = True
annualized = False

multiplier = 1
if pp:
    multiplier *= 100
if annualized:
    multiplier *= 12
# =============================================================================
#%% Summary Statistics - TABLE 1 (1)
# =============================================================================
title('Summary Statistics') 
print(start + ' - ' + end)
subtitle('Average excess return')

for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                          ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    print(port.mean() * multiplier, '\n')    

print('factors')
for f in factors:
    print(f)
    print(factors[f].mean() * multiplier, '\n')
    print('Annualized', factors[f].mean() * multiplier, '\n')

print('hedgeportfolios')
for h in hedgeportfolios:
    print(h)
    print(hedgeportfolios[h].mean() * multiplier, '\n')

subtitle('STDV')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    print(port.std() * multiplier, '\n')
    
subtitle('Unconditional Skew')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    print(port.skew(), '\n')
    
# Beta SKD
subtitle('Beta_SKD')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    skewness = BetaSKD(port, factors['mkt'])
    betas = skewness()
    print(betas, '\n')

# Beta squared
subtitle('Beta squared')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    skewness = BetaSquared(port, factors['mkt'])
    betas = skewness()
    print(betas, '\n')

# Beta LONG-SHORT
subtitle('Beta LONG-SHORT')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    skewness = BetaLongShortPort(port, LS_S)
    betas = skewness()
    print(betas, '\n')
    
# Beta Neg SKEW
subtitle('Beta Neg SKEW')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    skewness = BetaNegSkewPort(port, S_min)
    betas = skewness()
    print(betas, '\n')

# Beta CAPM
subtitle('Beta CAPM')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    capm = CAPM(port, factors['mkt'])
    betas = capm.timeseries_regression().estimates_df()
#    betas = capm.betas
    print(betas, '\n')


### SLIDE TABLE ####
# test portfolios
presentation_portfolios = [size_bm_port, size_port, mom_port, industry_port]
presentation_portfolio_keys = ['Size and B/M', 'Size', 'Momentum', 'Industry']
summary_stats = {}
se = {}

for port, portkey in zip(presentation_portfolios, presentation_portfolio_keys):
    # MEAN
    summary_stats[portkey] = pd.DataFrame(port.mean() * multiplier, 
                                 index=port.columns, 
                                 columns=['$\\mu^{e}$'])
    
    se[portkey] = pd.DataFrame(index=port.columns)
    
    # SDTV
    summary_stats[portkey]['$\\sigma$'] = port.std() * multiplier
    
    # SKEW
    summary_stats[portkey]['Skew'] = port.skew()
    
    # CAPM BETA
    capm = CAPM(port, factors['mkt'])
    reg_res = capm.timeseries_regression_statsmodels()
    betas = reg_res.betas
    summary_stats[portkey]['\\thead{$\\hat{\\beta}_{M}$}'] = betas[0, :-capm.K]
    se[portkey]['\\thead{$\\hat{\\beta}_{M}$}'] = 0
    for i in se[portkey].index:
        se[portkey].loc[i, '\\thead{$\\hat{\\beta}_{M}$}'] = Newey_West_se(reg_res.statsmodels_results[i])[1]
#    se[portkey]['\\thead{$\\hat{\\beta}_{M}$}'] = reg_res.standard_errrors()[1][:-capm.K]
    
    # BETA Squared
    skewness = BetaSquared(port, factors['mkt'])
    betas = skewness()
    summary_stats[portkey]['\\thead{$\\hat{\\beta}_{M^{2}}$}'] = betas / 100
    se[portkey]['\\thead{$\\hat{\\beta}_{M^{2}}$}'] = 0
    for i in se[portkey].index:
        se[portkey].loc[i, '\\thead{$\\hat{\\beta}_{M^{2}}$}'] = Newey_West_se(skewness._reg_results[i])[1] / 100
#        se[portkey].loc[i, '\\thead{$\\hat{\\beta}_{M^{2}}$}'] = skewness._reg_results[i].bse[1] / 100
    
    # BETA SKD
    skewness = BetaSKD(port, factors['mkt'])
    betas = skewness()
    summary_stats[portkey]['\\thead{$\\hat{\\beta}_{SKD}$}'] = betas
#    se[portkey]['\\thead{$\\hat{\\beta}_{SKD}$}'] = 0
#    for i in se[portkey].index:
#        se[portkey].loc[i, '\\thead{$\\hat{\\beta}_{SKD}$}'] = skewness._reg_results[i].bse[1] 
    
    # BETA SKS
    skewness = BetaLongShortPort(port, LS_S)
    betas = skewness()
    summary_stats[portkey]['\\thead{$\\hat{\\beta}_{SKS}$}'] = betas
    se[portkey]['\\thead{$\\hat{\\beta}_{SKS}$}'] = 0
    for i in se[portkey].index:
        se[portkey].loc[i, '\\thead{$\\hat{\\beta}_{SKS}$}'] = Newey_West_se(skewness._reg_results[i])[1]
#        se[portkey].loc[i, '\\thead{$\\hat{\\beta}_{SKS}$}'] = skewness._reg_results[i].bse[1] 

    # BETA S-
    skewness = BetaNegSkewPort(port, S_min)
    betas = skewness()    
    summary_stats[portkey]['\\thead{$\\hat{\\beta}_{S^{-}}$}'] = betas
    se[portkey]['\\thead{$\\hat{\\beta}_{S^{-}}$}'] = 0
    for i in se[portkey].index:
        se[portkey].loc[i, '\\thead{$\\hat{\\beta}_{S^{-}}$}'] = Newey_West_se(skewness._reg_results[i])[1]
#        se[portkey].loc[i, '\\thead{$\\hat{\\beta}_{S^{-}}$}'] = skewness._reg_results[i].bse[1] 
    
   # EXPORT TO LATEX
    portname = str(port.shape[1]) + ' ' + portkey + ' portfolios'
    table_name = 'sumstats_' + portname.replace(' ', '').replace('/', '')
    table_caption = 'Summary statistics for ' + portname
    latextable(summary_stats[portkey], table_name, table_caption, **{'to_latex':{'escape':False}})



rounding = 3
sum_sbm = pd.concat([summary_stats['Size and B/M'].iloc[:5], summary_stats['Size and B/M'].iloc[-5:]])
se_sbm = pd.concat([se['Size and B/M'].iloc[:5], se['Size and B/M'].iloc[-5:]])

f = pd.DataFrame(dtype=str).reindex_like(sum_sbm)
for i in se_sbm:
    for j in sum_sbm[i].index:
        f.loc[j, i] = latex_thead_se(sum_sbm[i][j], se_sbm[i][j])
for i in ['$\\mu^{e}$', '$\\sigma$', 'Skew']:
    for j in sum_sbm[i].index:
        r = str(rounding)
        fmt = "{0:." + r + "f}"
        f.loc[j, i] = latex_thead(fmt.format(sum_sbm[i][j]), '')

table_name = 'sumstats_extreme_s_bm' 
table_caption = 'Summary statistics for 10 Size and B/M portfolios' 
latextable(f, table_name, table_caption, **{'to_latex':{'escape': False}})       
    
rounding = 3
sum_sbm = summary_stats['Size']
se_sbm = se['Size']


f = pd.DataFrame(dtype=str).reindex_like(sum_sbm)
for i in se_sbm:
    for j in sum_sbm[i].index:
        f.loc[j, i] = latex_thead_se(sum_sbm[i][j], se_sbm[i][j])
for i in ['$\\mu^{e}$', '$\\sigma$', 'Skew', '\\thead{$\\hat{\\beta}_{SKD}$}']:
    for j in sum_sbm[i].index:
        r = str(rounding)
        fmt = "{0:." + r + "f}"
        f.loc[j, i] = latex_thead(fmt.format(sum_sbm[i][j]), '')

table_name = 'sumstats_extreme_s' 
table_caption = 'Summary statistics for 10 Size portfolios' 
latextable(f, table_name, table_caption, **{'to_latex':{'escape': False}})   


# Factors
factors_rf = pd.concat([factors, rf], axis=1)
factor_stats = pd.DataFrame(factors_rf.mean().values * multiplier, 
                            index=([i.upper() for i in factors.columns] + ['RF']), 
                            columns=['$\\mu^{e}$'])

factor_stats['$\\sigma$'] = (factors_rf * multiplier).std().values

table_name = 'sumstats_factors' 
table_caption = 'Average monthly return and STDV for Carhart factors and RF rate'
latextable(factor_stats.T, table_name, table_caption, **{'to_latex':{'escape': False}})

latextable.create_tabular_file(factor_stats.T, table_name, **{'escape': False})

# returns
#returns_desr = returns.describe().mean(axis=1)
#
#returns_stats = pd.DataFrame(index=['CRSP'], columns=returns_desr.index)
#
#for i in returns_stats:
#    print(i)
#    if i == 'count':
#        returns_stats[i] = returns_desr[i]
#    else:
#        returns_stats[i] = returns_desr[i] * multiplier
#    
#returns_stats.columns = ['N', '$\\mu^{e}$', '$\\sigma$', 'Min', 'Q1', 'Q2', 'Q3', 'Max']
#
#table_name = 'sumstats_returns' 
#table_caption = 'Summary statistics CRSP returns'
#latextable(returns_stats, table_name, table_caption, **{'to_latex':{'escape': False}})

# Aveareg spread hedgeportfolios
latexvalue(LS_S.mean() * multiplier, 'average_spread_S')
# =============================================================================
#%% Kernel plot - Figure 2 (1)
# =============================================================================
for port in [size_port['DEC1'], size_port['DEC10'], 
             size_bm_port['ME1 BM5'], size_bm_port['ME5 BM1']]:
    test = standardise(port)
    sns.kdeplot(test)
    x = np.linspace(-6, 6, 1000)
    y = norm.pdf(x)
    plt.plot(x, y, label='N(0, 1)')
    plt.xlim(-4, 4)
    plt.ylim(0, 0.5)
    plt.legend()
    plt.show()
    
# =============================================================================
# Analysis results
# =============================================================================
from Analysis import TS_regression_results, fm_res, CR_res_dic, wls_res, samples

# Time series
grs_df = pd.DataFrame(columns=pd.MultiIndex.from_product([presentation_portfolio_keys, 
                                                          list(TS_regression_results[presentation_portfolio_keys[0]]['ts'].keys())]),
                       index=pd.MultiIndex.from_product([['New', 'Original'], list(TS_regression_results[presentation_portfolio_keys[0]]['ts']['3F'].keys())]),
                       dtype=float)

for keys in presentation_portfolio_keys:
#    grs_df = pd.DataFrame(columns=list(TS_regression_results[keys]['ts'].keys()), 
#                          index=list(TS_regression_results[keys]['ts']['3F'].keys()),
#                          dtype=float)
    
    for c in grs_df[keys]:
#        print(c)
        for i in grs_df[keys][c].index:
#            print(i)
#            print(TS_regression_results[keys]['ts'][c][i[1]])
            grs_df[(keys, c)]['New'][i[1]] = TS_regression_results[keys]['ts'][c][i[1]]
 
    
    
    
grs_df.columns.set_levels(['$FF3$', '$FF3+S^{-}$'],level=1, inplace=True)    
idx = pd.IndexSlice
grs_df = grs_df.loc[idx[:, ['F-test', 'p-value']], idx[:, :]]
    
table_name = 'grs_test'
table_caption = 'GRS test results'
latextable(grs_df, table_name, table_caption, rounding=4, 
           **{'to_latex':{'escape':False, 'multicolumn':True, 
                          'multicolumn_format':'c', 'column_format':'ll|cccccccc',
                          'multirow':True,}})
#%%Fama- Mcbeth
#### Price error correlations
for keys in presentation_portfolio_keys:
    valuename = 'PE_corr' + keys.replace(' ', '').replace('/', '') 
    latexvalue(fm_res['corr'].loc[keys][0], valuename, 3)
    
##### RISK PREMIA - Adjusted R^2 
r_squared_df = pd.DataFrame(index=presentation_portfolio_keys, 
                            columns=CR_res_dic[presentation_portfolio_keys[0]].columns,
                            dtype=float)
for keys in presentation_portfolio_keys:
    r_squared_df.loc[keys] = CR_res_dic[keys].mean() * 100
    

r_squared_df.columns = ['CAPM', 'CAPM + $S^{-}$', 'CAPM + SKS', 'FF3', 'FF3 + $S^{-}$', 'FF3 + SKS']  
table_caption = 'Adjusted $R^{2}$ for different model specifications for the different testing portfolio groups'
table_name = 'adj_R2'
latextable(r_squared_df, table_name, table_caption, rounding=2,**{'to_latex':{'escape':False, 'column_format':'l|cccccc'}})   
    


#%% Risk premia estimation
risk_premia_dfs = {}
rounding = 3
r = str(rounding)

for model in ['FF3', 'FF3, S-', 'FF3, sks']:
    rp_df = pd.DataFrame(index=['full', 'T_24', '24_T_60', '60_T_90', '90_T'],
                         columns=list(wls_res[model]['full'].params.index), dtype=str)
    
    for s in  ['T_24', '24_T_60', '60_T_90', '90_T', 'full']:
        if model == 'FF3':
            t_df = samples[s].shape[1] - 3
        else:
            t_df = samples[s].shape[1] - 4
        for i in rp_df:
#            fmt = "{0:." + r + "f}"
#            coef_est = wls_res[model][s].params[i]
#            se_est = wls_res[model][s].bse[i]
#            
#            coef_str = fmt.format(coef_est)
#            se_str = fmt.format(se_est)
#            se_str = brace(se_str)
#            
#            p =  latex_thead(coef_str, se_str)
            p = latex_significance(wls_res[model][s].params[i], 
                                   wls_res[model][s].bse[i], t_df, rounding)
            rp_df.loc[s][i] = p
    
    rp_df.index = ['Full sample', '$T < 24$', '$24 \\leq T < 60 $', '$60 \\leq T < 90 $', '$T \\geq 90$']
    rp_df = rp_df.rename(columns= {'$\\hat{\\beta}^{Mkt}$':'$\\lambda_M$',
                                   '$\\hat{\\beta}^{SMB}$':'$\\lambda_{SMB}$',
                                   '$\\hat{\\beta}^{HML}$':'$\\lambda_{HML}$'})
    if model == 'FF3, S-':
        rp_df = rp_df.rename(columns= {'$\\hat{\\beta}^{f}$':'$\\lambda_{S^{-}}$'})
    elif model == 'FF3, sks':
        rp_df = rp_df.rename(columns= {'$\\hat{\\beta}^{f}$':'$\\lambda_{SKS}$'})
    
            
    risk_premia_dfs[model] = rp_df
       
    
    table_name = 'riskpremia' + model.replace(' ', '')
    table_caption = 'Estimation of risk premia'
    latextable(rp_df, table_name, table_caption)

final_rp_df = pd.concat({'FF3': risk_premia_dfs['FF3'], 'FF3 + S$^{-}$': risk_premia_dfs['FF3, S-'], 'FF3 + SKS':risk_premia_dfs['FF3, sks']}, axis=1)
final_rp_df['N'] = 0
for s, p in zip(['full', 'T_24', '24_T_60', '60_T_90', '90_T'],
                ['Full sample', '$T < 24$', '$24 \\leq T < 60 $', '$60 \\leq T < 90 $', '$T \\geq 90$']):
    final_rp_df.loc[p, 'N'] = samples[s].shape[1]

cols = final_rp_df.columns.tolist()
cols = cols[-1:] + cols[:-1]

final_rp_df = final_rp_df[cols]

table_name = 'riskpremia'
table_caption = 'Estimation of Risk premia'
latextable(final_rp_df, table_name, table_caption, **{'to_latex':{'escape':False, 'multicolumn':True, 'multicolumn_format':'c', 'column_format':'l|c|lll|llll|llll'}})


