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
import numpy as np

from DataImport import (size_bm_port, size_port, mom_port, industry_port,
                        factors)
from Settings import general_settings

start = general_settings['start']
end = general_settings['end']

from functions_support import title
from SkewnessClasses import (BetaSKD, BetaSquared, 
                             BetaNegSkewPort, BetaLongShortPort)

from Hedgeportfolio_construction import hedgeportfolios
from FactorModel import CAPM

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
title('Average excess return')

for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                          ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    print(port.mean() * multiplier, '\n')
    

print('factors')
for f in factors:
    print(f)
    print(factors[f].mean() * multiplier, '\n')

print('hedgeportfolios')
for h in hedgeportfolios:
    print(h)
    print(hedgeportfolios[h].mean() * multiplier, '\n')

title('STDV')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    print(port.std() * multiplier, '\n')
    
title('Unconditional Skew')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    print(port.skew(), '\n')
    
# Beta SKD
title('Beta_SKD')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    skewness = BetaSKD(port, factors['mkt'])
    betas = skewness()
    print(betas, '\n')

# Beta squared
title('Beta squared')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    skewness = BetaSquared(port, factors['mkt'])
    betas = skewness()
    print(betas, '\n')

# Beta LONG-SHORT
title('Beta LONG-SHORT')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    skewness = BetaLongShortPort(port, LS_S)
    betas = skewness()
    print(betas, '\n')
    
# Beta Neg SKEW
title('Beta Neg SKEW')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    skewness = BetaNegSkewPort(port, S_min)
    betas = skewness()
    print(betas, '\n')

# Beta CAPM
title('Beta CAPM')
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port],
                           ['Size and B/M', 'Size', 'Momentum', 'Industry']):
    print(port_name)
    capm = CAPM(port, factors['mkt'])
    betas = capm.timeseries_regression(True)
#    betas = capm.betas
    print(betas.T, '\n')

# =============================================================================
# Kernel plot - Figure 2 (1)
# =============================================================================
def standardise(series):
    standardised_series = (series - series.mean()) / series.std()
    return standardised_series


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
# TABLE II (1)
# =============================================================================
from TimeSeriesRegressions import TS_regression_results



