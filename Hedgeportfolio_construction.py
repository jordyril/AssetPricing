"""
Created on Mon Apr 15 14:09:15 2019

@author: jrilla
"""

# =============================================================================
#%% Package importing
# =============================================================================
#import os

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
import numpy as np

# Own files
#from latexclasses import *
from Criterium import Skewness
from SkewnessClasses import BetaSKD

from FactorPortfolio import LongShortPortfolios

from DataProcessor import DataProcessor

from Settings import hedge_port_settings, data_suffix, general_settings


#from DataCleaner import DropAllNan
# Data import
from DataImport import returns, size_df, factors

#from AssetSelection import High, Low
# Inititate assiting classes
dataprocessor = DataProcessor()

# =============================================================================
#%% Main data
# =============================================================================
start = general_settings['start'] 
end = general_settings['end'] 
log_returns = general_settings['log_returns']

mkt = factors['mkt']

in_sample = hedge_port_settings['in_sample_lenght']
out_sample = len(returns) - in_sample
fraction = hedge_port_settings['fraction']

returns
# =============================================================================
# Construct portofolio
# =============================================================================
#### THIS HAS TO BE CHANGED ####
#returns = returns.dropna(axis=1)
#size_df = size_df.dropna(axis=1)[returns.columns]
#%%
if hedge_port_settings['new_computation']:
        
    skewness_criterium = Skewness(BetaSKD, factor_series=mkt) 
    hedgeportfolios = LongShortPortfolios(skewness_criterium, 
                                          returns, 
                                          sizes=size_df, 
                                          in_sample_window=in_sample)()
#    test = LongShortPortfolios(skewness_criterium, returns, sizes=size_df, 
#                                          in_sample_window=in_sample)()
#    test = OneFactorportfolio(skewness_criterium, returns, High, sizes=size_df,
#                              in_sample_window=in_sample)()
#    test1 = OneFactorportfolio(skewness_criterium, returns, Low, sizes=size_df,
#                              in_sample_window=in_sample)()

    dataprocessor.save_to_pickle('hedgeportfolios' + data_suffix,
                                 hedgeportfolios)
else:
#    hedgeportfolios = dataprocessor.open_from_pickle('hedgeportfolios' 
#                                                     + data_suffix)
    hedgeportfolios = dataprocessor.open_from_pickle('hedgeportfolios_07011930_12312018')

    hedgeportfolios = hedgeportfolios[start:end]

if log_returns:
    hedgeportfolios = np.log(hedgeportfolios + 1)


