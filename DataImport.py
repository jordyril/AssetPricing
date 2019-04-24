"""
Created on Sat Apr 13 12:40:57 2019

@author: jrilla
"""

# =============================================================================
#%% Package importing
# =============================================================================
import pandas as pd
import wrds
import pandas_datareader.data as web
import numpy as np
#import os
"""
## Change local directory - Windows: change C:\ into C:/
# directory work desktop
directory = ("C:/Users\jrilla\OneDrive - bf.uzh.ch"
             "\Courses\Asset pricing\Project")

# directory home laptop
directory = ("D:/OneDrive - bf.uzh.ch\Courses"
             "\Asset pricing\Project")

os.chdir(directory)
"""

# Own files
from DataProcessor import DataProcessor

# Main settings file
from Settings import general_settings, data_suffix
from functions_support import title


# =============================================================================
#%% DATA IMPORT 
# =============================================================================
start = general_settings['start']
end = general_settings['end']

new_download = general_settings['new_download']
log_returns = general_settings['log_returns']

# =============================================================================
#%% WDRS - CRSP + FAMA-FRENCH
# =============================================================================
# Create daterange
daterange = pd.date_range(start, end, closed='right', freq='M')

# Inititalize DataProcessor object
dataprocessor = DataProcessor()

#download data
if new_download:

    # Connect to wdrs
    db = wrds.Connection(wrds_username='jordyril')
    
    #lists all libraries
    db.list_libraries() 
    
    # lists all tables withing library
    db.list_tables('ff')
    
    # describes table within library
    db.describe_table(library="crsp", table='msi')
    
    # describes table within library
    #data = db.get_table(library='crsp', table='msf', obs=10)


    columns = 'permno, date, prc, ret, shrout'
    database = 'crsp.msf'
    crsp = db.raw_sql('select ' + columns + ' from ' + database,
                      date_cols=['date'])
    
    dataprocessor.save_to_pickle('crsp_msf' + data_suffix, crsp)
    print('Downloading CRSP_msf: done')
    
else:
#    crsp = dataprocessor.open_from_pickle('crsp_msf' + data_suffix)
    crsp = dataprocessor.open_from_pickle('crsp_msf_07011930_12312018')
   
#reshape and select correct timewindow
crsp = crsp.pivot(index='date', columns='permno')[start:end]
crsp.index = daterange


# Market portfolio - some differences with french
"""
if new_download:
    columns = 'date, vwretd'
    database = 'crsp.msi'
    mkt = db.raw_sql('select ' + columns + ' from ' + database,
                      date_cols=['date'])
    dataprocessor.save_to_pickle('crsp_mkt', mkt)
    
else:
    mkt = dataprocessor.open_from_pickle('crsp_mkt')

mkt = mkt.set_index('date')[start:end]
mkt.index = daterange #set index
mkt = mkt.rename(columns={'vwretd':'mkt'}) #change columname
"""

## Fama-French: 4 factors (mkt, smb, hml, umd) - from wrds db
if new_download:
    columns = 'smb, hml, umd, mktrf , date, rf'
    database = 'ff.factors_monthly'
    wrds_ff = db.raw_sql('select ' + columns + ' from ' + database,
                      date_cols=['date'])
    
    wrds_ff = wrds_ff.set_index('date')
    wrds_ff = wrds_ff.rename(columns={'mktrf': 'mkt'})
    dataprocessor.save_to_pickle('crsp_ff' + data_suffix, wrds_ff)
    
    print('Downloading CRSP_ff: done')    
    
else:
#    wrds_ff = dataprocessor.open_from_pickle('crsp_ff' + data_suffix)
    wrds_ff = dataprocessor.open_from_pickle('crsp_ff_07011930_12312018')

# Select correct time window
wrds_ff = wrds_ff[start:end]
wrds_ff.index = daterange

## Riskfree asset (same as through French website)
#rf = wrds_ff['rf']
rf = dataprocessor.open_from_pickle('alternative_rf')['Riskfree Rate'][start:end] / 100

# Rename factors
factor_names = ['mkt', 'smb', 'hml', 'umd']
factors = wrds_ff[factor_names]

# listing all returns and making them excess returns
crsp['ret'] = crsp['ret'].subtract(rf, axis=0) 
returns = crsp['ret']

# size dataframe
size_df = np.abs(crsp['prc']) * crsp['shrout']

# =============================================================================
#%% Fama-French - FACTORS + PORTFOLIOS
# =============================================================================
## Fama-French: 3 factors (mkt, smb, hml)
"""
# two methods, through french's website or wrds => exact same series
# Monthly data is needed (index = 0), data in pct so we divide it by 100
FFF = web.DataReader(name='F-F_Research_Data_Factors', 
                     data_source='famafrench',
                     start=start, end=end)[0] / 100
"""
###  5 portfolio groups - index[0] (monthly + VW) - pct (divide by 100)
## 1) 30 VW Industry portfolios
if new_download:
    industry_port = web.DataReader(name='30_Industry_Portfolios', 
                              data_source='famafrench',
                              start=start, end=end)[0] / 100
                 
    # subtract Riskfree rate
    industry_port.index = daterange
    industry_port = industry_port.subtract(rf, axis=0)
    
    #save
    dataprocessor.save_to_pickle('ff_industries' + data_suffix, industry_port)
    
    print('Downloading ff_industries: done')
    

    ## 2) 25 VW FF portfolios sorted on size and B/M
    size_bm_port = web.DataReader(name='25_Portfolios_5x5', 
                                  data_source='famafrench',
                                  start=start, end=end)[0] / 100
    
    # subtract Riskfree rate - rename certain columns
    size_bm_port.index = daterange
    size_bm_port = size_bm_port.subtract(rf, axis=0)
    size_bm_port = size_bm_port.rename(columns={'SMALL LoBM':'ME1 BM1',
                                                'SMALL HiBM':'ME1 BM5',
                                                'BIG LoBM':'ME5 BM1',
                                                'BIG HiBM':'ME5 BM5'})
    
    #save
    dataprocessor.save_to_pickle('ff_size_bm' + data_suffix, size_bm_port)
    print('Downloading ff_size_bm: done')
    
    ## 3) 10 VW Momentum portfolios
    mom_port = web.DataReader(name='10_Portfolios_Prior_12_2', 
                                  data_source='famafrench',
                                  start=start, end=end)[0] / 100
                              
    # subtract Riskfree rate - rename certain columns
    mom_port.index = daterange
    mom_port = mom_port.subtract(rf, axis=0) 
    mom_port.columns = ['MOM' + str(i) for i in range(1,11)]   

    #save
    dataprocessor.save_to_pickle('ff_momentum' + data_suffix, mom_port) 
    print('Downloading ff_mom: done')
                   
    
    ## 4) 10 Size deciles
    size_port = web.DataReader(name='Portfolios_Formed_on_ME', 
                                  data_source='famafrench',
                                  start=start, end=end)[0] / 100
    
    size_port = size_port[['Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5', 
                           'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']]
    
    # subtract Riskfree rate - rename certain columns
    size_port.index = daterange
    size_port = size_port.subtract(rf, axis=0)      
    size_port.columns = ['DEC' + str(i) for i in range(1,11)]     

    #save
    dataprocessor.save_to_pickle('ff_size' + data_suffix, size_port) 
    print('Downloading ff_size: done')               
                           
    ## 5) 27 three-way sort, Size-B/M-Momentum (Carhart (1997))
    #TODO not on French  

else:
#    industry_port = dataprocessor.open_from_pickle('ff_industries' 
#                                                   + data_suffix)
#    size_bm_port = dataprocessor.open_from_pickle('ff_size_bm' + data_suffix)
#    mom_port = dataprocessor.open_from_pickle('ff_momentum' + data_suffix)
#    size_port = dataprocessor.open_from_pickle('ff_size' + data_suffix)
    
    industry_port = dataprocessor.open_from_pickle('ff_industries' +
                                                   '_07011930_12312018')
    size_bm_port = dataprocessor.open_from_pickle('ff_size_bm_07011930_12312018')
    mom_port = dataprocessor.open_from_pickle('ff_momentum_07011930_12312018')
    size_port = dataprocessor.open_from_pickle('ff_size_07011930_12312018')
    
    industry_port = industry_port[start:end]
    size_bm_port = size_bm_port[start:end]
    mom_port = mom_port[start:end]
    size_port = size_port[start:end]

if log_returns:
    returns = np.log(returns + 1)
    factors = np.log(factors + 1)
    
    industry_port = np.log(industry_port+ 1)
    size_port = np.log(size_port + 1)
    size_bm_port = np.log(size_bm_port + 1)
    mom_port = np.log(mom_port + 1)


# Create dictionary with all portfoliogroups
portfolio_dic = {}
for port, port_name in zip([size_bm_port, size_port, mom_port, industry_port, 
                            returns.dropna(axis=1)],
                          ['Size and B/M', 'Size', 'Momentum', 'Industry', 
                           'Individual']):
    portfolio_dic[port_name] = port


    
    
    
                


