# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:20:26 2019

@author: jrilla
"""

# =============================================================================
#%% Settings - general data
# =============================================================================
general_settings = {}

general_settings['start'] = '07-01-1958'
general_settings['end'] = '12-31-1993'
general_settings['new_download'] = False
general_settings['log_returns'] = False

data_suffix = ('_' 
               + general_settings['start'][:2]
               + general_settings['start'][3:5]
               + general_settings['start'][-4:]
               + '_' 
               + general_settings['end'][:2] 
               + general_settings['end'][3:5] 
               + general_settings['end'][-4:]) 



# =============================================================================
#%% Hedge portfolio construction 
# =============================================================================
hedge_port_settings = {}

hedge_port_settings['in_sample_lenght'] = 60
hedge_port_settings['fraction']= 0.3

hedge_port_settings['new_computation'] = False

