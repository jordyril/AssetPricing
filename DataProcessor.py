"""
Created on Sat Apr 13 12:56:08 2019

@author: jrilla
"""
# =============================================================================
#%% Package importing
# =============================================================================
import os
import pickle

# =============================================================================
#%% CLASS
# =============================================================================

class DataProcessor(object):
    def __init__(self):
        # create Data folder
        if not os.path.exists('Data'):
        	os.makedirs('Data')
            
        
    def save_to_pickle(self, filename, dic):
        """
        Saving an object to a pickle file
        """
        with open('Data/' + filename + '.pickle', 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return None
    
    def open_from_pickle(self, filename):
        """
        Opening pickle file
        """
        with open('Data/' + filename + '.pickle', 'rb') as handle:
            data = pickle.load(handle)
        return data