"""
Created on Sat Apr 13 18:41:59 2019

@author: jrilla
"""
# =============================================================================
# Packages import
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# Classes
# =============================================================================
class Weighting(object):
    """
    Weightingclass
    """
    def __init__(self, sizes=None):
        self.sizes= sizes
    
    def _set_settings(self, selection):
        self._selection = selection
        self.N = len(self._selection)
    
class EquallyWeighted(Weighting):
    def __call__(self, selection, position=None):
        self._set_settings(selection)
        
        weights = np.ones(self.N)
        
        weights /= self.N
        
        return pd.Series(weights, index=self._selection)
    
class ValueWeighted(Weighting):
    def __call__(self, selection, position):
        self._set_settings(selection)
        
        relevant_sizes = self.sizes[self._selection].iloc[position]
        
        total_size = relevant_sizes.sum()
        
        weights = relevant_sizes / total_size
        
        return pd.Series(weights, index=self._selection)
