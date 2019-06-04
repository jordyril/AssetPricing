"""
Created on Tue Apr 16 10:26:17 2019

@author: jrilla
"""

# =============================================================================
# Packages 
# =============================================================================

# =============================================================================
# DATACLEANER class
# =============================================================================
class DataCleaner(object):
    def __init__(self):
        pass
    
    def split_in_sample_out_sample(self, full_sample, out_sample_window=1):
        in_sample = full_sample.iloc[:-out_sample_window]
        out_sample = full_sample.iloc[-out_sample_window:]
        return in_sample, out_sample
    
class DropAllNan(DataCleaner):
    def __call__(self, full_sample, axis=1):
        cleaned_sample = full_sample.dropna(axis=1)
        return cleaned_sample
        