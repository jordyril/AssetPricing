"""
Created on Sat Apr 13 18:39:53 2019

@author: jrilla
"""
# =============================================================================
# CRITERIUM CLASS
# =============================================================================
class Criterium(object):
    def __init__(self):
        pass

class ClassCriterium(Criterium):
    def __init__(self, criterium_class):
        Criterium.__init__(self)
        self.criterium = criterium_class
        
class Skewness(ClassCriterium):
    """
    Criteriumclass accepting one of the CoskewnessClass classes as a criterium
    """
    def __init__(self, Coskewness_class, factor_series, returns_df=None):
#        self.criterium = Coskewness_class
        ClassCriterium.__init__(self, Coskewness_class)
        self.returns = returns_df
        self.factor = factor_series
        
    def _slice_factor(self, start=0, end=-1):
        try:
#            return self.returns.iloc[start:end], self.factor.iloc[start:end]
            return self.factor.iloc[start:end]
        except AttributeError:
            print("No returns and/or factor defined")
            return None
            
#        return self.returns.iloc[start:end], self.factor.iloc[start:end]
    
    def __call__(self, returns_df, start=0, end=-1):
           
#        returns, factor = self._slice(start, end)
        factor = self._slice_factor(start, end)
        
        return self.criterium(returns_df, factor)()