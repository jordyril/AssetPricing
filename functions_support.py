# =============================================================================
# PACKAGES
# =============================================================================
import statsmodels.api as sm
import numpy.linalg as la
import numpy as np

# =============================================================================
# FUNCTIONS
# =============================================================================

def title(title):
    print('\n' + '=' * 80)
    title = '|' + title + '|'
    print("{0: ^80}".format(title))
    print('=' * 80)
    return None 

def subtitle(subtitle):
    subtitle = '|' + subtitle + '|'
    n = len(subtitle)
    print('\n' + '-' * n)
    print(subtitle)
    print('-' * n)
    return None

def OLS_regression(endo, exo, constant=True):
    """
    Performs OLS regression, returns a resultswrapper, for extra information
    on the possible attributes, 
    use dir(results) or 
    check 
    http://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.RegressionResults.html
    'https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.html#statsmodels.regression.linear_model.OLSResults' or
    'https://www.statsmodels.org/dev/index.html#' or
    'https://www.statsmodels.org/stable/regression.html'
    """
    X = exo
    y = endo
    if constant:
        X = sm.add_constant(X)

	# Fit regression model
    results = sm.OLS(y, X).fit()
    return results

class LinRegEstimationBase(object):
    def __init__(self, y, x, constant=True):
        self.y = y
        if constant:
            self.x = sm.add_constant(x)
        else:
            self.x = x
    
    def _SStot(self):
        self.SStot = np.sum((self.y - np.mean(self.y))**2)
        return None
    
    def _SSreg(self):
        self._SStot()
        self.SSreg = self.SStot - self.SSres
        return None
    
    def R_squared(self):
        return self.SSreg / self.SStot    
    
    def adj_R_squared(self):
        self._compute_SS()
        n = len(self.y)
        p = self.x.shape[1]
        
        df_t = n - 1 
        df_e = n - p
        
        return 1 - (self.SSres / df_e) / (self.SStot / df_t)
    
    def _compute_SS(self):
        try:
            self.SStot
        except AttributeError:
            self._SStot()
            
        try:
            self.SSreg
        except AttributeError:
            self._SSreg()
    
    def param(self):
        return self.b
    
    def residuals(self):
        self.residuals = self.y - self.x @ self.b
        return self.residuals
        
    
class OLS(LinRegEstimationBase):
    def __init__(self, y, x, constant=True):
        LinRegEstimationBase.__init__(self, y, x, constant)          
        self.b, self.SSres = la.lstsq(self.x, self.y, rcond=None)[:2]
      
    

class FIML(LinRegEstimationBase):
    def __init__(self, y, x, constant=True):
        LinRegEstimationBase.__init__(self, y, x, constant) 
        
    
    
    

   

   