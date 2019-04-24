# =============================================================================
# PACKAGES
# =============================================================================
import statsmodels.api as sm

from dask import delayed, compute
import dask

from numba import types, jitclass, njit
# =============================================================================
# FUNCTIONS
# =============================================================================

def title(title):
    print('\n' + '=' * 80)
    title = '|' + title + '|'
    print("{0: ^80}".format(title))
    print('=' * 80)
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