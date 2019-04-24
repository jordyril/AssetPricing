"""
Created on Sat Apr 13 18:44:00 2019

@author: jrilla
"""

# =============================================================================
# 
# =============================================================================
class AssetSelection(object):
    """
    Based on a given pd.Series, index will be sorted and 
    top, bottom or middle percentage will be returned
    """
    def __init__(self, pandas_series, percentage=0.3):
        # Sort series
        self.ranking = pandas_series.sort_values(ascending=False)
        
        # Index of ranked series
        self.assets = list(self.ranking.index)
        
        # Total length of the series
        self._series_length = len(self.ranking)
        
        # Number of values to be selected
        self._selection_nbr = int(percentage * self._series_length)
                      
class High(AssetSelection):
    """
    Selecting those assets in the top 'percentage' of the total
    """
    def __call__(self):
        # selecting those assets in the top 'percentage' of the total
        selection = self.assets[:self._selection_nbr]
       
        return selection

class Low(AssetSelection):
    """
    Selecting those assets in the bottom 'percentage' of the total
    """
    def __call__(self):
        # selecting those assets in the Bottom 'percentage' of the total
        selection = self.assets[-self._selection_nbr:]
       
        return selection
    
class Middle(AssetSelection):
    """
    Selecting those assets in the middle 'percentage' of the total
    """
    def __call__(self):
        excluding_nbr = int((self._series_length - self._selection_nbr) / 2)
        
        # selecting those assets in the Middle 'percentage' of the total
        selection = self.assets[excluding_nbr:-excluding_nbr]
       
        return selection