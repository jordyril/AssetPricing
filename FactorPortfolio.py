"""
Created on Sat Apr 13 19:11:15 2019

@author: jrilla
"""
# =============================================================================
# Packages
# =============================================================================
import pandas as pd

from Weighting import EquallyWeighted, ValueWeighted
from AssetSelection import High, Low
from DataCleaner import DropAllNan


# =============================================================================
# CLASSES
# =============================================================================
class FactorPortfolio(object):
    def __init__(self, criterium_instance, returns_df, 
                 in_sample_window=1, percentage=0.3, sizes=None):
        
        self.criterium = criterium_instance
        self.returns = returns_df
        self.in_sample = in_sample_window
        self.out_sample =  len(self.returns) - self.in_sample
        
        self.proportion = percentage
        
        if sizes is None:
            self.weighting = EquallyWeighted()
        
        else:
            try:
                assert self.returns.shape == sizes.shape
                self.weighting = ValueWeighted(sizes)
            except AssertionError:
                print("Size df does not fit the given returns df")
       
    
class OneFactorportfolio(FactorPortfolio):
    def __init__(self, criterium_instance, returns_df, asset_selection_class,
                 in_sample_window=1, percentage=0.3, sizes=None):
        FactorPortfolio.__init__(self, criterium_instance, returns_df, 
                                 in_sample_window, percentage, sizes)
        
        self.portfolio = pd.Series(index=self.returns.index[self.in_sample:],
                                   dtype=float)
       
        self.selector = asset_selection_class
   
    def __call__(self, datacleaner_class=DropAllNan):
        datacleaner = datacleaner_class()
        
        self.returns = datacleaner.drop_all_na(self.returns)
        for t in range(self.out_sample):
            out_sample_timepoint = self.in_sample + t
            last_in_sample_timepoint = self.in_sample + t - 1
            
            returns_full = self.returns.iloc[t:out_sample_timepoint + 1]
            
            #clean data
            returns_full = datacleaner(returns_full)
            
            returns_in, _ = datacleaner.split_in_sample_out_sample(returns_full)
           
            betas = self.criterium(returns_in, t, self.in_sample + t)
            
            selection = self.selector(betas, self.proportion)()
            
            returns_out = self.returns[selection].iloc[out_sample_timepoint]
            
            weights = self.weighting(selection, last_in_sample_timepoint)
            
            self.portfolio.iloc[t] = weights @ returns_out
            
            print(str(t + 1) + ' done, ' + 
                  str(self.out_sample - t - 1) + ' to go')
     
        return self.portfolio

class LongShortPortfolios(FactorPortfolio):
    def __init__(self, criterium_instance, returns_df, 
                 in_sample_window=1, percentage=0.3, sizes=None):
        
        FactorPortfolio.__init__(self, criterium_instance, returns_df, 
                                 in_sample_window, percentage, sizes)
        
        self.portfolios = pd.DataFrame(index=self.returns
                                                 .index[self.in_sample:],
                                       columns=['High', 'Low', 
                                                'HL-LS', 'LH-LS'],
                                                dtype=float)
    
    def __call__(self, datacleaner_class=DropAllNan):
        datacleaner = datacleaner_class()
        
#        self.returns = datacleaner.drop_all_na(self.returns)
        for t in range(self.out_sample):
            out_sample_timepoint = self.in_sample + t
            last_in_sample_timepoint = self.in_sample + t - 1
            
            returns_full = self.returns.iloc[t:out_sample_timepoint + 1]
            
            # clean data
            returns_full = datacleaner(returns_full)
            
            returns_in, _ = datacleaner.split_in_sample_out_sample(returns_full)
           
            betas = self.criterium(returns_in, t, self.in_sample + t)
                      
            selection_H = High(betas, self.proportion)()
            selection_L = Low(betas, self.proportion)()
            
            returns_out_H = self.returns[selection_H].iloc[out_sample_timepoint]
            returns_out_L = self.returns[selection_L].iloc[out_sample_timepoint]
            
            weights_H = self.weighting(selection_H, last_in_sample_timepoint)
            weights_L = self.weighting(selection_L, last_in_sample_timepoint)
            

            self.portfolios['High'].iloc[t] = weights_H @ returns_out_H
            self.portfolios['Low'].iloc[t] = weights_L @ returns_out_L
            
            print(str(t + 1) + ' done, ' + 
                  str(self.out_sample - t - 1) + ' to go')
        
        self.portfolios['HL-LS'] = (self.portfolios['High'] 
                                    - self.portfolios['Low'])
        
        self.portfolios['LH-LS'] = (self.portfolios['Low'] 
                                    - self.portfolios['High'])
        
        
        return self.portfolios