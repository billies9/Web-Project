import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
import json



class Portfolio_Stats():
    def __init__(self, tickers_data, SPX_data):
        # Portfolio daily returns against SPX retruns
        self.tickers_data = tickers_data.dropna()

        self.SPX_data = SPX_data.dropna()

    def regression_stats(self, weights, y_data = None):
        if y_data == None:
            self.tickers_data.loc[:, 'Portfolio Return'] = np.dot(weights, self.tickers_data.T)
            y_data = self.tickers_data['Portfolio Return']
            slope, intercept, r_value, p_value, std_err = stats.linregress(self.SPX_data, y_data)
            self.tickers_data.drop('Portfolio Return', axis=1, inplace=True)
        return slope, intercept, r_value**2 # Beta, Alpha, R-squared
