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

    def VaR(self, weights):
        self.tickers_data.loc[:, 'Portfolio Return'] = np.dot(weights, self.tickers_data.T)
        mu = self.tickers_data['Portfolio Return'].mean()
        sigma = self.tickers_data['Portfolio Return'].std()

        self.tickers_data.drop('Portfolio Return', axis=1, inplace=True)
        P_value = 1e6

        VaR = []
        for c_level in [.90, .95, .975, .99]:
            alpha = stats.norm.ppf(1 - c_level, mu, sigma)
            VaR.append(P_value - P_value*(alpha + 1))
        return VaR
            # self.tickers_data.loc[:, "{c_level}% Confidence Level VaR".format(c_level = c_level * 100)]
