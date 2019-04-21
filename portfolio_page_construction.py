import pandas as pd
import numpy as np
import urllib3
from bs4 import BeautifulSoup
from collections import OrderedDict

mapping = {"Microsoft Corp.": "MSFT",
           "Amazon.com Inc.": "AMZN",
           "Facebook Inc. Cl A": "FB",
           "Tesla Inc.": "TSLA",
           "Under Armour Inc. Cl A": "UAA",
           "Alphabet Inc. Cl A": "GOOGL",
           "Apple Inc.": "AAPL",
           "S&P 500 Index": "SPX",
           "Dow Jones Industrial Average": "DJIA"}
#
def import_names():
    sheets = xw.Book(r'C:\Users\billies9\OneDrive\Documents\Python_Screwaround\Stock_Scraper\Good_Project\practice.xlsx').sheets

    sheet_names = []
    for name in sheets:
        name = str(name).split(']',1)[1].split('>',1)[0]
        if "Covar" not in name:
            sheet_names.append(name)
    return sheet_names

def construct_portfolio(dates, weights = None, close_df = None, returns_df = None):
    start_date = pd.to_datetime(dates[0])
    end_date = pd.to_datetime(dates[1])
    if 'on' in weights.values(): # synonymous with checkboxes
        # Select of securities
        _ = {}
        for key in weights.keys():
            if weights[key] == 'on':
                try:
                    dfs[key].set_index('DateTime', inplace=True)
                except: pass
                _[key] = [dfs[key].loc[start_date:end_date, "Percent Change"].mean(),] # Need to change to reflect end - beg / beg
        df = pd.DataFrame(_)

        cov_matrix = np.array(covariance_matrix(df.columns))
        ret_list = df.values.tolist()

        num_portfolios = 4000 # maybe allow user input in later versions...
        results = np.zeros((3 + len(df.columns), num_portfolios))
        nums = np.random.random(size = (num_portfolios, len(df.columns)))

        days = end_date - start_date
        for i in range(num_portfolios):
            weights = np.array(nums[i] / np.sum(nums[i]))

            port_return = np.sum(ret_list * weights) * (252/(days.days)) # Check returns list and match with weights in std deviation

            port_deviation = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252/days.days)

            results[0, i] = port_return
            results[1, i] = port_deviation
            results[2, i] = (results[0, i] - .03) / results[1, i] #extract risk free rate?
            for j in range(len(weights)):
                results[j + 3, i] = weights[j]

        results_frame = pd.DataFrame(results.T, columns = ['Portfolio Return', 'Portfolio Deviation', 'Sharpe Ratio'] + list(_.keys()))
    else:
        """Create the portfolio here, and when done, defined a new weights dictionary that houses 'on' as the signifier of a weight for a recursive definition"""
        # User has defined weights
        #results_frame = #####
        results = np.zeros((3 + len(returns_df.columns), 1))
        _ = {}
        for column in close_df.columns: # WIll run over twice becasue of close and returns - Do I need daily returns?
            ticker = column.split(' ')[0]
            if ticker != _.keys():
                return_over_pd = (close_df.loc[end_date, ticker + ' close'] - close_df.loc[start_date, ticker + ' close']) / close_df.loc[start_date, ticker + ' close']
                _[ticker] = [return_over_pd,]
        covar_df = pd.DataFrame(_)
        cov_matrix = np.array(covariance_matrix(covar_df.columns, close_df))

        ord_weights = OrderedDict(sorted(weights.items(), key=lambda k: k[0]))
        lst_weights = np.array([float(val) for key, val in ord_weights.items() if val != ''])

        results[0, 0] = returns_df.sum(axis = 1) *100 # annualize?
        results[1, 0] = np.sqrt(np.dot(lst_weights.T, np.dot(cov_matrix, lst_weights))) # Std Dev.
        results[2, 0] = (results[0, 0] - .03) / results[1, 0]
        for j in range(len(lst_weights)):
            results[j + 3, 0] = lst_weights[j]
        results_frame = pd.DataFrame(results.T, columns = ['Portfolio Return', 'Portfolio Deviation', 'Sharpe Ratio'] + list(_.keys()))
        print(results_frame)
    return results_frame

def covariance_matrix(df_columns, price_df = None):
    price_df = price_df.filter(regex='returns').dropna()
    result = price_df.reset_index(drop = True)
    try:
        return result.cov()
    except:
        return result.var()

# def Article_Scrape(keys, keywords_dict = link_matches, page = "https://www.wsj.com/news/markets"):
#     """ Currently this scraper identifies all articles present on the webpage - inefficient to say the least. Could reverse the process by taking each key
#         and using its keywords to match certain articles found in the soup.FindAll function. Then by using FindNextChild, etc. could find the relevant summary and image content."""
#     http = urllib3.PoolManager()
#     r = http.request("GET", page)
#     soup = BeautifulSoup(r.data, "html.parser")
#     article_dict = {}
#     #print(soup.findAll("a", {"class": "wsj-headline-link"}))
#     for _ in soup.findAll("a", {"class": "wsj-headline-link"}):
#         """ a is child of h3. h3 is sibling of div container holding image.
#         Aforementioned div is a container of div of meta with desired content url.
#         For desired summary, findAll p.wsj-summary sibling to h3.wsj-headline
#         """
#         article_dict[_.text] = _.get('href')
#     #print(article_dict)
#     rel_arts = {}
#     for key in keys:
#         if 'index' in key:
#             key = key.split('/')[1]
#         for title, link in article_dict.items():
#             past_keywords = []
#             for keyword in keywords_dict[key]:
#                 if keyword in title and check_list(past_keywords, title):
#                     try:
#                         rel_arts[key] = rel_arts[key] + [(title, link)]
#                     except:
#                         rel_arts[key] = [(title, link)]
#                 past_keywords.append(keyword)
#     #print(rel_arts)
#     return rel_arts

def check_list(lst, article):
    _ = True
    for word in lst:
        if word in article.split(' '):
            _ = False
    return _

if __name__ == '__main__':
    print(link_matches)
