import matplotlib.pyplot as plt
import pandas as pd
import io
import os
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Response
import numpy as np
import json
# from tiingo import TiingoClient
from http import client
# from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from bokeh.embed.standalone import json_item
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from portfolio_page_construction import construct_portfolio, covariance_matrix
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap

# Alpha_Vantage_API = '2IJFMFJU2M5IDU6N'
Tradier_API = 'Bearer qLUA59t6iQGIgASUpKY9AstSAiNC'

#dfs = pd.read_excel(r'C:\Users\billies9\OneDrive\Documents\Python_Screwaround\Stock_Scraper\Good_Project\practice.xlsx', sheetname=None, skiprows = 2)
mapping = {"Microsoft Corp.": "MSFT",
           "Amazon.com Inc.": "AMZN",
           "Facebook Inc. Cl A": "FB",
           "Tesla Inc.": "TSLA",
           "Under Armour Inc. Cl A": "UAA",
           "Alphabet Inc. Cl A": "GOOGL",
           "Apple Inc.": "AAPL",
           "S&P 500 Index": "SPX",
           "Dow Jones Industrial Average": "DJIA"}


# def intialize_tiingo():
#     config = {}
#     # To reuse the same HTTP Session across API calls (and have better performance), include a session key.
#     config['session'] = True
#     # If you don't have your API key as an environment variable,
#     # pass it in via a configuration dictionary.
#     config['api_key'] = "3649562a2e2ce6b9a81c6d37c26da48a3cb93870"
#     # Initialize
#     client = TiingoClient(config)
#     return client
class Security_Portfolio_data():
    def __init__(self, dates):
        # self.frame_title = frame_title
        self.dates = dates
        #maybe put dates, frame_title here

    def create_connection(self):
        connection = client.HTTPSConnection('proxy.server', 3128, timeout= 30)
        connection.set_tunnel('sandbox.tradier.com', 443) # need to add proxy.server:3128 here
        headers = {'Accept':'application/json'}
        Tradier_API = 'Bearer qLUA59t6iQGIgASUpKY9AstSAiNC'
        headers['Authorization'] = Tradier_API
        return connection, headers

    def type_security_content(self, frame_title, date_type = None):
        if date_type == 'hist':
            if frame_title == 'SPX': frame_title = 'SPY'
            elif frame_title == 'DJIA': frame_title = 'DIA'
            return self.hist_security_content(frame_title)
        else:
            if frame_title == 'SPX': frame_title = 'SPY'
            elif frame_title == 'DJIA': frame_title = 'DIA'
            return self.intra_security_content(frame_title)

    def intra_security_content(self, frame_title):
        connection, headers = self.create_connection()
        connection.request('GET', '/v1/markets/timesales?symbol={}&interval=1min&start={}&end={}&session_filter={}'.format(frame_title, self.dates[0], self.dates[1], 'open'), None, headers)

        data = self.load_security_content(connection)
        df = pd.DataFrame(data['series']['data'])
        df['time'] = pd.to_datetime(df['time'])
        return df.set_index('time')

    def hist_security_content(self, frame_title):
        connection, headers = self.create_connection()
        connection.request('GET', '/v1/markets/history?symbol={}&start={}&end={}'.format(frame_title, self.dates[0], self.dates[1]), None, headers)

        data = self.load_security_content(connection)
        df = pd.DataFrame(data['history']['day'])
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')

    def load_security_content(self, connection):
        response = connection.getresponse()
        content = response.read().decode("utf-8")
        data = json.loads(content)
        return data

    def portfolio_content(self, weights = None):
        if weights == None:
            print('error - no weights given')
        dfclose = pd.DataFrame(columns = [ticker + ' close' for ticker in weights.keys() if weights[ticker] != ''])
        dfreturns = pd.DataFrame({ticker + ' Weighted Return':[0] for ticker in weights.keys() if weights[ticker] != ''})
        i = 0
        for column in dfclose.columns:
            ticker = column.split(' ')[0]
            frame = self.hist_security_content(ticker)
            dfclose[ticker + ' close'] = frame['close']
            dfclose[ticker + ' returns'] = frame['close'].pct_change() * 100 # for covairance matrix calcs
            try:
                _ = (dfclose.loc[self.dates[1], ticker + ' close'] - dfclose.loc[self.dates[0], ticker + ' close']) / dfclose.loc[self.dates[0], ticker + ' close']
                dfreturns.loc[0, ticker + ' Weighted Return'] = _ * float(weights[ticker]) # Weighted
            except:
                self.dates[1] = pd.to_datetime(self.dates[1])-timedelta(1)
                _ = (dfclose.loc[self.dates[1], ticker + ' close'] - dfclose.loc[self.dates[0], ticker + ' close']) / dfclose.loc[self.dates[0], ticker + ' close']
                dfreturns.loc[0, ticker + ' Weighted Return'] = _ * float(weights[ticker]) # Weighted
        return dfclose, dfreturns

def tradier_content(frame_title, dates, date_type = 'hist', weights = None):
    connection = client.HTTPSConnection('sandbox.tradier.com', 443, timeout=30)
    headers={'Accept':'application/json'}
    headers['Authorization'] = Tradier_API
    if frame_title == 'Portfolio':
        # Create single dataframe based on weights inputs
        if weights == None:
            print('error - no weights given')
        dfclose = pd.DataFrame(columns = [ticker + ' close' for ticker in weights.keys() if weights[ticker] != ''])
        dfreturns = pd.DataFrame({ticker + ' Weighted Return':[0] for ticker in weights.keys() if weights[ticker] != ''})

        i = 0
        for column in dfclose.columns:
            ticker = column.split(' ')[0]
            connection.request('GET', '/v1/markets/history?symbol={}&start={}&end={}'.format(ticker, dates[0], dates[1]), None, headers)
            response = connection.getresponse()
            content = response.read().decode("utf-8")
            data = json.loads(content)
            frame = pd.DataFrame(data['history']['day'])
            if i != 0:
                frame.set_index(pd.to_datetime(frame['date']), inplace=True)
            dfclose[ticker + ' close'] = frame['close']
            dfclose[ticker + ' returns'] = frame['close'].pct_change() * 100
            if not isinstance(dfclose.index, pd.DatetimeIndex):
                dfclose.set_index(pd.to_datetime(frame['date']), inplace = True)
            try:
                _ = (dfclose.loc[dates[0], ticker + ' close'] - dfclose.loc[dates[1], ticker + ' close']) / dfclose.loc[dates[1], ticker + ' close']
                dfreturns.loc[0, ticker + ' Weighted Return'] = _ * float(weights[ticker])
            except:
                dates[1] = pd.to_datetime(dates[1])-timedelta(1)
                _ = (dfclose.loc[dates[0], ticker + ' close'] - dfclose.loc[dates[1], ticker + ' close']) / dfclose.loc[dates[1], ticker + ' close']
                dfreturns.loc[0, ticker + ' Weighted Return'] = _ * float(weights[ticker])
            i += 1
        return dfclose, dfreturns

    if date_type=='intra':
        # interval of tick, 1min, 5min, 15min (default)
        connection.request('GET', '/v1/markets/timesales?symbol={}&interval=1min&start={}&end={}&session_filter={}'.format(frame_title, dates[0], dates[1], 'open'))
        # Returns dictionary of dictionary of dictionary - keys: 'series', 'data'
    else:
        connection.request('GET', '/v1/markets/history?symbol={}&start={}&end={}'.format(frame_title, dates[0], dates[1]), None, headers) # default interval is daily
    # Returns dictionary of dictionary of dictionary - keys: 'history', 'day'
    response = connection.getresponse()
    content = response.read().decode("utf-8")
    data = json.loads(content)
    try:
        df = pd.DataFrame(data['history']['day'])
        return df.set_index('date')
    except:
        try:
            df = pd.DataFrame(data['series']['data'])
            return df.set_index('time')
        except:
            print('Error in retrieving information for ' + frame_title)
            return


def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31,
        29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return date.replace(day=d,month=m, year=y)

def build_interactive_graph(frame_title, type = None, weights = None, dates = None, date_type = 'hist'):
    """ Type: 'P' is Price graphing;
              'R' is Regression graphing;"""
    #qd.ApiConfig.api_key = 'GvpSgtQ8RzyZGAKhas4p'
    try: name = [name for name, ticker in mapping.items() if ticker == frame_title][0] # Doesn't work for Port_page - look into try/except
    except: pass
    if frame_title != 'Portfolio':
        if frame_title == 'SPX':
            # data = tradier_content('SPY', dates, date_type)
            # allow for date_type input........
            data = Security_Portfolio_data(dates).type_security_content(frame_title, date_type)
            data['close'] = data['close'] * 10
        elif frame_title == 'DJIA':
            # data = tradier_content('DIA', dates, date_type)
            data = Security_Portfolio_data(dates).type_security_content(frame_title, date_type)
            data['close'] = data['close'] * 100
        else:
            # data = tradier_content(frame_title, dates, date_type)
            data = Security_Portfolio_data(dates).type_security_content(frame_title, date_type)
        # data.index = pd.to_datetime(data.index)
        if type.upper() == 'P':
            # intraday will have to handled differently
            df = ColumnDataSource(data={
                'Date': data.index,
                'Price': data['close']
            })
            fig = figure(title=frame_title + ' Price', sizing_mode='fixed', plot_width=670, plot_height=450, toolbar_location='above', x_axis_type='datetime')
            fig.xaxis.axis_label = 'Date'
            fig.yaxis.axis_label = 'Price'
            fig.line(source=df, x='Date', y='Price', legend=frame_title + ' price')
            fig.add_tools(HoverTool(
                tooltips=[
                    ('DateTime', '@Date{%F}'),
                    ('Price', '$@Price{0.2f}')
                ],
                formatters={'Date': 'datetime'},
                mode='mouse'
            ))
            return fig
        elif type.upper() == 'R':
            data['Percent Change'] = data['close'].pct_change() * 100
            spx_data = tradier_content(frame_title = 'SPY', dates=dates) # Use SPY as proxy for SPX
            data['SPX Pct Change'] = (spx_data['close'] * 10).pct_change() * 100
            data.dropna(inplace=True)
            df = ColumnDataSource(data={
                'Date': data.index,
                'SPX Daily Return': data['SPX Pct Change'],
                frame_title + ' Daily Return': data['Percent Change']
            })
            fig = figure(title=frame_title + ' Regression', sizing_mode='fixed', plot_width=670, plot_height=450, toolbar_location='above')
            fig.xaxis.axis_label = 'SPX Pct Change'
            fig.yaxis.axis_label =  name + ' Pct Change'
            r1 = fig.scatter(source=data, x='SPX Pct Change', y='Percent Change', name='scatter')

            fig.add_tools(HoverTool(renderers=[r1],
                    tooltips=[
                        # ('DateTime', '@Date{%F}'),
                        ('SPX Daily', '@{SPX Pct Change}{%0.2f}%'),
                        (frame_title + ' Daily', '@{Percent Change}{0.6f}%')
                ],
                    formatters={
                        # 'Date': 'datetime',
                        'SPX Pct Change': 'printf',
                        frame_title + ' Daily': 'printf'},
                    mode='mouse'))

            fit_func = np.poly1d(np.polyfit(x=data['SPX Pct Change'], y=data['Percent Change'], deg = 1))
            r2 = (fig.line(x = data['SPX Pct Change'], y = fit_func(data['SPX Pct Change']), legend = 'Beta: ' + str(round(fit_func[1], 3))
                                                    + '\n' + "Alpha: " + str(round(fit_func[0], 3))))
            return fig
    else:
        start_date, end_date = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
        days = end_date.date() - start_date.date()
        # print(weights)
        fig = figure(title = frame_title, sizing_mode='fixed', plot_width=670, plot_height= 450, toolbar_location='right')
        fig.xaxis.axis_label = 'Std. Deviation'
        fig.yaxis.axis_label = 'Return'
        # tradier_content(frame_title, dates, date_type = 'hist', weights = weights)
        price, returns = Security_Portfolio_data(dates).portfolio_content(weights)
        frame_df = construct_portfolio(dates, weights = weights, close_df = price, returns_df = returns)
        df = ColumnDataSource(data = {
            'Deviation': frame_df['Portfolio Deviation'],
            'Return': frame_df['Portfolio Return']
        })
        fig.circle(source = df, x = 'Deviation', y ='Return')
        fig.add_tools(HoverTool(
            tooltips=[
                ('Risk', '@Deviation'),
                ('Return', '@Return{0.2f}%')
            ],
            mode='mouse'
        ))
        # Doesn't assign colors correctly
        # mapper = linear_cmap(field_name='Return', palette=Spectral6 ,low=min(frame_df['Sharpe Ratio']) ,high=max(frame_df['Sharpe Ratio']))
        return fig







#--------------------------------------------------------------------------------------------------------------------------
def calc_return_type(name, dates, return_type = 'D'):
    # Aggregate based on choice of Daily, Monthly, Yearly
    df_copy = dfs[name].copy().set_index('DateTime', inplace=True).loc[pd.to_datetime(dates[0]):pd.to_datetime(dates[1]), :] # Between dates and all columns
    print(df_copy)
    entry_len = len(df_copy)

    entry_arr = np.zeros((entry_len, 1))
    for i in range(entry_arr.shape[0]):
        entry[i][0] = 1 + df_copy.iloc[i]['Percent Change'] / 100
    df['1+r'] = entry

    if return_type == 'D':
        return df_copy
    elif return_type == 'M':
        for date in df.index:
            pass

        return
    else: #return_type=='Y'
        pass
    return

def build_static_graph(frame_title):
    img = io.BytesIO()
    dfs[[name for name, ticker in mapping.items() if ticker == frame_title][0]].plot(y = 'Price')
    plt.savefig(img, format='png', dpi= 100, bbox_inches='tight')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

def regression(frame_title):
    img = io.BytesIO()
    fig, ax = plt.subplots()
    x = dfs['S&P 500 Index']['Percent Change']
    y = dfs[[name for name, ticker in mapping.items() if ticker == frame_title][0]]['Percent Change']
    ax.scatter(x, y)
    fit_func = np.poly1d(np.polyfit(x, y, deg = 1))
    (ax.plot(x, fit_func(x), 'r', label = "Beta: " + str(round(fit_func[1], 3)) +
                                            '\n' + "Alpha: " + str(round(fit_func[0], 3))))
    ax.get_xaxis().get_label().set_visible(True)
    ax.set_title("Regression")
    ax.axhline(0, linestyle = '--', color = 'k', linewidth = .7)
    ax.axvline(0, linestyle = '--', color = 'k', linewidth = .7)
    ax.legend(loc = 0)

    plt.savefig(img,  format='png', dpi= 100, bbox_inches='tight')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

class YahooFinanceHistory:
    timeout = 2
    crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'

    def __init__(self, symbol, days_back=7):
        self.symbol = symbol
        self.session = requests.Session()
        self.dt = timedelta(days=days_back)

    def get_crumb(self):
        response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError('Could not get crumb from Yahoo Finance')
        else:
            self.crumb = match.group(1)

    def get_quote(self):
        if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
            self.get_crumb()
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb)
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])
