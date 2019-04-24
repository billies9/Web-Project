import pandas as pd
import base64
import numpy as np
import json
import requests
from collections import OrderedDict
from datetime import datetime, timedelta
from portfolio_page_construction import covariance_matrix
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import scipy.stats as stats

mapping = {"Microsoft Corp.": "MSFT",
           "Amazon.com Inc.": "AMZN",
           "Facebook Inc. Cl A": "FB",
           "Tesla Inc.": "TSLA",
           "Under Armour Inc. Cl A": "UAA",
           "Alphabet Inc. Cl A": "GOOGL",
           "Apple Inc.": "AAPL",
           "S&P 500 Index": "SPX",
           "Dow Jones Industrial Average": "DJIA"}

class Security_Portfolio_data():
    def __init__(self, ticker, dates_type):
        self.ticker = ticker
        self.dates = dates_type[0]
        self.date_type = dates_type[1]

    def load_content(self):
        r = requests.get('https://financialmodelingprep.com/api/company/historical-price/{ticker}?serietype=candle&datatype=json'.format(ticker = self.ticker))
        data = json.loads(r.text)
        df = pd.DataFrame(data['historical'])

        # Parse through date column
        df['date'] = df['date'].str.split(' ', n = 4, expand = False).str[1:4]
        df['date'] = pd.to_datetime(df['date'].str.join('-'))
        df['weekday'] = df['date'].dt.dayofweek
        df = df[df['weekday'] < 5]
        print(df)
        if df.empty:
            df = self.load_bad_data_content(self.dates[0], self.dates[1])
            print('df')
            print(df)
        elif df.empty is False and ('2019-01-01' <= self.dates[0] <= '2019-03-05' or '2019-01-01' <= self.dates[1] <= '2019-03-05'):
            extended_frame = self.load_bad_data_content(self.dates[0], self.dates[1])
            print(extended_frame)
        return df

    def load_bad_data_content(self, start, end):
        headers = {"Accept":"application/json",
                    "Authorization":"Bearer qLUA59t6iQGIgASUpKY9AstSAiNC"}
        r = (requests.get('https://sandbox.tradier.com/v1/markets/history?symbol={symbol}&interval=daily&start={start}&end={end}'.format(symbol=self.ticker, start=start, end=end),
                headers=headers))
        data = json.loads(r.text)

        frame = pd.DataFrame(data['history']['day'])
        return frame.drop('volume', axis=1).set_index('date', inplace=True)

    def parse_date_content(self, ticker = None):
        if ticker != None:
            self.ticker = ticker
        df = self.load_content()
        df.set_index('date', inplace = True)

        return df.loc[self.dates[0]:self.dates[1], :]

    def get_match_val(self):
        r = requests.get('https://financialmodelingprep.com/api/stock/list/all?datatype=json')
        data = json.loads(r.text)
        df = pd.DataFrame(data)
        df.set_index('Ticker', inplace = True)
        TickerName = df.loc[df.index[df.index == self.ticker], 'companyName'].tolist()[0]
        CompanyName = TickerName.split(' ')[0] # Find a way to target full name outside of Inc. / Corp. / etc.
        return CompanyName

    def portfolio_close_returns(self, weights = None):
        if weights == None:
            print('error - no weights given')
        dfclose = pd.DataFrame(columns = [ticker + ' close' for ticker in weights.keys() if weights[ticker] != ''])
        dfreturns = pd.DataFrame({ticker + ' Weighted Return':[0] for ticker in weights.keys() if weights[ticker] != ''})
        i = 0
        for column in dfclose.columns:
            ticker = column.split(' ')[0]
            frame = self.parse_date_content(ticker)
            dfclose[ticker + ' close'] = frame['close']
            dfclose[ticker + ' returns'] = frame['close'].pct_change() * 100 # for covariance matrix calcs
            try:
                _ = (dfclose.loc[self.dates[1], ticker + ' close'] - dfclose.loc[self.dates[0], ticker + ' close']) / dfclose.loc[self.dates[0], ticker + ' close']
                dfreturns.loc[0, ticker + ' Weighted Return'] = _ * float(weights[ticker]) # Weighted
            except:
                self.dates[1] = pd.to_datetime(self.dates[1])-timedelta(1)
                _ = (dfclose.loc[self.dates[1], ticker + ' close'] - dfclose.loc[self.dates[0], ticker + ' close']) / dfclose.loc[self.dates[0], ticker + ' close']
                dfreturns.loc[0, ticker + ' Weighted Return'] = _ * float(weights[ticker]) # Weighted
        return dfclose, dfreturns

    def portfolio_rand_user_weights(self, weights):
        # Defines random selection of securities and weights for use when selection or weight defined by user
        close_df, returns_df = self.portfolio_close_returns(weights)

        results = np.zeros((3 + len(returns_df.columns), 1))
        _, on_weights = {}, {}
        for column in close_df.columns: # WIll run over twice becasue of close and returns - Do I need daily returns?
            ticker = column.split(' ')[0]
            if ticker != _.keys():
                return_over_pd = (close_df.loc[self.dates[1], ticker + ' close'] - close_df.loc[self.dates[0], ticker + ' close']) / close_df.loc[self.dates[0], ticker + ' close']
                _[ticker] = [return_over_pd,]
                on_weights[ticker] = 'on'
        covar_df = pd.DataFrame(_)
        cov_matrix = np.array(covariance_matrix(covar_df.columns, close_df))

        ord_weights = OrderedDict(sorted(weights.items(), key=lambda k: k[0]))
        lst_weights = np.array([float(val) for key, val in ord_weights.items() if val != ''])

        results[0, 0] = returns_df.sum(axis = 1)  # annualize?
        results[1, 0] = np.sqrt(np.dot(lst_weights.T, np.dot(cov_matrix, lst_weights))) / 100
        results[2, 0] = (results[0, 0] - .03) / results[1, 0]
        for j in range(len(lst_weights)):
            results[j + 3, 0] = lst_weights[j]
        results_frame = pd.DataFrame(results.T, columns = ['Portfolio Return', 'Portfolio Deviation', 'Sharpe Ratio'] + list(sorted(_.keys())))
        print(results_frame)
        return results_frame, on_weights

    def portfolio_rand_rand_weights(self, weights):
        # Select of securities
        _ = {}
        pct_df = pd.DataFrame()
        for key in weights.keys():
            if weights[key] == 'on':
                frame = self.parse_date_content(key)
                pct_df[key + ' returns'] = frame['close'].pct_change().dropna() * 100
                _[key] = [(frame['close'][-1] - frame['close'][0]) / frame['close'][0]]

        df = pd.DataFrame.from_dict(_, orient='columns')
        cov_matrix = np.array(covariance_matrix(df.columns, pct_df))
        ret_list = df.values.tolist()
        num_portfolios = 1500 # maybe allow user input in later versions...
        results = np.zeros((3 + len(df.columns), num_portfolios))
        nums = np.random.random(size = (num_portfolios, len(df.columns)))

        days = pd.to_datetime(self.dates[1]) - pd.to_datetime(self.dates[0])
        for i in range(num_portfolios):
            weights = np.array(nums[i] / np.sum(nums[i]))
            port_return = np.sum(ret_list * weights)  #* (252/(days.days)) # Check returns list and match with weights in std deviation
            port_deviation = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) #* np.sqrt(252/days.days)

            results[0, i] = port_return
            results[1, i] = port_deviation / 100
            results[2, i] = (results[0, i] - .03) /  results[1, i]
            for j in range(len(weights)):
                results[j + 3, i] = weights[j]

        results_frame = pd.DataFrame(results.T, columns = ['Portfolio Return', 'Portfolio Deviation', 'Sharpe Ratio'] + list(_.keys()))
        return results_frame

    def get_company_info(self):
        r = requests.get('https://financialmodelingprep.com/api/company/profile/{}?datatype=json'.format(self.ticker))
        data = (json.loads(r.text))[self.ticker]
        frame = pd.DataFrame.from_dict(data, orient='index')
        rel_frame = frame.loc[['Beta', 'VolAvg', 'MktCap', 'LastDiv', 'Range', 'exchange', 'industry', 'website', 'description', 'CEO'], :]
        rel_frame.drop(['website'], axis=0, inplace=True)

        # Need to clean data - make dollar figures look like dollars, etc.
        (rel_frame.rename({'VolAvg': 'Volume Average', 'MktCap': 'Market Capitalization', 'LastDiv': 'Latest Dividend',
                                                    'exchange': 'Exchange', 'industry': 'Industry', 'description': 'Description'}, axis='index', inplace=True))
        # Clean Volume Average to be 0 decimal places with commas
        rel_frame.loc['Volume Average'] = '{:,.0f}'.format(float(rel_frame.loc['Volume Average', 0]))
        # Clean Market Capitalization, Latest Dividend to be 2 decimal places with commas / dollar signs
        for figure in ['Market Capitalization', 'Latest Dividend']:
            rel_frame.loc[figure] = '${:,.2f}'.format(float(rel_frame.loc[figure, 0]))
        # Clean Range for 2 decimal places with commas / dollar signs
        split_ = rel_frame.loc['Range', 0].split('-')
        lst = []
        for num in rel_frame.loc['Range', 0].split('-'):
            # rel_frame.loc['Range'] = '${:,.2f}'.format(float(num))
            lst.append('${:,.2f}'.format(float(num)))
            lst.append('-')
        lst.pop(-1)
        rel_frame.loc['Range'] = ' '.join(lst)
        return rel_frame

    def get_financial_ratios(self):
        r = requests.get('https://financialmodelingprep.com/api/financial-ratios/{}?datatype=json'.format(self.ticker))

        _ = (json.loads(r.text))['financialRatios']
        latest_data = _[str(list(_.keys())[-2])]
        rel_data = {'Liquidity': {}, 'Profitability': {}, 'Debt': {}, 'Operating': {}, 'Investment': {}}

        # Liquidity Measures - Current, Quick, Days of Payables Outstanding
        liq = ['currentRatio', 'quickRatio', 'daysofPayablesOutstanding']
        for measure in liq:
            if measure == liq[0]:
                measure_rename = 'Current Ratio'
            elif measure == liq[1]:
                measure_rename = 'Quick Ratio'
            else:
                measure_rename = 'Days of Payables Outstanding'
            try: figure = round(latest_data['liquidityMeasurementRatios'][measure], 4)
            except: figure = 'Null'
            rel_data['Liquidity'][measure_rename] = figure

        # Profitability Measures - Gross Profit, ROE, Effective Tax Rate
        prof = ['grossProfitMargin', 'returnOnEquity', 'effectiveTaxRate']
        for measure in prof:
            if measure == prof[0]:
                measure_rename = 'Gross Profit Margin'
            elif measure == prof[1]:
                measure_rename = 'Return on Equity'
            else:
                measure_rename = 'Effective Tax Rate'
            try: figure = round(latest_data['profitabilityIndicatorRatios'][measure], 4)
            except: figure = 'Null'
            rel_data['Profitability'][measure_rename] = figure

        # Debt Measures - Debt, Debt-to-Equity, Interest Coverage
        debt = ['debtRatio', 'debtEquityRatio', 'interestCoverageRatio']
        for measure in debt:
            if measure == debt[0]:
                measure_rename = 'Debt Ratio'
            elif measure == debt[1]:
                measure_rename = 'Debt-to-Equity Ratio'
            else:
                measure_rename = 'Interest Coverage Ratio'
            try: figure = round(latest_data['debtRatios'][measure], 4)
            except: figure = 'Null'
            rel_data['Debt'][measure_rename] = figure

        # Operating Performance - Asset Turnover
        ops = ['assetTurnover']
        for measure in ops:
            if measure == ops[0]:
                measure_rename = 'Asset Turnover Ratio'
            try: figure = round(latest_data['operatingPerformanceRatios'][measure], 4)
            except: figure = 'Null'
            rel_data['Operating'][measure_rename] = 'Null'

        # Investment Valuation - Price-to-Book, PE, Dividence Yield
        inv = ['priceBookValueRatio', 'priceEarningsRatio', 'dividendYield']
        multiplier = 1
        pct = ''
        for measure in inv:
            if measure == inv[0]:
                measure_rename = 'Price-to-Book'
            elif measure == inv[1]:
                measure_rename = 'Price-to-Earnings'
            else:
                measure_rename = 'Divident Yield'
                multiplier = 100
                pct = '%'
            try: figure = str(round(latest_data['investmentValuationRatios'][measure], 4)* multiplier) + pct
            except: figure = 'Null'
            rel_data['Investment'][measure_rename] = figure
        return rel_data

class Build_graph():
    def __init__(self, ticker, dates_type):
        self.dates = dates_type[0]
        self.date_type = dates_type[1]
        self.ticker = ticker
        if self.ticker =='SPX':
            self.ticker = 'SPY'
            self.multiplier = 10
        elif self.ticker == 'DJIA':
            self.ticker = 'DIA'
            self.multiplier = 100
        else:
            self.multiplier = 1

    def price_graph(self):
        data = Security_Portfolio_data(self.ticker, (self.dates, self.date_type)).parse_date_content()
        data['close'] = data['close'] * self.multiplier

        plot = go.Scatter(
                x = data.index,
                y = data['close'],
                mode = 'lines',
                hoverinfo = 'y'
        )
        layout = go.Layout(
                    title = self.ticker + ' Price Chart',
                    yaxis = go.layout.YAxis(
                        title = 'Price',
                        automargin = True,
                        mirror=True,
                        ticks='outside',
                        showline=True,
                    ),
                    xaxis = go.layout.XAxis(
                        title = 'Date',
                        automargin = True,
                        mirror=True,
                        ticks='outside',
                        showline=True,
                    ),
        )
        end_data = [plot]
        fig = go.Figure(data = end_data, layout = layout)
        graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graph

    def regression_graph(self):
        data = Security_Portfolio_data(self.ticker, (self.dates, self.date_type)).parse_date_content()
        data['Percent Change'] = data['close'].pct_change()
        # Use SPY as proxy for SPX
        spx_data = Security_Portfolio_data('SPY', (self.dates, self.date_type)).parse_date_content()
        data['SPX Pct Change'] = (spx_data['close'] * 10).pct_change()
        data.dropna(inplace=True)

        plot = go.Scatter(
                x = data['SPX Pct Change'],
                y = data['Percent Change'],
                mode = 'markers',
                hoverinfo='x+y',
                name = self.ticker + ' Daily Change'
        )
        slope, intercept, r_value, p_value, std_err = stats.linregress(data['SPX Pct Change'], data['Percent Change']) # Append to dataframe to graph as scatter line
        line = slope * data['SPX Pct Change'] + intercept
        fit = go.Scatter(
                  x=data['SPX Pct Change'],
                  y=line,
                  mode='lines',
                  hoverinfo = 'none',
                  marker=go.scatter.Marker(color='rgb(0, 0, 0)'),
                  name='Beta: ' + str(round(slope, 4)) + '\n' + 'Alpha: ' + str(round(intercept, 4)) # Maybe move to annotation
                  )
        end_data = [plot, fit]

        layout = go.Layout(
                    title = self.ticker + ' Regression Chart',
                    yaxis = go.layout.YAxis(
                        title = self.ticker + ' Daily Returns',
                        tickformat = ',.1%',
                        hoverformat = ',.4%',
                        automargin = True,
                        mirror=True,
                        ticks='outside',
                        showline=True,
                    ),
                    xaxis = go.layout.XAxis(
                        title = 'SPX Daily Return',
                        tickformat = ',.1%',
                        hoverformat = ',.4%',
                        automargin = True,
                        mirror=True,
                        ticks='outside',
                        showline=True,
                    ),
                    showlegend = True,
                    legend = dict(x=.1, y= .9),
        )
        fig = go.Figure(data = end_data, layout = layout)
        graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graph

    def portfolio_graph(self, weights):
        # try:
        user_data, _ = Security_Portfolio_data('', (self.dates, self.date_type)).portfolio_rand_user_weights(weights)
        print(user_data)
            # rand_data = Security_Portfolio_data('', (self.dates, self.date_type)).portfolio_rand_rand_weights(_)
            # print(user_data)
            # print(rand_data)
        # except:
            # rand_data = Security_Portfolio_data('', (self.dates, self.date_type)).portfolio_rand_rand_weights(weights)
            # print(rand_data)
        rand_plot = go.Scatter(
                x=rand_data['Portfolio Deviation'],
                y=rand_data['Portfolio Return'],
                mode='markers',
                hoverinfo='x+y',
                marker=dict(color='#1f77b4'),
                name='Random Portfolio'
        )
        try:
            user_plot = go.Scatter(
                x=user_data['Portfolio Deviation'],
                y=user_data['Portfolio Return'],
                mode='markers',
                hoverinfo='x+y',
                marker=dict(color='#accecd'),
                name='User Portfolio'
            )
            end_data = [rand_plot, user_plot]
        except:
            end_data = [rand_plot]
        layout = go.Layout(
                title = 'Portfolio Graph',
                yaxis = go.layout.YAxis(
                    title = 'Portfolio Return',
                    tickformat = ',.2%',
                    hoverformat = ',.4%',
                    automargin = True,
                    mirror=True,
                    ticks='outside',
                    showline=True,
                ),
                xaxis = go.layout.XAxis(
                    title = 'Portfolio Deviation',
                    tickformat = ',.2%',
                    hoverformat = ',.4%',
                    automargin = True,
                    mirror=True,
                    ticks='outside',
                    showline=True,
                ),
                showlegend = True,
                legend = dict(x=.9, y= .8),
        )
        fig = go.Figure(data = end_data, layout = layout)
        graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graph











#--------------------------------------------------------------------------------------------------------------------------

# def build_interactive_graph(frame_title, type = None, weights = None, dates = None, date_type = 'hist'):
#     """ Type: 'P' is Price graphing;
#               'R' is Regression graphing;"""
#     #qd.ApiConfig.api_key = 'GvpSgtQ8RzyZGAKhas4p'
#     try: name = [name for name, ticker in mapping.items() if ticker == frame_title][0] # Doesn't work for Port_page - look into try/except
#     except: pass
#     if frame_title != 'Portfolio':
#         if frame_title == 'SPX':
#             # data = tradier_content('SPY', dates, date_type)
#             # allow for date_type input........
#             data = Security_Portfolio_data(dates).type_security_content(frame_title, date_type)
#             data['close'] = data['close'] * 10
#         elif frame_title == 'DJIA':
#             # data = tradier_content('DIA', dates, date_type)
#             data = Security_Portfolio_data(dates).type_security_content(frame_title, date_type)
#             data['close'] = data['close'] * 100
#         else:
#             # data = tradier_content(frame_title, dates, date_type)
#             data = Security_Portfolio_data(dates).type_security_content(frame_title, date_type)
#         # data.index = pd.to_datetime(data.index)
#         if type.upper() == 'P':
#             # intraday will have to handled differently
#             df = ColumnDataSource(data={
#                 'Date': data.index,
#                 'Price': data['close']
#             })
#             fig = figure(title=frame_title + ' Price', sizing_mode='fixed', plot_width=670, plot_height=450, toolbar_location='above', x_axis_type='datetime')
#             fig.xaxis.axis_label = 'Date'
#             fig.yaxis.axis_label = 'Price'
#             fig.line(source=df, x='Date', y='Price', legend=frame_title + ' price')
#             fig.add_tools(HoverTool(
#                 tooltips=[
#                     ('DateTime', '@Date{%F}'),
#                     ('Price', '$@Price{0.2f}')
#                 ],
#                 formatters={'Date': 'datetime'},
#                 mode='mouse'
#             ))
#             return fig
#         elif type.upper() == 'R':
#             data['Percent Change'] = data['close'].pct_change() * 100
#             # Use SPY as proxy for SPX
#             spx_data = Security_Portfolio_data(dates).type_security_content('SPY', date_type)
#             data['SPX Pct Change'] = (spx_data['close'] * 10).pct_change() * 100
#             data.dropna(inplace=True)
#             df = ColumnDataSource(data={
#                 'Date': data.index,
#                 'SPX Daily Return': data['SPX Pct Change'],
#                 frame_title + ' Daily Return': data['Percent Change']
#             })
#             fig = figure(title=frame_title + ' Regression', sizing_mode='fixed', plot_width=670, plot_height=450, toolbar_location='above')
#             fig.xaxis.axis_label = 'SPX Pct Change'
#             fig.yaxis.axis_label =  name + ' Pct Change'
#             r1 = fig.scatter(source=data, x='SPX Pct Change', y='Percent Change', name='scatter')
#
#             fig.add_tools(HoverTool(renderers=[r1],
#                     tooltips=[
#                         # ('DateTime', '@Date{%F}'),
#                         ('SPX Daily', '@{SPX Pct Change}{%0.2f}%'),
#                         (frame_title + ' Daily', '@{Percent Change}{0.6f}%')
#                 ],
#                     formatters={
#                         # 'Date': 'datetime',
#                         'SPX Pct Change': 'printf',
#                         frame_title + ' Daily': 'printf'},
#                     mode='mouse'))
#
#             fit_func = np.poly1d(np.polyfit(x=data['SPX Pct Change'], y=data['Percent Change'], deg = 1))
#             r2 = (fig.line(x = data['SPX Pct Change'], y = fit_func(data['SPX Pct Change']), legend = 'Beta: ' + str(round(fit_func[1], 3))
#                                                     + '\n' + "Alpha: " + str(round(fit_func[0], 3))))
#             return fig
#     else:
#         self.dates[0], self.dates[1] = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
#         days = self.dates[1].date() - self.dates[0].date()
#         # print(weights)
#         fig = figure(title = frame_title, sizing_mode='fixed', plot_width=670, plot_height= 450, toolbar_location='right')
#         fig.xaxis.axis_label = 'Std. Deviation'
#         fig.yaxis.axis_label = 'Return'
#         # tradier_content(frame_title, dates, date_type = 'hist', weights = weights)
#         price, returns = Security_Portfolio_data(dates).portfolio_content(weights)
#         frame_df = construct_portfolio(dates, weights = weights, close_df = price, returns_df = returns)
#         df = ColumnDataSource(data = {
#             'Deviation': frame_df['Portfolio Deviation'],
#             'Return': frame_df['Portfolio Return']
#         })
#         fig.circle(source = df, x = 'Deviation', y ='Return')
#         fig.add_tools(HoverTool(
#             tooltips=[
#                 ('Risk', '@Deviation'),
#                 ('Return', '@Return{0.2f}%')
#             ],
#             mode='mouse'
#         ))
#         # Doesn't assign colors correctly
#         # mapper = linear_cmap(field_name='Return', palette=Spectral6 ,low=min(frame_df['Sharpe Ratio']) ,high=max(frame_df['Sharpe Ratio']))
#         return fig
#
# def tradier_content(frame_title, dates, date_type = 'hist', weights = None):
#     connection = client.HTTPSConnection('sandbox.tradier.com', 443, timeout=30)
#     headers={'Accept':'application/json'}
#     headers['Authorization'] = Tradier_API
#     if frame_title == 'Portfolio':
#         # Create single dataframe based on weights inputs
#         if weights == None:
#             print('error - no weights given')
#         dfclose = pd.DataFrame(columns = [ticker + ' close' for ticker in weights.keys() if weights[ticker] != ''])
#         dfreturns = pd.DataFrame({ticker + ' Weighted Return':[0] for ticker in weights.keys() if weights[ticker] != ''})
#
#         i = 0
#         for column in dfclose.columns:
#             ticker = column.split(' ')[0]
#             connection.request('GET', '/v1/markets/history?symbol={}&start={}&end={}'.format(ticker, dates[0], dates[1]), None, headers)
#             response = connection.getresponse()
#             content = response.read().decode("utf-8")
#             data = json.loads(content)
#             frame = pd.DataFrame(data['history']['day'])
#             if i != 0:
#                 frame.set_index(pd.to_datetime(frame['date']), inplace=True)
#             dfclose[ticker + ' close'] = frame['close']
#             dfclose[ticker + ' returns'] = frame['close'].pct_change() * 100
#             if not isinstance(dfclose.index, pd.DatetimeIndex):
#                 dfclose.set_index(pd.to_datetime(frame['date']), inplace = True)
#             try:
#                 _ = (dfclose.loc[dates[0], ticker + ' close'] - dfclose.loc[dates[1], ticker + ' close']) / dfclose.loc[dates[1], ticker + ' close']
#                 dfreturns.loc[0, ticker + ' Weighted Return'] = _ * float(weights[ticker])
#             except:
#                 dates[1] = pd.to_datetime(dates[1])-timedelta(1)
#                 _ = (dfclose.loc[dates[0], ticker + ' close'] - dfclose.loc[dates[1], ticker + ' close']) / dfclose.loc[dates[1], ticker + ' close']
#                 dfreturns.loc[0, ticker + ' Weighted Return'] = _ * float(weights[ticker])
#             i += 1
#         return dfclose, dfreturns
#
#     if date_type=='intra':
#         # interval of tick, 1min, 5min, 15min (default)
#         connection.request('GET', '/v1/markets/timesales?symbol={}&interval=1min&start={}&end={}&session_filter={}'.format(frame_title, dates[0], dates[1], 'open'))
#         # Returns dictionary of dictionary of dictionary - keys: 'series', 'data'
#     else:
#         connection.request('GET', '/v1/markets/history?symbol={}&start={}&end={}'.format(frame_title, dates[0], dates[1]), None, headers) # default interval is daily
#     # Returns dictionary of dictionary of dictionary - keys: 'history', 'day'
#     response = connection.getresponse()
#     content = response.read().decode("utf-8")
#     data = json.loads(content)
#     try:
#         df = pd.DataFrame(data['history']['day'])
#         return df.set_index('date')
#     except:
#         try:
#             df = pd.DataFrame(data['series']['data'])
#             return df.set_index('time')
#         except:
#             print('Error in retrieving information for ' + frame_title)
#             return
# def calc_return_type(name, dates, return_type = 'D'):
#     # Aggregate based on choice of Daily, Monthly, Yearly
#     df_copy = dfs[name].copy().set_index('DateTime', inplace=True).loc[pd.to_datetime(dates[0]):pd.to_datetime(dates[1]), :] # Between dates and all columns
#     print(df_copy)
#     entry_len = len(df_copy)
#
#     entry_arr = np.zeros((entry_len, 1))
#     for i in range(entry_arr.shape[0]):
#         entry[i][0] = 1 + df_copy.iloc[i]['Percent Change'] / 100
#     df['1+r'] = entry
#
#     if return_type == 'D':
#         return df_copy
#     elif return_type == 'M':
#         for date in df.index:
#             pass
#
#         return
#     else: #return_type=='Y'
#         pass
#     return
#
# def build_static_graph(frame_title):
#     img = io.BytesIO()
#     dfs[[name for name, ticker in mapping.items() if ticker == frame_title][0]].plot(y = 'Price')
#     plt.savefig(img, format='png', dpi= 100, bbox_inches='tight')
#     img.seek(0)
#     graph_url = base64.b64encode(img.getvalue()).decode()
#     plt.close()
#     return 'data:image/png;base64,{}'.format(graph_url)
#
# def regression(frame_title):
#     img = io.BytesIO()
#     fig, ax = plt.subplots()
#     x = dfs['S&P 500 Index']['Percent Change']
#     y = dfs[[name for name, ticker in mapping.items() if ticker == frame_title][0]]['Percent Change']
#     ax.scatter(x, y)
#     fit_func = np.poly1d(np.polyfit(x, y, deg = 1))
#     (ax.plot(x, fit_func(x), 'r', label = "Beta: " + str(round(fit_func[1], 3)) +
#                                             '\n' + "Alpha: " + str(round(fit_func[0], 3))))
#     ax.get_xaxis().get_label().set_visible(True)
#     ax.set_title("Regression")
#     ax.axhline(0, linestyle = '--', color = 'k', linewidth = .7)
#     ax.axvline(0, linestyle = '--', color = 'k', linewidth = .7)
#     ax.legend(loc = 0)
#
#     plt.savefig(img,  format='png', dpi= 100, bbox_inches='tight')
#     img.seek(0)
#     graph_url = base64.b64encode(img.getvalue()).decode()
#     plt.close()
#     return 'data:image/png;base64,{}'.format(graph_url)
#
# class YahooFinanceHistory:
#     timeout = 2
#     crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
#     crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
#     quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'
#
#     def __init__(self, symbol, days_back=7):
#         self.symbol = symbol
#         self.session = requests.Session()
#         self.dt = timedelta(days=days_back)
#
#     def get_crumb(self):
#         response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
#         response.raise_for_status()
#         match = re.search(self.crumble_regex, response.text)
#         if not match:
#             raise ValueError('Could not get crumb from Yahoo Finance')
#         else:
#             self.crumb = match.group(1)
#
#     def get_quote(self):
#         if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
#             self.get_crumb()
#         now = datetime.utcnow()
#         dateto = int(now.timestamp())
#         datefrom = int((now - self.dt).timestamp())
#         url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb)
#         response = self.session.get(url)
#         response.raise_for_status()
#         return pd.read_csv(StringIO(response.text), parse_dates=['Date'])
