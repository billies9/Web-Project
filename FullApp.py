from flask import Flask

app = Flask(__name__)

from flask import render_template, request, url_for, redirect, flash
from graph import Build_graph, Security_Portfolio_data
from bokeh.embed import components
from bokeh.resources import CDN
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
# from collections import OrderedDict
# from indices import link_matches
from newsapi import NewsApiClient

mapping = {"Microsoft": "MSFT",
           "Amazon": "AMZN",
           "Facebook": "FB",
           "Tesla": "TSLA",
           "Under Armour": "UAA",
           "Alphabet": "GOOGL",
           "Apple": "AAPL",
           "S&P 500 Index": "SPX",
           "Dow Jones Industrial Average": "DJIA"}

@app.route('/')
@app.route('/index')
# Creates association between URL given as argument and function
# Assigning two of them, Flask requests either of two URLs and returns value of function
def index():
    title = 'HomePage'
    ord_mapping = {key:val for key, val in sorted(mapping.items())}
    return render_template('index.html',
                            title=title,
                            stocks=ord_mapping)

class Security():
    def __init__(self, ticker):
        self.ticker = ticker
        self.news_api_key = '47c36eeeae194d00831b85ae1b7efaba'
        self.ord_mapping = {key:val for key, val in sorted(mapping.items())}

    def build_security(self):
        dates = (self.monthdelta(pd.to_datetime('today'), -1).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))

        script1 = Build_graph(self.ticker, (dates, 'hist')).price_graph()
        script2 = Build_graph(self.ticker, (dates, 'hist')).regression_graph()

        articles = self.Articles(self.news_api_key)

        if request.method == 'POST':
            dates = self.build_range_dates()
            if dates[0] is True:
                return render_template('securities/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        # div1=div1,
                                        regress=script2,
                                        # div2=div2,
                                        articles=articles,
                                        stocks=self.ord_mapping,
                                        error=dates[0],
                                        error_msg=dates[1],
                                        company_info=self.get_company_info(),
                                        financial_info=self.get_financial_ratios(),
                                        resources=CDN.render())
            elif dates[0] == None:
                return render_template('securities/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        # div1=div1,
                                        regress=script2,
                                        # div2=div2,
                                        articles=articles,
                                        stocks=self.ord_mapping,
                                        company_info=self.get_company_info(),
                                        financial_info=self.get_financial_ratios(),
                                        resources=CDN.render())
            else:
                script1 = Build_graph(self.ticker, dates).price_graph()
                script2 = Build_graph(self.ticker, dates).regression_graph()
        return render_template('securities/' + self.ticker + '.html',
                                title=self.ticker,
                                price=script1,
                                # div1=div1,
                                regress=script2,
                                # div2=div2,
                                articles=articles,
                                stocks=self.ord_mapping,
                                company_info=self.get_company_info(),
                                financial_info=self.get_financial_ratios(),
                                resources=CDN.render())

    def build_index(self):
        # Get Sector performances here for SPX, etc.
        dates = (self.monthdelta(pd.to_datetime('today'), -1).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))

        script1 = Build_graph(self.ticker, dates).price_graph()

        articles = self.Articles(self.news_api_key)
        if request.method == 'POST':
            dates = self.build_range_dates()
            if dates[0] is True:
                return render_template('indices/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        # div1=div1,
                                        articles=articles,
                                        stocks=self.ord_mapping,
                                        error=dates[0],
                                        error_msg=dates[1],
                                        resources=CDN.render())
            elif dates[0] == None:
                return render_template('indices/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        # div1=div1,
                                        articles=articles,
                                        stocks=self.ord_mapping,
                                        resources=CDN.render())
            else:
                script1 = Build_graph(self.ticker, dates).price_graph()
        return render_template('indices/' + self.ticker + '.html',
                                title=self.ticker,
                                price=script1,
                                # div1=div1,
                                articles=articles,
                                stocks=self.ord_mapping,
                                resources=CDN.render())

    def build_range_dates(self):
        range_dates = request.form
        date_type = 'hist'
        # Shouldn't be able to select daily in specified date change - only in daily area
        if range_dates['date1'] > range_dates['date2'] and range_dates['date2'] != '':
            error = True
            print(error)
            return error, 'Date Range Selector - End Date occurs before Beg. Date'
        if range_dates['date1'] != '':
            if range_dates['date2'] == '':
                # Default yesterday as date - ImmutableDict so must assign to dates tuple
                dates = (range_dates['date1'], pd.to_datetime(datetime.today() - timedelta(days = 1)).date())
                return dates, date_type
            dates = (range_dates['date1'], range_dates['date2'])
            return dates, date_type
        elif range_dates['range'] == '1D':
            # Need to figure out how to do daily delta change
            # Needs to be yesterday before market opens and today when market opens - Tradier handles this, but parse through datetimes
            date_type='intra'
            pass
        elif range_dates['range'] == '3M':
            dates = (self.monthdelta(pd.to_datetime('today'), -3).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))
        elif range_dates['range'] == '6M':
            dates = (self.monthdelta(pd.to_datetime('today'), -6).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))
        elif range_dates['range'] == '1Y':
            dates = (self.monthdelta(pd.to_datetime('today'), -12).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))
        elif range_dates['range'] == '5Y':
            dates = (self.monthdelta(pd.to_datetime('today'), -60).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))
        else: # Nothing specified - default one month
            dates = (self.monthdelta(pd.to_datetime('today'), -1).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))
        return dates, date_type

    def monthdelta(self, date, delta):
        m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
        if not m: m = 12
        d = min(date.day, [31, 29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
        return date.replace(day=d,month=m, year=y)

    def Articles(self, api_key):
        newsapi = NewsApiClient(api_key = api_key)
        # Get keywords for ticker based on its list matches
        match_val = [key for key, val in mapping.items() if val == self.ticker][0]
        days_25_prev = (datetime.today() - timedelta(25)).strftime("%Y-%m-%d")
        yesterday = (datetime.today() - timedelta(1)).strftime("%Y-%m-%d")
        all_articles = newsapi.get_everything(q = match_val,
                                            sources = 'the-wall-street-journal',
                                            domains = 'wsj.com',
                                            language = 'en',
                                            from_param = days_25_prev,
                                            to = yesterday
                                            )
        frame = pd.DataFrame(all_articles['articles'])
        if frame.empty:
            return None
        return frame[['title', 'description', 'url', 'urlToImage']]

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
        latest_data = (json.loads(r.text))['financialRatios']['2018-09'] # Find a way to review latest market data by date...
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
            rel_data['Liquidity'][measure_rename] = round(latest_data['liquidityMeasurementRatios'][measure], 4)

        # Profitability Measures - Gross Profit, ROE, Effective Tax Rate
        prof = ['grossProfitMargin', 'returnOnEquity', 'effectiveTaxRate']
        for measure in prof:
            if measure == prof[0]:
                measure_rename = 'Gross Profit Margin'
            elif measure == prof[1]:
                measure_rename = 'Return on Equity'
            else:
                measure_rename = 'Effective Tax Rate'
            rel_data['Profitability'][measure_rename] = round(latest_data['profitabilityIndicatorRatios'][measure], 4)

        # Debt Measures - Debt, Debt-to-Equity, Interest Coverage
        debt = ['debtRatio', 'debtEquityRatio', 'interestCoverageRatio']
        for measure in debt:
            if measure == debt[0]:
                measure_rename = 'Debt Ratio'
            elif measure == debt[1]:
                measure_rename = 'Debt-to-Equity Ratio'
            else:
                measure_rename = 'Interest Coverage Ratio'
            rel_data['Debt'][measure_rename] = round(latest_data['debtRatios'][measure], 4)

        # Operating Performance - Asset Turnover
        ops = ['assetTurnover']
        for measure in ops:
            if measure == ops[0]:
                measure_rename = 'Asset Turnover Ratio'
            rel_data['Operating'][measure_rename] = round(latest_data['operatingPerformanceRatios'][measure], 4)

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
            rel_data['Investment'][measure_rename] = str(round(latest_data['investmentValuationRatios'][measure], 4) * multiplier) + pct
        return rel_data

@app.route('/indices/SPX', methods = ['GET', 'POST'])
def SPX(ticker = 'SPX'):
    return Security(ticker).build_index()

@app.route('/indices/DJIA', methods = ['GET', 'POST'])
def DJIA(ticker = 'DJIA'):
    return Security(ticker).build_index()

@app.route('/securities/FB', methods = ['GET', 'POST'])
def FB(ticker = 'FB'):
    return Security(ticker).build_security()

@app.route('/securities/MSFT', methods = ['GET', 'POST'])
def MSFT(ticker = 'MSFT'):
    return Security(ticker).build_security()

@app.route('/securities/UAA', methods = ['GET', 'POST'])
def UAA(ticker = 'UAA'):
    return Security(ticker).build_security()

@app.route('/securities/AMZN', methods = ['GET', 'POST'])
def AMZN(ticker = 'AMZN'):
    return Security(ticker).build_security()

@app.route('/securities/AAPL', methods = ['GET', 'POST'])
def AAPL(ticker = 'AAPL'):
    return Security(ticker).build_security()

@app.route('/securities/GOOGL', methods = ['GET', 'POST'])
def GOOGL(ticker = 'GOOGL'):
    return Security(ticker).build_security()

@app.route('/securities/TSLA', methods=['GET', 'POST'])
def TSLA(ticker = 'TSLA'):
    return Security(ticker).build_security()

@app.route('/', methods=['GET', 'POST'])
def get_page():
    return redirect(url_for(request.form['security']))

@app.route('/portfolio/create', methods=['GET', 'POST'])
def create_portoflio():
    title = 'Portfolio Construction'
    # names = import_names()
    weights = {}
    if request.method =='POST':
        dates = (request.form.get('date1'), request.form.get('date2'))
        if dates[0] > dates[1]:
            # Want something to flash here...
            return render_template('portfolio/create.html',
                                    title=title,
                                    stocks=mapping,
                                    resources=CDN.render())
        weights = {ticker: weight for (ticker, weight) in request.form.items() if ticker in mapping.values()}
        script1, div1 = components(build_interactive_graph('Portfolio', 'Port', weights = weights, dates = dates))
        return render_template('portfolio/create.html',
                                title=title,
                                stocks=mapping,
                                weights=weights,
                                port=script1,
                                div1=div1,
                                resources=CDN.render())
    return render_template('portfolio/create.html',
                            title=title,
                            stocks=mapping)


if __name__ == '__main__':
    app.run()
