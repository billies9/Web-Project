from flask import Flask

app = Flask(__name__)

from flask import render_template, request, url_for, redirect, flash
from graph import build_interactive_graph, Build_graph
from bokeh.embed import components
from bokeh.resources import CDN
import pandas as pd
from datetime import datetime, timedelta
from indices import link_matches
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
    return render_template('index.html',
                            title=title,
                            stocks=mapping)

class Security():

    def __init__(self, ticker):
        self.ticker = ticker
        self.news_api_key = '47c36eeeae194d00831b85ae1b7efaba'

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
                                        stocks=mapping,
                                        error=dates[0],
                                        error_msg=dates[1],
                                        resources=CDN.render())
            elif dates[0] == None:
                return render_template('securities/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        # div1=div1,
                                        regress=script2,
                                        # div2=div2,
                                        articles=articles,
                                        stocks=mapping,
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
                                stocks=mapping,
                                resources=CDN.render())

    def build_index(self): # Re-work to resemble new codes
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
                                        stocks=mapping,
                                        error=dates[0],
                                        error_msg=dates[1],
                                        resources=CDN.render())
            elif dates[0] == None:
                return render_template('indices/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        # div1=div1,
                                        articles=articles,
                                        stocks=mapping,
                                        resources=CDN.render())
            else:
                script1 = Build_graph(self.ticker, dates).price_graph()
        return render_template('indices/' + self.ticker + '.html',
                                title=self.ticker,
                                price=script1,
                                # div1=div1,
                                articles=articles,
                                stocks=mapping,
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
        all_articles = newsapi.get_everything(q = match_val,
                                            sources = 'the-wall-street-journal',
                                            domains = 'wsj.com',
                                            language = 'en'
                                            # sort_by = 'popularity'
                                            )
        frame = pd.DataFrame(all_articles['articles'])
        if frame.empty:
            return None
        return frame[['title', 'description', 'url', 'urlToImage']]

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
