from flask import Flask

app = Flask(__name__)

# import routes
from flask import render_template, request, url_for, redirect, flash
import matplotlib.pyplot as plt
from graph import build_static_graph, regression, build_interactive_graph
from bokeh.embed import components
from bokeh.resources import CDN
from portfolio_page_construction import import_names, Article_Scrape
import pandas as pd
from datetime import datetime, timedelta

mapping = {"Microsoft Corp.": "MSFT",
           "Amazon.com Inc.": "AMZN",
           "Facebook Inc. Cl A": "FB",
           "Tesla Inc.": "TSLA",
           "Under Armour Inc. Cl A": "UAA",
           "Alphabet Inc. Cl A": "GOOGL",
           "Apple Inc.": "AAPL",
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
        #all_tickers = ['AAPL', 'AMZN', 'MSFT', 'TSLA', 'FB', 'UAA', 'GOOGL', 'SPX', 'DJIA']
        #self.articles = Article_Scrape(keys = all_tickers)

    def build_security(self):
        dates = (self.monthdelta(pd.to_datetime('today'), -1).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))
        script1, div1 = components(build_interactive_graph(self.ticker, 'P', dates = dates)) # Log default dates of last 1M
        script2, div2 = components(build_interactive_graph(self.ticker, 'R', dates = dates))
        articles = Article_Scrape(keys = [self.ticker,])
        # try: articles = Article_Scrape(keys = [self.ticker,])
        # except: articles = None
        if request.method == 'POST':
            dates = self.build_range_dates()
            if dates[0] is True:
                return render_template('securities/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        div1=div1,
                                        regress=script2,
                                        div2=div2,
                                        articles=articles,
                                        stocks=mapping,
                                        error=dates[0],
                                        error_msg=dates[1],
                                        resources=CDN.render())
            elif dates[0] == None:
                return render_template('securities/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        div1=div1,
                                        regress=script2,
                                        div2=div2,
                                        articles=articles,
                                        stocks=mapping,
                                        resources=CDN.render())
            else:
                script1, div1 = components(build_interactive_graph(frame_title = self.ticker, dates = dates[0], type = 'P', date_type = dates[1])) # Custom dates as given by the user
                script2, div2 = components(build_interactive_graph(frame_title = self.ticker, dates = dates[0], type = 'R', date_type = dates[1])) # Replace former graphs with custom date ranges
        return render_template('securities/' + self.ticker + '.html',
                                title=self.ticker,
                                price=script1,
                                div1=div1,
                                regress=script2,
                                div2=div2,
                                articles=articles,
                                stocks=mapping,
                                resources=CDN.render())

    def build_index(self): # Re-work to resemble new codes
        dates = (self.monthdelta(pd.to_datetime('today'), -1).strftime('%Y-%m-%d'), pd.to_datetime('today').strftime('%Y-%m-%d'))
        script1, div1 = components(build_interactive_graph(self.ticker, 'P', dates = dates))
        try: articles = Article_Scrape(keys = [self.ticker,])
        except: articles = None
        if request.method == 'POST':
            dates = self.build_range_dates()
            if dates[0] is True:
                return render_template('indices/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        div1=div1,
                                        articles=articles,
                                        stocks=mapping,
                                        error=dates[0],
                                        error_msg=dates[1],
                                        resources=CDN.render())
            elif dates[0] == None:
                return render_template('indices/' + self.ticker + '.html',
                                        title=self.ticker,
                                        price=script1,
                                        div1=div1,
                                        articles=articles,
                                        stocks=mapping,
                                        resources=CDN.render())
            else:
                script1, div1 = components(build_interactive_graph(frame_title = self.ticker, dates = dates, type = 'P')) # Custom dates as given by the user
        return render_template('indices/' + self.ticker + '.html',
                                title=self.ticker,
                                price=script1,
                                div1=div1,
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
            #flash('You must select a Beg. Date that occurs prior to the End Date.')
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
