url = 'https://markets.wsj.com/us'#'https://www.bloomberg.com/quote/INDU:IND',
    #'https://www.bloomberg.com/quote/SPX:IND',
    #'https://www.bloomberg.com/quote/AAPL:US',
    #'https://www.bloomberg.com/quote/AMZN:US',
    #'https://www.bloomberg.com/quote/NKE:US',
    #'https://www.bloomberg.com/quote/GOOGL:US',
    #'https://www.bloomberg.com/quote/UAA:US',
    #'https://www.bloomberg.com/quote/TSLA:US',
    #'https://www.bloomberg.com/quote/FB:US'
main_page = 'https://quotes.wsj.com/'

tickers = [ 'index/DJIA',
            'index/SPX',
            'AAPL',
            'GOOGL',
            'UAA',
            'TSLA',
            'FB',
            'AMZN',
            'MSFT'
            ]

webpage = ['https://www.bloomberg.com/markets',
           'https://www.bloomberg.com/technology',
           #'https://www.wsj.com/news/markets',
           'https://www.wsj.com/news/business',
           #'https://www.wsj.com/news/economy'
           ]
markets_page = "https://www.wsj.com/news/markets"

# Matches for article hits
Tesla_matches = ({"TSLA": ('Elon Musk', 'Musk', 'Elon', 'Tesla', 'Tesla Inc.', 'EV', 'Technology',"Stocks", "Electric", "Stock",
                           "Factory", "China", "Tech")})
Apple_matches = ({"AAPL": ('Apple', 'mobile', 'technology', "Tech", "Samsung", "Phones", "Phone", "FANG", "China", "Chinese")})
Amazon_matches = ({"AMZN": ('Amazon', 'Bezos', 'Jeff Bezos', "Retail", "Online", "FANG")})
Nike_matches = {"NKE": ('Nike', 'Athletic-wear', 'athletic', "Shoes", "Retail", "Online")}
Google_matches = {"GOOGL": ('Google', 'Alphabet Inc.', 'Alphabet', "Privacy", "Tech", "FANG")}
UA_matches = {"UAA": ('Under Armour', 'UA', "Athletic", "Retail")}
MSFT_matches = {"MSFT": ("Bill Gates", "Gates", "Windows")}
Facebook_matches = {"FB": ('Mark Zuckerberg', 'Zuckerberg', 'Facebook', 'FB', "Privacy", "Data")}
DJIA_matches = {"DJIA": ("Index", "Dow", "DJIA", "Dow Jones", "Industrials")}
SPX_matches = {"SPX": ("Index", "SPX", "S&P500")}
link_matches = (dict(Tesla_matches, **Apple_matches, **Amazon_matches, **Nike_matches, **Google_matches, **UA_matches,
                     **Facebook_matches, **MSFT_matches, **DJIA_matches, **SPX_matches))

# Matches within the articles themselves
word_dict = {'stocks': 0,
             'rise': 0,
             'higher':0,
             'raise': 0,
             'fall': 0 ,
             'lower': 0,
             'Trump': 0,
             'Administration':0,
             'Trump Administration': 0,
             'S&P 500 Index':0,
             'SPX':0,
             'Standards and Poors 500 Index': 0,
             'destroying': 0,
             'Dow':0,
             'Dow Jones Industrial Average':0,
             'DJIA': 0,
             'Beijing-based': 0


             }
#print(word_dict)
