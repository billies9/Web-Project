import pickle
import pandas as pd

#with open(r'C:\Users\billies9\OneDrive\Documents\Python_Screwaround\Stock_Scraper\Good_Project\Stocks.pickle', 'rb') as file:
#    dfs = pickle.load(file)
#print(dfs)

df=pd.read_excel(io=r'C:\Users\billies9\OneDrive\Documents\Python_Screwaround\Stock_Scraper\Good_Project\practice.xlsx', sheetname=None, skiprows=2, usecols = [0, 1, 2])

print(df)
