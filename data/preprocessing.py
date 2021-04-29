import os
import pandas as pd
import numpy as np
import statistics
import json

def preprocessing(datapath: str):

    # Full dataframe
    name = datapath.split('.')[0].split('/')[-1]
    stock_data = pd.read_csv(datapath)
    stock_data = stock_data.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
    
    # Get list of close prices
    prices = stock_data.loc[:, 'Close'].values

    # 6 day moving average
    stock_data['MA 7 Day'] = prices.rolling(window=6).mean()

    # 19 day moving average
    stock_data['MA 21 Day'] = prices.rolling(window=19).mean()

    # Bollinger bands with 20 day window using distance to upper band and distance to lower band columns
    past_20_sd = pd.stats.moments.rolling_std(prices, 20)
    stock_data['Upper Band'] = stock_data['MA 19 Day'] + (past_20_sd*2)
    stock_data['Lower Band'] = stock_data['MA 19 Day'] + (past_20_sd*2)

    # 12 day exponential moving average
    stock_data['EMA 12 Day'] = pd.ewma(prices, span=12)

    # 26 day exponential moving average
    stock_data['EMA 26 Day'] = pd.ewma(prices, span=26)

    # MACD line
    stock_data['MACD'] = stock_data['EMA 12 Day'] - stock_data['EMA 26 Day']
        
    return stock_data


def check_null(df):
    df = df.fillna(method='ffill', inplace=True)
    return df


def plot_input(df):
    cols = df.columns.values

if __name__=='__main__':
    stock_data = preprocessing('./MSFT.csv')
    check_null(stock_data)
    print(stock_data)
    stock_data.to_csv('./five_year.csv')
