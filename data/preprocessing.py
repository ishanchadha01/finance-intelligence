import os
import pandas as pd
import numpy as np
import json


def get_prices(datapath: str):
    stock_data = pd.read_csv(datapath)
    return stock_data['Close']



def preprocessing(datapath: str):
    
    # Get list of close prices
    stock_data = pd.DataFrame()
    stock_data['Prices'] = get_prices(datapath)
    prices = stock_data['Prices']

    # 6 day moving average
    stock_data['MA 7 Day'] = prices.rolling(window=6).mean()

    # 19 day moving average
    stock_data['MA 21 Day'] = prices.rolling(window=19).mean()

    # Bollinger bands with 20 day window using distance to upper band and distance to lower band columns
    past_20_sd = prices.rolling(window=20).std()
    stock_data['Upper Band'] = stock_data['MA 21 Day'] + (past_20_sd*2)
    stock_data['Lower Band'] = stock_data['MA 21 Day'] + (past_20_sd*2)

    # 12 day exponential moving average
    stock_data['EMA 12 Day'] = prices.ewm(span=12).mean()

    # 26 day exponential moving average
    stock_data['EMA 26 Day'] = prices.ewm(span=26).mean()

    # MACD line
    stock_data['MACD'] = stock_data['EMA 12 Day'] - stock_data['EMA 26 Day']

    # E-Mini S&P 500 Futures
    stock_data['S&P Futures'] = get_prices('./raw_data/ES_F.csv')

    # EURUSD
    stock_data['EURUSD'] = get_prices('./raw_data/EURUSD.csv')

    # EURJPY
    stock_data['EURJPY'] = get_prices('./raw_data/EURJPY.csv')

    # USDJPY
    stock_data['USDJPY'] = get_prices('./raw_data/USDJPY.csv')

    # Sentiment analysis result, 0 to 1 score

    return stock_data


def check_null(df):
    df = df.fillna(method='backfill', inplace=True)
    return df


def plot_input(df):
    cols = df.columns.values

if __name__=='__main__':
    stock_data = preprocessing('./raw_data/GE.csv')
    check_null(stock_data)
    print(stock_data)
    plot_input(stock_data)
    stock_data.to_csv('./input.csv')
