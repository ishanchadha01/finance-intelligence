import os
import pandas as pd
import numpy as np
import json
from transformers import AutoModel, AutoTokenizer
import sys
sys.path.append('../models/')
from sentiment_analysis import BertSentimentClassifier
import torch


def get_prices(datapath: str):
    stock_data = pd.read_csv(datapath)
    return stock_data['Close']


def get_date(datapath: str):
    stock_data = pd.read_csv(datapath)
    return stock_data['Date']


def sent_analysis(datapath: str, model, tokenizer, max_seq_len):
    # Load data and model
    df = pd.read_csv(datapath)
    data = df['Text']
    model.load_state_dict(torch.load('../models/saved_models/trained_bert.pth'))
    tokens = tokenizer.batch_encode_plus(
        data.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])

    # Get predictions for positive, neutral, or negative sentiment
    with torch.no_grad():
        preds = model(seq, mask)
    preds = np.argmax(preds, axis=1)
    
    # Add new column containing sentiments if doesn't already exist and write to csv
    sentiments = preds.numpy() - 1
    if 'Sentiments' in df.columns.values:
        df["Sentiments"] = sentiments
    else:
        df.insert(2, 'Sentiments', sentiments)
    df.to_csv(datapath, index=False)


def preprocessing(datapath: str):
    
    # Get list of close prices
    stock_data = pd.DataFrame()
    stock_data['Date'] = get_date(datapath)
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
    stock_data.fillna(method='backfill', inplace=True)

    # Sentiment analysis result, with -1,0,1 indicating negative, neutral, positive
    stock_data = stock_data.set_index('Date')
    sentiment_data = pd.read_csv('./raw_data/news_scraped_data.csv', index_col=0)
    sentiment_data = sentiment_data.drop('Text', axis=1)
    grouped = sentiment_data.groupby('Date')
    sums = grouped.sum()
    sentiments = sums.div(grouped.count(), axis=0)
    pd.to_datetime(stock_data.index)
    pd.to_datetime(sentiments.index)
    stock_data = stock_data.join(sentiments, on='Date', how='left')
    stock_data = stock_data.fillna(0)

    # Options ADV
    stock_data['Options ADV'] = pd.read_csv('./raw_data/GE_options_volumes.csv')['Volume']

    # Anomaly detection on currency basket

    return stock_data


def plot_input(df):
    cols = df.columns.values

if __name__=='__main__':
    # Write sentiment data if doesn't already exist
    # bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)
    # model = BertSentimentClassifier(bert)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
    # max_seq_len = 50
    # sent_analysis('./raw_data/news_scraped_data.csv', model, tokenizer, max_seq_len)

    stock_data = preprocessing('./raw_data/GE.csv')
    print(stock_data)
    plot_input(stock_data)
    stock_data.to_csv('./input.csv')
