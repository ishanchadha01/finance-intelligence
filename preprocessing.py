import pandas as pd
import numpy as np


def envByDates():
    df = pd.read_csv('data/prices.csv')
    date_dict = {}
    for idx, row in enumerate(df.iloc()):
        date = row['date'].split(' ')[0]
        if date in date_dict.keys():
            date_dict[date].append(row[1:].to_dict())
        else:
            date_dict[date] = [row[1:].to_dict()]

        # Only do first 100 days
        if idx == 100:
            break

    names = list(df['symbol'].unique())
    for date in date_dict:
        stocks = [ stock['symbol'] for stock in date_dict[date] ]
        for name in names:
            if name not in stocks:
                date_dict[date].append({
                    'symbol': name,
                    'open': 1.0, 
                    'close': 1.0, 
                    'low': 1.0, 
                    'high': 1.0, 
                    'volume': 0.0
                    })
    
    dates = list(date_dict.keys())
    dates.sort()

    return dates, date_dict

