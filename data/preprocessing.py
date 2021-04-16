import os
import pandas as pd
import numpy as np
import statistics
import json

def preprocessing_5year(datapath: str):

    # Full dataframe
    df = pd.DataFrame()
    
    # Iterate over data files and convert to dict
    for fname in os.listdir(datapath):
        fpath = os.path.join(datapath, fname)
        name = fname.split('.')[0]
        stock_data = pd.read_csv(fpath)
        stock_data = stock_data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
        
        # Get list of close prices
        prices = stock_data.loc[:, 'Close'].values

        # Calculate returns with (curr_price - prev_price) / prev_price
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.concatenate((np.zeros(1), returns))
        stock_data['Returns'] = returns

        # Bollinger bands with 20 day window using distance to upper band and distance to lower band columns
        upper_band = []
        lower_band = []
        for day, price in enumerate(prices):
            past_20 = prices[max(0, day-19) : day+1]
            mu = np.sum(past_20) / past_20.size
            var = np.sum([(x - mu) ** 2 for x in past_20]) / past_20.size
            upper_band.append(np.abs(mu + 2*var - price))
            lower_band.append(np.abs(mu - 2*var - price))
        upper_band = np.array(upper_band)
        lower_band = np.array(lower_band)
        stock_data['Upper Band'] = upper_band
        stock_data['Lower Band'] = lower_band

        # Exponential moving average with [12 day EMA, 26 day EMA, 50 day EMA]
        ema = [[price, price, price] for price in prices]
        for day, price in enumerate(prices):

            # Calculate smoothing params for each EMA length
            smoothing_12 = 2/(1 + 12)
            smoothing_26 = 2/(1 + 26)
            smoothing_50 = 2/(1 + 50)

            # 12 day ema
            if day == 13:
                ema_base = price
                for day_iter in range(day-1, day-14, -1):
                    ema_base = smoothing_12 * prices[day_iter] + (1-smoothing_12) * ema_base
                ema[day][0] = ema_base
            elif day > 13:
                ema[day][0] = smoothing_12 * price + (1-smoothing_12) * ema[day-1][0]

            # 26 day ema
            if day == 27:
                ema_base = price
                for day_iter in range(day-1, day-28, -1):
                    ema_base = smoothing_26 * prices[day_iter] + (1-smoothing_26) * ema_base
                ema[day][1] = ema_base
            elif day > 27:
                ema[day][1] = smoothing_26 * price + (1-smoothing_26) * ema[day-1][1]

            # 50 day ema
            if day == 51:
                ema_base = price
                for day_iter in range(day-1, day-52, -1):
                    ema_base = smoothing_50 * prices[day_iter] + (1-smoothing_50) * ema_base
                ema[day][2] = ema_base
            elif day > 51:
                ema[day][2] = smoothing_50 * price + (1-smoothing_50) * ema[day-1][2]

        # Convert ema to columns
        ema = np.array(ema).T
        stock_data['EMA 12 Day'] = ema[0]
        stock_data['EMA 26 Day'] = ema[1]
        stock_data['EMA 50 Day'] = ema[2]

        names = np.array([name] * 8)
        stats = stock_data.columns.values
        data = stock_data.to_numpy().T
        stock_data = pd.DataFrame(data.T, columns=pd.MultiIndex.from_tuples(zip(names, stats)))
        if df.empty:
            df = stock_data
        else:
            df = df.combine_first(stock_data)
    return df


if __name__=='__main__':
    stock_data = preprocessing_5year('./five_year')
    stock_data.to_csv('./five_year.csv')
