import os
import pandas as pd
import statistics
import json

def preprocessing_1year():

    data_dict = {}

    datapath = os.path.join('data', '1year')
    
    # Iterate over data files and convert to df
    for fname in os.listdir(datapath):
        fpath = os.path.join(datapath, fname)
        with open(fpath) as csv_file:
            name = fname.split('.')[0]
            stock_data = pd.read_csv(fpath)
            
            # Get list of prices where each price is ' $XX.XX'
            data_dict[name] = {
                'price': [float(stock[2:]) for stock in stock_data[' Close/Last'].to_list()]
            }

            # Calculate returns
            data_dict[name]['return'] = [0]
            for prev_price, curr_price in zip(data_dict[name]['price'][:-1], data_dict[name]['price'][1:]):
                data_dict[name]['return'].append((curr_price - prev_price) / prev_price)

            # Calculate volatility
            data_dict[name]['volatility'] = statistics.stdev(data_dict[name]['return']) * (len(data_dict[name]['price'])**0.5)

            # Assume risk free rate of .01% and find Sharpe ratio
            Rf = 0.0001
            data_dict[name]['sharpe'] = (sum(data_dict[name]['return']) - Rf) / data_dict[name]['volatility']

            # Bollinger bands with 20 day window using [distance to upper band, distance to lower band]
            data_dict[name]['bollinger_bands'] = []
            for day, price in enumerate(data_dict[name]['price']):
                past_20 = data_dict[name]['price'][max(0, day-19) : day+1]
                mu = sum(past_20) / len(past_20)
                var = sum((x - mu) ** 2 for x in past_20) / len(past_20)
                upper_band = mu + 2*var
                lower_band = mu - 2*var
                data_dict[name]['bollinger_bands'].append([abs(upper_band - price), abs(lower_band - price)])

            # Exponential moving average with [12 day EMA, 26 day EMA, 50 day EMA]
            data_dict[name]['ema'] = [[price, price, price] for price in data_dict[name][price]]
            for day, price in enumerate(data_dict[name]['price']):
                smoothing_12 = 2/(1 + 12)
                smoothing_26 = 2/(1 + 26)
                smoothing_50 = 2/(1 + 50)

                if day == 13:
                    ema_base = price
                    for day_iter in range(day-1, day-14, -1):
                        ema_base = smoothing_12 * data_dict[name]['price'][day_iter] + (1-smoothing_12) * ema_base
                    data_dict[name]['ema'][day][0] = ema_base
                elif day > 13:
                    data_dict[name]['ema'][day][0] = smoothing_12 * price + (1-smoothing_12) * data_dict[name]['ema'][day-1][0]

                if day == 27:
                    ema_base = price
                    for day_iter in range(day-1, day-28, -1):
                        ema_base = smoothing_26 * data_dict[name]['price'][day_iter] + (1-smoothing_26) * ema_base
                    data_dict[name]['ema'][day][1] = ema_base
                elif day > 27:
                    data_dict[name]['ema'][day][1] = smoothing_26 * price + (1-smoothing_26) * data_dict[name]['ema'][day-1][1]

                if day == 51:
                    ema_base = price
                    for day_iter in range(day-1, day-52, -1):
                        ema_base = smoothing_50 * data_dict[name]['price'][day_iter] + (1-smoothing_50) * ema_base
                    data_dict[name]['ema'][day][2] = ema_base
                elif day > 51:
                    data_dict[name]['ema'][day][2] = smoothing_50 * price + (1-smoothing_50) * data_dict[name]['ema'][day-1][2]

            # ARIMA
            data_dict[name]['arima'] = []

    return data_dict


if __name__=='__main__':
    stock_data = preprocessing_1year()
    json_file = json.dumps(stock_data)
    f = open("data.json","w")
    f.write(json_file)
    f.close()
