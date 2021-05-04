import numpy as np
import pandas as pd
import sys
sys.path.append('../models/')
from lstm import LSTM, test_LSTMs

def simulation(outputs):
    preds, tests = outputs[0].flatten(), outputs[1].flatten()
    for pred, actual in zip(preds, tests):
        yield pred, actual

def trader(market_sim):

    # Outputs passed in as (predictions, actual)
    money = 0
    shares = 0
    curr = next(market_sim, None)[1]
    tmrw = next(market_sim, None)
    while tmrw:
        pred, actual = tmrw

        # If stock going up tmrw, buy one share
        if pred >= curr:
            shares += 1
            money -= curr

        # If stock going down, sell all shares
        else:
            money += curr * shares
            shares = 0

        # Iterate
        curr = tmrw[1]
        tmrw = next(market_sim, None)
    money += shares * curr
    return money

if __name__=='__main__':
    data = pd.read_csv('../data/input.csv')
    model = LSTM()
    outputs = test_LSTMs('GE', model, data, 90000)
    market_sim = simulation(outputs)
    t0 = trader(market_sim)