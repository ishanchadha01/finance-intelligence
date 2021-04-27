import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class FirstModel():
    def __init__(self):
        pass

    def output(self, stock_name, stock):
        sharpe = stock['sharpe']
        if sharpe > 0:
            return 1
        elif sharpe < 0:
            return -1
        else:
            return 0


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_dim = 32
        self.num_layers = 2
        self.input_dim = 1
        self.output_dim = 1
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def train_LSTMs(model, data, num_epochs, optimizer, criterion, num_stocks):
    df = data
    cols_per_stock = len(df.columns) // num_stocks
    labels = df.iloc[0, 1:cols_per_stock+1].to_numpy()
    print(labels)

    # Train LSTM for each stock
    for stock in range(num_stocks):

        # Split training data where last 80% is training
        # X data consists of single day data with all 7 measures as inputs
        # Y data consists of next day close prices
        x_train = df.iloc[1+len(df)//5:len(df)-1, (1+stock*7):(1+(stock+1)*7)].to_numpy()
        x_train.resize(x_train.shape[0], x_train.shape[1], 1)
        y_train = df.iloc[2+len(df)//5:len(df), 1+stock*7].to_numpy()
        y_train.resize(y_train.shape[0], 1)

        # First 20% is test data
        x_test = df.iloc[1:1+len(df)//5, (1+stock*7):(1+(stock+1)*7)].to_numpy()
        x_test.resize(x_test.shape[0], x_test.shape[1], 1)
        y_test = df.iloc[2:2+len(df)//5, 1+stock*7].to_numpy()
        y_test.resize(y_test.shape[0], 1)

        # Create torch tensors from testing and training sets
        x_train = torch.from_numpy(np.asarray(x_train, dtype=np.float64)).type(torch.Tensor)
        x_test = torch.from_numpy(np.asarray(x_test, dtype=np.float64)).type(torch.Tensor)
        y_train = torch.from_numpy(np.asarray(y_train, dtype=np.float64)).type(torch.Tensor)
        y_test = torch.from_numpy(np.asarray(y_test, dtype=np.float64)).type(torch.Tensor)

        hist = np.zeros(num_epochs)
        start_time = time.time()
        pred = []
        actual = []
        for t in range(num_epochs):
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train)
            print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_time = time.time()-start_time
        print("Training time: {}".format(training_time))
        torch.save(model.state_dict(), './{}_lstm.pth'.format(stock))
        model.load_state_dict(torch.load('./{}_lstm.pth'.format(stock)))
        plt.plot(hist)
    plt.yscale('log')
    plt.show()

    


if __name__=='__main__':
    model = LSTM()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = pd.read_csv('../data/five_year.csv')
    train_LSTMs(model, data, 100, optimizer, criterion, 10)
