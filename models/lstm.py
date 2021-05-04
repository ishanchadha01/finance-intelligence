import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


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
        h0 = (torch.rand(self.num_layers, x.size(0), self.hidden_dim) * 1000).requires_grad_()
        c0 = (torch.rand(self.num_layers, x.size(0), self.hidden_dim) * 1000).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def train_LSTMs(name, model, data, num_epochs, optimizer, criterion):
    df = data

    # Split training data where last 80% is training
    # X data consists of single day data with all measures as inputs
    # Y data consists of next day close prices
    x_train = df.iloc[len(df)//5:-1, 1:].to_numpy()
    x_train.resize(x_train.shape[0], x_train.shape[1], 1)
    y_train = df.iloc[1+len(df)//5:, 1].to_numpy()
    y_train.resize(y_train.shape[0], 1)

    # Create torch tensors from training sets
    x_train = torch.from_numpy(np.asarray(x_train, dtype=np.float64)).type(torch.Tensor)
    y_train = torch.from_numpy(np.asarray(y_train, dtype=np.float64)).type(torch.Tensor)

    # Train LSTMs
    hist = np.zeros(num_epochs)
    start_time = time.time()
    curr_losses = []
    for t in tqdm(range(num_epochs)):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        hist[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print MSE ever 2000 Epochs
        if t % 100 == 0:
            print("Epoch ", t, "MSE: ", loss.item())

        # Save LSTM to file
        if t % (10000) == 0:
            torch.save(model.state_dict(), './saved_models/lstm/{}_lstm_{}.pth'.format(name, t))
            curr_losses.append(float(loss.item()))

            # Break at checkpoint if loss is small
            if loss.item() < .5:
                break
    
    # Write losses to loss csv
    with open('./results/LSTM/loss.csv', 'w', newline='') as loss_file:
        wr = csv.writer(loss_file, quoting=csv.QUOTE_ALL)
        wr.writerow([name] + curr_losses)

    training_time = time.time()-start_time
    torch.save(model.state_dict(), './saved_models/lstm/{}_lstm.pth'.format(name))
    print("Training time: {}".format(training_time))
    
    # Plot histogram of losses
    # plt.plot(hist)


def test_LSTMs(name, model, data, iter_num):
    df = data

    # First 20% is test data
    x_test = df.iloc[:len(df)//5, 1:].to_numpy()
    x_test.resize(x_test.shape[0], x_test.shape[1], 1)
    y_test = df.iloc[1:1+len(df)//5, 1].to_numpy()
    y_test.resize(y_test.shape[0], 1)

    # Create torch tensors from testing sets
    x_test = torch.from_numpy(np.asarray(x_test, dtype=np.float64)).type(torch.Tensor)
    y_test = torch.from_numpy(np.asarray(y_test, dtype=np.float64)).type(torch.Tensor)

    # Load trained models
    model.load_state_dict(torch.load('../models/saved_models/lstm/{}_lstm_{}.pth'.format(name, iter_num)))

    outputs = model(x_test)

    # Plot
    # plt.plot(outputs.detach().numpy(), label='Predictions')
    # plt.plot(y_test.numpy(), label='Actual Prices')
    # plt.xlabel('Days')
    # plt.ylabel('Prices')
    # plt.title('LSTM Predictions')
    # plt.show()

    return outputs.detach().numpy(), y_test.numpy()

    


if __name__=='__main__':
    data = pd.read_csv('../data/input.csv')
    model = LSTM()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    #train_LSTMs('GE', model, data, 1000000, optimizer, criterion)
    test_LSTMs('GE', model, data, 90000)
    plt.plot(pd.read_csv('./results/lstm/loss.csv', header=None).to_numpy()[0][1:])
    plt.title('LSTM Loss')
    plt.xlabel('Epochs')
    plt.show()
