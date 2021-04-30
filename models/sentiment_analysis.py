import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib
import matplotlib.pyplot as plt

'''
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
'''


def load_data(datapath: str):

    # Load training data
    df = pd.read_csv(datapath)
    train_x, temp_x, train_y, temp_y = train_test_split(
        df['text'], df['label'],
        test_size=0.4, 
        stratify=df['label']                                                                
    )

    # Load testing and validation data
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x,
        temp_y,
        test_size=0.5, 
        stratify=temp_y
    )

    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


    # get length of all the messages in the train set
    # seq_len = [len(i.split()) for i in train_y]
    # lens = pd.Series(seq_len).hist(bins = 30)
    # lens.plot()
    # plt.title('Message Lengths Sentiment Analysis')
    # plt.xlabel('Characters in Message')
    # plt.ylabel('Occurrences')
    # plt.savefig('../plots/message_lengths_sentiment_analysis.png')
    max_seq_len = 50


    # Tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_y.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # Tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_x.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # Tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_x.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # Train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_y.tolist())

    # Validation set
    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_y.tolist())

    # Test set
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_y.tolist())

    # Dataloaders
    batch_size = 32
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False

    return bert


class BertSentimentClassifier(nn.Module):

    def __init__(self, bert):
        super(BertSentimentClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu =  nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__=='__main__':
    print(load_data('../data/raw_data/financial_news_data.csv'))