import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('../plots')
from plot_classification_report import plot_classification_report


class BertSentimentClassifier(nn.Module):

    def __init__(self, bert):
        super(BertSentimentClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu =  nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def train_val_test_bert(model, tokenizer, optimizer, data, batch_size, num_epochs):
    
    df = data

    # Load training data
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
        train_x.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

    # Tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_x.tolist(),
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

    # Tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_x.tolist(),
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

    # Train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y_copy = train_y
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
    batch_size = batch_size
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    # Freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False

    # Get class weights and loss function
    class_wts = compute_class_weight('balanced', np.unique(train_y_copy), train_y_copy)
    weights = torch.tensor(class_wts,dtype=torch.float)
    cross_entropy_loss = nn.NLLLoss(weight=weights)
    
    # Perform training and validation
    best_valid_loss = float('inf')
    train_losses=[]
    valid_losses=[]

    # Iterate over epochs
    for epoch in range(num_epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, num_epochs))
        
        # Train model
        total_loss, total_accuracy = 0,0
        
        # Iterate over batches
        for step, batch in tqdm(enumerate(train_dataloader)):
            
            # Give progress every 50 batches
            if step % 20 == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

            # Clear gradient, compute loss, and backward pass for new gradient
            sent_id, mask, labels = batch
            model.zero_grad()
            preds = model(sent_id, mask)
            loss = cross_entropy_loss(preds, labels)
            total_loss = total_loss + loss.item()
            loss.backward()

            # Clip gradients to prevent exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Step forward
            optimizer.step()

        # Training loss of the epoch
        train_loss = total_loss / len(train_dataloader)
        
        # Evaluate model
        print("\nEvaluating...")
        model.eval()
        total_loss, total_accuracy = 0,0

        # Iterate over batches
        for step,batch in enumerate(val_dataloader):
            
            # Give progress every 50 batches
            if step % 20 == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            # Compute validation loss
            sent_id, mask, labels = batch
            with torch.no_grad():
                preds = model(sent_id, mask)
                loss = cross_entropy_loss(preds,labels)
                total_loss = total_loss + loss.item()

        valid_loss = total_loss / len(val_dataloader)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './saved_models/trained_bert.pth')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Sentiment Analysis Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('../plots/loss_sentiment_analysis.png')

    # Testing
    model.load_state_dict(torch.load('./saved_models/trained_bert.pth'))
    with torch.no_grad():
        preds = model(test_seq, test_mask)
    preds = np.argmax(preds, axis=1)
    cr = classification_report(test_y, preds)
    plot_classification_report(cr, '../plots/classification_report_sentiment_analysis.png')



if __name__=='__main__':
    bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
    batch_size = 32
    data = pd.read_csv('../data/raw_data/news_training_data.csv')
    model = BertSentimentClassifier(bert)
    optimizer = AdamW(model.parameters(), lr = 1e-3)
    num_epochs = 25
    train_val_test_bert(model, tokenizer, optimizer, data, batch_size, num_epochs)