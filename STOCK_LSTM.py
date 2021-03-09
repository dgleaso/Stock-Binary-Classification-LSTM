#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import math
import random

sp500 = yf.Ticker("^GSPC")

def normalize(x):
    x = (x - x.min())/(x.max() - x.min())
    return x

def preproc(data):
    data_open = normalize(data['Open'].values)
    data_close = normalize(data['Close'].values)
    data_high = normalize(data['High'].values)
    data_low = normalize(data['Low'].values)
    data_volume = normalize(data['Volume'].values)
    data_direction = data_open - data_close
    # Changed for binary classification
    data_direction[data_direction < 0] = 0
    data_direction[data_direction > 0] = 1
    x = np.stack([data_open, data_close, data_high, data_low, data_volume])
    return x, data_direction

class RecurrentNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(RecurrentNet, self).__init__()
        self.layer_1 = torch.nn.LSTM(D_in, H)
        self.hidden_1 = (torch.zeros(1, 1, H), torch.zeros(1, 1, H))
        self.extra_linear_layer = torch.nn.Linear(H,H)
        self.layer_2 = torch.nn.Linear(H,D_out)
    
    def forward(self, x):
        out, self.hidden_1 = self.layer_1(x, self.hidden_1)
        h_relu = out.clamp(min=0)
        out = self.extra_linear_layer(out)
        h_relu = out.clamp(min=0)
        out = self.layer_2(h_relu)
        y_pred = out
        return y_pred[-1].squeeze()
         

def create_sequences(x, y, seq_length):
    x_y_sequence = []
    positive_count = 0
    negative_count = 0
    for i in range(len(x) - seq_length):
        seq_x = x[i:i+seq_length]
        # This selects the day following the sequence
        seq_y = y[i+seq_length]
        if seq_y == 1:
            positive_count +=1
        else:
            negative_count +=1
        x_y_sequence.append((seq_x, seq_y))
    pos_weight = torch.tensor(negative_count/positive_count).float()
    return x_y_sequence, pos_weight

# start and end date in format 'year-month-day'
def prepare_data(start_date, end_date, sequence_length):
    sp500_data = sp500.history(start=start_date, end=end_date)
    x, y = preproc(sp500_data)
    x = x.T.reshape(x.shape[1], 1, x.shape[0])
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    
    data_sequence, positive_weight = create_sequences(x, y, sequence_length)
    return data_sequence, positive_weight

def evaluate_model(sequence, model):
    sig = nn.Sigmoid()
    
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for x, y_label in sequence:
        model.hidden_1 = (torch.zeros(1, 1, h_size),
                        torch.zeros(1, 1, h_size))
        model.hidden_2 = (torch.zeros(1, 1, 1),
                        torch.zeros(1, 1, 1))
        y_pred = model.forward(x)
        # puts through sigmoid after as loss contains sigmoid function
        y_pred = sig(y_pred)
        if y_pred > 0.5:
            if y_label == 1:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if y_label == 0:
                true_neg += 1
            else:
                false_neg += 1
    return true_pos, false_pos, true_neg, false_neg

def calculate_accuracy_both(tp, fp, tn, fn):
	tpr = tp/(tp + fn)
	tnr = tn/(tn+fp)
	return tpr, tnr

def confusion_string(true_pos, false_pos, true_neg, false_neg):
    return " |1 |0\n1|{0}|{1}\n0|{2}|{3}".format(true_pos, false_pos, false_neg ,true_neg)

def train_model(train_sequence, test_sequence, positive_weight, learning_rate, model, epochs):
    # loss containing sigmoid function
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    training_scores = []
    testing_scores = []
    iterations = []
    losses = []
    for epoch in range(epochs):
        loss_sum = 0
        random.shuffle(train_sequence)
        for x, y_label in train_sequence:
            optimizer.zero_grad()
       
            model.hidden_1 = (torch.zeros(1, 1, h_size),
                            torch.zeros(1, 1, h_size))
            model.hidden_2 = (torch.zeros(1, 1, 1),
                            torch.zeros(1, 1, 1))
    
            y_pred = model.forward(x)
            loss = criterion(y_pred, y_label)
            loss_sum += loss
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            average_loss = loss_sum/len(train_sequence)
            print("Epoch: {}".format(epoch))
            print(average_loss)
            losses.append(average_loss)
            tp, fp, tn, fn = evaluate_model(train_sequence, model)
            print("training")
            print(confusion_string(tp, fp, tn, fn))
            tpr, tnr = calculate_accuracy_both(tp, fp, tn, fn)
            print("Positive: {0}, Negative:{1}".format(tpr, tnr))
            training_scores.append((tp,fp,tn,fn))
            tp, fp, tn, fn = evaluate_model(test_sequence, model)
            print("testing")
            print(confusion_string(tp, fp, tn, fn))
            tpr, tnr = calculate_accuracy_both(tp, fp, tn, fn)
            print("Positive: {0}, Negative:{1}".format(tpr, tnr))
            testing_scores.append((tp,fp,tn,fn))
            
            iterations.append(epoch)
    print("Epoch: {}".format(epoch))
    tp, fp, tn, fn = evaluate_model(train_sequence, model)
    print("training")
    print(confusion_string(tp, fp, tn, fn))
    tpr, tnr = calculate_accuracy_both(tp, fp, tn, fn)
    print("Positive: {0}, Negative:{1}".format(tpr, tnr))
    training_scores.append((tp,fp,tn,fn))
    tp, fp, tn, fn = evaluate_model(test_sequence, model)
    print("testing")
    print(confusion_string(tp, fp, tn, fn))
    tpr, tnr = calculate_accuracy_both(tp, fp, tn, fn)
    print("Positive: {0}, Negative:{1}".format(tpr, tnr))
    testing_scores.append((tp,fp,tn,fn))
    iterations.append(epoch)
    return model, training_scores, testing_scores, iterations, losses

train_sequence, positive_weight = prepare_data('2010-01-01', '2020-12-31', 4)
test_sequence, _ = prepare_data('2004-01-01', '2007-12-31', 4)
h_size = 128
model = RecurrentNet(5, h_size, 1)
extra_weight = 0
model, training_scores, testing_scores, iterations, losses = train_model(train_sequence, test_sequence, positive_weight + extra_weight, 0.0001, model, 2000)



def save_data(model, training_scores, testing_scores, iterations, losses, name):
    torch.save(model.state_dict(), "{0}-lstm.pt".format(name))
    pickle.dump( training_scores, open( "{0}-training_scores.p".format(name), "wb" ) )
    pickle.dump( testing_scores, open( "{0}-testing_scores.p".format(name), "wb" ) )
    pickle.dump( iterations, open( "{0}-iterations.p".format(name), "wb" ) )
    pickle.dump( losses, open( "{0}-losses.p".format(name), "wb" ) )
    
def load_data(name):
    model.load_state_dict(torch.load("{0}-lstm.pt".format(name)))
    training_scores = pickle.load( open( "{0}-training_scores.p".format(name), "rb" ) )
    testing_scores = pickle.load( open( "{0}-testing_scores.p".format(name), "rb" ) )
    iterations = pickle.load( open( "{0}-iterations.p".format(name), "rb" ) )
    losses = pickle.load( open( "{0}-losses.p".format(name), "rb" ) )
    return model, training_scores, testing_scores, iterations, losses


save_data(model, training_scores, testing_scores, iterations, losses, "/path/to_save")
