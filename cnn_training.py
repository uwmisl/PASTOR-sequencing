import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import warnings
import pandas as pd
import scipy.signal as scisignal
import sklearn.utils.class_weight as class_weight
from sklearn.decomposition import PCA

# load data
segments_df = pd.read_json('data/pretty_segments_df.json')
channels_arr = list(np.load(open('data/channels_arr.npy','rb')))

# fixed held out test set runs
chosen_runs = ['20220824_run02_a', '20220826_run02_a', '20220826_run03_a', '20220907_run01_a', '20221213_run02_a', '20221214_run01_a']

# features and labels    
feature_cols = ['median', 'max', 'middle', 'mean_abs_deriv', 'median_abs_deriv','mean','raw_std', 'dip']
index_to_aa = [c for c in 'CSAGTVNQMILYWFPHRKDE']
aa_to_index = {aa:i for i, aa in enumerate(index_to_aa)}

# change putative deamidated N labels
for name, row in segments_df[segments_df.aa == 'N'][segments_df['max'] > 1.3].iterrows():
    segments_df.at[name, 'aa'] = 'D'
    
# stretch signals, to account for variable lenths
def stretch(arr, new_len):
    if len(arr) == new_len:
        return arr
    return torch.tensor([arr[int(i/new_len*len(arr))] for i in range(new_len)])

class MyDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# train / validation / test split
tr_segments = segments_df[segments_df.aa != ''][segments_df.pretty][~segments_df.run.isin(chosen_runs)]#[segments_df.channel > 120]

# Validation set used for testing architectures and hyperparameter tuning, but final model was trained with the entire training set
# vl_segments = segments_df[segments_df.aa != ''][segments_df.pretty][~segments_df.run.isin(chosen_runs)][segments_df.channel <= 120]
te_segments = segments_df[segments_df.aa != ''][segments_df.pretty][segments_df.run.isin(chosen_runs)]

batch_size = 16


    
def get_data(segment_len, batch_size, acids=index_to_aa):
    acid_to_index = {aa:i for i, aa in enumerate(acids)}

    train_input = tr_segments[tr_segments.aa.isin(acids)].transformed.apply(lambda s: stretch(s, segment_len)).apply(torch.tensor).values
#     val_input   = vl_segments[vl_segments.aa.isin(acids)].transformed.apply(lambda s: stretch(s, segment_len)).apply(torch.tensor).values
    test_input  = te_segments[te_segments.aa.isin(acids)].transformed.apply(lambda s: stretch(s, segment_len)).apply(torch.tensor).values
#     val_output    = vl_segments[vl_segments.aa.isin(acids)].aa.apply(lambda a: acid_to_index[a]).values
    train_output = tr_segments[tr_segments.aa.isin(acids)].aa.apply(lambda a: acid_to_index[a]).values
    test_output  = te_segments[te_segments.aa.isin(acids)].aa.apply(lambda a: acid_to_index[a]).values


    # Create instances of the dataset and dataloader
    train_dataset = MyDataset(train_input, train_output)
#     val_dataset = MyDataset(val_input, val_output)
    test_dataset = MyDataset(test_input, test_output)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=seed_worker)
#     vld_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=seed_worker)

    class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(train_output),y=train_output)
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    return train_loader, test_loader, class_weights  #vld_loader,

class CNN(nn.Module):
    def __init__(self, hidden_size, dropout, use_gru, segment_len, init, 
                 O_1=8, O_2=32, O_3=32, O_4=64, n_classes=20, 
                 K_1 = 2, K_2 = 1, K_3 = 4, K_4 = 2,
                 KP_1 = 4, KP_2 = 4, KP_3 = 1, KP_4 = 1,
                 act='Th', n_layers=2):
        
        reshape = segment_len
        self.act = act
        self.conv_linear_out = int(math.floor((math.floor((math.floor((math.floor((math.floor((reshape - K_1 + 1)/KP_1) 
                                                                       - K_2 + 1)/KP_2)
                                                                       - K_3 + 1)/KP_3) 
                                                                       - K_4 + 1)/KP_4))) * O_4)
        self.FN_1 = hidden_size
        self.use_gru = use_gru

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(1, O_1, K_1), nn.ReLU(),
                                   nn.MaxPool1d(KP_1))
        self.conv2 = nn.Sequential(nn.Conv1d(O_1, O_2, K_2), nn.ReLU(),
                                   nn.MaxPool1d(KP_2))
        self.conv3 = nn.Sequential(nn.Conv1d(O_2, O_3, K_3), nn.ReLU(),
                                   nn.MaxPool1d(KP_3))
        self.conv4 = nn.Sequential(nn.Conv1d(O_3, O_4, K_4), nn.ReLU(),
                                   nn.MaxPool1d(KP_4))
        
        self.gru = nn.GRU(input_size=self.conv_linear_out, hidden_size=self.FN_1, num_layers=n_layers, dropout=dropout)
        self.fc1 = nn.Linear(self.conv_linear_out, self.FN_1, nn.Dropout(dropout))
        
        self.fc2 = nn.Linear(self.FN_1, n_classes)
        
        if init:
            if init == "KN":
                nn.init.kaiming_normal_(self.conv1[0].weight)
                nn.init.kaiming_normal_(self.conv2[0].weight)
                nn.init.kaiming_normal_(self.conv3[0].weight)
                nn.init.kaiming_normal_(self.conv4[0].weight)
                nn.init.kaiming_normal_(self.fc1.weight)
                nn.init.kaiming_normal_(self.fc2.weight)
            elif init == 'XN':
                nn.init.xavier_normal_(self.conv1[0].weight)
                nn.init.xavier_normal_(self.conv2[0].weight)
                nn.init.xavier_normal_(self.conv3[0].weight)
                nn.init.xavier_normal_(self.conv4[0].weight)
                nn.init.xavier_normal_(self.fc1.weight)
                nn.init.xavier_normal_(self.fc2.weight)
            elif init == "KNC":
                nn.init.kaiming_normal_(self.conv1[0].weight)
                nn.init.kaiming_normal_(self.conv2[0].weight)
                nn.init.kaiming_normal_(self.conv3[0].weight)
                nn.init.kaiming_normal_(self.conv4[0].weight)
                
            elif init == 'XNC':
                nn.init.xavier_normal_(self.conv1[0].weight)
                nn.init.xavier_normal_(self.conv2[0].weight)
                nn.init.xavier_normal_(self.conv3[0].weight)
                nn.init.xavier_normal_(self.conv4[0].weight)

    def forward(self, x):
        x = x.float()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        
        x = x.view(len(x), -1)
        if self.use_gru:
            x = F.logsigmoid(self.gru(x)[0])
        else:
            x = F.logsigmoid(self.fc1(x)[0])
            
        x = self.fc2(x)
        return x


def train(model, optimizer,lmbd, epochs, criterion, train_loader): #, vld_loader
    use_cuda = False
    losses = []
    accs = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)

            reg_loss = None
            for param in model.parameters():
                if reg_loss is None:
                    reg_loss = 0.5 * torch.sum(param**2)
                else:
                    reg_loss = reg_loss + 0.5 * param.norm(2)**2

            loss += lmbd * reg_loss

            loss.backward()
            optimizer.step()

### below is only used for hyperparmater tuning
            
#         correct = 0.0
#         total = 0.0
#         running_vloss = 0
#         with torch.no_grad():
#             for i, data in enumerate(vld_loader):
#                 inputs, labels = data
#                 if use_cuda and torch.cuda.is_available():
#                     inputs = inputs.cuda()
#                     labels = labels.cuda()

#                 outputs = model(inputs.unsqueeze(1))
#                 _, predicted = torch.max(outputs.data, 1)
#                 running_vloss += criterion(outputs, labels).item()

#                 correct += (predicted == labels).sum().item()
#                 total += len(labels)
#                 running_vloss += criterion(outputs, labels).item()
                
#         avg_vloss = running_vloss / (i + 1)
#         losses.append(avg_vloss)
#         accs.append(correct / total * 100.)
#         if epoch %10 == 0:
#             print(f"{correct / total * 100.}%")
#     return losses, accs

def top_n_cnt(y_test, test_pred, k):
    cnt = 0
    for yt, pred in zip(y_test, test_pred):
        top_k = np.array(torch.topk(pred, k).indices)
        if int(yt) in top_k:
            cnt += 1
    return cnt

batch_size = 16
lmbd = 0
segment_len = 100
use_gru = True
dropout = 0.3
init = 'KN'
hidden_size = 128
momentum = 0.55
epochs = 200
use_weights = True
use_gru = True
act = 'Re'
lr = 0.01
n_layers = 2

acids = index_to_aa
train_loader, test_loader, class_weights = get_data(segment_len, batch_size, acids)   #vld_loader,

top_n_experiment = np.zeros((20,20))
for enum in range(20):

    model = CNN(hidden_size, dropout, use_gru, segment_len, init, n_classes=len(acids), act=act, n_layers=n_layers)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if use_weights else nn.CrossEntropyLoss()
    train(model, optimizer, lmbd, epochs, criterion, train_loader) #losses, accs =  vld_loader
    
    with torch.no_grad():
        for n_attempts in range(1,len(acids)+1):
            total = 0
            correct = 0
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs.unsqueeze(1))
                x, predicted = torch.max(outputs.data, 1)
                total += len(labels)
                correct += top_n_cnt(labels, outputs.data, n_attempts)
            top_n_experiment[enum][n_attempts-1] = correct/total
            
    np.save(f"top_n_experiment.npy", top_n_experiment, allow_pickle=True)
                        
