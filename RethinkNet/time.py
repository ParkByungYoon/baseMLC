import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import copy
import time
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as metrics
from skmultilearn import dataset
import sklearn.metrics as metrics
from Utils import Nadam, log_likelihood_loss, jaccard_score,  EarlyStopping

import sys
if len(sys.argv)> 3 :
    from Utils import ExternalDataset as Dataset
else:
    from Utils import MultilabelDataset as Dataset

from RethinkNet import RethinkNet

dataset_name = 'tmc2007_500'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 24
batch_size = 128

train_dataset = Dataset(dataset_name=dataset_name, opt='undivided_train', random_state=7)
test_dataset = Dataset(dataset_name=dataset_name, opt='undivided_test', random_state=7)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

model = RethinkNet(train_dataset.X.shape[1], train_dataset.y.shape[1], device=device).to(device).double()

criterion = nn.BCELoss()
optimizer =  Nadam([{'params': model.rnn.parameters(), 'lr': 0.0025, 'weight_decay': 0}, {'params':model.dec.parameters()}])

start_time = time.time()

for epoch in range(num_epochs):
    total_loss = 0
    for X, labels in train_loader:
        optimizer.zero_grad()
        X = X.to(device).double()
        labels = model.prep_Y(labels)
        labels = labels.to(device).double()

        output = model(X)
        ls = criterion(output, labels)
        ls.backward()
        optimizer.step()

end_time = time.time()
elapsed_time = end_time - start_time

"""ms_elapsed_time = elapsed_time.microseconds / 1000
print(ms_elapsed_time)"""
print(elapsed_time)