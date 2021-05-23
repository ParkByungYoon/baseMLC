import torch
import torch.nn as nn
import copy
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as metric
from skmultilearn import dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
import os
from CCRNN import CCRNN
from torch.nn.utils.rnn import pack_padded_sequence
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import metrics

def y2id(y):
    ids = []
    lens = []
    for i in range(y.shape[0]):
        id = np.argwhere(y[i] == 1)
        id = np.ravel(id, order='F')
        ids.append(id)
        lens.append(len(id))
    return ids, max(lens), lens

class CCRNNDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, opt = 'train', top = None, scaler = MinMaxScaler, random_state = 42, test_size = 0.25 ):
        X, y, _, _ = dataset.load_dataset(dataset_name, 'undivided')
        X, y = X.toarray(), y.toarray()

        if top is not None:
            example_num_per_label = y.sum(axis=0)
            top = 15

            asc_arg = np.argsort(example_num_per_label)
            des_arg = asc_arg[::-1]
            y = y[:, des_arg[:top]]

        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if scaler != None:
            scaler = scaler()
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_ts = scaler.transform(X_ts)

        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=random_state)
        if (opt == 'train'):
            X = X_tr;y = y_tr;
            del (X_ts);del (y_ts);del (X_val);del (y_val)
        elif (opt == 'valid'):
            X = X_val; y = y_val;
            del (X_tr);del (X_ts);del (y_tr);del (y_ts)
        else:
            X = X_ts; y = y_ts;
            del (X_tr);del (y_tr);del (X_val);del (y_val)

        self.X = torch.from_numpy(X)
        self.y = y
        self.start = y.shape[1]+2
        self.end = y.shape[1]+1
        self.pad = y.shape[1]
        self.ids, self.time_step, self.lens = y2id(y)
        self.length = X.shape[0]

    def __getitem__(self, idx):
        length = len(self.ids[idx])
        self.ids[idx] = np.insert(self.ids[idx], 0, self.start)
        self.ids[idx] = np.append(self.ids[idx], self.end)
        if(length < self.time_step) :
            for i in range(self.time_step - length) :
                self.ids[idx] = np.append(self.ids[idx], self.pad)
        return self.X[idx], torch.Tensor(self.ids[idx]), length+2

    def __len__(self):
        return self.length

train_dataset = CCRNNDataset(dataset_name='scene', opt='train', random_state=7)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=128,
                                              shuffle=False, num_workers=0)

test_dataset = CCRNNDataset(dataset_name='scene', opt='test', random_state=7)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = test_dataset.X.shape[1]
embed_size = 512
hidden_size = 1024
vocab_size = 9

model = CCRNN(input_size, embed_size, hidden_size, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for i, (X, label, lengths) in enumerate(train_loader):
        # Set mini-batch dataset
        X = X.to(device).float()
        label = label.to(device).long()
        targets = pack_padded_sequence(label, lengths, batch_first=True, enforce_sorted=False)[0]

        # Forward, backward and optimize
        outputs = model(X, label)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)[0]
        loss = criterion(outputs, targets)
        print(loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch%10 == 0) : print('EPOCH:'+str(epoch))

ids = None
y_true = None

for i, (X, targets, lengths) in enumerate(test_loader):
    start = torch.empty(X.size(0), 1).fill_(vocab_size-1).to(device).long()
    X = X.to(device).float()
    targets = targets.to(device).long()
    predicts = model.sample(X, start)

    if y_true == None :
        y_true, ids = targets, predicts
    else :
        y_true = torch.cat((y_true, targets), axis=0)
        ids = torch.cat((ids, predicts), axis=0)

y_true = (y_true.cpu()).detach().numpy()
ids = (ids.cpu()).detach().numpy()

prediction = np.zeros((len(test_dataset), int(test_dataset.pad)))

for i in range(ids.shape[0]) :
    predicted_ids = ids[i, 1:1+test_dataset.lens[i]].astype(np.int64)
    for j in predicted_ids :
        if(j >= int(test_dataset.pad)) : continue
        prediction[i, j] = 1

df = pd.DataFrame(prediction)
df.to_csv('./prediction.csv', index=False)

ema = metric.accuracy_score(test_dataset.y, prediction)
print(ema)