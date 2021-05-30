import torch
import torch.nn as nn
import copy
import numpy as np
import pandas as pd
import sys
import os
import sklearn
import sklearn.metrics as metric
from skmultilearn import dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Utils import BinaryRelevance, EarlyStopping
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import metrics

dataset_name = sys.argv[1]

EPSILON = 10e-18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.LR = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.LR(x)


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.length = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length

def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs, first_model):
    early_stopping = EarlyStopping(patience=20, verbose=True, path=dataset_name + ".pt")
    ep = 10000
    for epoch in range(1, num_epochs + 1):
        ######################
        #   train the model  #
        ######################
        model.train()
        total_loss = 0
        for idx, (X, label) in enumerate(train_loader):
            X = X.to(device).float()
            label = label.to(device).float()

            clm = first_model(X)
            output = model(torch.cat((X, clm), 1))

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(train_loader)

        ######################
        # validate the model #
        ######################
        model.eval()
        total_loss = 0
        for idx, (X, label) in enumerate(valid_loader):
            X = X.to(device).float()
            label = label.to(device).float()
            clm = first_model(X)
            output = model(torch.cat((X, clm), 1))

            loss = criterion(output, label)
            total_loss += loss.item() * X.size(0)
        valid_loss = total_loss / len(valid_loader)

        epoch_len = len(str(num_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            ep = epoch+1
            break

    final_model = torch.load("./" + dataset_name + ".pt")

    return  final_model, ep

def test(model, first_model, test_loader):
    prediction = np.expand_dims(np.zeros(test_loader.dataset.y.shape[1]), axis=0)
    y_true = np.expand_dims(np.zeros(test_loader.dataset.y.shape[1]), axis=0)

    for X, labels in test_loader:
        X = X.to(device).float()
        labels = labels.to(device).float()

        clm = first_model(X)
        predicted = model(torch.cat((X, clm), 1))

        frac_labels = (labels.cpu()).detach().numpy()
        y_true = np.concatenate((y_true, frac_labels), axis=0)

        frac_prediction = (predicted.cpu()).detach().numpy()
        prediction = np.concatenate((prediction, frac_prediction), axis=0)

    prediction = np.delete(prediction, 0, 0)
    prediction_proba = prediction.copy()
    y_true = np.delete(y_true, 0, 0)

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if (prediction[i, j] >= 0.5):
                prediction[i, j] = 1
            elif (prediction[i, j] < 0.5):
                prediction[i, j] = 0

    f1_micro_score = metric.f1_score(y_true, prediction, average='micro')
    f1_macro_score = metric.f1_score(y_true, prediction, average='macro')
    accuracy = metric.accuracy_score(y_true, prediction)
    cll_loss = metrics.log_likelihood_loss(y_true, prediction_proba)
    jaccard = metrics.jaccard_score(y_true, prediction)
    hamming_score = 1 - metric.hamming_loss(y_true, prediction)

    return prediction_proba, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score, cll_loss

max_epoch = 10000
batch_size = 128
learning_rate = [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075]
weight_decay = [0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001]
random_seed = [7, 14, 21, 28, 42]

"""max_epoch = 1
batch_size = 128
learning_rate = [0.01]
weight_decay = [0]
random_seed = [7]"""

if dataset_name == 'yahoo' :
    dataset_name = sys.argv[2]
    path_to_arff_file = '../yahoo/'+dataset_name+'1.arff'
    label_count = 26
    label_location = 'end'
    arff_file_is_sparse = False

    X, y = dataset.load_from_arff(
        path_to_arff_file,
        label_count=label_count,
        label_location=label_location,
        load_sparse=arff_file_is_sparse
    )
else :  X, y, _, _ = dataset.load_dataset(dataset_name, 'undivided')

X = X.toarray()
y = y.toarray()

if y.shape[1] > 15 and dataset_name != 'tmc2007_500':
    print("Label Number Changed %d -->" % (y.shape[1]), end=' ')

    example_num_per_label = y.sum(axis=0)
    top = 15

    asc_arg = np.argsort(example_num_per_label)
    des_arg = asc_arg[::-1]
    y = y[:, des_arg[:top]]
    print(y.shape[1])

model_num = 0
for seed in random_seed :
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.25, random_state=seed)

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ts = scaler.transform(X_ts)

    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)

    input_size = X_tr.shape[1]
    output_size = num_classes = y_tr.shape[1]

    BR = BinaryRelevance(LR, 'input_size', 'output_size', input_size, output_size,
                         [nn.BCELoss for i in range(output_size)],
                         [torch.optim.Adam for i in range(output_size)],
                         [{} for i in range(output_size)],
                         [{'lr': 0, 'weight_decay': 0} for i in range(output_size)]).to(device)
    model_names = ['/model'+(str(i) +'.pt') for i in range(num_classes)]
    BR.load_all('../../deepMLC/best_model/'+dataset_name+'/BR/seed'+str(seed), model_names)

    train_dataset = LoadDataset(X_tr, y_tr)
    valid_dataset = LoadDataset(X_val, y_val)
    test_dataset = LoadDataset(X_ts, y_ts)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    for lr in learning_rate:
        for wd in weight_decay:
            model_num += 1
            model = LR(input_size+output_size, output_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
            criterion = nn.BCELoss()

            model, epoch = train(model, criterion, optimizer, train_loader, valid_loader, max_epoch, first_model=BR)
            torch.save(model, "./models/" + dataset_name + "/model" + str(seed) + '_' + str(model_num) + ".pt")

            train_pred, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score, cll_loss = test(model, BR, train_loader)
            f = open('./' + dataset_name + str(seed) + '_train_Result.csv', 'a')
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(model_num, epoch, lr, wd, accuracy, jaccard, hamming_score,
                                                             f1_micro_score, f1_macro_score, cll_loss))
            f.close()

            test_pred, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score, cll_loss = test(model, BR, test_loader)
            f = open('./' + dataset_name + str(seed) + '_test_Result.csv', 'a')
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(model_num, epoch, lr, wd, accuracy, jaccard, hamming_score,
                                                             f1_micro_score, f1_macro_score, cll_loss))
            f.close()