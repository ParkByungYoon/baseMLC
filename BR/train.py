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
import Utils
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

def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    early_stopping = Utils.EarlyStopping(patience=100, verbose=True, path=dataset_name + ".pt")

    for epoch in range(1, num_epochs + 1):
        ######################
        #   train the model  #
        ######################
        model.train()
        total_loss = 0
        for idx, (X, label) in enumerate(train_loader):
            X = X.to(device).float()
            label = label.to(device).float().unsqueeze(1)

            output = model(X)

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
            label = label.to(device).float().unsqueeze(1)
            output = model(X)

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
            break

    final_model = torch.load("./" + dataset_name + ".pt")

    return final_model

def test(model, test_loader, proba = False):
    model.eval()
    y_true = np.array([])
    prediction = np.array([])
    loss = nn.BCELoss()
    cll_loss = 0

    for idx, (X, label) in enumerate(test_loader):
        X = X.to(device).float()
        label = label.to(device).float().unsqueeze(1)
        output = model(X)

        cll_loss += loss(output, label) * X.size(0)

        label = (label.cpu()).detach().numpy()
        y_true = np.concatenate((y_true, label), axis=None)

        output = (output.cpu()).detach().numpy()
        prediction = np.concatenate((prediction, output), axis=None)

    cll_loss = cll_loss / len(test_loader)

    if proba == False :
        for i in range(len(prediction)):
            if (prediction[i] >= 0.5 and prediction[i] < 1):
                prediction[i] = 1
            elif (prediction[i] < 0.5 and prediction[i] >= 0):
                prediction[i] = 0

    return prediction, cll_loss

def combine_result(best_model, data_loader, prediction):
    pred, _ = test(best_model, data_loader, proba=True)
    pred = np.expand_dims(pred, axis=1)
    if prediction is not None:
        prediction = np.concatenate([prediction, pred], axis=1)
    else:
        prediction = copy.deepcopy(pred)

    return prediction

max_epoch = 1
batch_size = 128
learning_rate = [0.001]
weight_decay = [0]
random_seed = [1011]

'''max_epoch = 10000
batch_size = 128
learning_rate = [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075]
weight_decay = [0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001]
random_seed = [7, 14, 21, 28, 42]'''

X, y, _, _ = dataset.load_dataset(dataset_name, 'undivided')
X = X.toarray()
y = y.toarray()

if y.shape[1] > 15:
    print("Label Number Changed %d -->" % (y.shape[1]), end=' ')

    example_num_per_label = y.sum(axis=0)
    top = 15

    asc_arg = np.argsort(example_num_per_label)
    des_arg = asc_arg[::-1]
    y = y[:, des_arg[:top]]
    print(y.shape[1])

for seed in random_seed :
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.25, random_state=seed)

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ts = scaler.transform(X_ts)

    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)

    input_size = X_tr.shape[1]
    output_size = num_classes = y_tr.shape[1]

    best_model_params = [{} for i in range(num_classes)]

    train_pred = valid_pred = test_pred = None

    for i in range(num_classes):
        minim = 999999
        train_dataset = LoadDataset(X_tr, y_tr[:, i])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)

        valid_dataset = LoadDataset(X_val, y_val[:, i])
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size)

        test_dataset = LoadDataset(X_ts, y_ts[:, i])
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

        for lr in learning_rate:
            for wd in weight_decay:
                model = LR(X_tr.shape[1], 1).to(device)
                torch.nn.init.xavier_uniform_(model.LR[0].weight)
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                model = train(model, criterion, optimizer, train_loader, valid_loader, max_epoch)
                _, cll_loss = test(model, test_loader)

                if cll_loss < minim:
                    best_model_params[i]['lr'] = lr
                    best_model_params[i]['wd'] = wd
                    best_model = copy.deepcopy(model)
                    minim = cll_loss
                del (model)

        torch.save(best_model, "./models/model" + str(i) + ".pt")

        train_pred = combine_result(best_model, train_loader, train_pred)
        valid_pred = combine_result(best_model, valid_loader, valid_pred)
        test_pred = combine_result(best_model, test_loader, test_pred)

    df = pd.DataFrame(test_pred)
    df.to_csv('./'+dataset_name+str(seed)+'.csv', index=False)

    prediction = copy.deepcopy(train_pred)
    for i in range(prediction.shape[0]):
        is_correct = 0
        for j in range(prediction.shape[1]):
            if (prediction[i, j] >= 0.5 and prediction[i, j] < 1):
                prediction[i, j] = 1
            elif (prediction[i, j] < 0.5 and prediction[i, j] >= 0):
                prediction[i, j] = 0

    ema = metric.accuracy_score(y_tr, prediction)
    jaccard_score = metrics.jaccard_score(y_tr, prediction)
    cll_loss = metrics.log_likelihood_loss(y_tr, train_pred)
    hamming_score = 1 - metric.hamming_loss(y_tr, prediction)
    f1_micro_score = metric.f1_score(y_tr, prediction, average='micro')
    f1_macro_score = metric.f1_score(y_tr, prediction, average='macro')

    f = open('./' + dataset_name + '_train_Result.csv', 'a')
    f.write('{},'.format(seed))
    for i in range(num_classes): f.write('{} '.format(best_model_params[i]['lr']))
    f.write(',')
    for i in range(num_classes): f.write('{} '.format(best_model_params[i]['wd']))
    f.write(',')
    f.write('{},{},{},{},{},{}\n'.format(ema, jaccard_score, hamming_score, f1_micro_score, f1_macro_score, cll_loss))
    f.close()

    prediction = copy.deepcopy(test_pred)
    for i in range(prediction.shape[0]):
        is_correct = 0
        for j in range(prediction.shape[1]):
            if (prediction[i, j] >= 0.5 and prediction[i, j] < 1):
                prediction[i, j] = 1
            elif (prediction[i, j] < 0.5 and prediction[i, j] >= 0):
                prediction[i, j] = 0

    ema = metric.accuracy_score(y_ts, prediction)
    jaccard_score = metrics.jaccard_score(y_ts, prediction)
    cll_loss = metrics.log_likelihood_loss(y_ts, test_pred)
    hamming_score = 1 - metric.hamming_loss(y_ts, prediction)
    f1_micro_score = metric.f1_score(y_ts, prediction, average='micro')
    f1_macro_score = metric.f1_score(y_ts, prediction, average='macro')

    f = open('./' + dataset_name + '_test_Result.csv', 'a')
    f.write('{},'.format(seed))
    for i in range(num_classes): f.write('{} '.format(best_model_params[i]['lr']))
    f.write(',')
    for i in range(num_classes): f.write('{} '.format(best_model_params[i]['wd']))
    f.write(',')
    f.write('{},{},{},{},{},{}\n'.format( ema, jaccard_score, hamming_score, f1_micro_score, f1_macro_score, cll_loss))
    f.close()