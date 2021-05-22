import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import copy
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as metrics
from skmultilearn import dataset
import sklearn.metrics as metrics
from Utils import Nadam, log_likelihood_loss, jaccard_score, MultilabelDataset, EarlyStopping
from RethinkNet import RethinkNet
import sys

dataset_name = sys.argv[1]

def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    early_stopping = EarlyStopping(patience=100, verbose=True, path=dataset_name + ".pt")
    ep= 10000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, labels in train_loader:
            optimizer.zero_grad()
            X = X.to(device).double()
            labels = model.prep_Y(labels)
            labels = labels.to(device).double()

            output = model(X)
            ls = criterion(output, labels)
            total_loss += ls.item() * X.size(0)
            ls.backward()
            optimizer.step()
        train_loss = total_loss / len(train_loader)

        model.eval()
        total_loss = 0
        for X, labels in valid_loader:
            X = X.to(device).double()
            labels = model.prep_Y(labels)
            labels = labels.to(device).double()

            output = model(X)
            ls = criterion(output, labels)
            total_loss += ls.item() * X.size(0)
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

    final_model = torch.load('./'+dataset_name+'.pt')

    return final_model, ep

def test(model, test_loader):
    prediction = np.expand_dims(np.zeros(test_loader.dataset.y.shape[1]), axis=0)
    y_true = np.expand_dims(np.zeros(test_loader.dataset.y.shape[1]), axis=0)

    for X, labels in test_loader:
        X = X.to(device).double()
        labels = labels.to(device).double()
        outputs = model.predict_proba(X)
        predicted = torch.squeeze(outputs[-1])
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

    f1_micro_score = metrics.f1_score(y_true, prediction, average='micro')
    f1_macro_score = metrics.f1_score(y_true, prediction, average='macro')
    accuracy = metrics.accuracy_score(y_true, prediction)
    cll_loss = log_likelihood_loss(y_true, prediction_proba)
    jaccard = jaccard_score(y_true, prediction)
    hamming_score = 1 - metrics.hamming_loss(y_true, prediction)

    return prediction_proba, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score, cll_loss

if __name__ ==  "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.device('cpu')
    #device = 'cpu'

    max_epoch = 10000
    batch_size = 128
    learning_rate = [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075]
    weight_decay = [0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001]
    random_seed = [7, 14, 21, 28, 42]

    '''max_epoch = 1
    batch_size = 128
    learning_rate = [0.001]
    weight_decay = [0]
    random_seed = [1011]'''

    for seed in random_seed:
        model_num = 0
        train_dataset = MultilabelDataset(dataset_name=dataset_name, opt='undivided_train', random_state=seed)
        valid_dataset = MultilabelDataset(dataset_name=dataset_name, opt='undivided_valid', random_state=seed)
        test_dataset = MultilabelDataset(dataset_name=dataset_name, opt='undivided_test', random_state=seed)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

        for lr in learning_rate :
            for wd in weight_decay :
                model_num += 1
                model = RethinkNet(train_dataset.X.shape[1], train_dataset.y.shape[1], device=device).to(device).double()
                criterion = nn.BCELoss()

                optimizer =  Nadam([{'params': model.rnn.parameters(), 'lr': lr, 'weight_decay': wd}, {'params':model.dec.parameters()}])

                model, epoch = train(model, criterion, optimizer, train_loader, valid_loader, max_epoch)
                torch.save(model, "./models/"+dataset_name+"/model" + str(seed)+'_'+str(model_num) + ".pt")

                train_pred, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score, cll_loss = test(model, train_loader)
                f = open('./' + dataset_name + str(seed)+'_train_Result.csv', 'a')
                f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(model_num,epoch, lr, wd,accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score, cll_loss))
                f.close()

                test_pred, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score, cll_loss = test(model, test_loader)
                f = open('./' + dataset_name+str(seed)+'_test_Result.csv', 'a')
                f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(model_num,epoch, lr, wd,accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score, cll_loss))
                f.close()
'''
    To Implement
        - Loss function (Reweighted CSMLC)
        - Recurrent Dropout???
'''