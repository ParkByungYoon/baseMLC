import torch
import torch.nn as nn
import copy
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as metric
import sys
import os
from Utils import CCRNNDataset, id2y, EarlyStopping
from CCRNN import CCRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import metrics


dataset_name = sys.argv[1]

def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    early_stopping = EarlyStopping(patience=100, verbose=True, path=dataset_name + ".pt")

    ep=num_epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (X, labels, lengths) in enumerate(train_loader):
            # Set mini-batch dataset
            X = X.to(device).float()
            labels = labels.to(device).long()
            targets = pack_padded_sequence(labels, lengths, batch_first=True, enforce_sorted=False)[0]
            # Forward, backward and optimize
            outputs = model(X, labels)
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)[0]
            loss = criterion(outputs, targets)

            total_loss += loss.item() * X.size(0)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = total_loss / len(train_loader)

        total_loss = 0
        for i, (X, labels, lengths) in enumerate(valid_loader):
            X = X.to(device).float()
            labels = labels.to(device).long()
            targets = pack_padded_sequence(labels, lengths, batch_first=True, enforce_sorted=False)[0]
            # Forward, backward and optimize
            outputs = model(X, labels)
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)[0]

            loss = criterion(outputs, targets)
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
            ep = epoch + 1
            break

    final_model = model
    #final_model = torch.load('./' + dataset_name + '.pt')

    return final_model, ep

def test(model, test_loader):
    y_true = prediction = None

    for i, (X, labels, lengths) in enumerate(test_loader):
        X = X.to(device).float()
        targets = labels.to(device).long()

        predicts = model.predict(X)

        if y_true == None:
            y_true, prediction = targets, predicts
        else:
            y_true = torch.cat((y_true, targets), axis=0)
            prediction = torch.cat((prediction, predicts), axis=0)

    y_true = (y_true.cpu()).detach().numpy()
    prediction = (prediction.cpu()).detach().numpy().astype(np.int64)

    ids = []
    for i in range(prediction.shape[0]):
        end_point = np.argwhere(prediction[i] == vocab_size - 1).astype(np.int64)
        if (len(end_point) == 0):
            ids.append(prediction[i]);
        else:
            end_point = end_point.squeeze(axis=1)
            ids.append(prediction[i, :end_point[0]])
        ids[i] = np.unique(ids[i])

    prediction = id2y(ids, test_loader.dataset.y.shape[1])
    y_ts = test_loader.dataset.y

    accuracy = metric.accuracy_score(y_ts, prediction)
    jaccard = metrics.jaccard_score(y_ts, prediction)
    #cll_loss = metrics.log_likelihood_loss(y_ts, prediction_proba)
    hamming_score = 1 - metric.hamming_loss(y_ts, prediction)
    f1_micro_score = metric.f1_score(y_ts, prediction, average='micro')
    f1_macro_score = metric.f1_score(y_ts, prediction, average='macro')

    return prediction, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 512
hidden_size = 1024
batch_size = 128

max_epoch = 100
learning_rate = [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075]
weight_decay = [0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001]
random_seed = [7]

"""max_epoch = 100
learning_rate = [0.001]
weight_decay = [0.0001]
random_seed = [7]"""

for seed in random_seed:
    model_num = 0
    train_dataset = CCRNNDataset(dataset_name=dataset_name, opt='train', random_state=seed)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    valid_dataset = CCRNNDataset(dataset_name=dataset_name, opt='valid', random_state=seed)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_dataset = CCRNNDataset(dataset_name=dataset_name, opt='test', random_state=seed)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=128,
                                                  shuffle=False)
    input_size = test_dataset.X.shape[1]
    vocab_size = test_dataset.y.shape[1] + 1

    for lr in learning_rate:
        for wd in weight_decay:
            model_num+=1
            model = CCRNN(input_size, embed_size, hidden_size, vocab_size).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, lr=lr)

            model, epoch = train(model, criterion, optimizer, train_loader, valid_loader, max_epoch)
            torch.save(model, "./models/" + dataset_name + "/model" + str(seed) + '_' + str(model_num) + ".pt")

            train_pred, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score= test(model,train_loader)
            f = open('./' + dataset_name + str(seed) + '_train_Result.csv', 'a')
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(model_num, epoch, lr, wd, accuracy, jaccard, hamming_score,
                                                             f1_micro_score, f1_macro_score))
            f.close()

            df = pd.DataFrame(train_pred)
            df.to_csv('./prediction/'+dataset_name+str(seed)+'_' + str(model_num)+'_train.csv', index=False)

            _, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score = test(model, valid_loader)
            f = open('./' + dataset_name + str(seed) + '_valid_Result.csv', 'a')
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(model_num, epoch, lr, wd, accuracy, jaccard, hamming_score,
                                                             f1_micro_score, f1_macro_score))

            test_pred, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score = test(model,test_loader)
            f = open('./' + dataset_name + str(seed) + '_test_Result.csv', 'a')
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(model_num, epoch, lr, wd, accuracy, jaccard, hamming_score,
                                                             f1_micro_score, f1_macro_score))
            f.close()

            df = pd.DataFrame(test_pred)
            df.to_csv('./prediction/' + dataset_name + str(seed) + '_' + str(model_num) + '_test.csv', index=False)