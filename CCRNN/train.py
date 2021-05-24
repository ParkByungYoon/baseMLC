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
    early_stopping = EarlyStopping(patience=20, verbose=True, path=dataset_name + ".pt")

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

            del X, labels, targets, outputs, loss
        train_loss = total_loss / len(train_loader)

        model.eval()
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

            del X, labels, targets, outputs, loss

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

    #final_model = torch.load('./' + dataset_name + '.pt')

    return model, ep

def test(model, test_loader):

    with torch.no_grad() :
        prediction = None

        for i, (X, labels, lengths) in enumerate(test_loader):
            X = X.to(device).float()
            targets = labels.to(device).long()

            predicts = model.predict(X)

            if prediction == None:  prediction = copy.deepcopy(predicts)
            else:   prediction = torch.cat((prediction, predicts), axis=0)

            del X, labels, targets, predicts

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
embed_size = 128
hidden_size = 256
batch_size = 128

max_epoch = 1000
learning_rate = [0.005, 0.01, 0.05]
weight_decay = [0, 0.00001, 0.00005, 0.0001]
random_seed = [7]
criterion = nn.CrossEntropyLoss()

for seed in random_seed:
    model_num = int(sys.argv[2])
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
    max_seq_length = test_dataset.y.shape[1]
    vocab_size = test_dataset.y.shape[1] + 1

    for lr in learning_rate:
        for wd in weight_decay:
            model_num+=1
            model = CCRNN(input_size, embed_size, hidden_size, vocab_size, max_seq_length).to(device)
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, lr=lr)

            model, epoch = train(model, criterion, optimizer, train_loader, valid_loader, max_epoch)
            torch.save(model, "./models/" + dataset_name + "/model" + str(seed) + '_' + str(model_num) + ".pt")

            train_pred, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score= test(model,train_loader)
            f = open('./' + dataset_name + str(seed) + '_train_Result.csv', 'a')
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(model_num, epoch, lr, wd, accuracy, jaccard, hamming_score,
                                                             f1_micro_score, f1_macro_score))
            f.close()

            _, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score = test(model, valid_loader)
            f = open('./' + dataset_name + str(seed) + '_valid_Result.csv', 'a')
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(model_num, epoch, lr, wd, accuracy, jaccard, hamming_score,
                                                             f1_micro_score, f1_macro_score))
            f.close()

            test_pred, accuracy, jaccard, hamming_score, f1_micro_score, f1_macro_score = test(model,test_loader)
            f = open('./' + dataset_name + str(seed) + '_test_Result.csv', 'a')
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(model_num, epoch, lr, wd, accuracy, jaccard, hamming_score,
                                                             f1_micro_score, f1_macro_score))
            f.close()

            df = pd.DataFrame(test_pred)
            df.to_csv('./prediction/' + dataset_name + str(seed) + '_' + str(model_num) + '_test.csv', index=False)

            del model
            torch.cuda.empty_cache()