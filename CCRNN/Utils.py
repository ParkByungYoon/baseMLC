import torch
import numpy as np
from skmultilearn import dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def y2id(y):
    ids = []
    lens = []
    for i in range(y.shape[0]):
        id = np.argwhere(y[i] == 1)
        id = np.ravel(id, order='F')
        ids.append(id)
        lens.append(len(id))
    return ids, max(lens), lens

def id2y(ids, num):
    y = np.zeros((len(ids),num))
    for i in range(len(ids)):
        y[i, ids[i]] = 1
    return y

class CCRNNDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, opt = 'train', top = None, scaler = MinMaxScaler, random_state = 42, test_size = 0.25 ):
        X, y, _, _ = dataset.load_dataset(dataset_name, 'undivided')
        X, y = X.toarray(), y.toarray()

        instances_per_label = y.sum(axis=0)
        asc_arg = np.argsort(instances_per_label)
        des_arg = asc_arg[::-1]
        y = y[:, des_arg]

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
        self.end = y.shape[1]
        self.pad = y.shape[1]+1
        self.ids, mlen, self.lens = y2id(y)
        self.max_len = mlen+1
        self.length = X.shape[0]

    def __getitem__(self, idx):
        self.ids[idx] = np.append(self.ids[idx], self.end)
        for i in range(self.max_len - len(self.ids[idx])) :
            self.ids[idx] = np.append(self.ids[idx], self.pad)

        return self.X[idx], self.ids[idx], self.lens[idx]+1

    def __len__(self):
        return self.length

class EarlyStopping:
    def __init__(self, patience = 100, verbose=False, tolerance=0.00001, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        # If tolereance is large value, then early stopping criterion is more strict.
        self.tolerance = tolerance
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss_last = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        if self.val_loss_min is np.Inf:
            self.save_checkpoint(val_loss, model)
            self.val_loss_last = val_loss

        elif  self.val_loss_last - val_loss < self.tolerance:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.val_loss_last = val_loss
            if(self.val_loss_min > val_loss):
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print('Least validation loss decreased (%.6f --> %.6f).  Saving model ...'%(self.val_loss_min, val_loss))

        torch.save(model, self.path)
        self.val_loss_min = val_loss
