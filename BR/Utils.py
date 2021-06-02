import torch
import math
import numpy as np
from skmultilearn import dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.optim import Optimizer
import torch.nn as nn
import copy

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