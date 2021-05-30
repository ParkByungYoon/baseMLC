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
import sys
import sklearn.metrics as metrics
from CCRNN import CCRNN
from Utils import ExtCCRNNDataset as CCRNNDataset

dataset_name = sys.argv[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 128
hidden_size = 256
batch_size = 128

train_dataset = CCRNNDataset(dataset_name=dataset_name, opt='train', random_state=7)

input_size = train_dataset.X.shape[1]
max_seq_length = train_dataset.y.shape[1]
vocab_size = train_dataset.y.shape[1] + 1

model = CCRNN(input_size, embed_size, hidden_size, vocab_size, max_seq_length, device=device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(model))