import torch
from torch.autograd import Variable
import torch.nn as nn
import copy

# Fully connected neural network with one hidden layer

######################## BinaryRelevance ########################

class BinaryRelevance(nn.Module):
    def __init__(self, base, input_attr, output_attr, input_size, output_size, criterions, optimizers, attr = [], opt_attr=[],device = None ):
        '''
        :param base: Class of PyTorch model for base-classifier
        :param input_attr: name of attribute for Input dimension
        :param output_attr: name of attribute for Output dimension
        :param input_size: dimension size of X
        :param output_size: dimension size of Y
        :param criterions : list of loss function for each classifier
        :param optimizers : list of optimizers for each classifier
        :param attr: list of dictionaries, each dictionary contains attributes to initialize base-classifier (excluding attributes for input size and output size)
        :param opt_attr: list of dictionaries, each dictionary contains attributes to initialize optimizer on each classifier
        '''

        super(BinaryRelevance, self).__init__()
        self.input_attr = input_attr
        self.input_size = input_size
        self.output_attr = output_attr
        self.output_size = output_size
        self.base_model = base
        self.models = nn.ModuleList()
        self.criterions = []
        self.optimizers = []

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else : self.device = device

        for i in range(output_size):
            #attr_init = copy.deepcopy(attr)
            attr_init = copy.deepcopy(attr[i]) # List of dictionaries
            attr_init[input_attr] = input_size
            attr_init[output_attr] = 1
            self.models.append(base(**attr_init))
            self.criterions.append(criterions[i]())

            #opt_attr_init = copy.deepcopy(opt_attr)
            opt_attr_init = copy.deepcopy(opt_attr[i]) # List of dictionaries
            opt_attr_init['params'] = self.models[-1].parameters()
            self.optimizers.append(optimizers[i](**opt_attr_init))
            del(attr_init)
            del(opt_attr_init)

    def fit(self, x, y, index):
        # Fit the classifier

        outputs = self.models[index](x)
        loss = self.criterions[index](outputs, y)
        self.optimizers[index].zero_grad()
        loss.backward()
        self.optimizers[index].step()
        return loss * x.size(0)

    def load_all(self, path, model_names):
        for i in range(self.output_size):
            self.load_model(path+model_names[i], i)
        return

    def load_model(self, path, index):
        self.models[index] = torch.load(path)
        return

    def forward(self, x, proba = True):
        # for prediction
        ret = []
        for i in range(self.output_size):
            out = self.models[i](x)
            ret.append(out)

        output = torch.cat(ret, 1)

        if proba == False :
            output = (output > Variable(torch.Tensor([0.5]).to(self.device))).double() * 1

        return output

