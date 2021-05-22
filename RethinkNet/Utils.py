import torch
import math
import numpy as np
from skmultilearn import dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.optim import Optimizer


class Nadam(Optimizer):
    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).
    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        schedule_decay (float, optional): momentum schedule decay (default: 4e-3)
    __ http://cs229.stanford.edu/proj2015/054_report.pdf
    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
        Originally taken from: https://github.com/pytorch/pytorch/pull/1408
        NOTE: Has potential issues but does work well on some problems.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=4e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, schedule_decay=schedule_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m_schedule'] = 1.
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                # Warming momentum schedule
                m_schedule = state['m_schedule']
                schedule_decay = group['schedule_decay']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                state['step'] += 1
                t = state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                momentum_cache_t = beta1 * \
                                   (1. - 0.5 * (0.96 ** (t * schedule_decay)))
                momentum_cache_t_1 = beta1 * \
                                     (1. - 0.5 * (0.96 ** ((t + 1) * schedule_decay)))
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                state['m_schedule'] = m_schedule_new

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_( grad, alpha=1. - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1. - beta2)
                exp_avg_sq_prime = exp_avg_sq / (1. - beta2 ** t)
                denom = exp_avg_sq_prime.sqrt_().add_(eps)

                p.data.addcdiv_( grad, denom, value = -group['lr'] * (1. - momentum_cache_t) / (1. - m_schedule_new))
                p.data.addcdiv_( exp_avg, denom, value = -group['lr'] * momentum_cache_t_1 / (1. - m_schedule_next))

        return loss

EPSILON = 10e-18

def log_likelihood_loss(y, y_prob):
    log_likelihood_loss = 0
    for row in range(y.shape[0]):
        if y.ndim == 1:
            if y[row] == 1:
                log_likelihood_loss += (-math.log(np.max(EPSILON, y_prob[row])))
            else:
                log_likelihood_loss += (-math.log(1 - np.min(1 - EPSILON, y_prob[row])))
        elif y.ndim == 2:
            for col in range(y.shape[1]):
                if y[row, col] == 1:
                    log_likelihood_loss += (-math.log(EPSILON if y_prob[row, col] == 0 else y_prob[row, col]))
                else:
                    log_likelihood_loss += (-math.log(1 - (EPSILON if y_prob[row, col] == 1 else y_prob[row, col])))
    return log_likelihood_loss


def jaccard_score(y, y_pred):
    y_ = y.tolist()
    y_pred = y_pred.tolist()
    jaccard_dis = 0
    all_zero  = 0
    for row in range(y.shape[0]):
        denom = [int(i)| int(j) for i, j in zip(y_[row], y_pred[row])]
        nom = [int(i) & int(j) for i, j in zip(y_[row], y_pred[row])]
        if(sum(denom) == 0): all_zero += 1
        else : jaccard_dis += sum(nom)/ sum(denom)
    jaccard_dis /= (y.shape[0] - all_zero)
    return jaccard_dis


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, opt='train', top = None, scaler = MinMaxScaler, random_state = 42, test_size = 0.25, embed_dir = None):

        opt = opt.split('_')
        X, y, _, _ = dataset.load_dataset(dataset_name, opt[0])
        X, y = X.toarray(), y.toarray()

        self.embed = None;
        if(embed_dir != None):
            embed = pd.read_csv(embed_dir, delimiter=',')
            embed = embed.to_numpy()
            self.embed = torch.from_numpy(embed)

        if top is not None:
            example_num_per_label = y.sum(axis=0)
            top = 15

            asc_arg = np.argsort(example_num_per_label)
            des_arg = asc_arg[::-1]
            y = y[:, des_arg[:top]]

        if(opt[0] == 'undivided') :
            X_tr, X_ts, y_tr, y_ts = train_test_split(X,y, test_size=test_size, random_state=random_state)
            if scaler != None:
                scaler = scaler()
                scaler.fit(X_tr)
                X_tr = scaler.transform(X_tr)
                X_ts = scaler.transform(X_ts)

            X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size = 0.25, random_state = random_state)

            if (opt[1] == 'train'):
                X = X_tr; y = y_tr; del(X_ts); del(y_ts) ; del(X_val); del(y_val)
            elif(opt[1] == 'valid'):
                X = X_val; y = y_val ; del(X_tr); del(X_ts); del(y_tr); del(y_ts)
            else:
                X = X_ts; y = y_ts; del(X_tr); del(y_tr); del(X_val); del(y_val)
        else :
            X_tr, y_tr, _, _ = dataset.load_dataset(dataset_name, 'train')
            X_tr = X_tr.toarray()

            if scaler != None:
                scaler = scaler()
                scaler.fit(X_tr)
                X = scaler.transform(X)
                del(X_tr)

            if top is not None:
                y_tr = y_tr.toarray()
                sum = y_tr.sum(axis=0)

                top_label_index = []

                for i in range(top):
                    largest_index = np.argmax(sum)
                    top_label_index.append(largest_index)
                    sum = np.delete(sum, largest_index)

                y = y[:, top_label_index]

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

        self.length = self.X.shape[0]

    def __getitem__(self, index):
        if(self.embed!= None) : return self.X[index], self.y[index], self.embed[index]
        return self.X[index], self.y[index]

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