import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import avg_pool2d
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class CCRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, vocab_size, max_seq_length=40):
        super(CCRNN, self).__init__()
        self.fembed = nn.Linear(input_size, embed_size)
        self.lembed = nn.Embedding(vocab_size+1, embed_size, padding_idx=vocab_size)
        self.lstm_cell = nn.LSTMCell(embed_size*2, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, X, labels):
        batch_size, time_step = labels.size()

        predicts = to_var(torch.zeros(batch_size, time_step, self.vocab_size))

        features = self.fembed(X)
        embeddings = self.lembed(labels)

        #print(features.size())
        #print(embeddings.size())
        hx = to_var(torch.zeros(batch_size, self.hidden_size))
        cx = to_var(torch.zeros(batch_size, self.hidden_size))

        for i in range(time_step):
            if(i==0): input = torch.cat((features, embeddings[:, -1, :]), -1)
            else :  input = torch.cat((features, embeddings[:, i-1, :]), -1)
            hx, cx = self.lstm_cell(input, (hx, cx))
            output = self.linear(hx)
            predicts[:, i, :] = output
        return predicts

    def predict(self, X):
        predict_ids = to_var(torch.zeros(X.size(0), self.max_seg_length))
        start = to_var(torch.zeros((X.size(0), self.embed_size)))
        features = self.fembed(X)

        inputs = torch.cat((features, start), -1)

        hx = to_var(torch.zeros(X.size(0), self.hidden_size))
        cx = to_var(torch.zeros(X.size(0), self.hidden_size))

        for i in range(self.max_seg_length):
            hx, cx = self.lstm_cell(inputs, (hx, cx))  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hx)  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            predict_ids[:, i] = predicted
            embeddings = self.lembed(predicted)  # inputs: (batch_size, embed_size)
            inputs = torch.cat((features, embeddings), -1)

        return predict_ids

    def predict_proba(self, X):
        predict_ids = to_var(torch.zeros(X.size(0), self.max_seg_length))
        prediction = to_var(torch.zeros(X.size(0), self.max_seg_length))
        start = to_var(torch.zeros((X.size(0), self.embed_size)))
        features = self.fembed(X)

        inputs = torch.cat((features, start), -1)

        hx = to_var(torch.zeros(X.size(0), self.hidden_size))
        cx = to_var(torch.zeros(X.size(0), self.hidden_size))

        for i in range(self.max_seg_length):
            hx, cx = self.lstm_cell(inputs, (hx, cx))  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hx)  # outputs:  (batch_size, vocab_size)
            predicted, predicted_id = outputs.max(1)  # predicted: (batch_size)
            predict_ids[:, i] = predicted_id
            prediction[:, i] = predicted
            embeddings = self.lembed(predicted_id)  # inputs: (batch_size, embed_size)
            inputs = torch.cat((features, embeddings), -1)

        return predict_ids, prediction