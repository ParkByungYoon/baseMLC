import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import avg_pool2d
from torch.autograd import Variable

class CCRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, vocab_size, max_seq_length=40, device='cpu'):
        super(CCRNN, self).__init__()
        self.fembed = nn.Linear(input_size, embed_size)
        self.lembed = nn.Embedding(vocab_size+1, embed_size, padding_idx=vocab_size)
        self.lstm_cell = nn.LSTMCell(embed_size*2, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, X, labels):
        batch_size, time_step = labels.size()

        predicts = torch.zeros(batch_size, time_step, self.vocab_size).to(self.device).float()

        features = self.fembed(X)
        features = self.sigmoid(features)
        embeddings = self.lembed(labels)
        embeddings = self.sigmoid(embeddings)

        #print(features.size())
        #print(embeddings.size())
        hx = torch.zeros(batch_size, self.hidden_size).to(self.device).float()
        cx = torch.zeros(batch_size, self.hidden_size).to(self.device).float()

        for i in range(time_step):
            if(i==0): input = torch.cat((features, embeddings[:, -1, :]), -1)
            else :  input = torch.cat((features, embeddings[:, i-1, :]), -1)
            hx, cx = self.lstm_cell(input, (hx, cx))
            output = self.linear(hx)
            predicts[:, i, :] = output

        del hx, cx

        return predicts

    def predict(self, X):
        predict_ids = torch.zeros(X.size(0), self.max_seg_length).to(self.device).float()
        start = torch.zeros(X.size(0), self.embed_size).to(self.device).float()
        features = self.fembed(X)
        features = self.sigmoid(features)

        inputs = torch.cat((features, start), -1)

        hx = torch.zeros(X.size(0), self.hidden_size).to(self.device).float()
        cx = torch.zeros(X.size(0), self.hidden_size).to(self.device).float()

        for i in range(self.max_seg_length):
            hx, cx = self.lstm_cell(inputs, (hx, cx))  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hx)  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            predict_ids[:, i] = predicted
            embeddings = self.lembed(predicted)  # inputs: (batch_size, embed_size)
            embeddings = self.sigmoid(embeddings)
            inputs = torch.cat((features, embeddings), -1)

        del start, hx, cx

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
            outputs = self.softmax(outputs)
            predicted, predicted_id = outputs.max(1)  # predicted: (batch_size)
            predict_ids[:, i] = predicted_id
            prediction[:, i] = predicted
            embeddings = self.lembed(predicted_id)  # inputs: (batch_size, embed_size)
            inputs = torch.cat((features, embeddings), -1)

        del start, hx, cx

        return predict_ids, prediction


if __name__ == "__main__":
    import sys
    import copy
    import numpy as np
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = sys.argv[1]
    embed_size = 128
    hidden_size = 256
    batch_size = 128
    max_epoch = 100
    criterion = nn.CrossEntropyLoss()

    print("Example")
    from Utils import CCRNNDataset, id2y, EarlyStopping

    train_dataset = CCRNNDataset(dataset_name=dataset_name, opt='train', random_state=7)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_dataset = CCRNNDataset(dataset_name=dataset_name, opt='test', random_state=7)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    input_size = test_dataset.X.shape[1]
    max_seq_length = test_dataset.y.shape[1]
    vocab_size = test_dataset.y.shape[1] + 1

    model = CCRNN(input_size, embed_size, hidden_size, vocab_size, max_seq_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=0.001)

    for epoch in range(100):
        model.train()
        for i, (X, labels, lengths) in enumerate(train_loader):
            # Set mini-batch dataset
            X = X.to(device).float()
            labels = labels.to(device).long()
            targets = pack_padded_sequence(labels, lengths, batch_first=True, enforce_sorted=False)[0]
            # Forward, backward and optimize
            outputs = model(X, labels)
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)[0]
            loss = criterion(outputs, targets)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            del X, labels, targets, outputs, loss

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

        import sklearn.metrics as metric
        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        import metrics

        f1_micro_score = metric.f1_score(y_ts, prediction, average='micro')
        f1_macro_score = metric.f1_score(y_ts, prediction, average='macro')
        accuracy = metric.accuracy_score(y_ts, prediction)
        jaccard = metrics.jaccard_score(y_ts, prediction)
        hamming_score = 1 - metric.hamming_loss(y_ts, prediction)
        print('ema = %.5f' % (accuracy))

