from os import getcwd
from os.path import join, exists

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

torch.manual_seed(1)
EMBEDDING_DIM_OR_PATH = "resources/embeddings/glove.6B.50d.txt"
HIDDEN_DIM = 64
UNKNOWN_TOKEN = "UNK"


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix[UNKNWON_TOKEN] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def map_unks(data, to_ix):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] not in to_ix:
                data[i][j] = UNKNOWN_TOKEN
    return data
    

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim_or_path, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        if type(embedding_dim_or_path) is int:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            if len(vocab_size) == 0:
                raise ValueError("Need vocab size from from-scratch embedding declaration")
            self.word_to_ix = None
        elif type(embedding_dim_or_path) is str:
            # glove vectors

            existing = pd.read_csv(embedding_dim_or_path, sep=" ", header=None, quoting=3, index_col=0)
            embedding_dim = existing.values.shape[-1]
            existing.loc[UNKNOWN_TOKEN] = np.zeros(embedding_dim)
            self.word_to_ix = {x: i for (i,x) in enumerate(existing.index.tolist())}

            existing = torch.Tensor(existing.values)
            self.word_embeddings = nn.Embedding.from_pretrained(existing)


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def map_data(data):
    word_to_ix, tag_to_ix = {}, {}
    # For each words-list (sentence) and tags-list in each tuple of training_data
    for sent in data:
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                # Assign each word with a unique index
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def predict(model, data, tags, w2i):
    preds = []
    gt = []
    with torch.no_grad():
        for dd, tt in zip(data, tags):
            inputs = prepare_sequence(dd, w2i)
            pred = model(inputs)
            pred = pred.argmax(axis=1)
            # exclude padding
            pad_idx = torch.where(pred == -100)[0]

            if len(pad_idx) == 0:
                pad_idx = 0
            preds.append(pred[pad_idx:])
            gt.append(tt[pad_idx:])
    
    gt = np.concatenate(gt)
    preds = np.concatenate(preds)
    r = classification_report(y_true=gt, y_pred=preds)
    print(r)
    return preds

def train(model, data, tags, word_to_ix, tag_to_ix, num_epochs=50, lr=0.01):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()

    # again, normally you would NOT do 300 epochs, it is toy data
    for epoch in range(num_epochs):
        print("Performance at epoch", epoch)
        predict(model, data, tags, word_to_ix)
        for batch_idx, (sentence, tg) in enumerate(zip(data, tags)):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tg, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            if batch_idx > 0 and batch_idx % 100 == 0:
                print("Batch / epoch / loss", batch_idx, epoch, loss)
    print("Performance at end of training")
    predict(model, data, tags, word_to_ix)

    return model

def run_baseline(data, tags, model=None):
    if model is None:

        t2i = {x: i for (i, x) in enumerate(set(np.concatenate(tags)))}
        w2i = {}
        model = LSTMTagger(EMBEDDING_DIM_OR_PATH, HIDDEN_DIM,
                       len(w2i), len(t2i))
        if model.word_to_ix is None:
            w2i = map_data(data)
        else:
            w2i = model.word_to_ix
        data = map_unks(data, w2i)
        model = train(model, data, tags, w2i, t2i)
        print("Perf. after training on training data")
        predict(model, data, tags, w2i)
    else:
        model, w2i, t2i = model
        data = map_unks(data, w2i)

    print("Performance")
    preds = predict(model, data, tags, w2i)

    return model, w2i, t2i
