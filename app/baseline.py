from os import getcwd
from os.path import join

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
EMBEDDING_DIM = 32
HIDDEN_DIM = 64


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

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


def make_model(train_data, train_labels, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
    word_to_ix, tag_to_ix = {}, {}
    # For each words-list (sentence) and tags-list in each tuple of training_data
    for sent, tags in zip(train_data, train_labels):
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                # Assign each word with a unique index
                word_to_ix[word] = len(word_to_ix)
        for t in tags:
            if t not in tag_to_ix:
                tag_to_ix[t] = len(tag_to_ix)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,
                       len(word_to_ix), len(tag_to_ix))
    return model, word_to_ix, tag_to_ix


def train(model, data, labels, word_to_ix, tag_to_ix, num_epochs=20, lr=0.01):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        for d in data:
            inputs = prepare_sequence(d, word_to_ix)
            tag_scores = model(inputs)
            print(tag_scores)

    # again, normally you would NOT do 300 epochs, it is toy data
    for epoch in range(num_epochs):
        for sentence, tags in zip(data, labels):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        for d in data:
            inputs = prepare_sequence(d, word_to_ix)
            tag_scores = model(inputs)

            # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
            # for word i. The predicted tag is the maximum scoring tag.
            # Here, we can see the predicted sequence below is 0 1 2 0 1
            # since 0 is index of the maximum value of row 1,
            # 1 is the index of maximum value of row 2, etc.
            # Which is DET NOUN VERB DET NOUN, the correct sequence!
            print(tag_scores)


def create_training_data():
    path = join(getcwd(), "resources", "data", "adu")
    train_data = pd.read_csv(path, sep="\t", index_col=None, header=0)
    examples, labels = [], []
    instance = []
    instance_labels = []
    for idx, row in train_data.iterrows():
        if row.isnull().all():
            examples.append(instance)
            labels.append(instance_labels)
            instance = []
            instance_labels = []
            continue
        instance.append(row[0])
        instance_labels.append(row[1])
    return examples, labels


data, tags = create_training_data()
model, w2i, t2i = make_model(data, tags)
train(model, data, tags, w2i, t2i)
