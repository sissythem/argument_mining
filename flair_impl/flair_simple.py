import random
import os
from os import getcwd
from os.path import join
from pathlib import Path
from typing import List

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim.adam import Adam


def train(base_path, embeddings, tag_dictionary, tag_type, corpus, hidden_size, mini_batch_size, num_workers,
          optimizer=Adam, learning_rate=0.01, rnn_layers=1, max_epochs=150, train_with_dev=False, use_crf=True,
          shuffle=False, save_final_model=True):
    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=use_crf,
                                            rnn_layers=rnn_layers)
    # 6. initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=optimizer)

    # 7. start training
    trainer.train(base_path,
                  learning_rate=learning_rate,
                  mini_batch_size=mini_batch_size,
                  max_epochs=max_epochs,
                  train_with_dev=train_with_dev,
                  save_final_model=save_final_model,
                  num_workers=num_workers,
                  shuffle=shuffle,
                  monitor_test=True)


def get_base_path(path, hidden_size, rnn_layers, use_crf, optimizer, learning_rate, mini_batch_size):
    # Create a base path:
    embedding_names = 'bert-greek'
    base_path = path + '-' + '-'.join([
        str(embedding_names),
        'hs=' + str(hidden_size),
        'hl=' + str(rnn_layers),
        'crf=' + str(use_crf),
        str(optimizer.__name__),
        'lr=' + str(learning_rate),
        'bs=' + str(mini_batch_size)
    ])
    try:
        # os.mkdir(base_path, 0o755)
        os.makedirs(base_path)
    except (OSError, Exception):
        pass
    return base_path


def main():
    properties = {
        "hidden_size": 256,
        "rnn_layers": 2,
        "use_crf": True,
        "learning_rate": 0.0001,
        "mini_batch_size": 32,
        "num_workers": 8
    }
    curr_dir = Path(getcwd())
    curr_dir = str(curr_dir) if str(curr_dir).endswith("mining") else str(curr_dir.parent)
    data_folder = join(curr_dir, "resources")

    # define columns
    columns = {0: 'text', 1: 'ner'}
    path = join(curr_dir, "output")
    base_path = get_base_path(path=path, hidden_size=properties["hidden_size"], rnn_layers=properties["rnn_layers"],
                              use_crf=properties["use_crf"], optimizer=Adam, learning_rate=properties["learning_rate"],
                              mini_batch_size=properties["mini_batch_size"])

    # 1. get the corpus
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.csv',
                                  test_file='train.csv',
                                  dev_file='train.csv')

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        TransformerWordEmbeddings('nlpaueb/bert-base-greek-uncased-v1')
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embedding_types)

    train(base_path=base_path, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type, corpus=corpus,
          hidden_size=properties["hidden_size"], mini_batch_size=properties["mini_batch_size"],
          num_workers=properties["num_workers"], optimizer=Adam, learning_rate=properties["learning_rate"],
          rnn_layers=properties["rnn_layers"], max_epochs=properties["max_epochs"], use_crf=properties["use_crf"],
          train_with_dev=False, shuffle=False, save_final_model=True)


if __name__ == '__main__':
    random.seed(2020)
    main()
