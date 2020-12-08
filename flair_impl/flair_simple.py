import hashlib
import json
import random
from os.path import join
from typing import List

import flair
import torch
from ellogon import tokeniser
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, BertEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim.adam import Adam

from flair_impl import utils


def train(base_path, embeddings, tag_dictionary, tag_type, corpus, hidden_size, mini_batch_size, num_workers,
          optimizer=Adam, learning_rate=0.01, rnn_layers=1, max_epochs=150, use_crf=True, train_with_dev=False,
          shuffle=False, save_final_model=True, patience=50, use_tensorboard=True):
    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=use_crf,
                                            rnn_layers=rnn_layers)
    # 6. initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus, use_tensorboard=use_tensorboard, optimizer=optimizer)

    # 7. start training
    trainer.train(base_path,
                  patience=patience,
                  learning_rate=learning_rate,
                  mini_batch_size=mini_batch_size,
                  max_epochs=max_epochs,
                  train_with_dev=train_with_dev,
                  save_final_model=save_final_model,
                  num_workers=num_workers,
                  shuffle=shuffle,
                  monitor_test=True)


def adu_train(properties, data_folder, base_path):
    random.seed(2020)
    flair.device = torch.device(utils.configure_device())

    # define columns
    columns = {0: 'text', 1: 'ner'}

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
        BertEmbeddings('nlpaueb/bert-base-greek-uncased-v1')
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embedding_types)

    train(base_path=base_path, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type, corpus=corpus,
          hidden_size=properties["hidden_size"], mini_batch_size=properties["mini_batch_size"],
          num_workers=properties["num_workers"], optimizer=Adam, learning_rate=properties["learning_rate"],
          rnn_layers=properties["rnn_layers"], max_epochs=properties["max_epochs"], use_crf=properties["use_crf"],
          train_with_dev=False, shuffle=False, save_final_model=True)


def predict(model_path, input_file_path, out_dir):
    # load the model you trained
    model = SequenceTagger.load(model_path)
    with open(input_file_path, "r") as f:
        data = json.load(f)
    data = data["data"]["documents"]
    if data:
        for document in data:
            segment_counter = 0
            initial_json = _get_initial_doc(document)
            sentences = tokeniser.tokenise(document["text"])
            for sentence in sentences:
                sentence = Sentence(" ".join(sentence))
                model.predict(sentence)
                segment_text, segment_type = _get_args_from_sentence(sentence)
                if segment_text and segment_type:
                    segment_counter += 1
                    seg = {
                        "id": "T{}".format(segment_counter),
                        "type": segment_type,
                        "starts": str(document["text"].index(segment_text)),
                        "ends": str(document["text"].index(segment_text) + len(segment_text)),
                        "segment": segment_text
                    }
                    initial_json["annotations"]["ADUs"].append(seg)
            file_path = join(out_dir, document["name"])
            with open(file_path, "w") as f:
                json.dump(initial_json, f)


def _get_args_from_sentence(sentence: Sentence):
    tagged_string = sentence.to_tagged_string()
    tagged_string_split = tagged_string.split()
    words, labels = [], []
    for tok in tagged_string_split:
        if tok.startswith("<"):
            tok = tok.replace("<", "")
            tok = tok.replace(">", "")
            labels.append(tok)
        else:
            words.append(tok)
    idx = 0
    segment_text, segment_type = "", ""
    while idx < len(labels):
        label = labels[idx]
        if label.startswith("B-"):
            segment_type = label.replace("B-", "")
            segment_text = words[idx]
            next_correct_label = "I-{}".format(segment_type)
            idx += 1
            if idx > len(labels):
                break
            next_label = labels[idx]
            while next_label == next_correct_label:
                segment_text += " {}".format(words[idx])
                idx += 1
                if idx > len(labels):
                    break
                next_label = labels[idx]
        else:
            idx += 1
    return segment_text, segment_type


def _get_initial_doc(document):
    hash_id = hashlib.md5(document["name"].encode())
    return {
        "id": hash_id.hexdigest(),
        "link": "",
        "description": "",
        "date": "",
        "tags": [],
        "document_link": "",
        "publishedAt": "",
        "crawledAt": "",
        "domain": "",
        "netloc": "",
        "content": document["text"],
        "annotations": {
            "ADUs": [],
            "Relations": [],
            "Stance": []
        }
    }
