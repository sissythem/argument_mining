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

from base import utils


class AduModel:

    def __init__(self, app_config):
        random.seed(2020)
        self.app_config = app_config
        self.properties = self.app_config.properties
        self.resources_path = self.app_config.resources_path
        self.dev_file = self.app_config.dev_csv
        self.train_file = self.app_config.train_csv
        self.test_file = self.app_config.test_csv
        self.eval_file = self.app_config.eval_doc
        self.base_path = self.app_config.base_path
        self.model_file = "best-model.pt"
        self.model = None
        flair.device = torch.device(app_config.device_name)

    def train(self):
        # define columns
        columns = {0: 'text', 1: 'ner'}

        # 1. get the corpus
        corpus: Corpus = ColumnCorpus(self.resources_path, columns,
                                      train_file=self.train_file,
                                      test_file=self.test_file,
                                      dev_file=self.dev_file)

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
        self._train(embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type,
                    corpus=corpus, train_with_dev=False, shuffle=False, save_final_model=True)

    def predict(self):
        self.load()
        data = self._load_eval_doc()
        if data:
            for document in data:
                segment_counter = 0
                initial_json = utils.get_initial_json(name=document["name"], text=document["text"])
                sentences = tokeniser.tokenise(document["text"])
                for sentence in sentences:
                    sentence = Sentence(" ".join(sentence))
                    self.model.predict(sentence)
                    segment_text, segment_type = self._get_args_from_sentence(sentence)
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
                file_path = join(self.app_config.out_files_path, document["name"])
                with open(file_path, "w") as f:
                    json.dump(initial_json, f)

    @staticmethod
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

    def load(self):
        model_path = join(self.base_path, self.model_file)
        self.model = SequenceTagger.load(model_path)

    def _train(self, embeddings, tag_dictionary, tag_type, corpus, train_with_dev=False, shuffle=False,
               save_final_model=True, patience=50, use_tensorboard=True):
        properties = self.properties["adu_model"]
        hidden_size = properties["hidden_size"]
        use_crf = properties["use_crf"]
        rnn_layers = properties["rnn_layers"]
        optimizer = self._configure_optimizer()

        # 5. initialize sequence tagger
        tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=use_crf,
                                                rnn_layers=rnn_layers)
        # 6. initialize trainer
        trainer: ModelTrainer = ModelTrainer(tagger, corpus, use_tensorboard=use_tensorboard, optimizer=optimizer)

        learning_rate = properties["learning_rate"]
        mini_batch_size = properties["mini_batch_size"]
        max_epochs = properties["max_epochs"]
        num_workers = properties["num_workers"]

        # 7. start training
        trainer.train(self.base_path,
                      patience=patience,
                      learning_rate=learning_rate,
                      mini_batch_size=mini_batch_size,
                      max_epochs=max_epochs,
                      train_with_dev=train_with_dev,
                      save_final_model=save_final_model,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      monitor_test=True)
        self.model = trainer.model

    def _configure_optimizer(self):
        optimizer_name = self.properties["adu_model"]["optimizer"]
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD
        else:
            optimizer = torch.optim.Adam
        return optimizer

    def _load_eval_doc(self):
        filepath = join(self.resources_path, self.eval_file)
        with open(filepath, "r") as f:
            data = json.load(f)
        return data["data"]["documents"]
