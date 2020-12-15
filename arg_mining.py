import hashlib
import json
import random
from os.path import join
from typing import List

import flair
import torch
from ellogon import tokeniser
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus, CSVClassificationCorpus
from flair.embeddings import TokenEmbeddings, BertEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.nn import Model
from flair.trainers import ModelTrainer
from torch.optim.optimizer import Optimizer

from utils import AppConfig


class Classifier:

    def __init__(self, app_config, dev_csv, train_csv, test_csv, base_path, model_name):
        random.seed(2020)
        self.app_config: AppConfig = app_config
        self.app_logger = app_config.app_logger
        self.properties: dict = self.app_config.properties
        self.resources_path: str = self.app_config.resources_path
        self.dev_file: str = dev_csv
        self.train_file: str = train_csv
        self.test_file: str = test_csv
        self.base_path: str = base_path
        self.model_file: str = "best-model.pt"
        self.model: Model = None
        self.optimizer: Optimizer = self.get_optimizer()
        flair.device = torch.device(app_config.device_name)

        if model_name == "adu":
            self.model_properties: dict = self.properties["adu_model"]
        elif model_name == "rel" or model_name == "stance":
            self.model_properties: dict = self.properties["rel_model"]

    def get_optimizer(self):
        optimizer_name = self.properties["adu_model"]["optimizer"]
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD
        else:
            optimizer = torch.optim.Adam
        return optimizer

    def train(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class AduModel(Classifier):

    def __init__(self, app_config):
        super(AduModel, self).__init__(app_config=app_config, dev_csv=app_config.adu_dev_csv,
                                       train_csv=app_config.adu_train_csv, test_csv=app_config.adu_test_csv,
                                       base_path=app_config.adu_base_path, model_name="adu")

        self.hidden_size: int = self.model_properties["hidden_size"]
        self.use_crf: bool = self.model_properties["use_crf"]
        self.rnn_layers: int = self.model_properties["rnn_layers"]
        self.learning_rate: float = self.model_properties["learning_rate"]
        self.mini_batch_size: int = self.model_properties["mini_batch_size"]
        self.max_epochs: int = self.model_properties["max_epochs"]
        self.num_workers: int = self.model_properties["num_workers"]
        self.patience: int = self.model_properties["patience"]
        self.use_tensorboard: bool = self.model_properties["use_tensorboard"]
        self.train_with_dev: bool = self.model_properties["train_with_dev"]
        self.save_final_model: bool = self.model_properties["save_final_model"]
        self.shuffle: bool = self.model_properties["shuffle"]

    def train(self):
        # define columns
        columns = {0: 'text', 1: 'ner'}
        # 1. get the corpus
        corpus: Corpus = ColumnCorpus(self.resources_path, columns,
                                      train_file=self.train_file,
                                      test_file=self.test_file,
                                      dev_file=self.dev_file)

        self.app_logger.info("Corpus created")
        self.app_logger.info("First training sentence: {}".format(corpus.train[0]))
        # 2. what tag do we want to predict?
        tag_type = 'ner'

        # 3. make the tag dictionary from the corpus
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        self.app_logger.info("Tag dictionary created")
        self.app_logger.debug(tag_dictionary.idx2item)

        # 4. initialize embeddings
        embedding_types: List[TokenEmbeddings] = [
            BertEmbeddings('nlpaueb/bert-base-greek-uncased-v1')
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embedding_types)

        # 5. initialize sequence tagger
        tagger: SequenceTagger = SequenceTagger(hidden_size=self.hidden_size,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=self.use_crf,
                                                rnn_layers=self.rnn_layers)
        # 6. initialize trainer
        trainer: ModelTrainer = ModelTrainer(tagger, corpus, use_tensorboard=self.use_tensorboard,
                                             optimizer=self.optimizer)

        self.app_logger.debug("Starting training with ModelTrainer")
        self.app_logger.debug("Model configuration properties: {}".format(self.model_properties))
        # 7. start training
        trainer.train(self.base_path, patience=self.patience, learning_rate=self.learning_rate,
                      mini_batch_size=self.mini_batch_size, max_epochs=self.max_epochs,
                      train_with_dev=self.train_with_dev, save_final_model=self.save_final_model,
                      num_workers=self.num_workers, shuffle=self.shuffle, monitor_test=True)
        self.model = trainer.model

    def load(self):
        model_path = join(self.base_path, self.model_file)
        self.app_logger.info("Loading ADU model from path: {}".format(model_path))
        self.model = SequenceTagger.load(model_path)


class RelationsModel(Classifier):

    def __init__(self, app_config, dev_csv, train_csv, test_csv, base_path, model_name):
        super(RelationsModel, self).__init__(app_config=app_config, dev_csv=dev_csv, train_csv=train_csv,
                                             test_csv=test_csv, base_path=base_path, model_name=model_name)

        self.hidden_size: int = self.model_properties["hidden_size"]
        self.use_crf: bool = self.model_properties["use_crf"]
        self.layers: int = self.model_properties["layers"]
        self.learning_rate: float = self.model_properties["learning_rate"]
        self.mini_batch_size: int = self.model_properties["mini_batch_size"]
        self.max_epochs: int = self.model_properties["max_epochs"]
        self.num_workers: int = self.model_properties["num_workers"]
        self.use_tensorboard: bool = self.model_properties["use_tensorboard"]
        self.train_with_dev: bool = self.model_properties["train_with_dev"]
        self.save_final_model: bool = self.model_properties["save_final_model"]
        self.shuffle: bool = self.model_properties["shuffle"]

    def train(self):
        # define columns
        column_name_map = {0: "text", 1: "label_topic"}
        # 1. create Corpus
        corpus: Corpus = CSVClassificationCorpus(data_folder=self.resources_path, column_name_map=column_name_map,
                                                 skip_header=True, delimiter="\t", train_file=self.train_file,
                                                 test_file=self.test_file, dev_file=self.dev_file)
        self.app_logger.info("Corpus created")
        self.app_logger.info("First training sentence: {}".format(corpus.train[0]))

        # 2. make label dictionary
        label_dictionary = corpus.make_label_dictionary()

        # 3. initialize embeddings
        # document_embeddings = TransformerDocumentEmbeddings('nlpaueb/bert-base-greek-uncased-v1')
        # document_embeddings.tokenizer.model_max_length = 512
        bert_embeddings = BertEmbeddings("nlpaueb/bert-base-greek-uncased-v1")

        # initialize the document embeddings, mode = mean
        document_embeddings = DocumentPoolEmbeddings([bert_embeddings])

        # 4. create the TextClassifier
        classifier = TextClassifier(document_embeddings=document_embeddings, label_dictionary=label_dictionary,
                                    multi_label=True)

        # 5. create the ModelTrainer
        trainer: ModelTrainer = ModelTrainer(classifier, corpus, use_tensorboard=self.use_tensorboard,
                                             optimizer=self.optimizer)

        self.app_logger.debug("Starting training with ModelTrainer")
        self.app_logger.debug("Model configuration properties: {}".format(self.model_properties))
        # 7. start training
        trainer.train(self.base_path, learning_rate=self.learning_rate,
                      mini_batch_size=self.mini_batch_size, max_epochs=self.max_epochs,
                      train_with_dev=self.train_with_dev, save_final_model=self.save_final_model,
                      num_workers=self.num_workers, shuffle=self.shuffle, monitor_test=True)
        self.model = trainer.model

    def load(self):
        model_path = join(self.base_path, self.model_file)
        self.app_logger.info("Loading Relations model from path: {}".format(model_path))
        self.model = TextClassifier.load(model_path)


class ArgumentMining:

    def __init__(self, app_config):
        self.app_config: AppConfig = app_config
        self.app_logger = app_config.app_logger

    def predict(self, document):
        json_obj = self._predict_adus(document=document)
        segments = json_obj["annotations"]["ADUs"]
        major_claims, claims, premises = [], [], []
        for segment in segments:
            text = segment["segment"]
            segment_id = segment["id"]
            if segment["type"] == "major_claim":
                major_claims.append((text, segment_id))
            elif segment["type"] == "claim":
                claims.append((text, segment_id))
            else:
                premises.append((text, segment_id))
        json_obj = self._predict_relations(major_claims=major_claims, claims=claims, premises=premises,
                                           json_obj=json_obj)
        json_obj = self._predict_stance(major_claims=major_claims, claims=claims, json_obj=json_obj)
        self._save_data(filename=document["title"] + ".json", json_obj=json_obj)

    def _predict_adus(self, document):
        # init document id & annotations
        hash_id = hashlib.md5(document["title"].encode())
        document["id"] = hash_id.hexdigest()
        document["annotations"] = {
            "ADUs": [],
            "Relations": []
        }

        # load ADU model
        adu_model = AduModel(app_config=self.app_config)
        adu_model.load()

        self.app_logger.debug(
            "Processing document with id: {} and name: {}".format(document["id"], document["title"]))

        segment_counter = 0
        sentences = tokeniser.tokenise_no_punc(document["content"])
        for sentence in sentences:
            self.app_logger.debug("Predicting labels for sentence: {}".format(sentence))
            sentence = list(sentence)
            sentence = Sentence(" ".join(sentence).strip())
            adu_model.model.predict(sentence)
            self.app_logger.debug("Output: {}".format(sentence.to_tagged_string()))
            segment_text, segment_type = self._get_args_from_sentence(sentence)
            if segment_text and segment_type:
                self.app_logger.debug("Segment text: {}".format(segment_text))
                self.app_logger.debug("Segment type: {}".format(segment_type))
                segment_counter += 1
                try:
                    start_idx = document["content"].index(segment_text)
                except(Exception, BaseException):
                    try:
                        start_idx = document["content"].index(segment_text[:4])
                    except(Exception, BaseException):
                        start_idx = None
                if start_idx:
                    end_idx = start_idx + len(segment_text)
                else:
                    start_idx, end_idx = "", ""
                seg = {
                    "id": "T{}".format(segment_counter),
                    "type": segment_type,
                    "starts": str(start_idx),
                    "ends": str(end_idx),
                    "segment": segment_text
                }
                document["annotations"]["ADUs"].append(seg)
        return document

    def _predict_relations(self, major_claims, claims, premises, json_obj):

        # load Relations model
        rel_model = RelationsModel(app_config=self.app_config, dev_csv=self.app_config.rel_dev_csv,
                                   train_csv=self.app_config.rel_train_csv, test_csv=self.app_config.rel_test_csv,
                                   base_path=self.app_config.rel_base_path, model_name="rel")
        rel_model.load()

        rel_counter = 0
        for major_claim in major_claims:
            for claim in claims:
                sentence_pair = "[CLS] " + claim[0] + " [SEP] " + major_claim[0]
                self.app_logger.debug("Predicting relation for sentence pair: {}".format(sentence_pair))
                sentence = Sentence(sentence_pair)
                rel_model.model.predict(sentence)
                # TODO check get_labels
                label = sentence.get_labels()
                if label != "other":
                    rel_counter += 1
                    rel_dict = {
                        "id": "R{}".format(rel_counter),
                        "type": label,
                        "arg1": claim[1],
                        "arg2": major_claim[1]
                    }
                    json_obj["annotations"]["Relations"].append(rel_dict)
        for claim in claims:
            for premise in premises:
                sentence_pair = "[CLS] " + premise[0] + " [SEP] " + claim[0]
                self.app_logger.debug("Predicting relation for sentence pair: {}".format(sentence_pair))
                sentence = Sentence(sentence_pair)
                rel_model.model.predict(sentence)
                label = sentence.get_labels()
                if label != "other":
                    rel_counter += 1
                    rel_dict = {
                        "id": "R{}".format(rel_counter),
                        "type": label,
                        "arg1": premise[1],
                        "arg2": claim[1]
                    }
                    json_obj["annotations"]["Relations"].append(rel_dict)
        return json_obj

    def _predict_stance(self, major_claims, claims, json_obj):
        # load Stance model
        stance_model = RelationsModel(app_config=self.app_config, dev_csv=self.app_config.stance_dev_csv,
                                      train_csv=self.app_config.stance_train_csv,
                                      test_csv=self.app_config.stance_test_csv,
                                      base_path=self.app_config.stance_base_path, model_name="stance")
        stance_model.load()

        stance_counter = 0
        for major_claim in major_claims:
            for claim in claims:
                sentence_pair = "[CLS] " + claim[0] + " [SEP] " + major_claim[0]
                self.app_logger.debug("Predicting stance for sentence pair: {}".format(sentence_pair))
                sentence = Sentence(sentence_pair)
                stance_model.model.predict(sentence)
                label = sentence.get_labels()
                # TODO check label
                if label != "other":
                    stance_counter += 1
                    stance_list = [{
                        "id": "A{}".format(stance_counter),
                        "type": label
                    }]
                    for segment in json_obj["annotations"]["ADUs"]:
                        if segment["id"] == claim[1]:
                            segment["stance"] = stance_list
        return json_obj

    def _save_data(self, filename, json_obj):
        file_path = join(self.app_config.out_files_path, filename)
        self.app_logger.debug("Writing output to json file")
        with open(file_path, "w", encoding='utf8') as f:
            f.write(json.dumps(json_obj, indent=4, sort_keys=False, ensure_ascii=False))

    def _get_args_from_sentence(self, sentence: Sentence):
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
        self.app_logger.debug("Words and labels for current sentence: {}".format(words, labels))
        self.app_logger.debug("Extracting ADU from sentence...")
        idx = 0
        segment_text, segment_type = "", ""
        while idx < len(labels):
            label = labels[idx]
            self.app_logger.debug("Current label: {}".format(label))
            if label.startswith("B-"):
                segment_type = label.replace("B-", "")
                self.app_logger.debug("Found ADU with type: {}".format(segment_type))
                segment_text = words[idx]
                next_correct_label = "I-{}".format(segment_type)
                idx += 1
                if idx >= len(labels):
                    break
                next_label = labels[idx]
                while next_label == next_correct_label:
                    segment_text += " {}".format(words[idx])
                    idx += 1
                    if idx >= len(labels):
                        break
                    next_label = labels[idx]
            else:
                idx += 1
        return segment_text, segment_type
