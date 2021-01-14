import random
from os.path import join
from typing import List

import flair
import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus, CSVClassificationCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings
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
        self.model_file: str = "best-model.pt" if self.properties["eval"]["model"] == "best" else "final-model.pt"
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
        self.dropout: float = self.model_properties["dropout"]
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

    def load(self):
        model_path = join(self.base_path, self.model_file)
        self.app_logger.info("Loading ADU model from path: {}".format(model_path))
        self.model = SequenceTagger.load(model_path)
        self.model.eval()


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
        self.patience: int = self.model_properties["patience"]
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
        # document_embeddings = TransformerDocumentEmbeddings('nlpaueb/bert-base-greek-uncased-v1', fine_tune=True)
        # document_embeddings.tokenizer.model_max_length = 512

        # 3. initialize the document embeddings, mode = mean
        bert_embeddings = BertEmbeddings('nlpaueb/bert-base-greek-uncased-v1')
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
        trainer.train(self.base_path, learning_rate=self.learning_rate, patience=self.patience,
                      mini_batch_size=self.mini_batch_size, max_epochs=self.max_epochs,
                      train_with_dev=self.train_with_dev, save_final_model=self.save_final_model,
                      num_workers=self.num_workers, shuffle=self.shuffle, monitor_test=True)

    def load(self):
        model_path = join(self.base_path, self.model_file)
        self.app_logger.info("Loading Relations model from path: {}".format(model_path))
        self.model = TextClassifier.load(model_path)
        self.model.eval()