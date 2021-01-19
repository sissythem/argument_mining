import random
from os.path import join
from typing import List

import flair
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from ellogon import tokeniser
from flair.data import Corpus
from flair.datasets import ColumnCorpus, CSVClassificationCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.nn import Model
from flair.trainers import ModelTrainer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
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

        self.app_logger.info("Starting training with ModelTrainer")
        self.app_logger.info("Model configuration properties: {}".format(self.model_properties))
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

        self.app_logger.info("Starting training with ModelTrainer")
        self.app_logger.info("Model configuration properties: {}".format(self.model_properties))
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


class TopicModel:

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.device_name = app_config.device_name

    def get_topics(self, sentences: List[str]):
        if not sentences:
            return {}
        model = SentenceTransformer("distiluse-base-multilingual-cased-v2").to(self.device_name)
        embeddings = model.encode(sentences, show_progress_bar=True)
        self.app_logger.debug(f"Sentence embeddings shape: {embeddings.shape}")
        # reduce document dimensionality
        k = int(len(sentences) / 2)
        umap_embeddings = umap.UMAP(n_neighbors=k,
                                    n_components=5,
                                    metric='cosine').fit_transform(embeddings)

        # clustering
        cluster = hdbscan.HDBSCAN(min_cluster_size=k,
                                  metric='euclidean',
                                  cluster_selection_method='eom').fit(umap_embeddings)

        docs_df = pd.DataFrame(sentences, columns=["Sentence"])
        docs_df['Topic'] = cluster.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Sentence': ' '.join})
        tf_idf, count = self._c_tf_idf(docs_per_topic.Sentence.values, m=len(sentences))
        top_n_words = self._extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=5)
        topic_sizes = self._extract_topic_sizes(docs_df).head(5)
        topic_ids = topic_sizes["Topic"]
        topics = {}
        for topic in topic_ids:
            words = []
            for word_score_tuple in top_n_words[topic]:
                words.append(word_score_tuple[0])
            topics[topic] = words
        return topics

    @staticmethod
    def _c_tf_idf(sentences, m, ngram_range=(1, 1)):
        greek_stopwords = tokeniser.stop_words()
        count = CountVectorizer(ngram_range=ngram_range, stop_words=greek_stopwords).fit(sentences)
        t = count.transform(sentences).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)
        return tf_idf, count

    @staticmethod
    def _extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                       enumerate(labels)}
        return top_n_words

    @staticmethod
    def _extract_topic_sizes(df):
        topic_sizes = (df.groupby(['Topic'])
                       .Sentence
                       .count()
                       .reset_index()
                       .rename({"Topic": "Topic", "Sentence": "Size"}, axis='columns')
                       .sort_values("Size", ascending=False))
        return topic_sizes

    @staticmethod
    def visualize_topics(cluster, embeddings):
        # Prepare data
        umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        result = pd.DataFrame(umap_data, columns=['x', 'y'])
        result['labels'] = cluster.labels_

        # Visualize clusters
        # fig, ax = plt.subplots(figsize=(20, 10))
        plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
        plt.colorbar()
