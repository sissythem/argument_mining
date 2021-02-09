import random
from os import mkdir
from os.path import join, exists
from typing import List

import flair
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from ellogon import tokeniser
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus, CSVClassificationCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from transformers import AutoModel, AutoTokenizer
from utils.config import AppConfig


class Classifier:
    """
    Abstract class representing a classification model
    """

    def __init__(self, app_config: AppConfig, dev_csv: str, train_csv: str, test_csv: str, base_path: str,
                 model_name: str):
        """
        Classifier class constructor

        Args
            | app_config (AppConfig): the application configuration object
            | dev_csv (str): the name of the dev csv file
            | train_csv (str): the name of the train csv file
            | test_csv (str): the name of the test csv file
            | base_path (str): full path to the folder where the model will be stored
            | model_name (str): the name of the model
        """
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
        self.model = None
        self.optimizer: torch.optim.Optimizer = self.get_optimizer()
        flair.device = torch.device(app_config.device_name)

        if model_name == "adu":
            self.model_properties: dict = self.properties["adu_model"]
        elif model_name == "rel" or model_name == "stance":
            self.model_properties: dict = self.properties["rel_model"]

    def get_optimizer(self):
        """
        Define the model's optimizer based on the application properties

        Returns
            optimizer: the optimizer class
        """
        optimizer_name = self.properties["adu_model"]["optimizer"]
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD
        else:
            optimizer = torch.optim.Adam
        return optimizer

    def train(self):
        """
        Define the training process of the model. All subclasses should implement this method
        """
        raise NotImplementedError

    def load(self):
        """
        Define the way to load the trained model. All subclasses should implement this method
        """
        raise NotImplementedError


class AduModel(Classifier):
    """
    Class for the ADU sequence model
    """

    def __init__(self, app_config):
        """
        Constructor for the AduModel class

        Args
            app_config (AppConfig): the application configuration object
        """
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
        """
        ADU training method. It uses the flair library.
        """
        # define columns
        columns = {0: 'text', 1: 'ner'}
        data_folder = join(self.resources_path, "data")
        # 1. get the corpus
        corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=self.train_file, test_file=self.test_file,
                                      dev_file=self.dev_file)

        self.app_logger.info("Corpus created")
        self.app_logger.info(f"First training sentence: {corpus.train[0]}")
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
        self.app_logger.info(f"Model configuration properties: {self.model_properties}")
        # 7. start training
        trainer.train(self.base_path, patience=self.patience, learning_rate=self.learning_rate,
                      mini_batch_size=self.mini_batch_size, max_epochs=self.max_epochs,
                      train_with_dev=self.train_with_dev, save_final_model=self.save_final_model,
                      num_workers=self.num_workers, shuffle=self.shuffle, monitor_test=True)

    def load(self):
        """
        Loads the trained ADU model for the folder specified during the training
        """
        model_path = join(self.base_path, self.model_file)
        self.app_logger.info(f"Loading ADU model from path: {model_path}")
        self.model = SequenceTagger.load(model_path)
        self.model.eval()


class RelationsModel(Classifier):
    """
    Class representing the model for relations and stance prediction
    """

    def __init__(self, app_config, dev_csv, train_csv, test_csv, base_path, model_name):
        """
        Constructor of the RelationsModel class

        Args
            | app_config (AppConfig): the application configuration object
            | dev_csv (str): the name of the dev csv file
            | train_csv (str): the name of the train csv file
            | test_csv (str): the name of the test csv file
            | base_path (str): full path to the folder where the model will be stored
            | model_name (str): the name of the model
        """
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
        """
        Function to train a relations or stance prediction model. Uses the flair library
        """
        data_folder = join(self.resources_path, "data")
        # define columns
        column_name_map = {0: "text", 1: "label_topic"}
        # 1. create Corpus
        corpus: Corpus = CSVClassificationCorpus(data_folder=data_folder, column_name_map=column_name_map,
                                                 skip_header=True, delimiter="\t", train_file=self.train_file,
                                                 test_file=self.test_file, dev_file=self.dev_file)
        self.app_logger.info("Corpus created")
        self.app_logger.info(f"First training sentence: {corpus.train[0]}")

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
        self.app_logger.info(f"Model configuration properties: {self.model_properties}")
        # 7. start training
        trainer.train(self.base_path, learning_rate=self.learning_rate, patience=self.patience,
                      mini_batch_size=self.mini_batch_size, max_epochs=self.max_epochs,
                      train_with_dev=self.train_with_dev, save_final_model=self.save_final_model,
                      num_workers=self.num_workers, shuffle=self.shuffle, monitor_test=True)

    def load(self):
        """
        Load the relations or stance model
        """
        model_path = join(self.base_path, self.model_file)
        self.app_logger.info(f"Loading Relations model from path: {model_path}")
        self.model = TextClassifier.load(model_path)
        self.model.eval()


class Clustering:

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.device_name = app_config.device_name
        model_id = "nlpaueb/bert-base-greek-uncased-v1"
        self.bert_model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_clusters(self, sentences, n_clusters):
        try:
            for sentence in sentences:
                tokens = self.tokenizer.encode(sentence)
            tokens = self.tokenizer.encode()
            input_ids = torch.tensor(tokens).unsqueeze(0)
            outputs = self.bert_model(input_ids)
            embeddings = outputs[1][-1]
            # embeddings = outputs[1]
            # model = SentenceTransformer("distiluse-base-multilingual-cased-v2").to(self.device_name)
            # embeddings = model.encode(sentences, show_progress_bar=True)
            self.app_logger.debug(f"Sentence embeddings shape: {embeddings.shape}")
            # reduce document dimensionality
            umap_embeddings = umap.UMAP(n_neighbors=n_clusters,
                                        metric='cosine').fit_transform(embeddings)

            # clustering
            clusters = hdbscan.HDBSCAN(min_cluster_size=n_clusters,
                                       metric='euclidean',
                                       cluster_selection_method='eom').fit_predict(umap_embeddings)
            return clusters
        except (BaseException, Exception) as e:
            self.app_logger.error(e)

    def get_content_per_cluster(self, n_clusters: int, clusters, sentences, doc_ids):
        cluster_lists = []
        for i in range(n_clusters):
            cluster_lists.append([])
        for idx, cluster in enumerate(clusters):
            sentence = sentences[idx]
            doc_id = doc_ids[idx]
            cluster_lists[idx].append((sentence, doc_id))
        self.print_clusters(cluster_lists=cluster_lists)

    def print_clusters(self, cluster_lists):
        for idx, cluster in enumerate(cluster_lists):
            self.app_logger.debug(f"Content of Cluster {idx}")
            for pair in cluster:
                self.app_logger.debug(f"Sentence {pair[0]} in document with id {pair[1]}")


class TopicModel(Clustering):
    """
    Class for topic modeling
    """

    def __init__(self, app_config: AppConfig):
        """
        Constructor of the TopicModel class

        Args
            app_config (AppConfig): the application configuration object
        """
        super(TopicModel, self).__init__(app_config=app_config)

    def get_topics(self, sentences: List[str]):
        """
        Function that uses BERT model and clustering to find the topics of the document

        Args
            sentences (list): the list of sentences of the document

        Returns
            list: a list of topics
        """
        topics = []
        try:
            n_clusters = int(len(sentences) / 2)
            clusters = self.get_clusters(sentences=sentences, n_clusters=n_clusters)
            docs_df = pd.DataFrame(sentences, columns=["Sentence"])
            docs_df['Topic'] = clusters.labels_
            docs_df['Doc_ID'] = range(len(docs_df))
            docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Sentence': ' '.join})
            tf_idf, count = self._c_tf_idf(docs_per_topic.Sentence.values, m=len(sentences))
            top_n_words = self._extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=10)
            topic_sizes = self._extract_topic_sizes(docs_df).head(2)
            topic_ids = topic_sizes["Topic"]
            for topic in topic_ids:
                for word_score_tuple in top_n_words[topic]:
                    if word_score_tuple[0].isdigit():
                        continue
                    topics.append(word_score_tuple[0])
            return topics
        except (BaseException, Exception) as e:
            self.app_logger.error(e)
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
