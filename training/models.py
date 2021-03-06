import random
from itertools import combinations
from os import mkdir
from os.path import join, exists
from typing import List, Tuple

import flair
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from flair.data import Corpus
from flair.datasets import ColumnCorpus, CSVClassificationCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings, \
    TransformerDocumentEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer

from utils.config import AppConfig
from utils.utils import Utilities


class Model:
    """
    Super class for all model classes
    """

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.properties: dict = app_config.properties
        self.resources_path: str = self.app_config.resources_path
        self.model_file: str = "best-model.pt" if self.properties["eval"]["model"] == "best" else "final-model.pt"


class SupervisedModel(Model):
    """
    Abstract class representing a supervised learning model
    """

    def __init__(self, app_config: AppConfig, model_name: str):
        """
        Classifier class constructor

        Args
            | app_config (AppConfig): the application configuration object
            | model_name (str): the name of the model
        """
        super(SupervisedModel, self).__init__(app_config=app_config)
        # define training / dev / test CSV files
        self.dev_csv, self.train_csv, self.test_csv = self._get_csv_file_names(model_name=model_name)
        self.model_properties: dict = self._get_model_properties(model_name=model_name)
        self.bert_name = self._get_bert_model_name(model_name=model_name)
        self.base_path: str = self._get_base_path(model_name=model_name)
        self.model = None
        self.optimizer: torch.optim.Optimizer = self.get_optimizer(model_name=model_name)
        self.device_name = app_config.device_name
        flair.device = torch.device(self.device_name)

    def download_model(self, model_name) -> str:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        path = join(self.app_config.output_path, "models", "facebook")
        if not exists(path):
            mkdir(path)
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
        return path

    def _get_csv_file_names(self, model_name: str) -> Tuple[str, str, str]:
        if model_name == "adu":
            return self.app_config.adu_dev_csv, self.app_config.adu_train_csv, self.app_config.adu_test_csv
        elif model_name == "sim":
            return self.app_config.sim_dev_csv, self.app_config.sim_train_csv, self.app_config.sim_test_csv
        elif model_name == "rel":
            return self.app_config.rel_dev_csv, self.app_config.rel_train_csv, self.app_config.rel_test_csv
        elif model_name == "stance":
            return self.app_config.stance_dev_csv, self.app_config.stance_train_csv, self.app_config.stance_test_csv

    def _get_base_path(self, model_name: str) -> str:
        if model_name == "adu":
            return self.app_config.adu_base_path
        elif model_name == "sim":
            return self.app_config.sim_base_path
        elif model_name == "rel":
            return self.app_config.rel_base_path
        elif model_name == "stance":
            return self.app_config.stance_base_path

    def _get_model_properties(self, model_name: str) -> dict:
        if model_name == "adu":
            return self.properties["seq_model"]
        elif model_name == "rel" or model_name == "stance" or model_name == "sim":
            return self.properties["class_model"]

    def _get_bert_model_name(self, model_name: str, download: bool = False) -> str:
        self.bert_kind = self.app_config.get_bert_kind(bert_kind_props=self.model_properties["bert_kind"],
                                                       model_name=model_name)
        if self.bert_kind == "base":
            return "bert-base-uncased"
        elif self.bert_kind == "aueb":
            return "nlpaueb/bert-base-greek-uncased-v1"
        elif self.bert_kind == "nli":
            if download:
                path = self.download_model(model_name="facebook/bart-large-mnli")
                return path
            else:
                return "facebook/bart-large-mnli"
        elif self.bert_kind == "base-multi":
            return "bert-base-multilingual-uncased"

    def get_optimizer(self, model_name: str) -> torch.optim.Optimizer:
        """
        Define the model's optimizer based on the application properties

        Args
            | model_name (str): the name of the model

        Returns
            optimizer: the optimizer class
        """
        if model_name == "adu":
            properties = self.properties["seq_model"]
        else:
            properties = self.properties["class_model"]
        optimizer_name = properties["optimizer"]
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


class UnsupervisedModel(Model):
    """
    Abstract class representing an unsupervised model
    """

    def __init__(self, app_config: AppConfig):
        super(UnsupervisedModel, self).__init__(app_config=app_config)
        self.device_name = app_config.device_name
        self.utilities = Utilities(app_config=app_config)


class AduModel(SupervisedModel):
    """
    Class for the ADU sequence model
    """

    def __init__(self, app_config: AppConfig, model_name="adu"):
        """
        Constructor for the AduModel class

        Args
            app_config (AppConfig): the application configuration object
        """
        super(AduModel, self).__init__(app_config=app_config, model_name=model_name)

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
        corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=self.train_csv, test_file=self.test_csv,
                                      dev_file=self.dev_csv)

        self.app_logger.info("Corpus created")
        self.app_logger.info(f"First training sentence: {corpus.train[0]}")
        # 2. what tag do we want to predict?
        tag_type = 'ner'

        # 3. make the tag dictionary from the corpus
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        self.app_logger.info("Tag dictionary created")
        self.app_logger.debug(tag_dictionary.idx2item)

        # 4. initialize embeddings
        embedding_types: List[TokenEmbeddings] = [BertEmbeddings(self.bert_name)]

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


class RelationsModel(SupervisedModel):
    """
    Class representing the model for relations and stance prediction
    """

    def __init__(self, app_config: AppConfig, model_name: str):
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
        super(RelationsModel, self).__init__(app_config=app_config, model_name=model_name)
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
                                                 skip_header=True, delimiter="\t", train_file=self.train_csv,
                                                 test_file=self.test_csv, dev_file=self.dev_csv)
        self.app_logger.info("Corpus created")
        self.app_logger.info(f"First training sentence: {corpus.train[0]}")

        # 2. make label dictionary
        label_dictionary = corpus.make_label_dictionary()

        # 3. initialize embeddings
        document_embeddings = TransformerDocumentEmbeddings(self.bert_name, fine_tune=True)
        document_embeddings.tokenizer.model_max_length = 512

        # 3. initialize the document embeddings, mode = mean
        # bert_embeddings = BertEmbeddings(self.bert_name)
        # document_embeddings = DocumentPoolEmbeddings([bert_embeddings])

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


class Clustering(UnsupervisedModel):

    def __init__(self, app_config: AppConfig):
        super(Clustering, self).__init__(app_config=app_config)
        model_id = "nlpaueb/bert-base-greek-uncased-v1"
        self.bert_model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def run_clustering(self, n_clusters, sentences, adu_ids, doc_ids):
        clusters = self.get_clusters(n_clusters=n_clusters, sentences=sentences)
        relations = self.get_cross_document_relations(clusters=clusters, sentences=sentences, adu_ids=adu_ids,
                                                      doc_ids=doc_ids)
        relations_ids = []
        for relation in relations:
            # TODO uncomment save to Elasticsearch -- change index inside the save_relation function!!
            # self.app_config.elastic_save.save_relation(relation=relation)
            relations_ids.append(relation["id"])
        return relations_ids

    def get_clusters(self, n_clusters, sentences):
        try:
            # model = SentenceTransformer("distiluse-base-multilingual-cased-v2").to(self.device_name)
            # embeddings = model.encode(sentences, show_progress_bar=True)
            sentence_embeddings = []
            for sentence in sentences:
                tokens = self.tokenizer.encode(sentence)
                input_ids = torch.tensor(tokens).unsqueeze(0)
                outputs = self.bert_model(input_ids)
                embeddings = outputs[1][-1].detach().numpy()
                sentence_embeddings.append(embeddings)
            embeddings = np.asarray(sentence_embeddings)
            self.app_logger.debug(f"Sentence embeddings shape: {embeddings.shape}")

            # reduce document dimensionality
            umap_embeddings = umap.UMAP(n_neighbors=n_clusters, metric='cosine').fit_transform(embeddings)

            # clustering
            clusters = hdbscan.HDBSCAN(min_cluster_size=n_clusters, metric='euclidean',
                                       cluster_selection_method='eom').fit(umap_embeddings)
            return clusters
        except (BaseException, Exception) as e:
            self.app_logger.error(e)

    def get_cross_document_relations(self, clusters, sentences, adu_ids, doc_ids):
        cluster_dict = self.get_content_per_cluster(clusters=clusters, sentences=sentences, adu_ids=adu_ids,
                                                    doc_ids=doc_ids)
        relations = []
        for cluster, pairs in cluster_dict.items():
            cluster_combinations = list(combinations(pairs, r=2))
            for pair_combination in cluster_combinations:
                arg1 = pair_combination[0]
                arg2 = pair_combination[1]
                relation = {
                    "id": f"{arg1[1]};{arg2[1]};{arg1[0]};{arg2[0]}",
                    "cluster": cluster,
                    "source": arg1[0],
                    "source_doc": arg1[1],
                    "target": arg2[0],
                    "target_doc": arg2[1]
                }
                relations.append(relation)
        return relations

    def get_content_per_cluster(self, clusters, sentences, doc_ids, adu_ids, print_clusters=True):
        clusters_dict = {}
        for idx, cluster in enumerate(clusters.labels_):
            if cluster not in clusters_dict.keys():
                clusters_dict[cluster] = []
            adu_id = adu_ids[idx]
            doc_id = doc_ids[idx]
            sentence = sentences[idx]
            clusters_dict[cluster].append((adu_id, doc_id, sentence))
        if print_clusters:
            self.print_clusters(cluster_lists=clusters_dict)
        return clusters_dict

    def print_clusters(self, cluster_lists):
        for idx, cluster_list in cluster_lists.items():
            self.app_logger.debug(f"Content of Cluster {idx}")
            for pair in cluster_list:
                self.app_logger.debug(f"Sentence {pair[0]} in document with id {pair[1]}")
                self.app_logger.debug(f"Sentence content: {pair[2]}")


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

    def get_topics(self, content):
        """
        Function that uses BERT model and clustering to find the topics of the document

        Args
            sentences (list): the list of sentences of the document

        Returns
            list: a list of topics
        """
        topics = []
        sentences = self.utilities.tokenize(text=content)
        sentences = [" ".join(s) for s in sentences]
        self.app_logger.debug(f"Sentences fed to TopicModel: {sentences}")
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

    def _c_tf_idf(self, sentences, m, ngram_range=(1, 1)):
        greek_stopwords = self.utilities.get_greek_stopwords()
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
