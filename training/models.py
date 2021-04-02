from os import mkdir
from os.path import join, exists
from typing import List, Tuple, Type, Union

import flair
try:
    import hdbscan
    import umap
except (BaseException, Exception):
    pass
import numpy as np
import pandas as pd
import torch
from flair.data import Corpus, Dictionary, Sentence
from flair.datasets import ColumnCorpus, CSVClassificationCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, TransformerWordEmbeddings, FastTextEmbeddings, \
    TransformerDocumentEmbeddings, WordEmbeddings, BytePairEmbeddings, DocumentPoolEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from torch.optim import SGD, Adam, Optimizer

from utils.config import AppConfig
from utils.utils import Utilities


class Model:
    """
    Super class for all model classes
    """

    def __init__(self, app_config: AppConfig, model_name: str):
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.model_name = model_name
        self.properties: dict = app_config.properties
        self.model_properties: dict = self._get_model_properties()
        self.resources_path: str = self.app_config.resources_path
        self.utilities = Utilities(app_config=app_config)
        self.model_file: str = "best-model.pt" if self.properties["eval"]["model"] == "best" else "final-model.pt"

    def _get_model_properties(self) -> dict:
        if self.model_name == "adu":
            return self.properties["seq_model"]
        elif self.model_name == "rel" or self.model_name == "stance" or self.model_name == "sim":
            return self.properties["class_model"]
        elif self.model_name == "clustering":
            return self.properties["clustering"]


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
        super(SupervisedModel, self).__init__(app_config=app_config, model_name=model_name)
        # define training / dev / test CSV files
        self.dev_csv, self.train_csv, self.test_csv = self.get_data_files()
        self.use_tensorboard = self.model_properties.get("use_tensorboard", True)
        self._set_bert_model_names()
        self.base_path: str = self._get_base_path()
        self.model = None
        self.optimizer: Optimizer = self.get_optimizer(model_name=model_name)
        self.device_name = app_config.device_name
        flair.device = torch.device(self.device_name)

    def train(self):
        """
        Define the training process of the model
        """
        # 1. get the corpus
        corpus = self.get_corpus()
        self.app_logger.info("Corpus created")
        self.app_logger.info(f"First training sentence: {corpus.train[0]}")

        # 2. make the dictionary from the corpus
        dictionary: Dictionary = self.get_dictionary(corpus=corpus)

        # 3. initialize embeddings
        embeddings = self.get_embeddings()

        # 4. get flair model: SequenceTagger or TextClassifier
        flair_model = self.get_flair_model(dictionary=dictionary, embeddings=embeddings)

        # 5. initialize the ModelTrainer
        trainer: ModelTrainer = self.get_model_trainer(corpus=corpus, flair_model=flair_model)

        trainer.train(self.base_path, learning_rate=self.model_properties["learning_rate"],
                      patience=self.model_properties["patience"], max_epochs=self.model_properties["max_epochs"],
                      mini_batch_size=self.model_properties["mini_batch_size"], monitor_test=True,
                      train_with_dev=self.model_properties["train_with_dev"],
                      save_final_model=self.model_properties["save_final_model"],
                      num_workers=self.model_properties["num_workers"], shuffle=self.model_properties["shuffle"])

    def get_corpus(self) -> Corpus:
        raise NotImplementedError

    def get_dictionary(self, corpus: Corpus) -> Dictionary:
        raise NotImplementedError

    def get_embeddings(self):
        raise NotImplementedError

    def get_flair_model(self, dictionary: Dictionary, embeddings) -> flair.nn.Model:
        raise NotImplementedError

    def load(self):
        """
        Define the way to load the trained model
        """
        raise NotImplementedError

    def get_model_trainer(self, corpus: Corpus, flair_model: flair.nn.Model) -> ModelTrainer:
        # 5. initialize the ModelTrainer
        trainer: ModelTrainer = ModelTrainer(flair_model, corpus, use_tensorboard=self.use_tensorboard,
                                             optimizer=self.optimizer)
        self.app_logger.info("Starting training with ModelTrainer")
        self.app_logger.info(f"Model configuration properties: {self.model_properties}")
        return trainer

    def download_model(self, model_name, dir_name) -> str:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        path = join(self.app_config.output_path, "models", dir_name)
        if not exists(path):
            mkdir(path)
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
        return path

    def get_optimizer(self, model_name: str) -> Union[Type[Optimizer], Optimizer]:
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
        optimizer = Adam
        if optimizer_name == "Adam":
            optimizer = Adam
        elif optimizer_name == "SGD":
            optimizer = SGD
        return optimizer

    def get_data_files(self) -> Tuple[str, str, str]:
        if self.model_name == "adu":
            return self.app_config.adu_dev_csv, self.app_config.adu_train_csv, self.app_config.adu_test_csv
        elif self.model_name == "sim":
            return self.app_config.sim_dev_csv, self.app_config.sim_train_csv, self.app_config.sim_test_csv
        elif self.model_name == "rel":
            return self.app_config.rel_dev_csv, self.app_config.rel_train_csv, self.app_config.rel_test_csv
        elif self.model_name == "stance":
            return self.app_config.stance_dev_csv, self.app_config.stance_train_csv, self.app_config.stance_test_csv

    def _get_base_path(self) -> str:
        if self.model_name == "adu":
            return self.app_config.adu_base_path
        elif self.model_name == "sim":
            return self.app_config.sim_base_path
        elif self.model_name == "rel":
            return self.app_config.rel_base_path
        elif self.model_name == "stance":
            return self.app_config.stance_base_path

    def _set_bert_model_names(self, download: bool = False):
        bert_kinds = self.app_config.get_bert_kind(bert_kind_props=self.model_properties["bert_kind"],
                                                   model_name=self.model_name)
        if bert_kinds:
            self.bert_model_names = self.utilities.get_bert_model_names(bert_kinds=bert_kinds)
        else:
            self.bert_model_names = ["nlpaueb/bert-base-greek-uncased-v1"]
        if download:
            for idx, pair in enumerate(self.bert_model_names):
                model, kind = pair[0], pair[1]
                path = self.download_model(model_name=model, dir_name=kind)
                self.bert_model_names[idx] = (path, kind)


class SequentialModel(SupervisedModel):

    def __init__(self, app_config: AppConfig, model_name: str):
        super(SequentialModel, self).__init__(app_config=app_config, model_name=model_name)
        self.hidden_size: int = self.model_properties["hidden_size"]
        self.use_crf: bool = self.model_properties["use_crf"]
        self.rnn_layers: int = self.model_properties["rnn_layers"]
        # what tag do we want to predict?
        self.tag_type = 'ner'

    def get_corpus(self) -> Corpus:
        columns = {0: 'text', 1: 'ner'}
        data_folder = join(self.resources_path, "data")
        corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=self.train_csv, test_file=self.test_csv,
                                      dev_file=self.dev_csv)
        return corpus

    def get_dictionary(self, corpus: Corpus) -> Dictionary:
        tag_dictionary = corpus.make_tag_dictionary(tag_type=self.tag_type)
        self.app_logger.info("Tag dictionary created")
        self.app_logger.debug(tag_dictionary.idx2item)
        return tag_dictionary

    def get_embeddings(self):
        embedding_types: List[TokenEmbeddings] = [TransformerWordEmbeddings(bert_name[0]) for bert_name in
                                                  self.bert_model_names]

        embeddings: StackedEmbeddings = StackedEmbeddings(embedding_types)
        return embeddings

    def get_flair_model(self, dictionary: Dictionary, embeddings) -> flair.nn.Model:
        # 5. initialize sequence tagger
        tagger: SequenceTagger = SequenceTagger(hidden_size=self.hidden_size,
                                                embeddings=embeddings,
                                                tag_dictionary=dictionary,
                                                tag_type=self.tag_type,
                                                use_crf=self.use_crf,
                                                rnn_layers=self.rnn_layers)
        return tagger

    def load(self):
        """
        Define the way to load the trained model
        """
        model_path = join(self.base_path, self.model_file)
        self.app_logger.info(f"Loading model from path: {model_path}")
        self.model = SequenceTagger.load(model_path)


class ClassificationModel(SupervisedModel):

    def __init__(self, app_config: AppConfig, model_name: str):
        super(ClassificationModel, self).__init__(app_config=app_config, model_name=model_name)

    def get_corpus(self) -> Corpus:
        data_folder = join(self.resources_path, "data")
        # define columns
        column_name_map = {0: "text", 1: "label_topic"}
        # create Corpus
        corpus: Corpus = CSVClassificationCorpus(data_folder=data_folder, column_name_map=column_name_map,
                                                 skip_header=True, delimiter="\t", train_file=self.train_csv,
                                                 test_file=self.test_csv, dev_file=self.dev_csv)
        return corpus

    def get_dictionary(self, corpus: Corpus) -> Dictionary:
        # make label dictionary
        return corpus.make_label_dictionary()

    def get_embeddings(self):
        # initialize the document embeddings
        bert_name = self.bert_model_names[0][0]
        document_embeddings = TransformerDocumentEmbeddings(bert_name, fine_tune=True)
        return document_embeddings

    def get_flair_model(self, dictionary: Dictionary, embeddings) -> flair.nn.Model:
        # create the TextClassifier
        classifier = TextClassifier(document_embeddings=embeddings, label_dictionary=dictionary,
                                    multi_label=dictionary.multi_label)
        return classifier

    def load(self):
        """
        Define the way to load the trained model
        """
        model_path = join(self.base_path, self.model_file)
        self.app_logger.info(f"Loading model from path: {model_path}")
        self.model = TextClassifier.load(model_path)


class UnsupervisedModel(Model):
    """
    Abstract class representing an unsupervised model
    """

    def __init__(self, app_config: AppConfig, model_name: str):
        super(UnsupervisedModel, self).__init__(app_config=app_config, model_name=model_name)
        self.device_name = app_config.device_name


class Clustering(UnsupervisedModel):

    def __init__(self, app_config: AppConfig, model_name="clustering"):
        super(Clustering, self).__init__(app_config=app_config, model_name=model_name)
        self.n_clusters = self.model_properties["n_clusters"]
        embedding_kind = self.model_properties["embeddings"]
        if embedding_kind == "fasttext":
            path_to_embeddings = join(self.resources_path, "embeddings", "wiki.el.bin")
            self.document_embeddings = DocumentPoolEmbeddings([
                FastTextEmbeddings(path_to_embeddings, use_local=True)
            ])
            # self.document_embeddings = DocumentPoolEmbeddings(
            #     [
            #         # standard FastText word embeddings for English
            #         WordEmbeddings('en'),
            #         # Byte pair embeddings for English
            #         BytePairEmbeddings('en'),
            #     ]
            # )
        else:
            self.bert_model_names = self.utilities.get_bert_model_names(bert_kinds=[embedding_kind])
            bert_name = self.bert_model_names[0][0]
            self.document_embeddings = TransformerDocumentEmbeddings(bert_name)

    def get_embeddings(self, sentences):
        sentence_embeddings = []
        for sentence in sentences:
            flair_sentence = Sentence(sentence)
            self.document_embeddings.embed(flair_sentence)
            embeddings = flair_sentence.get_embedding()
            embeddings = embeddings.to("cpu")
            embeddings = embeddings.detach().numpy()
            sentence_embeddings.append(embeddings)
        embeddings = np.asarray(sentence_embeddings)
        return embeddings

    def get_clusters(self, sentences, n_clusters=None):
        raise NotImplementedError


class Hdbscan(Clustering):

    def __init__(self, app_config: AppConfig):
        super(Hdbscan, self).__init__(app_config=app_config)

    def get_clusters(self, sentences, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        embeddings = self.get_embeddings(sentences=sentences)
        try:
            self.app_logger.debug(f"Sentence embeddings shape: {embeddings.shape}")
            # reduce document dimensionality
            umap_embeddings = umap.UMAP(n_neighbors=n_clusters, metric='cosine').fit_transform(embeddings)
            # clustering
            clusters = hdbscan.HDBSCAN(min_cluster_size=n_clusters, metric='euclidean',
                                       cluster_selection_method='eom').fit(umap_embeddings)
            return clusters
        except (BaseException, Exception) as e:
            self.app_logger.error(e)


class KmeansClustering(Clustering):

    def __init__(self, app_config):
        super(KmeansClustering, self).__init__(app_config=app_config)
        self.kmeans_model = KMeans(n_clusters=self.n_clusters)

    def get_clusters(self, sentences, n_clusters=None):
        embeddings = self.get_embeddings(sentences=sentences)
        clusters = self.kmeans_model.fit(embeddings)
        return clusters


class Agglomerative(Clustering):

    def __init__(self, app_config: AppConfig):
        super(Agglomerative, self).__init__(app_config=app_config)

    def get_clusters(self, sentences, n_clusters=None):
        n_clusters = len(sentences) / 2
        agglomerative_model = AgglomerativeClustering(linkage="complete", affinity="cosine",
                                                      n_clusters=n_clusters, compute_distances=True)
        embeddings = self.get_embeddings(sentences=sentences)
        clusters = agglomerative_model.fit(embeddings)
        return clusters


class TopicModel(Hdbscan):
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
