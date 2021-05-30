import logging
from os.path import join
from typing import List, Type, Union, AnyStr, Dict

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
    TransformerDocumentEmbeddings, DocumentPoolEmbeddings  # , DocumentTFIDFEmbeddings
# , WordEmbeddings, BytePairEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, OPTICS
from sklearn.feature_extraction.text import CountVectorizer
from torch.optim import SGD, Adam, Optimizer

from utils.config import AppConfig
from utils import utils


class Model:
    """
    Super class for all model classes
    """

    def __init__(self, app_config: AppConfig, model_name: AnyStr):
        self.app_config: AppConfig = app_config
        self.app_logger: logging.Logger = app_config.app_logger
        self.model_name: AnyStr = model_name
        self.properties: Dict = app_config.properties
        self.model_properties: Dict = self._get_model_properties()
        self.transformer_name = self.model_properties["bert_kind"][self.model_name]
        self.resources_path: AnyStr = self.app_config.resources_path
        self.model_file: AnyStr = "best-model.pt" if self.properties["eval"]["model"] == "best" else "final-model.pt"

    def _get_model_properties(self) -> Dict:
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

    def __init__(self, app_config: AppConfig, model_name: AnyStr):
        """
        Classifier class constructor

        Args
            | app_config (AppConfig): the application configuration object
            | model_name (str): the name of the model
        """
        super(SupervisedModel, self).__init__(app_config=app_config, model_name=model_name)
        # define training / dev / test CSV files
        self.data_folder: AnyStr = join(self.app_config.dataset_folder, model_name)
        self.use_tensorboard: bool = self.model_properties.get("use_tensorboard", True)
        self.base_path: AnyStr = self._get_base_path()
        self.model = None
        self.optimizer: Optimizer = self.get_optimizer(model_name=model_name)
        self.device_name: AnyStr = app_config.device_name
        flair.device = torch.device(self.device_name)

    def train(self):
        """
        Define the training process of the model
        """
        # 1. get the corpus
        corpus: Corpus = self.get_corpus()
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

    def get_optimizer(self, model_name: AnyStr) -> Union[Type[Optimizer], Optimizer]:
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

    def _get_base_path(self) -> AnyStr:
        if self.model_name == "adu":
            return self.app_config.adu_base_path
        elif self.model_name == "sim":
            return self.app_config.sim_base_path
        elif self.model_name == "rel":
            return self.app_config.rel_base_path
        elif self.model_name == "stance":
            return self.app_config.stance_base_path


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
        # TODO rename test & dev datasets
        corpus: Corpus = ColumnCorpus(self.data_folder, columns, train_file="train_oversample.csv",
                                      test_file="train_oversample.csv",
                                      dev_file="train_oversample.csv")
        return corpus

    def get_dictionary(self, corpus: Corpus) -> Dictionary:
        tag_dictionary = corpus.make_tag_dictionary(tag_type=self.tag_type)
        self.app_logger.info("Tag dictionary created")
        self.app_logger.debug(tag_dictionary.idx2item)
        return tag_dictionary

    def get_embeddings(self):
        embedding_types: List[TokenEmbeddings] = [TransformerWordEmbeddings(self.transformer_name)]
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
        # define columns
        column_name_map = {0: "text", 1: "label_topic"}
        # create Corpus
        # TODO rename test & dev datasets
        corpus: Corpus = CSVClassificationCorpus(data_folder=self.data_folder, column_name_map=column_name_map,
                                                 skip_header=True, delimiter="\t", train_file="train_oversample.csv",
                                                 test_file="train_oversample.csv", dev_file="train_oversample.csv")
        return corpus

    def get_dictionary(self, corpus: Corpus) -> Dictionary:
        # make label dictionary
        return corpus.make_label_dictionary()

    def get_embeddings(self):
        # initialize the document embeddings
        document_embeddings = TransformerDocumentEmbeddings(self.transformer_name)
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
        self.embedding_kind = self.model_properties["embeddings"]
        if self.embedding_kind == "fasttext":
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
        elif self.embedding_kind == "tfidf":
            pass
        else:
            bert_path = self.model_properties.get("bert_path", None)
            local_files_only = True if bert_path is not None else False
            self.bert_model_names = utils.get_bert_model_names(bert_kinds=[self.embedding_kind], local_path=bert_path)
            bert_name = self.bert_model_names[0][0]
            self.document_embeddings = TransformerDocumentEmbeddings(bert_name)

    def get_embeddings(self, sentences):
        # if self.embedding_kind == "tfidf":
        #     self.document_embeddings = DocumentTFIDFEmbeddings(train_dataset=sentences)
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
        n_clusters = int(len(sentences) / 2)
        agglomerative_model = AgglomerativeClustering(linkage="complete", affinity="cosine",
                                                      n_clusters=n_clusters, compute_distances=True)
        embeddings = self.get_embeddings(sentences=sentences)
        clusters = agglomerative_model.fit(embeddings)
        return clusters


class BirchClustering(Clustering):

    def __init__(self, app_config: AppConfig):
        super(BirchClustering, self).__init__(app_config=app_config)

    def get_clusters(self, sentences, n_clusters=None):
        n_clusters = int(len(sentences) / 2)
        birch_model = Birch(threshold=0.7, n_clusters=n_clusters)
        embeddings = self.get_embeddings(sentences=sentences)
        clusters = birch_model.fit(embeddings)
        return clusters


class OpticsClustering(Clustering):

    def __init__(self, app_config: AppConfig):
        super(OpticsClustering, self).__init__(app_config=app_config)

    def get_clusters(self, sentences, n_clusters=None):
        # n_clusters = int(len(sentences) / 2)
        optics_model = OPTICS(metric="cosine")
        embeddings = self.get_embeddings(sentences=sentences)
        clusters = optics_model.fit(embeddings)
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
        if type(content) == str:
            sentences = utils.tokenize(text=content)
            sentences = [" ".join(s) for s in sentences]
        else:
            sentences = content
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
        greek_stopwords = utils.get_greek_stopwords()
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
