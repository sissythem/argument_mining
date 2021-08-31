from typing import List, Type, Union, AnyStr, Dict

import flair

try:
    import hdbscan
    import umap
except (BaseException, Exception):
    pass
import pandas as pd
import torch
from flair.data import Corpus, Dictionary, Sentence
from flair.datasets import ColumnCorpus, CSVClassificationCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, TransformerWordEmbeddings, \
    TransformerDocumentEmbeddings
from flair.models.similarity_learning_model import CosineSimilarity
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from torch.optim import SGD, Adam, Optimizer

import logging
from os.path import join

import numpy as np
from src.utils.config import AppConfig
from src.utils import utils
from itertools import combinations


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
        if "bert_kind" in self.model_properties.keys():
            self.transformer_name = self.model_properties["bert_kind"][self.model_name] if \
                self.model_name != "clustering" else self.model_properties["bert_kind"]
        self.resources_path: AnyStr = self.app_config.resources_path
        self.model_file: AnyStr = "best-model.pt" if self.properties["eval"]["model"] == "best" else "final-model.pt"

    def _get_model_properties(self) -> Dict:
        if self.model_name == "adu":
            return self.properties["seq_model"]
        elif self.model_name == "rel" or self.model_name == "stance" or self.model_name == "sim":
            return self.properties["class_model"]
        elif self.model_name == "clustering":
            return self.properties["clustering"]
        elif self.model_name == "alignment":
            return self.properties["alignment"]


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
        self.optimizer: Optimizer = self.get_optimizer()
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

    def load(self, model_path=None):
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

    def get_optimizer(self) -> Union[Type[Optimizer], Optimizer]:
        """
        Define the model's optimizer based on the application properties

        Args
            | model_name (str): the name of the model

        Returns
            optimizer: the optimizer class
        """
        optimizer_name = self.model_properties.get("optimizer", None)
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
        columns = {0: 'text', 1: self.tag_type}
        corpus: Corpus = ColumnCorpus(self.data_folder, columns, train_file="train.csv", test_file="test.csv",
                                      dev_file="dev.csv")
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

    def load(self, model_path=None):
        """
        Define the way to load the trained model
        """
        if not model_path:
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
        corpus: Corpus = CSVClassificationCorpus(data_folder=self.data_folder, column_name_map=column_name_map,
                                                 skip_header=True, delimiter="\t", train_file="train.csv",
                                                 test_file="test.csv", dev_file="dev.csv")
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

    def load(self, model_path=None):
        """
        Define the way to load the trained model
        """
        if not model_path:
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
        self.document_embeddings = TransformerDocumentEmbeddings(self.transformer_name)
        self.document_embeddings.tokenizer.model_max_length = 512

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

    def get_clusters(self, sentences, n_clusters=None, **kwargs):
        raise NotImplementedError


class Hdbscan(Clustering):

    def __init__(self, app_config: AppConfig):
        super(Hdbscan, self).__init__(app_config=app_config)

    def get_clusters(self, sentences, n_clusters=None, **kwargs):
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


class Agglomerative(Clustering):

    def __init__(self, app_config: AppConfig):
        super(Agglomerative, self).__init__(app_config=app_config)

    def get_clusters(self, sentences, n_clusters=None, **kwargs):
        n_clusters = int(len(sentences) / 2)
        agglomerative_model = AgglomerativeClustering(linkage="complete", affinity="cosine",
                                                      n_clusters=n_clusters, compute_distances=True)
        embeddings = self.get_embeddings(sentences=sentences)
        clusters = agglomerative_model.fit(embeddings)
        return clusters


class CustomAgglomerative(Clustering):

    def __init__(self, app_config: AppConfig):
        super(CustomAgglomerative, self).__init__(app_config=app_config)
        self.cosine_similarity = CosineSimilarity()

    def get_clusters(self, sentences, n_clusters=None, **kwargs):
        self.app_logger.info("Starting clustering")
        embeddings = self.get_embeddings(sentences=sentences)
        data_pairs = self._agglomerative_clustering(sentences=sentences, embeddings=embeddings,
                                                    docs_ids=kwargs["doc_ids"], sentences_ids=kwargs["sentences_ids"])
        self.app_logger.info(f"Finished clustering with {len(data_pairs)} of pairs")
        return data_pairs

    def _agglomerative_clustering(self, sentences: List[AnyStr], embeddings, docs_ids: List[AnyStr],
                                  sentences_ids: List[AnyStr]):
        sims = cosine_similarity(embeddings)
        mask = np.zeros(sims.shape, dtype=bool)
        # purge diagonal
        for k in range(len(sims)):
            mask[k, k] = True
        pairs = []
        simils = []

        # disable pairs from the same doc
        for doc in set(docs_ids):
            idx = [i for i in range(len(docs_ids)) if docs_ids[i] == doc]
            for i, j in combinations(idx, 2):
                mask[i, j] = mask[j, i] = True

        while False in mask:
            # get max sim
            a = np.ma.array(sims, mask=mask)
            idx = np.argmax(a)
            idxs = np.unravel_index(idx, sims.shape)
            row = idxs[0]
            if len(idxs) > 1:
                col = idxs[1]
                if docs_ids[row] == docs_ids[col]:
                    raise ValueError("Found matching pairs with same docids!")
                pairs.append((row, col))
                simils.append(sims[row, col])
                for x in (row, col):
                    mask[:, x] = True
                    mask[x, :] = True

        self.app_logger.info(f"Number of pairs: {len(pairs)}")
        self.app_logger.info(list(zip(pairs, simils)))
        count = 1
        data_pairs = []
        for pair, sim in zip(pairs, simils):
            if sim < 0.5:
                continue
            elif sim > 1.0:
                sim = 1.0
            data = {}
            idx_1 = pair[0]
            idx_2 = pair[1]
            sentence1 = sentences[idx_1]
            sentence2 = sentences[idx_2]
            vector1 = embeddings[idx_1]
            vector2 = embeddings[idx_2]
            doc_id1 = docs_ids[idx_1]
            doc_id2 = docs_ids[idx_2]
            sentence1_id = sentences_ids[idx_1]
            sentence2_id = sentences_ids[idx_2]
            data["sentence1"] = sentence1
            data["sentence2"] = sentence2
            data["embeddings1"] = vector1
            data["embeddings2"] = vector2
            data["doc_id1"] = doc_id1
            data["doc_id2"] = doc_id2
            data["sentence1_id"] = sentence1_id
            data["sentence2_id"] = sentence2_id
            data["cluster"] = count
            data["type"] = "similar"
            data["score"] = sim
            data_pairs.append(data)
            self.app_logger.info(f"Similar pair no {count} with similarity {sim}")
            self.app_logger.info(f"Sentence1: {sentence1}")
            self.app_logger.info(f"Sentence2: {sentence2}")
            self.app_logger.info(
                "=====================================================================================================")
            count += 1
        return data_pairs


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


# class EmbeddingAlignment(Model):
#
#     def __init__(self, app_config, model_name="alignment"):
#         super(EmbeddingAlignment, self).__init__(app_config=app_config, model_name=model_name)
#         self.data_folder: AnyStr = join(self.app_config.dataset_folder, model_name)
#         self.teacher_model_name = self.model_properties["teacher_model"]
#         self.student_model_name = self.model_properties["student_model"]
#         self.train_file = join(self.data_folder, "train.tsv")
#         self.dev_file = join(self.data_folder, "dev.tsv")
#         self.test_file = join(self.data_folder, "test.tsv")
#         self.data_limit = np.Inf
#         self.teacher_model = None
#         self.student_model = None
#
#     def train(self):
#         self.teacher_model = self.get_teacher_model()
#         self.student_model = self.get_student_model()
#         train_dataloader, train_loss = self.get_train_dataloader_and_loss()
#         dev_src, dev_trg = self.read_tsv(self.dev_file)
#         test_src, test_trg = self.read_tsv(self.test_file)
#         evaluators = self.get_evaluators(dev_src=dev_src, dev_trg=dev_trg, test_src=test_src, test_trg=test_trg)
#         self.student_model.fit(train_objectives=[(train_dataloader, train_loss)],
#                                evaluator=evaluation.SequentialEvaluator(evaluators,
#                                                                         main_score_function=lambda scores: np.mean(
#                                                                             scores)),
#                                epochs=self.model_properties["num_epochs"],
#                                warmup_steps=self.model_properties["num_warmup_steps"],
#                                evaluation_steps=self.model_properties["num_evaluation_steps"],
#                                output_path=self.app_config.student_path,
#                                save_best_model=True,
#                                optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
#                                )
#
#     def read_tsv(self, path):
#         src, trg = [], []
#         self.app_logger.info(f"Reading data from {path}.")
#         with open(path, 'r', encoding='utf8') as f_in:
#             for line in f_in:
#                 splits = line.strip().split('\t')
#                 if splits[0] != "" and splits[1] != "":
#                     src.append(splits[0])
#                     trg.append(splits[1])
#                 if len(src) >= self.data_limit:
#                     break
#         self.app_logger.info(f"Got {len(src)} data.")
#         return src, trg
#
#     def get_teacher_model(self):
#         self.app_logger.info("Load teacher model")
#         return SentenceTransformer(self.teacher_model_name)
#
#     def get_student_model(self):
#         self.app_logger.info("Create student model from scratch")
#         word_embedding_model = models.Transformer(self.student_model_name,
#                                                   max_seq_length=self.model_properties["max_seq_length"])
#         pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#         return SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
#
#     def get_train_dataloader_and_loss(self):
#         train_data = ParallelSentencesDataset(student_model=self.student_model, teacher_model=self.teacher_model,
#                                               batch_size=self.model_properties["inference_batch_size"],
#                                               use_embedding_cache=True)
#         train_data.load_data(self.train_file,
#                              max_sentences=min(self.model_properties["max_sentences_per_language"], self.data_limit),
#                              max_sentence_length=self.model_properties["train_max_sentence_len"])
#
#         train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.model_properties["train_batch_size"])
#         train_loss = losses.MSELoss(model=self.student_model)
#         return train_dataloader, train_loss
#
#     def get_evaluators(self, dev_src, dev_trg, test_src, test_trg):
#         evaluators = []
#
#         # dev evaluators
#         dev_mse = evaluation.MSEEvaluator(dev_src, dev_trg, name=os.path.basename(self.dev_file),
#                                           teacher_model=self.teacher_model,
#                                           batch_size=self.model_properties["inference_batch_size"])
#         # TranslationEvaluator computes the embeddings for all parallel sentences.
#         # It then check if the embedding of source[i] is the closest to target[i] out of all available target
#         sentences
#         dev_trans_acc = evaluation.TranslationEvaluator(dev_src, dev_trg, name=os.path.basename(self.dev_file),
#                                                         batch_size=self.model_properties["inference_batch_size"])
#         evaluators.extend([dev_mse, dev_trans_acc])
#
#         # test evaluators
#
#         test_mse = evaluation.MSEEvaluator(
#             test_src, test_trg, name="test_mse.csv", teacher_model=self.teacher_model,
#             batch_size=self.model_properties["inference_batch_size"])
#         test_trans_acc = evaluation.TranslationEvaluator(
#             test_src, test_trg, name="test_transl.csv", batch_size=self.model_properties["inference_batch_size"])
#
#         # test_evaluator = evaluation.EmbeddingSimilarityEvaluator(test_src, test_trg, data['scores'],
#         #                                                             batch_size=inference_batch_size, name="test",
#         #                                                             show_progress_bar=False)
#         evaluators.extend([test_mse, test_trans_acc])
#         return evaluators
