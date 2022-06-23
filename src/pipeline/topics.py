from typing import List, Type, Union, AnyStr, Dict
from os.path import join
from src import utils
import pandas as pd
import warnings
import spacy
import logging
import nltk
import functools

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.pipeline.clustering import Hdbscan


class TFIDFClusterDescriptor:
    """
    Class to extract representations from topic clusters
    """


class TopicModel(Hdbscan):
    """
    Class for topic modeling
    """

    def __init__(self, config, embedder=None):
        super(TopicModel, self).__init__(config, embedder=embedder)
        self.nlp_model = spacy.load("el_core_news_sm")
        self.greek_stopwords = utils.get_greek_stopwords()

    @functools.lru_cache(maxsize=5000)
    def analyze_token(self, word):
        return self.nlp_model(word)[0]

    def pos_tag_word(self, word):
        return self.analyze_token(word).pos_

    def is_valid_topic_keyword(self, token):
        # noun and some alphanumeric
        return token.pos_ in ("NOUN",) and any(c.isalpha() for c in token.text)

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
            sentences = [" ".join([str(x) for x in s]) for s in sentences]
        else:
            sentences = content

        # n_clusters = max(2, int(len(sentences) / 2))
        # at most <num_sentences> clusters, at least 2
        n_clusters = max(2, min(self.n_clusters, len(sentences) // 2))
        assignments = self.get_cluster_assignments(sentences=sentences, n_clusters=n_clusters)
        if assignments is None:
            return []
        docs_df = pd.DataFrame(sentences, columns=["Sentence"])
        docs_df['Topic'] = assignments
        # docs_df['Topic'] = clusters.labels_
        docs_df['sentence_ID'] = range(len(docs_df))
        docs_df = docs_df[~docs_df["Topic"].isna()]
        docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Sentence': ' '.join})
        try:
            warnings.filterwarnings("ignore", category=UserWarning)
            tf_idf, count = self._c_tf_idf(docs_per_topic.Sentence.values, m=len(sentences))
        except Exception as ex:
            logging.error(f"Could not extract topics from content: {content}, due to: {ex}")
            return []
        top_n_words = self._extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=10)
        topic_sizes = self._extract_topic_sizes(docs_df).head(2)
        topic_ids = topic_sizes["Topic"]
        for topic in topic_ids:
            for word_score_tuple in top_n_words[topic]:
                if word_score_tuple[0].isdigit():
                    continue
                topics.append(word_score_tuple[0])
        return topics

    def _c_tf_idf(self, sentences, m, ngram_range=(1, 1)):
        docs = [self.nlp_model(sent) for sent in sentences]
        # lemmatize, drop invalid POS
        texts = [" ".join([tok.lemma_ for tok in doc if self.is_valid_topic_keyword(tok)]) for doc in docs]
        # vectorizer_class = CountVectorizer
        vectorizer_class = TfidfVectorizer
        # get original vocabulary
        count = vectorizer_class(ngram_range=ngram_range, stop_words=self.greek_stopwords).fit(texts)
        weights = count.transform(sentences).toarray().T
        # t = count.transform(sentences).toarray()
        # w = t.sum(axis=1)
        # tf = np.divide(t.T, w)
        # sum_t = t.sum(axis=0)
        # idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        # tf_idf = np.multiply(tf, idf)
        return weights, count

    @staticmethod
    def _extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names_out()
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
