import logging
from itertools import combinations
from os.path import join

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from training.models import KmeansClustering, Agglomerative, Hdbscan, BirchClustering, OpticsClustering
from utils.config import AppConfig
from utils import utils


def get_logger():
    app_logger = app_config.app_logger
    app_logger.setLevel(logging.DEBUG)
    return app_logger


def get_clustering_model():
    clustering_properties = app_config.properties["clustering"]
    algorithm = clustering_properties["algorithm"]
    if algorithm == "hdbscan":
        clustering_model = Hdbscan(app_config=app_config)
    elif algorithm == "kmeans":
        clustering_model = KmeansClustering(app_config=app_config)
    elif algorithm == "agglomerative":
        clustering_model = Agglomerative(app_config=app_config)
    elif algorithm == "birch":
        clustering_model = BirchClustering(app_config=app_config)
    else:
        clustering_model = OpticsClustering(app_config=app_config)
    return clustering_model


def get_sentences():
    path_to_csv = join(app_config.output_path, "test", "claims.csv")
    df = pd.read_csv(path_to_csv, sep="\t", header=0, index_col=None)
    sentences = list(df["claims"])
    return sentences


def get_topic_words(num_words, topic, vocab):
    topic = enumerate(topic)
    topic = list(sorted(topic, key=lambda x: x[1], reverse=True))
    topic = topic[:num_words]
    topic_words_weights = []
    for idx, weight in topic:
        word = vocab[idx]
        topic_words_weights.append((word, weight))
    logger.info(topic_words_weights)


def lda(sentences):
    greek_stopwords = utils.get_greek_stopwords()
    count_vec = TfidfVectorizer(stop_words=greek_stopwords, max_df=0.1)
    output = count_vec.fit_transform(sentences)
    lda_model = LatentDirichletAllocation(n_components=150)
    result = lda_model.fit_transform(output)
    vocab = count_vec.vocabulary_
    vocab2 = {v: k for (k, v) in vocab.items()}
    topics = list(lda_model.components_)
    for i in range(len(topics)):
        topic = topics[i]
        get_topic_words(num_words=5, topic=topic, vocab=vocab2)
    for idx, sentence in enumerate(sentences):
        logger.info(sentence)
        res = result[idx]
        res = enumerate(res)
        res = list(sorted(res, key=lambda x: x[1], reverse=True))[:5]
        for topic_idx, topic_weight in res:
            logger.info(f"Topic weight: {topic_weight}")
            topic = topics[topic_idx]
            get_topic_words(num_words=5, topic=topic, vocab=vocab2)


def clustering(sentences):
    clustering_model = get_clustering_model()
    embeddings = clustering_model.get_embeddings(sentences=sentences)
    clusters = clustering_model.get_clusters(sentences=sentences)
    clusters = clusters.labels_
    clusters_dict = {}
    result_df = pd.DataFrame(columns=["sentence", "x", "y", "cluster"])
    counter = 0
    for idx, cluster in enumerate(clusters):
        if cluster not in clusters_dict.keys():
            clusters_dict[cluster] = []
        sentence = sentences[idx]
        embedding = embeddings[idx]
        clusters_dict[cluster].append((sentence, embedding))

    cluster_sims_dict = {}
    for cluster, sentences in clusters_dict.items():
        logger.info(f"Number of instances in the cluster {cluster}: {len(sentences)}")
        logger.info(f"finding cosine similarities for cluster {cluster}")
        for sentence in sentences:
            logger.info(f"Sentence: {sentence[0]}")
        cluster_combinations = list(combinations(sentences, r=2))
        cluster_sims = []
        for pair_combination in cluster_combinations:
            sentence1, embedding1 = pair_combination[0]
            sentence2, embedding2 = pair_combination[1]
            embedding1, embedding2 = embedding1.reshape(1, -1), embedding2.reshape(1, -1)
            sim = list(list(cosine_similarity(embedding1, embedding2))[0])[0]
            # logger.debug(f"For sentence {sentence1} and sentence {sentence2} cosine similarity is: {sim}")
            cluster_sims.append(sim)
        cluster_sims.sort()
        cluster_sims_dict[cluster] = cluster_sims
    logger.info(f"Number of clusters: {len(cluster_sims_dict.keys())}")
    for cluster, sims in cluster_sims_dict.items():
        logger.debug(f"Similarities for cluster {cluster}:")
        logger.debug(sims)
        if sims:
            avg_per_cluster = sum(sims) / len(sims)
            min_per_cluster = min(sims)
            max_per_cluster = max(sims)
            logger.info(f"Average similarity for cluster {cluster} is {avg_per_cluster}")
            logger.info(f"Min similarity in cluster {cluster} is {min_per_cluster}")
            logger.info(f"Max similarity in cluster {cluster} is {max_per_cluster}")


def main():
    sentences = get_sentences()
    clustering(sentences=sentences)


if __name__ == '__main__':
    app_config = AppConfig()
    logger = get_logger()
    main()
