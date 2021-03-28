import logging
from itertools import combinations
from os.path import join

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from training.models import KmeansClustering, Agglomerative, Hdbscan
from utils.config import AppConfig

app_config = AppConfig()
logger = app_config.app_logger
logger.setLevel(logging.DEBUG)
clustering_properties = app_config.properties["clustering"]
algorithm = clustering_properties["algorithm"]
if algorithm == "hdbscan":
    clustering_model = Hdbscan(app_config=app_config)
elif algorithm == "kmeans":
    clustering_model = KmeansClustering(app_config=app_config)
else:
    clustering_model = Agglomerative(app_config=app_config)

path_to_csv = join(app_config.output_path, "test", "claims.csv")
df = pd.read_csv(path_to_csv, sep="\t", header=0, index_col=None)
sentences = list(df["claims"])
embeddings = clustering_model.get_embeddings(sentences=sentences)
clusters = clustering_model.get_clusters(sentences=sentences)
clusters = clusters.labels_
clusters_dict = {}
for idx, cluster in enumerate(clusters):
    if cluster not in clusters_dict.keys():
        clusters_dict[cluster] = []
    sentence = sentences[idx]
    embedding = embeddings[idx]
    clusters_dict[cluster].append((sentence, embedding))

cluster_sims_dict = {}
for cluster, sentences in clusters_dict.items():
    logger.info(f"finding cosine similarities for cluster {cluster}")
    cluster_combinations = list(combinations(sentences, r=2))
    cluster_sims = []
    for pair_combination in cluster_combinations:
        sentence1, embedding1 = pair_combination[0]
        sentence2, embedding2 = pair_combination[1]
        embedding1, embedding2 = embedding1.reshape(1, -1), embedding2.reshape(1, -1)
        sim = list(list(cosine_similarity(embedding1, embedding2))[0])[0]
        logger.debug(f"For sentence {sentence1} and sentence {sentence2} cosine similarity is: {sim}")
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
