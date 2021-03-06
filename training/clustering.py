from itertools import combinations
from os.path import join

import hdbscan
import numpy as np
import pandas as pd
import umap
from flair.data import Sentence

from training.models import UnsupervisedModel, RelationsModel
from utils.config import AppConfig


class Clustering(UnsupervisedModel):

    def __init__(self, app_config: AppConfig):
        super(Clustering, self).__init__(app_config=app_config)
        self.sim_model = RelationsModel(app_config=app_config, model_name="sim")
        self.sim_model.load()
        self.document_embeddings = self.sim_model.model.document_embeddings

    def run_clustering(self):
        n_clusters = 10
        path_to_csv = join(self.resources_path, "data", "UKP_ASPECT.tsv")
        df = pd.read_csv(path_to_csv, sep="\t", header=0, index_col=None)
        sentences = list(df["sentence_1"]) + list(df["sentence_2"])
        sentences = list(set(sentences))
        clusters = self.get_clusters(n_clusters=n_clusters, sentences=sentences)
        self.get_content_per_cluster(clusters=clusters, sentences=sentences, df=df)

    def get_clusters(self, n_clusters, sentences):
        try:
            sentence_embeddings = []
            for sentence in sentences:
                flair_sentence = Sentence(sentence)
                self.document_embeddings.embed(flair_sentence)
                embeddings = flair_sentence.get_embedding()
                embeddings = embeddings.to("cpu")
                embeddings = embeddings.numpy()
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

    def get_content_per_cluster(self, clusters, sentences, df, print_clusters=True):
        clusters_dict = {}
        for idx, cluster in enumerate(clusters.labels_):
            if cluster not in clusters_dict.keys():
                clusters_dict[cluster] = []
            sentence = sentences[idx]
            clusters_dict[cluster].append(sentence)
        if print_clusters:
            self.print_clusters(cluster_lists=clusters_dict, df=df)
        return clusters_dict

    def print_clusters(self, cluster_lists, df):
        for idx, cluster_list in cluster_lists.items():
            self.app_logger.debug(f"Content of Cluster {idx}")
            for sentence in cluster_list:
                self.app_logger.debug(f"Sentence content: {sentence}")
        sim_clusters = {}
        for cluster, sentences in cluster_lists.items():
            sim_clusters[cluster] = {"DTORCD": 0, "NS": 0, "SS": 0, "HS": 0}
            cluster_combinations = list(combinations(sentences, r=2))
            for pair_combination in cluster_combinations:
                sentence1 = pair_combination[0]
                sentence2 = pair_combination[1]
                sentence1_row = df[(df == sentence1).any(axis=1)]
                for index, row in sentence1_row.iterrows():
                    sent1 = row["sentence_1"]
                    sent2 = row["sentence_2"]
                    label = row["label"]
                    if sentence2 == sent2 or sentence2 == sent1:
                        sim_clusters[cluster][label] += 1
        for cluster, labels_dict in sim_clusters.items():
            self.app_logger.info(f"For cluster {cluster}: number of sentences for each label:")
            self.app_logger.info(f"{labels_dict}")
