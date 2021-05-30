from itertools import combinations
from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ElectraConfig, ElectraTokenizer, BertTokenizer, BertForSequenceClassification, \
    ElectraForSequenceClassification

from utils.config import AppConfig


class Clustering:

    def __init__(self, model_name="electra"):
        self.app_config: AppConfig = app_config
        self.app_logger = logger
        if model_name == "electra":
            self.output_path = app_config.output_path
            tokenizer_path = join(self.output_path, "models", "tokenizer")
            self.config = ElectraConfig(vocab_size=100000,
                                        embedding_size=768,
                                        hidden_size=768,
                                        num_hidden_layers=12,
                                        num_attention_heads=4,
                                        intermediate_size=3072,
                                        hidden_act="gelu",
                                        hidden_dropout_prob=0.1,
                                        attention_probs_dropout_prob=0.1,
                                        max_position_embeddings=512,
                                        position_embedding_type="absolute")
            self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer_path)
            self.model = ElectraForSequenceClassification.from_pretrained(
                join(self.output_path, "models", "greek_electra", "discriminator"),
                config=self.config)
        else:
            self.tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
            self.model = BertForSequenceClassification.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

    def get_embeddings(self, sentences):
        sentence_embeddings = []
        all_input_ids = []
        for sentence in sentences:
            input_ids = self.tokenizer.encode(sentence)
            all_input_ids.append(input_ids)
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            output = self.model(input_ids, output_hidden_states=True)
            embeddings = output.hidden_states[-1].squeeze(0)[0].detach().numpy()
            sentence_embeddings.append(embeddings)
        embeddings = np.asarray(sentence_embeddings)
        return embeddings

    def get_clusters(self, sentences):
        n_clusters = int(len(sentences) / 2)
        agglomerative_model = AgglomerativeClustering(linkage="complete", affinity="cosine",
                                                      n_clusters=n_clusters, compute_distances=True)
        embeddings = self.get_embeddings(sentences=sentences)
        clusters = agglomerative_model.fit(embeddings)
        return clusters

    def get_manual_clusters(self, sentences, embeddings):
        sims = cosine_similarity(embeddings)
        # purge diagonal
        for k in range(len(sims)):
            sims[k, k] = -1
        pairs = []
        simils = []
        m = np.zeros(sims.shape, dtype=bool)
        while False in m:
            # get max sim
            a = np.ma.array(sims, mask=m)
            idx = np.argmax(a)
            idxs = np.unravel_index(idx, sims.shape)
            row = idxs[0]
            if len(idxs) > 1:
                col = idxs[1]
                pairs.append((row, col))
                simils.append(sims[row, col])
                for x in (row, col):
                    m[:, x] = True
                    m[x, :] = True

        logger.info(f"Number of pairs: {len(pairs)}")
        logger.info(list(zip(pairs, simils)))
        count = 1
        for pair, sim in zip(pairs, simils):
            idx_1 = pair[0]
            idx_2 = pair[1]
            sentence1 = sentences[idx_1]
            sentence2 = sentences[idx_2]
            vector1 = embeddings[idx_1]
            vector2 = embeddings[idx_2]
            logger.info(f"Similar pair no {count} with similarity {sim}")
            logger.info(f"Sentence1: {sentence1}")
            logger.info(f"Sentence2: {sentence2}")
            logger.info(
                "===========================================================================================================")
            count += 1


def get_sentences():
    path_to_csv = join(app_config.output_path, "test", "claims.csv")
    df = pd.read_csv(path_to_csv, sep="\t", header=0, index_col=None)
    sentences = list(df["claims"])
    sentences = list(set(sentences))
    return sentences


def manual_clustering(sentences, model_name="electra"):
    clustering_model = Clustering(model_name=model_name)
    embeddings = clustering_model.get_embeddings(sentences=sentences)
    clustering_model.get_manual_clusters(sentences=sentences, embeddings=embeddings)


def clustering(sentences, model_name="electra"):
    clustering_model = Clustering(model_name=model_name)
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
    model_name = "aueb"
    sentences = get_sentences()
    clustering(sentences=sentences)
    # manual_clustering(sentences=sentences, model_name=model_name)


if __name__ == '__main__':
    app_config = AppConfig()
    logger = app_config.app_logger
    main()