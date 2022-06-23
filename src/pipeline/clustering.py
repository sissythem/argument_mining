import os
import logging
import json
import math
import umap
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from os.path import join
from src.utils import MODEL_VERSION

from transformers import AutoModel, AutoTokenizer


class UnsupervisedModel:
    """
    Abstract class representing an unsupervised model
    """

    def __init__(self, config, embedder=None):

        self.model_name = config.get("model_name", "nlpaueb/bert-base-greek-uncased-v1")
        if embedder is None:
            logging.info(f"Using {self.model_name} for clustering embedding generation.")
            self.model = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            # preloaded embedder
            logging.info(f"Using preloaded text embedder for clustering embedding generation.")
            self.model, self.tokenizer = embedder.model, embedder.tokenizer

        self.n_clusters = int(config.get("n_clusters", 10))
        self.device = config.get("device_name", "cuda")


class Clustering(UnsupervisedModel):

    def __init__(self, config, embedder=None):
        super(Clustering, self).__init__(config, embedder)

    def get_embeddings(self, sentences):
        sentence_embeddings = []
        for sentence in sentences:
            tokenized = self.tokenizer(sentence,
                                       padding="max_length",
                                       max_length=128,
                                       truncation=True,
                                       return_tensors="pt")
            # get CLS output
            embeddings = self.model(**tokenized)[1]
            embeddings = embeddings.to("cpu").detach().numpy()
            sentence_embeddings.append(embeddings)
        embeddings = np.asarray(sentence_embeddings)
        if len(embeddings.shape) > 2:
            embeddings = embeddings.squeeze()
        return embeddings

    def get_cluster_assignments(self, sentences, n_clusters=None, **kwargs):
        raise NotImplementedError


class Hdbscan(Clustering):

    def __init__(self, config, embedder=None):
        super(Hdbscan, self).__init__(config, embedder=embedder)

    def compute_pairs(self, sentences, **kwargs):
        assignments = self.get_cluster_assignments(sentences)
        adu_ids = kwargs["sentence_ids"]
        doc_ids = kwargs["doc_ids"]

        # get_content_per_cluster
        clusters_dict = defaultdict(list)

        # group sentences to clusters
        for idx, cluster in enumerate(assignments):
            if cluster is None:
                continue
            clusters_dict[cluster].append(idx)

        pairs = []
        # from
        for cluster, idxs in clusters_dict.items():
            n_pairs = 0
            cluster_combinations = list(combinations(idxs, r=2))
            for (i, j) in cluster_combinations:
                # make a pair from each combo in the cluster
                pair = make_crossdoc_relation_dict(i, j, 0.0, cluster, adu_ids, doc_ids, sentences)
                # skip intra-doc
                if pair['doc1_id'] == pair['doc2_id']:
                    continue
                pairs.append(pair)
                n_pairs += 1
            logging.info(f"Got {n_pairs} pairs in cluster {cluster}")
        return pairs

    def get_cluster_assignments(self, sentences, n_clusters=None, **kwargs):
        if not sentences:
            return []
        if n_clusters is None:
            n_clusters = self.n_clusters
        # we should want 'eom' that tends to make 2 clusters, e.g. one for
        # related and one for non-related
        cluster_selection_method = kwargs.get('cluster_selection_method', 'eom')
        prob_threshold = kwargs.get('probability_threshold', 0.0)
        do_noise_filtering = kwargs.get('filter_noise', True)
        embeddings = self.get_embeddings(sentences=sentences)
        logging.debug(f"Sentence embeddings shape: {embeddings.shape}")
        cluster_size = max(math.ceil(len(sentences) // n_clusters), 1)
        # reduce document dimensionality

        try:
            umap_embeddings = umap.UMAP(metric='cosine',
                                        n_neighbors=max(15, min(15, len(sentences) - 1))).fit_transform(
                embeddings)
            # clustering
            try:
                import hdbscan
                clusters = hdbscan.HDBSCAN(min_cluster_size=cluster_size, metric='euclidean',
                                           cluster_selection_method=cluster_selection_method).fit(umap_embeddings)
                assignments = clusters.labels_
                # apply prob. threshold
                under_thr = np.where(0 < (clusters.probabilities_ < prob_threshold))[0]
                logging.debug(f"Dropping {len(under_thr)} / {len(assignments)} via prob. threshold {prob_threshold}")
                noisy = np.where(assignments < 0)[0] if do_noise_filtering else []
                logging.debug(f"Dropping {len(noisy)} / {len(assignments)} noisy samples")
                assignments = assignments.tolist()
                drop = [x for x in np.concatenate((noisy, under_thr))]
                assignments = [x if i not in drop else None for i, x in enumerate(assignments)]
                logging.debug(f"HDBSCAN clustering resulted in {len(set(assignments))} clusters.")
            except ValueError as ve:
                logging.error(f"HSBSCAN failed: {ve} -- falling back to KMeans")
                assignments = self.kmeans_fallback(n_clusters, umap_embeddings)
        except Exception as ex:
            print(f"Failed to produce topics: {ex}")
            assignments = self.kmeans_fallback(n_clusters, embeddings)
        return assignments

    def kmeans_fallback(self, n_clusters, vectors):
        # def get_pairwise_format_from_clusters(self, clusters_dict):
        km = KMeans(n_clusters=n_clusters).fit(vectors)
        # centers, assignments = km.cluster_centers_, km.predict(umap_embeddings)
        assignments = km.predict(vectors)
        return assignments


class Agglomerative(Clustering):

    def get_cluster_assignments(self, sentences, n_clusters=None, **kwargs):
        n_clusters = int(len(sentences) / 2)
        agglomerative_model = AgglomerativeClustering(linkage="complete", affinity="cosine",
                                                      n_clusters=n_clusters, compute_distances=True)
        embeddings = self.get_embeddings(sentences=sentences)
        clusters = agglomerative_model.fit(embeddings)
        return clusters


class PairwiseSimilarity(Clustering):

    def compute_pairs(self, sentences, **kwargs):
        if len(sentences) <= 1:
            return []
        logging.info(f"Mapping {len(sentences)} input sentences for clustering")
        embeddings = self.get_embeddings(sentences=sentences)
        logging.info(f"Obtained embeddings of {embeddings.shape} for clustering")
        threshold = kwargs.get("threshold", 0.9)
        data_pairs = self._agglomerative_clustering(sentences=sentences, embeddings=embeddings,
                                                    doc_ids=kwargs["doc_ids"], sentences_ids=kwargs["sentence_ids"],
                                                    sim_threshold=threshold)
        logging.info(f"Completed clustering -- generated {len(data_pairs)} pairs")
        return data_pairs

    def _agglomerative_clustering(self, sentences, embeddings, doc_ids, sentences_ids, sim_threshold=0.9,
                                  limit_adu_pair_membership=False):
        """
        Compute agglomerative clustering
        :param sentences: Text sentences
        :param embeddings: Sentence vectors
        :param doc_ids: Corresponding document ids
        :param sentences_ids: Corresponding document ids
        :param sim_threshold: Threshold for sentence similarity
        :param limit_adu_pair_membership: If true, restrict an adu to at most one similarity pair
        :return:
        """
        logging.info(f"Computing agglomerative clustering with a sim. threshold {sim_threshold}")
        logging.info(f"Calculating cosine similarity between row vector collection of shape: {embeddings.shape}")
        sims = cosine_similarity(embeddings)

        mask = np.zeros(sims.shape, dtype=bool)
        # purge diagonal
        for k in range(len(sims)):
            mask[k, k] = True
        pairs, simils = [], []

        # disable pairs from the same doc
        for doc in set(doc_ids):
            idx = [i for i in range(len(doc_ids)) if doc_ids[i] == doc]
            for i, j in combinations(idx, 2):
                mask[i, j] = mask[j, i] = True

        mask[sims < sim_threshold] = True
        if limit_adu_pair_membership:
            while False in mask:
                # get max sim
                a = np.ma.array(sims, mask=mask)
                idx = np.argmax(a)
                idxs = np.unravel_index(idx, sims.shape)
                row = idxs[0]
                if len(idxs) > 1:
                    col = idxs[1]
                    if doc_ids[row] == doc_ids[col]:
                        raise ValueError("Found matching pairs with same docids!")
                    pairs.append((row, col))
                    simils.append(sims[row, col])
                    # disable future inclusion of these adus in similarity pairs
                    for x in (row, col):
                        mask[:, x] = True
                        mask[x, :] = True
        else:
            idx = np.where(~mask)
            simils = sims[idx]
            pairs = list(zip(*idx))

        logging.debug(f"Number of pairs: {len(pairs)}")
        logging.debug(list(zip(pairs, simils)))
        data_pairs = []
        for pair, sim in zip(pairs, simils):
            logging.debug(f"Similar pair no {len(data_pairs)}/{len(pairs)} with similarity {sim}")
            if sim < sim_threshold:
                continue
            elif sim > 1.0:
                sim = 1.0
            i, j = pair
            doc1, doc2 = doc_ids[i], doc_ids[j]
            if doc1 == doc2:
                continue
            dat = make_crossdoc_relation_dict(i, j, sim, len(data_pairs) + 1, sentences_ids, doc_ids, sentences)
            data_pairs.append(dat)
            logging.debug(f"Sentence1: {sentences[i]}")
            logging.debug(f"Sentence2: {sentences[j]}")
            logging.debug(
                "=====================================================================================================")
        return data_pairs


def run_clustering(documents, config, embedder=None):
    """
    Cross-document relations pipeline

    Args
        | documents (list): list of json documents extracted from the SocialObservatory Elasticsearch & updated by
        the Argument Mining pipeline
        | embedder (tuple):
        | document_ids (list): list of the ids of the valid documents
    """
    model = config.pop('clustering')
    if model == "similarity":
        clustering_model = PairwiseSimilarity(config, embedder=embedder)
    else:
        raise ValueError(f"Unknown clustering model {model}")

    logging.info("Fetching relevant ADUs for cross-document clustering -- only claims.")
    adus, adu_ids, doc_ids = [], [], []
    for i, document in enumerate(documents):
        for adu in document["annotations"]["ADUs"]:
            if adu["type"] == "claim":
                adus.append(adu["segment"])
                adu_ids.append(adu["id"])
                doc_ids.append(document["id"])

    data_pairs = clustering_model.compute_pairs(adus, sentence_ids=adu_ids, doc_ids=doc_ids, **config)
    return data_pairs
    #
    # relations = []
    # for pair in data_pairs:
    #     relation_id = f"{pair['doc1_id']};{pair['doc2_id']};{pair['sentence1_id']};{pair['sentence2_id']}"
    #     # logging.debug(f"Saving cross document relation with id:{relation_id}")
    #     relations.append({
    #         "id": relation_id,
    #         "cluster": pair['cluster_id'],
    #         "source": pair["sentence1_id"],
    #         "source_doc": pair["doc1_id"],
    #         "source_segment": pair["sentence1"],
    #         "target": pair["sentence2_id"],
    #         "target_doc": pair["doc2_id"],
    #         "target_segment": pair["sentence2"],
    #         "type": pair["type"],
    #         "score": pair["score"]}
    #     )
    # return relations


def make_crossdoc_relation_dict(src_idx, target_idx, score, cluster_id, adu_ids, doc_ids, sentences):
    i, j = src_idx, target_idx
    d1, d2 = doc_ids[i], doc_ids[j]
    s1, s2 = adu_ids[i], adu_ids[j]
    id_ = f"{d1};{d2};{s1};{s2}"
    pair = {
        "id": id_,
        "cluster_id": str(cluster_id),
        "sentence1_id": s1,
        "doc1_id": d1,
        "sentence2_id": s2,
        "doc2_id": d2,
        "sentence1": sentences[i],
        "sentence2": sentences[j],
        "type": "similar",
        "score": str(score),
        "model_version": MODEL_VERSION
    }
    return pair
