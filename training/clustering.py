from os.path import join
import numpy as np
import pandas as pd
import hdbscan
import umap
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from training.models import UnsupervisedModel, RelationsModel
from utils.config import AppConfig


class Clustering(UnsupervisedModel):

    def __init__(self, app_config: AppConfig):
        super(Clustering, self).__init__(app_config=app_config)
        self.sim_model = RelationsModel(app_config=app_config, model_name="sim")
        self.sim_model.load()
        self.path_to_transformer_model = join(self.app_config.sim_base_path, self.model_file)
        # init embeddings from your trained LM
        self.document_embeddings = TransformerDocumentEmbeddings(self.path_to_transformer_model)

    def run_clustering(self):
        n_clusters = 10
        path_to_csv = join(self.resources_path, "data", "train_sim.csv")
        df = pd.read_csv(path_to_csv, sep="\t")
        sentences = list(df[0])
        clusters = self.get_clusters(n_clusters=n_clusters, sentences=sentences)
        self.get_content_per_cluster(clusters=clusters, sentences=sentences)

    def get_clusters(self, n_clusters, sentences):
        try:
            # model = SentenceTransformer("distiluse-base-multilingual-cased-v2").to(self.device_name)
            # embeddings = model.encode(sentences, show_progress_bar=True)
            sentence_embeddings = []
            for sentence in sentences:
                flair_sentence = Sentence(sentence)
                # tokens = self.tokenizer.encode(sentence)
                # input_ids = torch.tensor(tokens).unsqueeze(0)
                # outputs = self.bert_model(input_ids)
                # embeddings = outputs[1][-1].detach().numpy()
                embeddings = self.document_embeddings.embed(flair_sentence)
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

    def get_content_per_cluster(self, clusters, sentences, print_clusters=True):
        clusters_dict = {}
        for idx, cluster in enumerate(clusters.labels_):
            if cluster not in clusters_dict.keys():
                clusters_dict[cluster] = []
            sentence = sentences[idx]
            clusters_dict[cluster].append(sentence)
        if print_clusters:
            self.print_clusters(cluster_lists=clusters_dict)
        return clusters_dict

    def print_clusters(self, cluster_lists):
        for idx, cluster_list in cluster_lists.items():
            self.app_logger.debug(f"Content of Cluster {idx}")
            for sentence in cluster_list:
                self.app_logger.debug(f"Sentence content: {sentence}")


if __name__ == '__main__':
    conf_app = AppConfig()
    clustering = Clustering(app_config=conf_app)
    clustering.run_clustering()
