from typing import List

import umap
from sentence_transformers import SentenceTransformer


def train(documents: List[str]):
    model = SentenceTransformer("nlpaueb/bert-base-greek-uncased-v1")
    embeddings = model.encode(documents, show_progress_bar=True)

    # reduce document dimensionality
    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=5,
                                metric='cosine').fit_transform(embeddings)

    # clustering

