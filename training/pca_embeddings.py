import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from sklearn.manifold import TSNE

csv_file = "../output/test/claims.csv"
document_embeddings = TransformerDocumentEmbeddings("nlpaueb/bert-base-greek-uncased-v1", fine_tune=False)
df = pd.read_csv(csv_file, sep="\t", header=0, index_col=None)
sentences = df["claims"]

sentence_embeddings = []
for sentence in sentences:
    flair_sentence = Sentence(sentence)
    document_embeddings.embed(flair_sentence)
    embeddings = flair_sentence.get_embedding()
    embeddings = embeddings.to("cpu")
    embeddings = embeddings.numpy()
    sentence_embeddings.append(embeddings)
embeddings = np.asarray(sentence_embeddings)
print(embeddings.shape)

tnse_embeddings = TSNE(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x=tnse_embeddings[:, 0], y=tnse_embeddings[:, 1],
    # hue="y",
    palette=sns.color_palette("hls", 10),
    # data=embeddings,
    legend="full",
    alpha=0.3
)
plt.show()

pca_embeddings = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x=pca_embeddings[:, 0], y=pca_embeddings[:, 1],
    # hue="y",
    palette=sns.color_palette("hls", 10),
    # data=embeddings,
    legend="full",
    alpha=0.3
)
plt.show()
