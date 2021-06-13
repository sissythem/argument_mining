from os import getcwd
from os.path import join

import pandas as pd
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

path_to_csv = join(getcwd(), "output", "test", "claims.csv")
df = pd.read_csv(path_to_csv, sep="\t", header=0, index_col=None)
sentences = list(set(list(df["claims"])))
document_embeddings = TransformerDocumentEmbeddings("bert-base-multilingual-uncased", fine_tune=False)
sentence_embeddings = []
for sentence in sentences:
    flair_sentence = Sentence(sentence)
    document_embeddings.embed(flair_sentence)
    embeddings = flair_sentence.get_embedding()
    embeddings = embeddings.to("cpu")
    embeddings = embeddings.numpy()
    sentence_embeddings.append(embeddings)
data_vecs = np.asarray(sentence_embeddings)

# data_vecs = np.random.rand(100, 50)
sims = cosine_similarity(data_vecs)

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

print(len(pairs), "pairs")
print(list(zip(pairs, simils)))
count = 1
for pair, sim in zip(pairs, simils):
    idx_1 = pair[0]
    idx_2 = pair[1]
    sentence1 = sentences[idx_1]
    sentence2 = sentences[idx_2]
    vector1 = data_vecs[idx_1]
    vector2 = data_vecs[idx_2]
    print(f"Similar pair no {count} with similarity {sim}")
    print(f"Sentence1: {sentence1}")
    print(f"Sentence2: {sentence2}")
    print("===========================================================================================================")
    count += 1
