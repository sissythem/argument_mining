import re
from string import punctuation
from typing import Union, List, Dict, Tuple, AnyStr, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

try:
    from ellogon import tokeniser
except (Exception, BaseException):
    print("Missing ellogon")
    pass


# ******************************** Generic functions *****************************************
def get_greek_stopwords():
    return tokeniser.stop_words()


def tokenize(text, punct=True):
    return list(tokeniser.tokenise_no_punc(text)) if not punct else list(tokeniser.tokenise(text))


def tokenize_with_spans(text):
    return tokeniser.tokenise_spans(text)


def get_punctuation_symbols() -> Set[AnyStr]:
    """
    Function to get a set with punctuation symbols

    Returns
        | set: a set with all the punctuation symbols
    """
    punc = list(set(punctuation))
    punc += ["´", "«" "»"]
    punc = set(punc)
    return punc


def join_sentences(tokenized_sentences: List[Tuple[AnyStr]]) -> List[AnyStr]:
    """
    Function to create a correct string (punctuation in the correct position - correct spaces)

    Args
        | tokenized_sentences (list): a list of sentences. Each sentence is a tuple with the respective tokens

    Returns
        | list: a list of strings (i.e. the sentences)
    """
    sentences = []
    for sentence in tokenized_sentences:
        sentence = join_sentence(sentence=sentence)
        sentences.append(sentence)
    return sentences


def join_sentence(sentence: Union[List[AnyStr], Tuple[AnyStr]]) -> AnyStr:
    punc = get_punctuation_symbols()
    sentence = "".join(w if set(w) <= punc else f" {w}" for w in sentence).lstrip()
    sentence = sentence.replace("( ", " (")
    sentence = sentence.replace("« ", " «")
    sentence = sentence.replace(" »", "» ")
    sentence = sentence.replace('" ', ' "')
    sentence = sentence.replace("\n", " ")
    sentence = re.sub(" +", " ", sentence)
    return sentence


def replace_multiple_spaces_with_single_space(text):
    return re.sub(' +', ' ', text)


def is_empty(obj: Union[List, Tuple, str, Dict]):
    if obj is None:
        return True
    if type(obj) == str:
        return obj == ""
    elif type(obj) == list:
        return obj == []
    elif type(obj) == dict:
        return obj == {}
    elif type(obj) == tuple:
        return obj == ()
    return False


def name_exceeds_bytes(self, name):
    """
    Checks if a string exceeds the 255 bytes

    Args
        name (str): the name of a file

    Returns
        bool: True/False
    """
    return self._utf8len(name) >= 255


def _utf8len(s):
    """
    Find the length of the encoded filename

    Args
        s (str): the filename to encode

    Returns
        int: the length of the encoded filename
    """
    return len(s.encode('utf-8'))


# ************************************ Preprocessing *****************************************
def is_old_annotation(attributes):
    for attribute in attributes:
        name = attribute["name"]
        if name == "premise_type" or name == "premise" or name == "claim":
            return True
    return False


def collect_relation_pairs(parents, children, relation_pairs):
    new_relation_pairs = []
    count_relations = 0
    for p_id, p_text in parents.items():
        for c_id, c_text in children.items():
            key = (c_id, p_id)
            if key in relation_pairs.keys():
                count_relations += 1
            relation = relation_pairs.get(key, "other")
            new_relation_pairs.append((c_text, p_text, relation))
    return new_relation_pairs


def bio_tagging(sentences, label, other_label="O"):
    new_sentences, sentences_labels = [], []
    for sentence in sentences:
        sentence_labels = []
        tokens = []
        for token in sentence:
            if token:
                tokens.append(token)
                if label == other_label:
                    sentence_labels.append(other_label)
                else:
                    if sentence.index(token) == 0:
                        sentence_labels.append(f"B-{label}")
                    else:
                        sentence_labels.append(f"I-{label}")
        new_sentences.append(tokens)
        sentences_labels.append(sentence_labels)
    return new_sentences, sentences_labels


def bio_tag_lbl_per_token(tokens_labels_tuple, other_label="O"):
    previous_label = None
    tokens, labels = [], []
    for token, label in tokens_labels_tuple:
        if token is None or token == "":
            continue
        tokens.append(token)
        if label == other_label:
            labels.append(label)
        else:
            if previous_label == label:
                labels.append(f"I-{label}")
            else:
                labels.append(f"B-{label}")
        previous_label = label
    tokens = tuple(tokens)
    labels = tuple(labels)
    return tokens, labels


# **************************** Segment Extraction **************************************
def get_label_with_max_conf(labels):
    max_lbl, max_conf = "", 0.0
    if labels:
        for label in labels:
            lbl = label.value
            conf = label.score
            if conf > max_conf:
                max_lbl = lbl
                max_conf = conf
    return max_lbl, max_conf


def locate_end(adu, content, end_idx):
    window_len = min(len(adu), 10)
    adu_slice = adu[-window_len:]
    jitter_length = 5
    candidates = list(range(end_idx - jitter_length, end_idx + jitter_length + 1))
    variable_end_idx = end_idx
    while True:
        content_slice = content[variable_end_idx - window_len:variable_end_idx]
        if adu_slice == content_slice:
            return variable_end_idx
        if not candidates:
            return None
        variable_end_idx = candidates.pop(0)


def find_segment_in_text(segment, sentence):
    segment_tokens = segment["tokens"]
    start_idx, end_idx = -1, -1
    first_token = segment_tokens[0]
    last_token = segment_tokens[-1]
    for token in sentence.tokens:
        if first_token == token.text:
            start_idx = token.start
        elif start_idx != -1 and last_token == token.text:
            end_idx = token.end
            break
    if start_idx < sentence.start_idx:
        start_idx = sentence.start_idx
    if end_idx > sentence.end_idx or end_idx < sentence.start_idx:
        end_idx = sentence.end_idx
    return start_idx, end_idx


def get_args_from_sentence(sentence):
    if sentence.tokens:
        segments = []
        idx = None
        while True:
            segment, idx = get_next_segment(sentence.tokens, current_idx=idx)
            if segment:
                segment["mean_conf"] = np.mean(segment["confidences"])
                segments.append(segment)
            if idx is None:
                break
        return segments


def get_next_segment(tokens, current_idx=None, current_label=None, segment=None):
    if current_idx is None:
        current_idx = 0
    if current_idx >= len(tokens):
        return segment, None
    token = tokens[current_idx]
    raw_label = token.get_tag(label_type=None)
    lbl_txt = raw_label.value
    confidence = raw_label.score
    label_parts = lbl_txt.split("-")
    if len(label_parts) > 1:
        label_type, label = label_parts[0], label_parts[1]
    else:
        label_type, label = None, lbl_txt

    # if we're already tracking a contiguous segment:
    if current_label is not None:
        if label_type == "I" and current_label == label:
            # append to the running collections
            segment["text"] += " " + token.text
            segment["confidences"].append(confidence)
            segment["tokens"].append(token)
            return get_next_segment(tokens, current_idx + 1, current_label, segment)
        else:
            # new segment, different than the current one
            # next function call should start at the current_idx
            return segment, current_idx
    else:
        # only care about B-tags to start a segment
        if label_type == "B":
            segment = {"text": token.text, "label": label, "tokens": [token], "confidences": [confidence]}
            return get_next_segment(tokens, current_idx + 1, label, segment)
        else:
            return get_next_segment(tokens, current_idx + 1, None, segment)


def get_adus(segments):
    major_claims = [(segment["segment"], segment["id"]) for segment in segments if segment["type"] == "major_claim"]
    claims = [(segment["segment"], segment["id"]) for segment in segments if segment["type"] == "claim"]
    premises = [(segment["segment"], segment["id"]) for segment in segments if segment["type"] == "premise"]
    return major_claims, claims, premises


# *************************************** Clustering *******************************************
def collect_adu_for_clustering(documents, document_ids):
    adus, adu_ids, doc_ids = [], [], []
    for document in documents:
        if document["id"] in document_ids:
            for adu in document["annotations"]["ADUs"]:
                if adu["type"] == "claim":
                    adus.append(adu["segment"])
                    adu_ids.append(adu["id"])
                    doc_ids.append(document["id"])
    return adus, doc_ids, adu_ids


def preprocess_sentences(sentences, top_n_words=100):
    preprocessed_sentences = []
    greek_stopwords = get_greek_stopwords()
    tf_idf_vectorizer = TfidfVectorizer(stop_words=greek_stopwords)
    res = tf_idf_vectorizer.fit_transform(sentences)
    vocab = tf_idf_vectorizer.vocabulary_
    word_weights = zip(res.indices, res.data)
    word_weights = sorted(word_weights, key=lambda k: k[1], reverse=True)
    keep_words = word_weights[:top_n_words]
    keep_vocab = []
    for weight_tuple in keep_words:
        word_index, word_weight = weight_tuple
        for word, index in vocab.items():
            if index == word_index:
                keep_vocab.append(word)
    for sentence in sentences:
        tokens = tokenize(sentence)
        tokens = [token for token in tokens if token in keep_vocab]
        sentence = " ".join(tokens)
        sentence = replace_multiple_spaces_with_single_space(text=sentence)
        preprocessed_sentences.append(sentence)
    return preprocessed_sentences


def visualize_topics(cluster, embeddings):
    # Prepare data
    result = pd.DataFrame(embeddings, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    # fig, ax = plt.subplots(figsize=(20, 10))
    plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()


def visualize_embeddings(method, embeddings):
    if method == "tnse":
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
    elif method == "pca":
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