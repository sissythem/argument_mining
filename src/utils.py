import random

from os import listdir, makedirs
from os.path import join, isdir, dirname
import json
from itertools import product
import logging
import unicodedata
import threading
import difflib
import datetime
from uuid import uuid4
from transformers import logging as transformers_logging

from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.tokenize.util import string_span_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
import numpy as np
import torch
from filelock import FileLock, Timeout
from ellogon import tokeniser

MODEL_VERSION = "0.2"


class ObjectLockContext(object):
    """
    Object-locking context manager
    """

    def __init__(self, lock=None):
        if lock is None:
            lock = threading.Lock()
        self.lock = lock

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.lock.release()


class LockContext(object):
    """
    File-locking context manager
    """

    def __init__(self, path, timeout=None):
        self.path = path
        self.timeout = timeout
        self.lock = FileLock(path + ".lock")

    def __enter__(self):
        try:
            self.lock.acquire(self.timeout)
            return self
        except Timeout as tm:
            return ValueError(f"Unable to initiate training: {str(tm)}")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.lock.release()


def make_segment_id():
    """ID generator"""
    return str(uuid4())


def get_id_counter(ids_string_list: list):
    """

    :param ids_string_list: List of ids in the form <string><number>
    :return: the id count
    """
    int_ids = [int(x[1:]) for x in ids_string_list]
    num_ids = max(int_ids)
    assert len(ids_string_list) == num_ids, f"Unexpected max id count: {num_ids} for ids {sorted(ids_string_list)}"
    return num_ids


def skip_content_insertion(key: str, document: str, can_overwrite: bool = True, id_getter=None):
    """
    Utility to check skipping on document containers
    :param key:
    :param container:
    :param can_overwrite:
    :param id_getter:
    :return:
    """
    if id_getter is None:
        id_getter = lambda x: x['id']
    if key in document and document[key]:
        if can_overwrite:
            logging.info(f"Overwriting existing [{key}] items for document {id_getter(document)}")
            return False
        else:
            logging.info(f"Skipping [{key}] extraction due to existing items for document {id_getter(document)}")
            return True


def get_greek_stopwords():
    return tokeniser.stop_words()


class DocumentJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, np.ndarray):
                lobj = obj.tolist()
                if isinstance(lobj[0], float):
                    lobj = [str(x) for x in lobj]
                return lobj
            # number
            if type(obj) in (float, np.float32):
                return str(obj)
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)


def read_json_str_or_file(inp):
    try:
        return json.loads(inp)
    except Exception:
        with open(inp) as f:
            return json.load(f)


def set_seed(seed, n_gpu=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def document_contained_in(document: dict, existing: list, use_version: bool = False):
    """
    Checks if a document is in a collection based on document ID
    Args:
        document: The document
        existing: List of existing documents
        use_version: Whether to check the model version as well

    Returns:

    """
    ids = [doc["id"] for doc in existing]
    if use_version:
        ids = zip(ids, [doc["model_version"] for doc in existing])
    docid = document["id"] if not use_version else (document["id"], document["model_version"])
    return docid in ids


def timestamp():
    return datetime.datetime.now().strftime('%m%d%Y_%H%M%S')


def strip_accents_and_lowercase(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn').lower()


def normalize_newlines(text):
    return "\n".join(text.splitlines())


def preprocess_text(text):
    text = text.replace("\u200b", " ")
    return text


def tokens_to_text(tokens, how="ellogon"):
    if how == "ellogon":
        return expanded_tokens_to_text(tokens)


def remove_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def postprocess_tokens(toks, blacklist_types=None, blacklist_tokens=None):
    if blacklist_types is None:
        blacklist_types = ['UNICODE_PRIVATE_USE']
    if blacklist_tokens is None:
        blacklist_tokens = ['\xad']
    res = []
    for s, sent in enumerate(toks):
        sent_tokens, s, e = sent
        # drop type and apply blacklist
        sent_tokens = [s for s in sent_tokens if
                       s[1] not in blacklist_types and s[0] not in blacklist_tokens]
        # only keep non-empty sentences
        if sent_tokens:
            res.append([sent_tokens, s, e])
    return res


def tokenize_with_spans(text, how="ellogon"):
    if how == "ellogon":
        # text = preprocess_text(text)
        toks_raw = tokeniser.tokenise_type_spans(text)
        # everything to list, cast to string
        toks_raw = [[[[str(t[0])] + list(t[1:]) for t in tups], int(x), int(y)] for (tups, x, y) in list(toks_raw)]
        verify_tokenization(toks_raw, text)
        toks_raw = postprocess_tokens(toks_raw)
        toks_fixed = []
        curr_idx = 0
        for sent in toks_raw:
            tk = inject_missing_gaps(sent, starting_idx=curr_idx, reference_text=text)
            toks_fixed.append(tk)
            curr_idx = tk[-1]
        return toks_fixed, toks_raw

    elif how == "nltk":
        sentok = PunktSentenceTokenizer()
        sents = list(sentok.span_tokenize(text))
        res = []
        for s in sents:
            s1, s2 = s
            sent = text[s1: s2]
            word_spans = list(twt().span_tokenize(sent))
            words = [(sent[x:y], "token_type", s1 + x, s1 + y) for (x, y) in word_spans]
            res.append((words, s1, s2))

        res = postprocess_tokens(res)
        verify_tokenization(res, text)

        expanded = []
        curr_idx = 0
        for t in res:
            tk = inject_missing_gaps(t, starting_idx=curr_idx, reference_text=text)
            expanded.append(tk)
            curr_idx = tk[-1]
        return expanded, res


def propagate_offset(text, tokens, sent_idx, tok_idx, offset):
    # for each relevant sentence
    for s in range(sent_idx, len(tokens)):
        start_t = tok_idx if sent_idx == s else 0
        sent_tokens = tokens[s]
        if start_t == 0:
            tokens[sent_idx][-1] += offset
        tokens[sent_idx][-2] += offset
        for t in range(start_t, len(sent_tokens[0])):
            sent_tokens[0][t][-1] += offset
            sent_tokens[0][t][-2] += offset


def verify_tokenization(toks, text):
    # verify offsets are correct
    for sent_idx, sentence_info in enumerate(toks):
        sent_tokens, sent_start, sent_end = sentence_info
        for tok_idx, tok in enumerate(sent_tokens):
            # type checks
            txt, ttype, s, e = tok
            sent_tokens[tok_idx] = [str(txt), str(ttype), int(s), int(e)]
            tok = sent_tokens[tok_idx]
            txt, ttype, s, e = tok
            txt = str(txt)
            original = text[s:e]
            if original != txt:
                logging.error("TOKENIZATION ERROR: Mismatch in raw tokenization offsets!")
                # just replace the token
                sent_tokens[tok_idx][0] = original
                logging.error(
                    f"Replacing problematic [token]: [{txt}], len {len(txt)} with its spanned source [{original}], len {len(original)}")
                continue
                # sent_tokens[tok_idx] = [original, s, s + len(original)]
                # offset = len(txt) - len(original)
                # if offset != 0:
                #     # modify current token end
                #     # apply offset to all subsequent tokens
                #     propagate_offset(text, toks, sent_idx, tok_idx + 1, offset)


def inject_missing_gaps(token_idx_list, starting_idx=0, reference_text=None):
    """Modify list of tokens so that all indices are consequtive

    Args:
        token_idx_list (list): List of tuples, each element is
        starting_idx (int): Initial index
    """

    if reference_text is None:
        reference_text = " " * token_idx_list[-1]

    if type(token_idx_list[0][0]) is not str:
        l = []
        for part in token_idx_list:
            if type(part) is not int:
                # it's a tuple of tokens
                part = inject_missing_gaps(
                    part, starting_idx=starting_idx, reference_text=reference_text)
            else:
                pass
            l.append(part)
        # update starting index to the modified tuple
        l[1] = l[0][0][-2]
        return tuple(l)
    else:
        res = []
        current = starting_idx
        # we are at a lowest-level tuple
        for part_idx, part in enumerate(token_idx_list):
            txt, ttype, start, end = part
            # print(txt, start, end, "---", "[" + reference_text[start:end] + "]")
            diff = start - current
            if diff != 0:
                whitespace_slice = reference_text[current:current + diff]
                if len(whitespace_slice.strip()) > 0:
                    logging.debug(
                        f"Non-empty whitespace in tokenization-injection: [{whitespace_slice}]!")
                res.append([whitespace_slice, "injected_whitespace", current, start])
            res.append(part)
            current = end
        return tuple(res)


def tokenize(text, punct=True):
    return list(tokeniser.tokenise_no_punc(text)) if not punct else list(tokeniser.tokenise(text))


def expanded_tokens_to_text(expanded_tokens):
    tokens_text = get_sentence_raw_tokens(expanded_tokens)
    txt = join_sentence_tokens(tokens_text)
    return txt


def suspend_logging_verbosity():
    # This is used for disabling warnings for unused weights in model checkpoint
    transformers_logging.set_verbosity_error()
    logging.getLogger().setLevel(logging.ERROR)


def get_sentence_raw_tokens(expanded_tokens):
    tokens_tuples, _, _ = expanded_tokens
    tokens_text = [x[0] for x in tokens_tuples]
    return tokens_text


def join_sentence_tokens(toks):
    return "".join(toks)


def join_sentences(tokenized_sentences):
    """
    Function to create a correct string (punctuation in the correct position - correct spaces)

    Args
        | tokenized_sentences (list): a list of sentences. Each sentence is a tuple with the respective tokens

    Returns
        | list: a list of strings (i.e. the sentences)
    """
    sentences = []
    for sentence in tokenized_sentences:
        sentence = expanded_tokens_to_text(sentence)
        # sentence = join_sentence(sentence=sentence)
        sentences.append(sentence)
    return sentences


def align_expanded_tokens(tokens, expanded_tokens, best_effort=False):
    """ align token sequence with the expanded token sequence, to have a perfect match for reconstructed text

    Args:
        tokens (list of strings): Smaller sequence to match
        expanded_tokens (list of strings): Reference sequence to match onto. Has to be >= than the tokens.

    Returns:
        [type]: Tuple of start and end index (inclusive), so that tokens == expanded_tokens[start: end+1]
    """

    # match the start and end
    start = [i for i, tok in enumerate(expanded_tokens) if tok == tokens[0]]
    end = [i for i, tok in enumerate(expanded_tokens) if tok == tokens[-1]]

    # handle singletons

    orig_combos = list(product(start, end))

    # start <= end
    combos = [(s, e) for (s, e) in orig_combos if s <= e]

    # expaned seqlen >= original seqlen
    if len(tokens) > 1:
        combos = [(s, e) for (s, e) in combos if e - s + 1 >= len(tokens)]

    if len(combos) > 1:
        # get closest match
        candidates = [expanded_tokens[i:j + 1] for (i, j) in combos]
        match = difflib.get_close_matches(tokens, candidates, n=1)[0]
        ix = candidates.index(match)
        combos = [combos[ix]]

    if len(combos) != 1:
        raise ValueError(
            "Ambiguous / missing token alignment to expanded collection!")
    start, end = combos[0]
    return start, end


def collect_adu_types(adus):
    mcs = [x for x in adus if x['type'] == "major_claim"]
    claims = [x for x in adus if x['type'] == "claim"]
    premises = [x for x in adus if x['type'] == "premise"]
    return mcs, claims, premises


def strip_document_extra_info(doc_json, keys_to_keep=None):
    """
    Remove superfluous annotation keys from a document dict
    Args:
        doc_json: Document dict
        keys_to_keep: Key-values to retain

    Returns:
        The updated dict
    """
    if keys_to_keep is None:
        keys_to_keep = ("starts", "ends", "segment", "id", "confidence", "type", "stance")
    annotations = doc_json["annotations"]["ADUs"]
    for i, annot in enumerate(annotations):
        annotations[i] = {k: v for (k, v) in annot.items() if k in keys_to_keep}
    return doc_json


def listfolders(path: str):
    """
    Get list of subfolders from a given path
    """
    subpaths = [join(path, x) for x in listdir(path)]
    return [x for x in subpaths if isdir(x)]


def lock_write_file(path: str, message: str):
    """
    Lock-safe write message to file
    """
    makedirs(dirname(path), exist_ok=True)
    with LockContext(path):
        with open(path, "w") as f:
            f.write(message)


def lock_read_file(path: str):
    """
    Lock-safe write message to file
    """
    with LockContext(path):
        with open(path, "r") as f:
            msg = f.read()
    return msg
