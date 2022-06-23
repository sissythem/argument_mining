from statistics import mean
from src.utils import document_contained_in
import os
import filelock
import json
import logging


def read_promising_docs_for_annotation(path="promising_docs.json"):
    """
    Read promising documents to manually annotate
    Args:
        path (str): Path to docs
    """
    lock = filelock.FileLock(path + ".lock")
    with lock.acquire():
        try:
            with open(path) as f:
                docs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            docs = []
    return docs


def save_promising_docs_for_annotation(documents, path="promising_docs.json"):
    """
    Identify promising documents to manually annotate
    Args:
        documents (list): List of documents
        path (str): Path to save docs
    """
    if not documents:
        return
    lock = filelock.FileLock(path + ".lock")
    with lock.acquire():
        total = documents
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
            new_docs = [doc for doc in documents if not document_contained_in(doc, existing, use_version=True)]
            logging.info(f"Read {len(existing)} existing promising docs, added {len(new_docs)} new ones.")
            total = existing + new_docs
        with open(path, "w") as f:
            logging.info(f"Writing a total of {len(total)} existing promising docs.")
            json.dump(total, f)


def empty_promising_docs_for_annotation(path="promising_docs.json", docs_to_delete=None):
    """
    Identify promising documents to manually annotate
    Args:
        path (str): Path to docs
    """
    lock = filelock.FileLock(path + ".lock")
    with lock.acquire():
        if docs_to_delete is None:
            del_func = lambda x: True
        else:
            del_func = lambda x: document_contained_in(x, docs_to_delete)
        with open(path) as f:
            docs = json.load(f)
        docs_keep = [doc for doc in docs if not del_func(doc)]
        logging.info(
            f"Will keep {len(docs_keep)} / {len(docs)} of promising docs, after invoking deletion with input: {len(docs_to_delete) if docs_to_delete is not None else '<none>'}")

        with open(path, "w") as f:
            json.dump(docs_keep, f)


def identify_promising_docs_for_annotation(documents, how="confidence", confidence_threshold=0.95):
    """
    Identify promising documents to manually annotate
    Args:
        documents (list): List of documents
        how (str): How to estimate promise.
            'confidence': Use confidence scores

    Returns:
        promising (list): List of promising documents for manual annotation
    """
    promising = []
    if how == "confidence":
        for doc in documents:
            try:
                confs = []
                for adu in doc['annotations']['ADUs']:
                    confs.append(float(adu["confidence"]))
                if confs:
                    doc_confidence = mean(confs)
                    if doc_confidence > confidence_threshold:
                        logging.debug(
                            f"Marking promising document ({doc_confidence} > {confidence_threshold}): [{doc['title']}] --  {doc['link']}")
                        promising.append(doc)
            except KeyError:
                pass
    return promising
