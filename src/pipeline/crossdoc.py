import time
import requests
import json
import os
import logging
from src.utils import DocumentJSONEncoder, document_contained_in
import filelock


def empty_additional_cross_doc_inputs(path):
    lock = filelock.FileLock(path + ".lock")
    with lock.acquire():
        os.remove(path)


def read_additional_cross_doc_inputs(path, max_wait=15):
    lock = filelock.FileLock(path + ".lock")
    with lock.acquire(timeout=max_wait):
        # read contents
        with open(path) as f:
            existing_docs = json.load(f)
        # flush them
        with open(path, "w") as f:
            json.dump([], f)
    return existing_docs


def post_additional_cross_doc_inputs(url, documents, token):
    response = requests.post(url, json=documents, params={"access_token": token})
    logging.info(f"POSTed additional docs for cross-doc relation extraction, response: {response}")


def write_additional_cross_doc_inputs(path, documents, max_wait=15):
    docs_to_write = [x for x in documents]
    lock = filelock.FileLock(path + ".lock")
    with lock.acquire(timeout=max_wait):
        if os.path.exists(path):
            # append
            with open(path) as f:
                existing_docs = json.load(f)
                other_docs = [x for x in documents if not document_contained_in(x, existing_docs, use_version=True)]
                logging.info(
                    f"Adding {len(other_docs)} new docs to existing {len(existing_docs)}, for future cross-doc relation discovery.")
                # only retain documents not already in the current collection, with respect to id
                docs_to_write += other_docs
        with open(path, "w") as f:
            logging.info(
                f"Dumping a total of {len(documents)} documents to {path} for future cross-doc clustering")
            json.dump(docs_to_write, f, cls=DocumentJSONEncoder)
    return docs_to_write
