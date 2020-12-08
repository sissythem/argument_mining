import pickle
from os.path import join

import pandas as pd


def adu_preprocess(app_config):
    logger = app_config.app_logger
    logger.debug("Running ADU preprocessing")
    resources = app_config.resources_path
    documents_path = join(resources, app_config.documents_pickle)
    logger.debug("Loading documents from pickle file")
    with open(documents_path, "rb") as f:
        documents = pickle.load(f)
    logger.debug("Documents are loaded")
    df = pd.DataFrame()
    row_counter = 0
    sentence_counter = 0
    for document in documents:
        logger.debug("Processing document with id: {}".format(document.document_id))
        doc_sentence_counter = 0
        for idx, sentence in enumerate(document.sentences):
            logger.debug("Processing sentence: {}".format(sentence))
            labels = document.sentences_labels[idx]
            for token, label in zip(sentence, labels):
                is_arg = "Y" if label != "O" else "N"
                sp = "SP: {}".format(doc_sentence_counter)
                sentence_counter_str = "Sentence: {}".format(sentence_counter)
                document_str = "Doc: {}".format(document.document_id)
                df.loc[row_counter] = [token, label, is_arg, sp, sentence_counter_str, document_str]
                row_counter += 1
                sentence_counter += 1
                doc_sentence_counter += 1
            df.loc[row_counter] = ["", "", "", "", "", ""]
            row_counter += 1
    logger.debug("Finished building dataframing. Saving...")
    out_file_path = join(resources, "train.csv")
    df.to_csv(out_file_path, sep='\t', index=False, header=None)
    logger.debug("Dataframe saved!")


def preprocess_relations():
    pass
